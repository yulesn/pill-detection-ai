"""알약 검출용 PyTorch Dataset."""

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def build_category_mapping(ann_dir):
    """
    ann_dir 안의 모든 JSON에서 실제 약품 코드(dl_mapping_code)를 수집하여
    RetinaNet 학습에 사용할 연속적인 label 번호로 변환한다.

    예:
        K-003483 -> 1
        K-025367 -> 2
        K-027733 -> 3
        ...

    현재 데이터에는 총 56개의 약품 class가 존재한다.
    """
    drug_codes = set()
    drug_names = {}

    for root, _, files in os.walk(ann_dir):
        for f in files:
            if not f.endswith(".json"):
                continue

            try:
                with open(
                    os.path.join(root, f),
                    "r",
                    encoding="utf-8"
                ) as fp:
                    data = json.load(fp)

                images = data.get("images", [])

                if not images:
                    continue

                image_info = images[0]

                # 실제 약품 종류는 categories가 아니라 images의 dl_mapping_code를 기준으로 구분함.
                drug_code = image_info.get("dl_mapping_code")
                drug_name = image_info.get("dl_name")

                if drug_code:
                    drug_codes.add(drug_code)

                    if drug_name:
                        drug_names[drug_code] = drug_name

            except (json.JSONDecodeError, OSError):
                continue

    # 약품 코드를 정렬한 뒤 1부터 연속적인 label 부여
    code_to_label = {
        code: idx + 1
        for idx, code in enumerate(sorted(drug_codes))
    }

    # label -> 약품 이름 매핑
    label_to_name = {
        label: drug_names.get(code, code)
        for code, label in code_to_label.items()
    }

    return code_to_label, label_to_name


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    한 이미지에 여러 개의 알약이 있을 수 있으므로,
    이미지에 포함된 모든 (label, bbox) 쌍을 반환하도록 구현한다.

    현재 데이터에서는 이미지당 2~4개의 알약이 존재한다.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        img_size=512,
        is_train=True
    ):
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train

        # 폴더를 따로 나누지않고 data/processed 안에 다 넣었어서 경로를 하나로 통일함
        self.img_dir = data_dir
        self.annot_dir = data_dir

        # 실제 약품 코드 기준으로 class mapping 생성
        self.category_id_to_label, self.label_to_name = (
            build_category_mapping(self.annot_dir)
        )

        self.samples = []

        ##### 이미지 파일 목록 수집 과정
        img__extensions = [
            "*.jpg",
            "*.png",
            "*.jpeg",
            "*.JPG",
            "*.PNG"
        ]

        img_paths = []

        for ext in img__extensions:
            img_paths.extend(
                glob.glob(
                    os.path.join(self.img_dir, ext)
                )
            )

        # JSON 파일을 file_name 기준으로 그룹화
        # 하나의 실제 이미지에는 여러 json이 존재함.
        # 따라서 json 하나만 선택하지 않고 같은 file_name을 가진 모든 json을 묶음

        json_groups = defaultdict(list)

        json_paths = glob.glob(
            os.path.join(self.annot_dir, "*.json")
        )

        for json_path in json_paths:
            try:
                with open(
                    json_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    ann_data = json.load(f)

                images = ann_data.get("images", [])

                if not images:
                    continue

                file_name = images[0].get("file_name")

                if file_name:
                    json_groups[file_name].append(
                        ann_data
                    )

            except (json.JSONDecodeError, OSError):
                continue

        # 이미지와 JSON 매칭

        for img_path in sorted(img_paths):

            img_filename = os.path.basename(img_path)

            # train_images_ 접두사를 제거하여
            # JSON의 file_name과 동일한 형태로 만든다.
            core_name = img_filename.replace(
                "train_images_",
                "",
                1
            )

            json_files_for_image = json_groups.get(
                core_name,
                []
            )

            # 해당 이미지와 연결된 JSON이 없다면 건너뜀
            if not json_files_for_image:
                continue

            boxes = []
            labels = []

            # 같은 이미지에 연결된 모든 JSON 처리

            for ann_data in json_files_for_image:

                images = ann_data.get("images", [])

                if not images:
                    continue

                image_info = images[0]

                # 실제 약품 class
                drug_code = image_info.get(
                    "dl_mapping_code"
                )

                if not drug_code:
                    continue

                if drug_code not in self.category_id_to_label:
                    continue

                label = self.category_id_to_label[
                    drug_code
                ]

                anns = ann_data.get(
                    "annotations",
                    []
                )

                for ann in anns:

                    bbox = ann.get("bbox")

                    if not bbox or len(bbox) != 4:
                        continue

                    # AI Hub 원본 annotation의 좌표 오류 보정
                    if (
                        drug_code == "K-018357"
                        and bbox == [6567, 625, 311, 315]
                    ):
                        bbox = [657, 625, 311, 315]

                    x, y, w, h = map(
                        float,
                        bbox
                    )

                    if w <= 0 or h <= 0:
                        continue

                    # AI Hub bbox:
                    # [x, y, width, height]
                    # RetinaNet:
                    # [x1, y1, x2, y2]

                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

                    boxes.append(
                        [x1, y1, x2, y2]
                    )

                    labels.append(label)

            # 유효한 bbox가 있는 이미지에 대해서만 sample 생성
            if boxes:
                self.samples.append(
                    {
                        "img_path": img_path,
                        "boxes": boxes,
                        "labels": labels
                    }
                )

        # 데이터 증강 설정
        if self.transform is None:

            if self.is_train:
                self.transform = A.Compose(
                    [
                        A.Resize(
                            self.img_size,
                            self.img_size
                        ),

                        A.HorizontalFlip(
                            p=0.5
                        ),

                        A.RandomBrightnessContrast(
                            p=0.2
                        ),

                        A.HueSaturationValue(
                            p=0.2
                        ),

                        A.ShiftScaleRotate(
                            shift_limit=0.05,
                            scale_limit=0.1,
                            rotate_limit=15,
                            p=0.3,
                            border_mode=cv2.BORDER_CONSTANT
                        ),

                        # RetinaNet 내부에서 ImageNet mean/std
                        # 정규화를 수행하므로 여기서는 Normalize를 사용하지 않음.
                        # ToTensorV2는 이미지 데이터를 float tensor로 변환하기 위해 유지함.
                        
                        A.ToFloat(
                            max_value=255.0
                        ),

                        ToTensorV2()
                    ],
                    bbox_params=A.BboxParams(
                        format="pascal_voc",
                        label_fields=["labels"]
                    )
                )

            else:
                self.transform = A.Compose(
                    [
                        A.Resize(
                        self.img_size,
                        self.img_size
                    ),

                    A.ToFloat(
                        max_value=255.0
                    ),

                    ToTensorV2()
                    ],
                    bbox_params=A.BboxParams(
                        format="pascal_voc",
                        label_fields=["labels"],
                        min_visibility=0.0
                    )
                )

        # Dataset 생성 결과 확인

        print("=" * 60)
        print("PillDataset")
        print("=" * 60)
        print(f"Images       : {len(img_paths)}")
        print(f"JSON files   : {len(json_paths)}")
        print(f"Samples      : {len(self.samples)}")
        print(f"Drug classes : {len(self.category_id_to_label)}")
        print(f"Image size   : {self.img_size}x{self.img_size}")
        print(f"Train mode   : {self.is_train}")
        print("=" * 60)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        # 이미지 읽기

        image = Image.open(
            sample["img_path"]
        ).convert("RGB")

        image_np = np.array(image)

        orig_w, orig_h = image.size

        # bbox 범위를 원본 이미지 크기 안으로 제한

        boxes = []
        labels = []

        # bbox와 label을 같은 순서로 처리하여
        # 유효하지 않은 bbox가 제거될 때 해당 label도 함께 제거함
        for box, label in zip(
            sample["boxes"],
            sample["labels"]
        ):

            x1, y1, x2, y2 = box

            x1 = max(
                0.0,
                min(float(x1), float(orig_w))
            )

            y1 = max(
                0.0,
                min(float(y1), float(orig_h))
            )

            x2 = max(
                0.0,
                min(float(x2), float(orig_w))
            )

            y2 = max(
                0.0,
                min(float(y2), float(orig_h))
            )

            if x2 > x1 and y2 > y1:
                boxes.append(
                    [x1, y1, x2, y2]
                )
                labels.append(label)

        # 이미지와 bbox에 동일한 augmentation 적용

        transformed = self.transform(
            image=image_np,
            bboxes=boxes,
            labels=labels
        )

        image_tensor = transformed["image"]

        t_boxes = transformed["bboxes"]
        t_labels = transformed["labels"]

        # augmentation 이후 bbox가 모두 제거된 경우
        # 가짜 bbox를 만들지 않고 해당 sample을 사용할 수 없도록 오류 처리
        if not t_boxes:
            raise RuntimeError(
                f"augmentation 이후 bbox가 모두 사라졌습니다: "
                f"{sample['img_path']}"
            )

        # 최종 bbox 범위 안정화

        final_boxes = []
        final_labels = []

        # bbox와 label을 함께 확인하여 최종적으로 유효한 객체만 유지함
        for box, label in zip(
            t_boxes,
            t_labels
        ):

            bx1, by1, bx2, by2 = box

            bx1 = max(
                0.0,
                min(
                    float(bx1),
                    float(self.img_size)
                )
            )

            by1 = max(
                0.0,
                min(
                    float(by1),
                    float(self.img_size)
                )
            )

            bx2 = max(
                0.0,
                min(
                    float(bx2),
                    float(self.img_size)
                )
            )

            by2 = max(
                0.0,
                min(
                    float(by2),
                    float(self.img_size)
                )
            )

            if bx2 <= bx1 or by2 <= by1:
                continue

            final_boxes.append(
                [bx1, by1, bx2, by2]
            )

            final_labels.append(
                int(label)
            )

        if not final_boxes:
            raise RuntimeError(
                f"유효한 bbox가 남지 않았습니다: "
                f"{sample['img_path']}"
            )

        # bbox와 label은 같은 객체를 가리켜야 하므로
        # 최종적으로 개수가 일치하는지 확인함
        if len(final_boxes) != len(final_labels):
            raise RuntimeError(
                f"augmentation 이후 bbox와 label 개수가 "
                f"일치하지 않습니다: {sample['img_path']}"
            )

        # RetinaNet target 생성

        target = {
            "boxes": torch.tensor(
                final_boxes,
                dtype=torch.float32
            ),

            "labels": torch.tensor(
                final_labels,
                dtype=torch.int64
            ),

            "image_id": torch.tensor(
                [idx],
                dtype=torch.int64
            )
        }

        return image_tensor, target


# 이미지와 JSON 라벨 파일을 누락 없이 찾아내도록 매칭 로직 만듦
# ---> 기존 코드는 경로가 조금만 어긋나도 파일 못찾거나 에러났음,
#      이 부분을 같은 file_name을 가진 여러 JSON을 하나의 이미지에 연결하도록 수정함

# 알약의 원본 카테고리 ID가 뒤죽박죽 일수도 있어서
# 모델이 학습하기 좋은 순차적 레이블 부여하며 매핑함
# ---> 실제 약품 식별에는 dl_mapping_code를 사용함,
#      이걸 1부터 시작하는 연속적인 label로 변환함

# 알부멘테이션 증강 도입 -> 바운딩박스까지 동시에 변환
# ---> 좌우반전, 밝기 및 대비 조절, 이동 및 확대 축소와 회전 /
#      이미지와 바운딩 박스에 동일한 변환 적용

# 512x512 고정 해상도 및 박스 좌표 안정화
# ---> 객체 검출 모델은 박스 좌표가 이미지 범위를 벗어나면
#      학습 중 오류 발생할 수 있어서 이미지 범위 안으로 제한함


### 현재 데이터는 클래스별 개수 차이 존재하지만 RetinaNet은 Focal Loss 사용해서
### 초기 학습에는 별도의 오버샘플링 없이 실제 데이터 분포 유지함
### 먼저 train/val 성능 확인 후 필요한 경우에는 클래스 불균형 대응 방법을 추가할 예정