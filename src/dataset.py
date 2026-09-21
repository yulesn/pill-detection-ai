"""알약 검출용 PyTorch Dataset."""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

# transform=None일 때 최소한 텐서 변환은 되도록 기본값 제공
DEFAULT_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# EDA에서 찾은 라벨 누락 의심 이미지 목록 (select_valid_images.py가 생성)
EXCLUDED_IMAGES_PATH = Path("data/processed/excluded_images.json")


def load_excluded_names() -> set:
    """제외 대상 이미지 이름 목록을 읽는다. 파일이 없으면 아무것도 제외하지 않는다."""
    if not EXCLUDED_IMAGES_PATH.exists():
        return set()
    with open(EXCLUDED_IMAGES_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))


class CustomCOCO:
    def __init__(self, annotation_dir, excluded_names: set = None):
        self.images = {}
        self.annotations = {}
        self.categories = {}
        excluded_names = excluded_names or set()

        for annotation_file in Path(annotation_dir).rglob("*.json"):
            if annotation_file.stem in excluded_names:
                continue

            with open(annotation_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                image = content["images"][0]
                image_id = image["id"]
                annotation = content["annotations"][0]
                category = content["categories"][0]
                category_id = category["id"]
                if image_id not in self.images:
                    self.images[image_id] = image
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append(annotation)
                if category_id not in self.categories:
                    self.categories[category_id] = category

    def loadImgs(self, ids):
        return [self.images[i] for i in ids if i in self.images]

    def getAnnIds(self, imgIds):
        ann_ids = []
        for img_id in imgIds:
            if img_id in self.annotations:
                ann_ids.extend([ann["id"] for ann in self.annotations[img_id]])
        return ann_ids


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    __getitem__은 (image, target)을 반환한다.
    - image: tv_tensors.Image, shape (3, H, W), float32, [0, 1]
    - target (train): {"image_id": LongTensor(1,), "boxes": BoundingBoxes(N, 4) XYXY,
      "labels": LongTensor(N,)} — N은 이미지당 알약 개수(가변)
    - target (test): {} (라벨 없음)

    train=True일 때, EDA에서 확인된 라벨 누락 의심 이미지
    (data/processed/excluded_images.json)는 자동으로 제외된다.

    id(dl_idx, 원본 알약 코드)는 값의 범위가 넓어 분류기 클래스 수와
    맞지 않으므로, 0(배경)을 제외한 1~N 사이 연속 번호(label)로 재매핑하여 사용한다.
    raw_id_to_label: 원본 id -> 재매핑된 label
    label_to_raw_id: 재매핑된 label -> 원본 id (예측 결과 해석 시 사용)

    subset: train=True일 때만 의미 있음.
        None(기본값) - 전체 사용 (분리 없음)
        "train" - 학습용 부분만 (val_ratio만큼 제외)
        "val"   - 검증용 부분만 (val_ratio만큼만)
        같은 val_ratio/split_seed를 쓰면 "train"과 "val"은 서로 겹치지 않음.
    """

    def __init__(
        self,
        data_dir: str,
        train: bool,
        transform=DEFAULT_TRANSFORM,
        subset: str = None,
        val_ratio: float = 0.1,
        split_seed: int = 42,
    ):
        self.transform = transform
        self.train = "train" if train else "test"
        self.subset = subset
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        data_dir = Path(data_dir)
        self.image_path = data_dir / f"{self.train}_images"
        self.categories = {}
        if train:
            annotation_path = data_dir / f"{self.train}_annotations"
            excluded_names = load_excluded_names()
            self.coco = CustomCOCO(annotation_path, excluded_names=excluded_names)
            for cat_id, cat in self.coco.categories.items():
                self.categories[cat_id] = cat["name"]

            sorted_ids = sorted(self.categories.keys())
            self.raw_id_to_label = {raw_id: i + 1 for i, raw_id in enumerate(sorted_ids)}
            self.label_to_raw_id = {v: k for k, v in self.raw_id_to_label.items()}

        self.data = self._load_data()

    def _split_img_ids(self, img_ids):
        """subset 설정에 따라 img_id 목록을 학습/검증용으로 나눈다.
        같은 알약 조합(각도만 다른 사진)이 train/val에 걸쳐 섞이지 않도록,
        조합 단위(파일명에서 각도 이전 부분)로 묶어서 나눈다."""
        if self.subset is None:
            return img_ids

        def combo_key(img_id):
            file_name = self.coco.images[img_id]["file_name"]
            return file_name.split("_0_2_0_2_")[0]

        combos = {}
        for img_id in img_ids:
            combos.setdefault(combo_key(img_id), []).append(img_id)

        combo_keys = sorted(combos.keys())
        random.Random(self.split_seed).shuffle(combo_keys)
        n_val_combos = max(1, int(len(combo_keys) * self.val_ratio))
        val_combo_keys = set(combo_keys[:n_val_combos])

        val_ids = {img_id for k in val_combo_keys for img_id in combos[k]}

        if self.subset == "val":
            return [i for i in img_ids if i in val_ids]
        elif self.subset == "train":
            return [i for i in img_ids if i not in val_ids]
        else:
            raise ValueError(f"subset은 None/'train'/'val' 중 하나여야 합니다: {self.subset}")

    def _load_data(self):
        """데이터셋의 [(image, target)] list를 로드하는 함수."""
        data = []
        if self.train == "test":
            for image_file in self.image_path.rglob("*.png"):
                image = Image.open(image_file)
                data.append((image, {}))
        elif self.train == "train":
            img_ids = self._split_img_ids(list(self.coco.images.keys()))
            for img_id in img_ids:
                img_info = self.coco.images[img_id]
                image_file = self.image_path / img_info["file_name"]
                image = Image.open(image_file)
                boxes, labels = [], []
                for ann in self.coco.annotations[img_id]:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self.raw_id_to_label[ann["category_id"]])
                target = {
                    "image_id": torch.LongTensor([img_id]),
                    "boxes": torch.FloatTensor(boxes),
                    "labels": torch.LongTensor(labels),
                }
                data.append((image, target))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        image, target = self.data[index]
        if self.train == "train":
            img_w, img_h = image.size
            target["boxes"] = tv_tensors.BoundingBoxes(
                target["boxes"], format="XYXY", canvas_size=(img_h, img_w)
            )
            image, target = self.transform(image, target)
        else:
            image = self.transform(image)
        return image, target