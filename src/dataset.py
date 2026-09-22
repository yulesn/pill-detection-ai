"""알약 검출용 PyTorch Dataset."""

import os
import json
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def build_category_mapping(ann_dir):
    """ann_dir 안의 모든 JSON에서 category_id -> label(1부터), label -> name 매핑을 만든다."""
    category_ids = set()
    category_names = {}

    for root, _, files in os.walk(ann_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for cat in data.get("categories", []):
                    category_ids.add(cat["id"])
                    category_names[cat["id"]] = cat["name"]

    id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(category_ids))}
    label_to_name = {idx: category_names[cat_id] for cat_id, idx in id_to_label.items()}
    return id_to_label, label_to_name


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    한 이미지에 최대 4개의 알약이 있으므로, 이미지당 최대 4개의
    (label, bbox) 쌍을 반환하도록 구현한다.
    """

    def __init__(self, data_dir: str, transform=None, img_size=512, is_train=True):
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train

        target_dir = data_dir
        if not os.path.exists(os.path.join(target_dir, "images")):
            if os.path.exists("data/raw/sprint_ai_project1_data/train_images"):
                target_dir = "data/raw/sprint_ai_project1_data"
            elif os.path.exists("data/processed/images"):
                target_dir = "data/processed"

        self.img_dir = os.path.join(target_dir, "train_images")
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(target_dir, "images")

        self.annot_dir = os.path.join(target_dir, "train_annotations")
        if not os.path.exists(self.annot_dir):
            self.annot_dir = os.path.join(target_dir, "annotations")

        self.category_id_to_label, self.label_to_name = build_category_mapping(self.annot_dir)

        self.samples = []

        img_extensions = ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"]
        img_paths = []
        for ext in img_extensions:
            img_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))

        for img_path in sorted(img_paths):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(self.annot_dir, f'{base_name}.json')

            if not os.path.exists(json_path):
                found = False
                for root, _, files in os.walk(self.annot_dir):
                    if f'{base_name}.json' in files:
                        json_path = os.path.join(root, f'{base_name}.json')
                        found = True
                        break
                if not found:
                    continue

            with open(json_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)

            anns = ann_data if isinstance(ann_data, list) else ann_data.get('annotations', [ann_data])
            boxes = []
            labels = []

            for ann in anns:
                bbox = ann.get('bbox', ann.get('bounding_box', None))
                cat_id = ann.get('category_id', ann.get('label', None))
                if bbox and len(bbox) == 4 and cat_id in self.category_id_to_label:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(self.category_id_to_label[cat_id])

            if boxes:
                self.samples.append({
                    'img_path': img_path,
                    'boxes': boxes,
                    'labels': labels
                })

        if self.is_train and self.samples:
            class_counts = {}
            for sample in self.samples:
                for lbl in sample['labels']:
                    class_counts[lbl] = class_counts.get(lbl, 0) + 1

            if class_counts:
                avg_count = sum(class_counts.values()) / len(class_counts)
                rare_threshold = max(1, int(avg_count * 0.5))
                rare_classes = {cls for cls, cnt in class_counts.items() if cnt < rare_threshold}

                extra_samples = []
                for sample in self.samples:
                    if any(lbl in rare_classes for lbl in sample['labels']):
                        extra_samples.append(sample)
                self.samples.extend(extra_samples)

        if self.transform is None:
            if self.is_train:
                self.transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.HueSaturationValue(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3, border_mode=0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        image_np = np.array(image)
        orig_w, orig_h = image.size

        boxes = []
        for box in sample["boxes"]:
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(x1), float(orig_w)))
            y1 = max(0.0, min(float(y1), float(orig_h)))
            x2 = max(0.0, min(float(x2), float(orig_w)))
            y2 = max(0.0, min(float(y2), float(orig_h)))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

        labels = sample['labels']

        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]

        transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
        image_tensor = transformed['image']

        t_boxes = transformed['bboxes']
        t_labels = transformed['labels']

        if not t_boxes:
            t_boxes = [[0.0, 0.0, 1.0, 1.0]]
            t_labels = [1]

        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h

        final_boxes = []
        for box in t_boxes:
            bx1, by1, bx2, by2 = box
            bx1 = max(0.0, min(bx1, float(self.img_size)))
            by1 = max(0.0, min(by1, float(self.img_size)))
            bx2 = max(0.0, min(bx2, float(self.img_size)))
            by2 = max(0.0, min(by2, float(self.img_size)))
            if bx2 <= bx1: bx2 = min(float(self.img_size), bx1 + 1.0)
            if by2 <= by1: by2 = min(float(self.img_size), by1 + 1.0)
            final_boxes.append([bx1, by1, bx2, by2])

        target = {
            'boxes': torch.tensor(final_boxes, dtype=torch.float32),
            'labels': torch.tensor(t_labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target


# 이미지와 JSON 라벨 파일을 누락없이 찾아내도록 매칭 로직 만듦 ---> 기존 코드는 경로가 조금만 어긋나도 파일 못찾거나 에러남, 알약의 원본 카테고리 ID가 뒤죽박죽 일 수 있어서 모델이 학습하기 좋은 순차적 레이블 부여하며 매핑함
# 알부멘테이션 증강 도입(바운딩박스까지 동시에 변환해줌) ---> 좌우반전, 밝기 및 대비 조절, 이동 및 확대 축소와 회전 즉 이미지 증강 도입
# 512x512 고정 해상도 및 박스 좌표 안정화 ---> 객체 검출 모델은 박스 좌표가 이미지 범위 벗어나면 치명적인 런타임에러 발생함. 이를 사전 차단을 해서 학습 중 멈추는 현상을 원천 차단함

# 클래스별 개수 파악 ---> 전체 데이터셋을 훑어보면서 각 알약 종류 즉, 클래스별로 데이터가 몇개씩 있는지 카운팅함
# 부족한 데이터 판별 ---> 평균 개수보다 턱없이 부족하게 (절반 미만으로) 들어있는 알약 클래스 찾아냄
# 데이터 복제 및 추가 (오버샘플링) ---> 그 부족한 알약이 포함된 이미지만 골라서 self.samples 리스트에 한 번 더 복제해줌
# 오버셈플링을 이렇게 한 이유는 어떤 알약은 사진이 엄청 많고 어떤 알약은 몇 장 없어서 모델이 흔한 알약만 잘 맞추고 희귀한 알약은 아예 못맞추는 클래스 불균형이 생기기 마련임
# 오버샘플링 코드는 부족한 알약 데이터를 더 자주 보게 만들어줌으로써 모델이 모든 알약을 골고루 잘 찾도록 균형 잡아주는 역할을 함 