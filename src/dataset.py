"""알약 검출용 PyTorch Dataset."""

import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class PillDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        # 이미지 파일 전체 경로 수집 (하위 폴더 포함해서 탐색)
        self.img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(root, f))

        # 어노테이션 json 파일을 base_name 기준 매핑 (하위 폴더 포함)
        self.ann_map = {}
        for root, _, files in os.walk(self.ann_dir):
            for f in files:
                if f.endswith('.json'):
                    base_name = os.path.splitext(f)[0]
                    self.ann_map[base_name] = os.path.join(root, f)

        # category_id는 듬성듬성한 값이라 1부터 시작하는 라벨로 재매핑
        self.category_id_to_label, self.label_to_name = self._build_category_mapping()

    def _build_category_mapping(self):
        category_ids = set()
        category_names = {}
        for json_path in self.ann_map.values():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for cat in data.get('categories', []):
                category_ids.add(cat['id'])
                category_names[cat['id']] = cat['name']

        id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(category_ids))}
        label_to_name = {idx: category_names[cat_id] for cat_id, idx in id_to_label.items()}
        return id_to_label, label_to_name

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # 이미지 파일명과 같은 base_name의 json 찾기
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = self.ann_map.get(base_name)

        boxes = []
        labels = []

        if json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ann in data.get('annotations', []):
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.category_id_to_label[ann['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def get_train_paths():
    base = RAW_DIR / "sprint_ai_project1_data"
    return base / "train_images", base / "train_annotations"
