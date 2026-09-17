"""알약 검출용 PyTorch Dataset."""

import os
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


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
            # 카테고리 ID, 이름 수집하는 과정
            for cat in data.get("categories", []):
                category_ids.add(cat["id"])
                category_names[cat["id"]] = cat["name"]
    # 카테고리 ID -> 1부터 시작하는 순차적 라벨로 매핑하는 과정
    id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted(category_ids))}
    label_to_name = {idx: category_names[cat_id] for cat_id, idx in id_to_label.items()}
    return id_to_label, label_to_name


def collate_fn(batch):
    """객체 검출용 배치 묶음 (리스트 형태로 유지)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "train_images")
        self.ann_dir = os.path.join(data_dir, "train_annotations")

        # 전처리 파이프라인 설정
        if transform is not None:
            self.transform = transform

            # 계속해서 데이터 증강 강화시킴 
        else:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 색상 변환 (조명이 어둡거나 밝거나 색감이 다 다르기 때문에 다양한 환경에 적응할 수 있도록)
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 회전시키거나 크기 조절을 무작위로 적용함 (각도가 틀어지거나 크기가 다 달라도 알약을 잘 찾아낼 수 있도록)
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5), # 블러 효과를 줌 (흔들리거나 초적 흐린 악조건 상황에서도 알약 인식 할 수 있도록)
                T.ToTensor(),
            ])

        self.img_paths = []
        for root, _, files in os.walk(self.img_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.img_paths.append(os.path.join(root, f))

        self.ann_map = {}
        for root, _, files in os.walk(self.ann_dir):
            for f in files:
                if f.endswith(".json"):
                    base_name = os.path.splitext(f)[0]
                    self.ann_map.setdefault(base_name, []).append(os.path.join(root, f))

        self.category_id_to_label, self.label_to_name = build_category_mapping(self.ann_dir)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        image = Image.open(img_path).convert("RGB")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_paths = self.ann_map.get(base_name, [])

        # 바운딩 박스, 라벨 추출 과정
        boxes, labels = [], []
        for json_path in json_paths:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ann in data.get("annotations", []):
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h]) # 원본 [x,y,w,h] 형태 -> 모델이 학습할 수 있는 시작점과 끝점 형태로 변환
                labels.append(self.category_id_to_label[ann["category_id"]])

        # 알약 탐지되지 않는 경우를 대비해서 torch.zeros로 처리해서 에러 방지함 
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        # 좌표, 라벨, 이미지 아이디 -> 딕셔너리로 묶은 후 전처리 거쳐서 반환
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }

        # transform이 존재할 경우 적용
        if self.transform is not None:
            image = self.transform(image)
        return image, target