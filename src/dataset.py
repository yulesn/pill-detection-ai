"""알약 검출용 PyTorch Dataset."""

import json
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
            if annotation_file.stem in excluded_names:  # ← 추가된 부분
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
    """

    def __init__(self, data_dir: str, train: bool, transform=DEFAULT_TRANSFORM):
        self.transform = transform
        self.train = "train" if train else "test"
        data_dir = Path(data_dir)
        self.image_path = data_dir / f"{self.train}_images"
        self.categories = {}
        if train:
            annotation_path = data_dir / f"{self.train}_annotations"
            excluded_names = load_excluded_names()  # ← 추가된 부분
            self.coco = CustomCOCO(annotation_path, excluded_names=excluded_names)
            for cat_id, cat in self.coco.categories.items():
                self.categories[cat_id] = cat["name"]
        self.data = self._load_data()

    def _load_data(self):
        """데이터셋의 [(image, target)] list를 로드하는 함수."""
        data = []
        if self.train == "test":
            for image_file in self.image_path.rglob("*.png"):
                image = Image.open(image_file)
                data.append((image, {}))
        elif self.train == "train":
            for img_id, img_info in self.coco.images.items():
                image_file = self.image_path / img_info["file_name"]
                image = Image.open(image_file)
                boxes, labels = [], []
                for ann in self.coco.annotations[img_id]:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann["category_id"])
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