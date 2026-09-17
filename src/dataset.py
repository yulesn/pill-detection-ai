"""알약 검출용 PyTorch Dataset."""

import json
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

# transform=None일 때 최소한 텐서 변환은 되도록 기본값 제공
DEFAULT_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

class CustomCOCO:
    def __init__(self, annotation_dir):
        self.images = {}
        self.annotations = {}
        self.categories = {}
        for annotation_file in Path(annotation_dir).rglob("*.json"):
            with open(annotation_file, "r") as f:
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


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    __getitem__은 (image, target)을 반환한다.
    - image: tv_tensors.Image, shape (3, H, W), float32, [0, 1]
    - target (train): {"image_id": LongTensor(1,), "boxes": BoundingBoxes(N, 4) XYXY,
      "labels": LongTensor(N,) 0-index 클래스 번호} — N은 이미지당 알약 개수(가변)
    - target (test): {} (라벨 없음)

    라벨 인덱스 ↔ 원본 category_id 매핑은 self.cat_id_to_label / self.label_to_cat_id
    (train 인스턴스에만 존재)를 사용한다. 제출 등 원본 category_id가 필요한 곳에서는
    train 데이터셋의 label_to_cat_id를 재사용해야 한다 (test 인스턴스는 매핑을 만들 수 없음).
    """

    def __init__(self, data_dir: str, train: bool, transform=DEFAULT_TRANSFORM):
        self.transform = transform
        self.train = "train" if train else "test"
        data_dir = Path(data_dir)
        self.image_path = data_dir / f"{self.train}_images"
        self.categories = {}
        if train:
            annotation_path = data_dir / f"{self.train}_annotations"
            self.coco = CustomCOCO(annotation_path)
            for cat_id, cat in self.coco.categories.items():
                self.categories[cat_id] = cat["name"]
            self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(sorted(self.categories))}
            self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}
        self.data = self._load_data()

    def _load_data(self):
        """데이터셋의 [{image Tensor, bbox list, category list}] list를 로드하는 함수."""
        data = []
        if self.train == "test":
            for image_file in self.image_path.rglob("*.png"):
                image = Image.open(image_file)
                data.append((image, {}))
        elif self.train == "train":
            for img_id, img_info in self.coco.images.items():
                image_file = self.image_path / img_info["file_name"]
                image = Image.open(image_file)
                boxes = []
                labels = []
                for ann in self.coco.annotations[img_id]:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x,y,x+w, y+h])
                    labels.append(self.cat_id_to_label[ann["category_id"]])
                target = {
                    "image_id": torch.LongTensor([img_id]),
                    "boxes": torch.FloatTensor(boxes),
                    "labels": torch.LongTensor(labels)
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
