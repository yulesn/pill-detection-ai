"""알약 검출용 PyTorch Dataset."""

from PIL import Image
from pathlib import Path
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors

# transform=None일 때 최소한 텐서 변환은 되도록 기본값 제공
DEFAULT_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    train 데이터는 prepare_split.py가 만든 표준 COCO 형식
    (data_dir/images/, data_dir/annotations.json)을 입력으로 받는다.

    __getitem__은 (image, target)을 반환한다.
    - image: tv_tensors.Image, shape (3, H, W), float32, [0, 1]
    - target (train): {"image_id": LongTensor(1,), "boxes": BoundingBoxes(N, 4) XYXY,
      "labels": LongTensor(N,) 0-index 클래스 번호} — N은 이미지당 알약 개수(가변)
    - target (test): {} (라벨 없음)
    """

    def __init__(self, data_dir: str, train: bool, transform=DEFAULT_TRANSFORM):
        self.transform = transform
        self.train = train
        data_dir = Path(data_dir)
        self.categories = {}
        if train:
            self.image_path = data_dir / "images"
            annotation_path = data_dir / "annotations.json"
            self.coco = COCO(str(annotation_path))
            for cat_id, cat in self.coco.cats.items():
                self.categories[cat_id] = cat["name"]
            self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(sorted(self.categories))}
            self.label_to_cat_id = {i: cat_id for cat_id, i in self.cat_id_to_label.items()}
        else:
            self.image_path = data_dir / "test_images"
        self.data = self._load_data()

    def _load_data(self):
        """데이터셋의 [{image Tensor, bbox list, category list}] list를 로드하는 함수."""
        data = []
        if not self.train:
            for image_file in self.image_path.rglob("*.png"):
                image = Image.open(image_file)
                data.append((image, {}))
        else:
            for img_id in sorted(self.coco.imgs):
                img_info = self.coco.imgs[img_id]
                image_file = self.image_path / img_info["file_name"]
                image = Image.open(image_file)
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
                boxes = []
                labels = []
                for ann in anns:
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
        if self.train:
            img_w, img_h = image.size
            target["boxes"] = tv_tensors.BoundingBoxes(
                target["boxes"], format="XYXY", canvas_size=(img_h, img_w)
            )
            image, target = self.transform(image, target)
        else:
            image = self.transform(image)
        return image, target
