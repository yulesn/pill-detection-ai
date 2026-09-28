"""알약 검출용 PyTorch Dataset.

학습용(train=True): make_dataset.py 가 만든 data/processed/train_coco_final.json 을 읽는다.
  - 조합 단위로 미리 나눠 둔 train/val 을 그대로 쓴다 (subset="train" / "val").
  - 사진은 경로만 들고 있다가 필요할 때 연다 (7천 장을 미리 열어 두면 메모리가 터짐).
  - label_to_raw_id: 모델 라벨(1~N) -> 대회 category_id. 예측 결과를 제출용 번호로 바꿀 때 쓴다.
  - 배경이 라벨 0번이라, 모델의 num_classes 는 클래스 수 + 1 이어야 한다.

시험용(train=False): 예전 그대로 data_dir/test_images 의 사진을 읽는다.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from pill_common import load_train_samples

# transform=None일 때 최소한 텐서 변환은 되도록 기본값 제공
DEFAULT_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])


class PillDataset(Dataset):
    """이미지와 (클래스, bbox) 라벨을 반환하는 Dataset.

    __getitem__은 (image, target)을 반환한다.
    - image: tv_tensors.Image, shape (3, H, W), float32, [0, 1]
    - target (train): {"image_id": LongTensor(1,), "boxes": BoundingBoxes(N, 4) XYXY,
      "labels": LongTensor(N,)} — N은 이미지당 알약 개수(가변)
    - target (test): {} (라벨 없음)

    subset: train=True일 때만 의미 있음. None(전체) / "train" / "val"
    (val_ratio, split_seed 는 예전 코드와 호환하려고 받기만 하고 쓰지 않는다. 분할은 make_dataset.py 에서 정해진다.)
    """

    def __init__(
        self,
        data_dir: str,
        train: bool,
        transform=DEFAULT_TRANSFORM,
        subset: str = None,
        val_ratio: float = None,
        split_seed: int = None,
    ):
        self.transform = transform
        self.train = "train" if train else "test"
        self.subset = subset
        data_dir = Path(data_dir)
        self.image_path = data_dir / f"{self.train}_images"
        self.categories = {}
        self.label_to_raw_id = {}
        self.raw_id_to_label = {}

        if train:
            (self.samples, self.label_to_raw_id,
             self.categories, self.raw_id_to_label) = load_train_samples(subset)
            self.data = self.samples
        else:
            self.data = []
            for image_file in self.image_path.rglob("*.png"):
                image = Image.open(image_file)
                self.data.append((image, {}))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        if self.train == "test":
            image, target = self.data[index]
            return self.transform(image), target

        s = self.samples[index]
        image = Image.open(s["path"]).convert("RGB")
        img_w, img_h = image.size
        boxes = torch.as_tensor(s["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(s["labels"], dtype=torch.int64)
        target = {
            "image_id": torch.as_tensor([s["image_id"]], dtype=torch.int64),
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img_h, img_w)),
            "labels": labels,
        }
        image, target = self.transform(image, target)
        return image, target