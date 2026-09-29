import json
from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import nms
from tqdm import tqdm


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "test_images"
)

TRAIN_ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_annotations"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "retinanet"
    / "retinanet_resnet50_fpn_v2_epoch30.pth"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "retinanet_submission.csv"
)

NUM_CLASSES = 56

# YOLO와 비슷하게 너무 낮은 confidence 제거
CONF_THRESHOLD = 0.30

# 한 이미지에서 최대 객체 수
MAX_DETECTIONS = 4

# NMS IoU
NMS_IOU_THRESHOLD = 0.5


# ============================================================
# 2. Test Dataset
# ============================================================

class TestDataset(Dataset):

    def __init__(self, image_dir):

        self.image_dir = Path(image_dir)

        self.images = sorted(
            list(self.image_dir.glob("*.png"))
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"테스트 이미지를 찾을 수 없습니다:\n"
                f"{self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        image_np = np.array(image)

        image_tensor = (
            torch.from_numpy(image_np)
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        return image_tensor, image_path


# ============================================================
# 3. Collate
# ============================================================

def collate_fn(batch):

    images, paths = zip(*batch)

    return list(images), list(paths)


# ============================================================
# 4. RetinaNet 모델
# ============================================================

def create_model():

    model = retinanet_resnet50_fpn_v2(
        weights=None
    )

    old_head = (
        model.head.classification_head
    )

    model.head.classification_head = (
        RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=old_head.num_anchors,
            num_classes=NUM_CLASSES
        )
    )

    return model


# ============================================================
# 5. 모델 로드
# ============================================================

def load_model(device):

    print("모델 로드 중...")

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:
        checkpoint = checkpoint[
            "state_dict"
        ]

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    print(
        f"모델 로드 완료: {MODEL_PATH}"
    )

    return model


# ============================================================
# 6. Category ID 매핑
# ============================================================
def load_category_mapping():

    print("\nCategory mapping 생성 중...")

    # preprocess.py와 동일하게
    # 모든 annotation JSON에서 category_id 수집

    category_ids = set()

    for root, dirs, files in os.walk(
        TRAIN_ANNOTATION_DIR
    ):

        for file in files:

            if not file.endswith(".json"):
                continue

            json_path = Path(root) / file

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            for ann in data.get(
                "annotations",
                []
            ):

                category_ids.add(
                    ann["category_id"]
                )

    # preprocess.py와 동일
    sorted_category_ids = sorted(
        category_ids
    )

    if len(sorted_category_ids) != NUM_CLASSES:

        raise RuntimeError(
            f"Category 수가 예상과 다릅니다.\n"
            f"현재: {len(sorted_category_ids)}\n"
            f"예상: {NUM_CLASSES}"
        )

    # class_id -> 원본 category_id
    class_to_category = {
        class_id: category_id
        for class_id, category_id
        in enumerate(
            sorted_category_ids
        )
    }

    print(
        f"총 category 수: "
        f"{len(sorted_category_ids)}"
    )

    print("\nClass → Category mapping")
    print("-" * 50)

    for class_id, category_id in list(
        class_to_category.items()
    )[:10]:

        print(
            f"class_id {class_id:2d}"
            f" -> category_id {category_id}"
        )

    print("-" * 50)

    return class_to_category

# ============================================================
# 7. image_id 추출
# ============================================================

def extract_image_id(image_path):

    # 파일명 예:
    # K-003xxx-....png
    #
    # competition test image의 image_id가
    # 파일명과 별도인 경우에는 기존 submission 로직과
    # 맞춰야 함.
    #
    # 숫자만 있는 파일명이라면 그대로 사용.

    stem = image_path.stem

    try:
        return int(stem)

    except ValueError:

        # 기본적으로 파일명에서 마지막 숫자 그룹 추출
        import re

        numbers = re.findall(
            r"\d+",
            stem
        )

        if not numbers:

            raise ValueError(
                f"image_id를 추출할 수 없습니다: "
                f"{image_path.name}"
            )

        return int(
            numbers[-1]
        )


# ============================================================
# 8. Prediction
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device,
    class_to_category
):

    rows = []

    annotation_id = 0

    print("\nTest prediction 시작")
    print("=" * 60)

    for images, image_paths in tqdm(
        loader,
        desc="Predicting"
    ):

        images_device = [
            image.to(device)
            for image in images
        ]

        outputs = model(
            images_device
        )

        for output, image_path in zip(
            outputs,
            image_paths
        ):

            boxes = output["boxes"]
            labels = output["labels"]
            scores = output["scores"]

            # confidence filtering
            keep = (
                scores
                >= CONF_THRESHOLD
            )

            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if len(boxes) == 0:
                continue

            # NMS
            keep = nms(
                boxes,
                scores,
                NMS_IOU_THRESHOLD
            )

            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            # confidence 높은 순
            order = torch.argsort(
                scores,
                descending=True
            )

            boxes = boxes[order]
            labels = labels[order]
            scores = scores[order]

            # 최대 객체 수
            boxes = boxes[
                :MAX_DETECTIONS
            ]

            labels = labels[
                :MAX_DETECTIONS
            ]

            scores = scores[
                :MAX_DETECTIONS
            ]

            image_id = extract_image_id(
                image_path
            )

            for box, label, score in zip(
                boxes,
                labels,
                scores
            ):

                x1, y1, x2, y2 = (
                    box.tolist()
                )

                class_id = int(
                    label.item()
                )

                category_id = (
                    class_to_category[
                        class_id
                    ]
                )

                width = x2 - x1
                height = y2 - y1

                rows.append(
                    {
                        "annotation_id":
                            annotation_id,

                        "image_id":
                            image_id,

                        "category_id":
                            category_id,

                        "bbox_x":
                            x1,

                        "bbox_y":
                            y1,

                        "bbox_w":
                            width,

                        "bbox_h":
                            height,

                        "score":
                            float(
                                score.item()
                            )
                    }
                )

                annotation_id += 1

    return rows


# ============================================================
# 9. Main
# ============================================================

def main():

    print("=" * 60)
    print("RetinaNet Test Prediction")
    print("=" * 60)

    print(
        f"Test images : {TEST_IMAGE_DIR}"
    )

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Output      : {OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device      : {device}"
    )

    # --------------------------------------------------------
    # 파일 확인
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\n모델 파일이 없습니다:\n"
            f"{MODEL_PATH}"
        )

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"\nTest image 폴더가 없습니다:\n"
            f"{TEST_IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = TestDataset(
        TEST_IMAGE_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(
        f"Test images : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Category mapping
    # --------------------------------------------------------

    class_to_category = (
        load_category_mapping()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    rows = predict(
        model,
        loader,
        device,
        class_to_category
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows,
        columns=[
            "annotation_id",
            "image_id",
            "category_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "score"
        ]
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("Prediction 완료")
    print("=" * 60)

    print(
        f"Test images : {len(dataset)}"
    )

    print(
        f"Predictions : {len(df)}"
    )

    print(
        f"CSV         : {OUTPUT_CSV}"
    )

    print("\nCSV preview:")
    print(df.head())

    print("\nPrediction 완료.")


if __name__ == "__main__":
    main()