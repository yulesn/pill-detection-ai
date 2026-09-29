import torch
import torchvision
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


# ============================================================
# 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "processed"

VAL_IMAGE_DIR = DATA_ROOT / "images" / "val"
VAL_LABEL_DIR = DATA_ROOT / "labels" / "val"

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "retinanet"
    / "retinanet_resnet50_fpn_v2_epoch30.pth"
)

NUM_CLASSES = 56
BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RetinaNet Validation Evaluation")
print("=" * 60)
print("Device    :", device)
print("Checkpoint:", CHECKPOINT)
print("Val images:", VAL_IMAGE_DIR)
print()


# ============================================================
# Dataset
# ============================================================

class PillRetinaNetDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = self.images[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        image = torch.from_numpy(
            np.array(image)
        ).permute(2, 0, 1).float() / 255.0

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id, xc, yc, w, h = map(
                        float,
                        parts
                    )

                    x1 = (xc - w / 2) * width
                    y1 = (yc - h / 2) * height
                    x2 = (xc + w / 2) * width
                    y2 = (yc + h / 2) * height

                    boxes.append([
                        x1, y1, x2, y2
                    ])

                    labels.append(
                        int(class_id)
                    )

        if boxes:
            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )
        else:
            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# Dataset / DataLoader
# ============================================================

val_dataset = PillRetinaNetDataset(
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("Val images :", len(val_dataset))
print("Val batches:", len(val_loader))
print()


# ============================================================
# Model
# ============================================================

model = retinanet_resnet50_fpn_v2(
    weights=None
)

old_head = model.head.classification_head

model.head.classification_head = RetinaNetClassificationHead(
    in_channels=256,
    num_anchors=old_head.num_anchors,
    num_classes=NUM_CLASSES
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print("Classes:", NUM_CLASSES)
print()


# ============================================================
# IoU
# ============================================================

def box_iou(boxes1, boxes2):

    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2))
        )

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        *
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        *
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    lt = torch.max(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    rb = torch.min(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    wh = (rb - lt).clamp(min=0)

    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(min=1e-6)


# ============================================================
# Collect predictions / GT
# ============================================================

all_predictions = []
all_targets = []

print("Running validation...")

with torch.no_grad():

    for images, targets in val_loader:

        images = [
            image.to(device)
            for image in images
        ]

        outputs = model(images)

        for output, target in zip(outputs, targets):

            all_predictions.append({
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            })

            all_targets.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            })


print("Validation inference complete.")
print()


# ============================================================
# AP 계산
# ============================================================

def calculate_ap(
    predictions,
    targets,
    class_id,
    iou_threshold
):

    detections = []
    total_gt = 0

    # GT 준비
    gt_by_image = {}

    for image_idx, target in enumerate(targets):

        mask = (
            target["labels"] == class_id
        )

        gt_boxes = target["boxes"][mask]

        gt_by_image[image_idx] = gt_boxes

        total_gt += len(gt_boxes)

    if total_gt == 0:
        return None

    # Prediction 준비
    for image_idx, prediction in enumerate(predictions):

        mask = (
            prediction["labels"] == class_id
        )

        boxes = prediction["boxes"][mask]
        scores = prediction["scores"][mask]

        for box, score in zip(boxes, scores):

            detections.append({
                "image": image_idx,
                "box": box,
                "score": float(score)
            })

    if len(detections) == 0:
        return 0.0

    # confidence 내림차순
    detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {
        image_idx: torch.zeros(
            len(gt_by_image[image_idx]),
            dtype=torch.bool
        )
        for image_idx in gt_by_image
    }

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for det_idx, detection in enumerate(detections):

        image_idx = detection["image"]

        gt_boxes = gt_by_image[image_idx]

        if len(gt_boxes) == 0:
            fp[det_idx] = 1
            continue

        ious = box_iou(
            detection["box"].unsqueeze(0),
            gt_boxes
        )[0]

        best_iou, best_idx = torch.max(
            ious,
            dim=0
        )

        if (
            best_iou >= iou_threshold
            and not matched[image_idx][best_idx]
        ):

            tp[det_idx] = 1
            matched[image_idx][best_idx] = True

        else:
            fp[det_idx] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / max(total_gt, 1)
    precisions = tp / np.maximum(
        tp + fp,
        1e-12
    )

    # COCO-style 101-point interpolation
    recall_points = np.linspace(
        0.0,
        1.0,
        101
    )

    ap = 0.0

    for recall_point in recall_points:

        valid = recalls >= recall_point

        if np.any(valid):
            precision = np.max(
                precisions[valid]
            )
        else:
            precision = 0.0

        ap += precision

    ap /= 101.0

    return ap


# ============================================================
# mAP 계산
# ============================================================

def calculate_map(iou_threshold):

    aps = []

    for class_id in range(1, NUM_CLASSES + 1):

        ap = calculate_ap(
            all_predictions,
            all_targets,
            class_id,
            iou_threshold
        )

        if ap is not None:
            aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps))


print("Calculating mAP...")

map50 = calculate_map(0.50)
map75 = calculate_map(0.75)

map5095_values = []

for iou in np.arange(
    0.50,
    0.96,
    0.05
):

    ap = calculate_map(
        round(float(iou), 2)
    )

    map5095_values.append(ap)

map5095 = float(
    np.mean(map5095_values)
)


# ============================================================
# 결과
# ============================================================

print()
print("=" * 60)
print("RetinaNet Validation Results")
print("=" * 60)

print(
    f"Val mAP50     : {map50:.4f}"
)

print(
    f"Val mAP75     : {map75:.4f}"
)

print(
    f"Val mAP50-95  : {map5095:.4f}"
)

print("=" * 60)