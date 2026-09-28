"""검증 데이터(val)에 대해 mAP를 계산하는 스크립트.
표준 지표(mAP50, mAP50-95 등)와 대회 기준 지표(mAP@[0.75:0.95])를 모두 출력한다.

사용 예:
    python src/evaluate.py --config configs/kimgun_t2_fasterrcnn.yaml --checkpoint outputs/checkpoints/kimgun_t2_fasterrcnn_best.pt
"""

import argparse

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from dataset import PillDataset
from model import build_model
from utils import load_config


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def build_coco_gt_and_predictions(model, val_loader, device):
    """검증 데이터로부터 COCO 형식의 정답(gt)과 예측(predictions)을 만든다."""
    gt = {"images": [], "annotations": [], "categories": []}
    seen_categories = set()
    ann_id = 1

    # 정답(ground truth) 구성
    for images, targets in val_loader:
        target = targets[0]
        image_id = target["image_id"].item()
        _, h, w = images[0].shape
        gt["images"].append({"id": image_id, "width": w, "height": h})
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            gt["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
            ann_id += 1
            seen_categories.add(int(label))
    gt["categories"] = [{"id": c, "name": str(c)} for c in sorted(seen_categories)]

    # 예측(prediction) 구성
    predictions = []
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            image_id = targets[0]["image_id"].item()
            output = model([images[0].to(device)])[0]
            for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": image_id, "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
                })

    return gt, predictions


def run_eval(coco_gt: COCO, predictions: list, iou_thrs=None, label: str = "") -> None:
    """COCOeval을 돌리고 결과를 출력한다. iou_thrs가 None이면 COCO 기본값(0.5~0.95, 12개 지표) 사용."""
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if iou_thrs is not None:
        coco_eval.params.iouThrs = iou_thrs
    coco_eval.evaluate()
    coco_eval.accumulate()

    print(f"\n{'=' * 20} {label} {'=' * 20}")
    coco_eval.summarize()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config["train"]["device"])

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    data_root = config["data"]["raw_dir"] + "/sprint_ai_project1_data"
    val_dataset = PillDataset(data_root, train=True, subset="val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    gt, predictions = build_coco_gt_and_predictions(model, val_loader, device)

    if not predictions:
        print("예측 결과가 없어 평가를 진행할 수 없음 (score_threshold를 확인해줘)")
        return

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    # 1) 표준 지표: mAP50-95(stats[0]), mAP50(stats[1]), mAP75(stats[2]) 등 12개를 한 번에 출력
    run_eval(coco_gt, predictions, iou_thrs=None, label="표준 지표 (mAP50 / mAP50-95 포함)")

    # 2) 대회 기준 지표: mAP@[0.75:0.95]
    comp_iou_thrs = [round(0.75 + 0.05 * i, 2) for i in range(5)]
    run_eval(coco_gt, predictions, iou_thrs=comp_iou_thrs, label="대회 기준 mAP@[0.75:0.95]")


if __name__ == "__main__":
    main()
