"""학습된 모델을 validation 세트로 평가하는 스크립트 (COCOeval 기반 mAP).

사용 예:
    python src/evaluate.py --config configs/fasterrcnn_test.yaml \
        --checkpoint outputs/checkpoints/default/best.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from dataset import PillDataset
from model import build_model
from predict import run_inference
from train import collate_fn, run_epoch
from utils import load_config, load_model_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def compute_val_loss(model, val_dataset: PillDataset, device, batch_size: int) -> float:
    """train.py의 run_epoch를 그대로 재사용해 val loss를 계산한다.

    torchvision detection 모델은 train() 모드일 때만 loss_dict를 반환하므로,
    train.py에서 val_loss를 구할 때와 동일하게 model.train() 상태에서
    (역전파는 하지 않고 no_grad로) 호출해야 한다.
    """
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.train()
    with torch.no_grad():
        return run_epoch(model, val_loader, device, optimizer=None)


def collect_val_predictions(
    model,
    val_dataset: PillDataset,
    device,
    score_threshold: float,
    max_objects_per_image: int,
) -> list[dict]:
    """COCOeval이 요구하는 형식(image_id, category_id, bbox, score)의 예측 리스트를 만든다.

    category_id는 val_dataset.coco와 같은 라벨 공간(1-index, category_mapping.json
    기준)을 그대로 쓴다 — 제출용 CSV와 달리 원본 category_id로 되돌리면 GT와
    비교가 불가능해지므로 변환하지 않는다.
    """
    results = []
    for image, target in val_dataset:
        image_id = target["image_id"].item()
        boxes, labels, scores = run_inference(
            model, image, device, score_threshold, max_objects_per_image
        )
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": image_id,
                    "category_id": label.item(),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score.item(),
                }
            )
    return results


def evaluate_map(
    model,
    val_dataset: PillDataset,
    device,
    score_threshold: float,
    max_objects_per_image: int,
) -> COCOeval:
    """val_dataset의 GT(COCO)와 모델 예측을 COCOeval로 비교해 mAP를 계산하고 출력한다."""
    results = collect_val_predictions(
        model, val_dataset, device, score_threshold, max_objects_per_image
    )
    coco_dt = val_dataset.coco.loadRes(results)
    coco_eval = COCOeval(val_dataset.coco, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def evaluate_torchvision(config: dict, checkpoint: str) -> None:
    device = torch.device(config["train"]["device"])

    val_dir = Path(config["data"]["processed_dir"]) / "val"
    val_dataset = PillDataset(val_dir)

    model = build_model(config)
    load_model_weights(model, checkpoint, device)
    model.to(device)

    val_loss = compute_val_loss(model, val_dataset, device, config["train"]["batch_size"])
    print(f"val_loss={val_loss:.4f}")

    model.eval()
    evaluate_map(
        model,
        val_dataset,
        device,
        config["predict"]["score_threshold"],
        config["data"]["max_objects_per_image"],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    framework = config["model"]["framework"]

    if framework == "yolo":
        # TODO(YOLO 트랙): ultralytics 모델은 model.val(...)로 자체 평가를 제공하므로
        # 그걸 그대로 쓰거나, mAP 리포트 형식을 맞추려면 별도 구현 필요
        raise NotImplementedError("YOLO 평가 로직을 구현해주세요.")

    if framework == "torchvision":
        evaluate_torchvision(config, args.checkpoint)
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
