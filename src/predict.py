"""학습된 모델로 data/raw/.../test_images 전체를 추론하는 스크립트.

사용 예:
    python src/predict.py --config configs/default.yaml \
        --checkpoint outputs/checkpoints//default/best.pt

출력:
    outputs/predictions/<experiment_name>.csv — annotation_id, image_id, category_id,
    bbox_x, bbox_y, bbox_w, bbox_h, score 컬럼의 제출용 CSV
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from dataset import DEFAULT_TRANSFORM
from model import build_model
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def predict_image(model, image_path: Path, device, score_threshold: float) -> list[dict]:
    # test 이미지 파일명(예: "1.png")이 곧 image_id
    image_id = int(image_path.stem)

    image = Image.open(image_path).convert("RGB")
    input_tensor = DEFAULT_TRANSFORM(image).to(device)

    with torch.no_grad():
        output = model([input_tensor])[0]

    results = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        category_id = label.item()
        x1, y1, x2, y2 = box.tolist()
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": round(x1, 2),
                "bbox_y": round(y1, 2),
                "bbox_w": round(x2 - x1, 2),
                "bbox_h": round(y2 - y1, 2),
                "score": round(score.item(), 4),
            }
        )
    return results


def predict_torchvision(
    model, checkpoint: str, test_images_dir: Path, device, score_threshold: float
) -> list[dict]:
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    results = []
    image_paths = sorted(test_images_dir.glob("*.png"), key=lambda p: int(p.stem))
    for image_path in image_paths:
        results.extend(predict_image(model, image_path, device, score_threshold))
    return results


def save_predictions_csv(results: list[dict], output_path: Path) -> None:
    """예측 결과를 annotation_id를 붙여 제출용 CSV로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for annotation_id, result in enumerate(results, start=1):
            writer.writerow({"annotation_id": annotation_id, **result})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    framework = config["model"]["framework"]
    model = build_model(config)

    if framework == "yolo":
        # TODO(YOLO 트랙): model.predict(args.image, conf=...) 결과를 파싱해서
        # (클래스, bbox) 리스트로 출력
        raise NotImplementedError("YOLO 추론 로직을 구현해주세요.")

    if framework == "torchvision":
        device = torch.device(config["train"]["device"])
        test_images_dir = Path(config["data"]["raw_dir"]) / "sprint_ai_project1_data" / "test_images"

        results = predict_torchvision(
            model, args.checkpoint, test_images_dir, device, args.score_threshold
        )
        experiment_name = config["output"]["experiment_name"]
        output_path = Path(config["output"]["dir"]) / "predictions" / f"{experiment_name}.csv"
        save_predictions_csv(results, output_path)
        print(f"{len(results)}개 예측 결과 저장: {output_path}")
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
