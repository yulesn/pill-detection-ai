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
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image

from dataset import DEFAULT_TRANSFORM
from model import build_model
from utils import load_config


def load_label_to_category_id(processed_dir: Path) -> dict[int, int]:
    """prepare_split.py가 저장한 category_mapping.json을 읽어 모델이 예측하는
    라벨(1-index)을 원본 category_id로 되돌리는 매핑을 만든다."""
    mapping_path = processed_dir / "splits" / "category_mapping.json"
    with open(mapping_path) as f:
        mapping = json.load(f)
    return {entry["label"]: entry["category_id"] for entry in mapping}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def predict_image(
    model,
    image_path: Path,
    device,
    score_threshold: float,
    max_objects_per_image: int,
    label_to_category_id: dict[int, int],
    nms_iou_threshold: float = 0.5,
) -> list[dict]:
    # test 이미지 파일명(예: "1.png")이 곧 image_id
    image_id = int(image_path.stem)

    image = Image.open(image_path).convert("RGB")
    input_tensor = DEFAULT_TRANSFORM(image).to(device)

    with torch.no_grad():
        output = model([input_tensor])[0]

    boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
    keep = scores >= score_threshold
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # torchvision 모델의 NMS는 클래스별로만 적용되어, 같은 물체를 다른 카테고리로
    # 예측해 겹치는 박스가 남을 수 있다. 클래스 무관 NMS를 한 번 더 적용해서 걸러낸다.
    # nms()는 살아남은 인덱스를 score 내림차순으로 반환하므로, 이어서 상위
    # max_objects_per_image개만 자르면 이미지당 최대 알약 개수 제한도 함께 적용된다.
    keep = torchvision.ops.nms(boxes, scores, nms_iou_threshold)[:max_objects_per_image]
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    results = []
    for box, label, score in zip(boxes, labels, scores):
        category_id = label_to_category_id[label.item()]
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
    model,
    checkpoint: str,
    test_images_dir: Path,
    device,
    score_threshold: float,
    max_objects_per_image: int,
    label_to_category_id: dict[int, int],
) -> list[dict]:
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    results = []
    image_paths = sorted(test_images_dir.glob("*.png"), key=lambda p: int(p.stem))
    for image_path in image_paths:
        results.extend(
            predict_image(
                model,
                image_path,
                device,
                score_threshold,
                max_objects_per_image,
                label_to_category_id,
            )
        )
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
        label_to_category_id = load_label_to_category_id(Path(config["data"]["processed_dir"]))

        results = predict_torchvision(
            model,
            args.checkpoint,
            test_images_dir,
            device,
            config["predict"]["score_threshold"],
            config["data"]["max_objects_per_image"],
            label_to_category_id,
        )
        experiment_name = config["output"]["experiment_name"]
        output_path = Path(config["output"]["dir"]) / "predictions" / f"{experiment_name}.csv"
        save_predictions_csv(results, output_path)
        print(f"{len(results)}개 예측 결과 저장: {output_path}")
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
