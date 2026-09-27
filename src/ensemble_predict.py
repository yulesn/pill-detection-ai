"""YOLO와 RF-DETR 예측을 Weighted Boxes Fusion(WBF)으로 합쳐서 제출용 CSV를 만드는 스크립트.

configs/ensemble.yaml의 models 목록에 있는 모델들을 전부 불러와서, 이미지 한 장마다
각 모델의 예측을 다 돌린 뒤 WBF로 박스를 합친다. category_id(원본 약 코드)가
모델 간 공통 라벨 역할을 하므로, 다른 category_id끼리는 절대 합쳐지지 않는다.

사용 예:
    python src/ensemble_predict.py --config configs/ensemble.yaml --test_dir data/raw/sprint_ai_project1_data/test_images
"""

import argparse

from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
from ultralytics import YOLO

from model import build_model
from predict import (
    detections_to_rows,
    list_image_paths,
    load_category_id_map,
    rfdetr_detect,
    yolo_detect,
)
from utils import load_config, save_predictions_csv


def load_model_entry(entry: dict) -> dict:
    """configs/ensemble.yaml의 models 항목 하나를 로드해서 예측에 필요한 값들을 묶어 반환."""
    config = load_config(entry["config"])
    framework = config["model"]["framework"]
    weight = entry.get("weight", 1.0)

    if framework == "yolo":
        model = YOLO(entry["checkpoint"])
        data_file = load_config(config["data"]["yaml_path"])
        return {"framework": "yolo", "model": model, "label_map": data_file["names"], "weight": weight}

    if framework == "rfdetr":
        config["model"]["checkpoint"] = entry["checkpoint"]
        model = build_model(config)
        category_id_map = load_category_id_map(config["data"]["processed_dir"])
        return {
            "framework": "rfdetr",
            "model": model,
            "class_names": model.class_names,
            "category_id_map": category_id_map,
            "weight": weight,
        }

    raise ValueError(f"앙상블에서 지원하지 않는 framework입니다: {framework}")


def detect_one(entry: dict, image_path: str) -> list[dict]:
    if entry["framework"] == "yolo":
        return yolo_detect(entry["model"], image_path, entry["label_map"])
    return rfdetr_detect(entry["model"], image_path, entry["class_names"], entry["category_id_map"])


def fuse_detections(
    per_model_detections: list[list[dict]], weights: list[float], width: int, height: int,
    iou_thr: float, skip_box_thr: float,
) -> list[dict]:
    """모델별 detection 리스트(픽셀 xyxy)를 WBF로 합쳐서 다시 픽셀 xyxy로 돌려준다."""
    boxes_list, scores_list, labels_list = [], [], []
    for detections in per_model_detections:
        boxes, scores, labels = [], [], []
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])
            scores.append(det["score"])
            labels.append(det["category_id"])
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
    )

    return [
        {
            "category_id": int(category_id),
            "bbox_xyxy": (x1 * width, y1 * height, x2 * width, y2 * height),
            "score": float(score),
        }
        for (x1, y1, x2, y2), score, category_id in zip(fused_boxes, fused_scores, fused_labels)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml")
    parser.add_argument("--test_dir", type=str, required=True, help="이미지 파일 또는 이미지가 담긴 폴더 경로")
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensemble_config = load_config(args.config)

    entries = [load_model_entry(m) for m in ensemble_config["models"]]
    weights = [entry["weight"] for entry in entries]
    iou_thr = ensemble_config["ensemble"]["iou_thr"]
    skip_box_thr = ensemble_config["ensemble"]["skip_box_thr"]

    image_paths = list_image_paths(args.test_dir)
    results = []

    for image_path in image_paths:
        image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
        with Image.open(image_path) as img:
            width, height = img.size

        per_model_detections = [detect_one(entry, str(image_path)) for entry in entries]
        fused = fuse_detections(per_model_detections, weights, width, height, iou_thr, skip_box_thr)
        results.extend(detections_to_rows(image_id, fused))

    output_csv = args.output_csv or f"outputs/predictions/{ensemble_config['output']['experiment_name']}.csv"
    save_predictions_csv(results, output_csv)
    print(f"{len(image_paths)}장 이미지에서 예측 {len(results)}건을 {output_csv}에 저장했습니다.")


if __name__ == "__main__":
    main()
