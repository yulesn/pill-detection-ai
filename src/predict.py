"""학습된 모델로 추론하는 스크립트.

사용 예:
    # YOLO
    python src/predict.py --config configs/exp2_yolo11n.yaml --checkpoint outputs/exp1_yolo11n/weights/best.pt --test_dir data/raw/sprint_ai_project1_data/test_images --output_csv outputs/predictions/submission.csv
    # RF-DETR
    python src/predict.py --config configs/rfdetr.yaml --checkpoint outputs/rfdetr_nano/checkpoints/checkpoint_best_ema.pth --test_dir data/raw/sprint_ai_project1_data/test_images
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from model import build_model
from utils import load_config, save_predictions_csv

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True, help="이미지 파일 또는 이미지가 담긴 폴더 경로")
    parser.add_argument("--output_csv", type=str, default=None, help="결과 CSV 경로 (기본: outputs/predictions/<experiment_name 또는 submission>.csv)")
    return parser.parse_args()


def list_image_paths(image_arg: str) -> list[Path]:
    path = Path(image_arg)
    if path.is_dir():
        paths = [p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        return sorted(paths, key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    return [path]


def load_category_id_map(processed_dir: str) -> dict[str, int]:
    """category_mapping.json(label/category_id/name)에서 클래스명 -> 제출용 category_id 매핑을 만든다."""
    mapping_path = Path(processed_dir) / "splits" / "category_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {entry["name"]: entry["category_id"] for entry in mapping}


def yolo_detect(model, image_path: str, label_map: dict) -> list[dict]:
    """한 이미지에 대한 YOLO 추론 결과를 {category_id, bbox_xyxy, score} 리스트로 반환."""
    results = model.predict(source=image_path, conf=0.25, save=False, iou=0.1, max_det=4, verbose=False)
    detections = []
    for box in results[0].boxes:
        category_idx = int(box.cls[0].item())    # tensor 객체를 가져와 int로 변환
        category_id = int(label_map[category_idx].split('-')[1])
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({"category_id": category_id, "bbox_xyxy": (x1, y1, x2, y2), "score": score})
    return detections


def rfdetr_detect(
    model, image_path: str, class_names: dict, category_id_map: dict, threshold: float = 0.5
) -> list[dict]:
    """한 이미지에 대한 RF-DETR 추론 결과를 {category_id, bbox_xyxy, score} 리스트로 반환."""
    result = model.predict(image_path, threshold=threshold)
    detections = []
    for class_id, bbox, score in zip(result.class_id, result.xyxy, result.confidence):
        x1, y1, x2, y2 = bbox.tolist()
        class_name = class_names[int(class_id)]
        detections.append(
            {"category_id": category_id_map[class_name], "bbox_xyxy": (x1, y1, x2, y2), "score": float(score)}
        )
    return detections


def detections_to_rows(image_id: int | str, detections: list[dict]) -> list[dict]:
    """detect 함수들이 반환한 결과를 save_predictions_csv용 row로 변환 (bbox: xyxy -> xywh)."""
    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        rows.append(
            {
                "image_id": image_id,
                "category_id": det["category_id"],
                "bbox_x": round(x1),
                "bbox_y": round(y1),
                "bbox_w": round(x2 - x1),
                "bbox_h": round(y2 - y1),
                "score": round(det["score"], 4),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    framework = config["model"]["framework"]
    image_paths = list_image_paths(args.test_dir)

    if framework == "yolo":
        data_file = load_config(config["data"]["yaml_path"])
        label_map = data_file["names"]  # 모델 예측 결과를 데이터 제출 형식에 맞춤
        model = YOLO(args.checkpoint)

        results = []
        for image_path in image_paths:
            image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
            results.extend(detections_to_rows(image_id, yolo_detect(model, str(image_path), label_map)))

        output_csv = args.output_csv or "outputs/predictions/submission.csv"
        save_predictions_csv(results, output_csv)
        print(f"{len(image_paths)}장 이미지에서 예측 {len(results)}건을 {output_csv}에 저장했습니다.")
        return

    if framework == "torchvision":
        # TODO(torchvision 트랙): args.checkpoint 로드, 이미지 전처리,
        # 커스텀 추론 후 (클래스, bbox) 리스트로 출력
        raise NotImplementedError("torchvision 추론 로직을 구현해주세요.")

    if framework == "rfdetr":
        config["model"]["checkpoint"] = args.checkpoint
        model = build_model(config)
        category_id_map = load_category_id_map(config["data"]["processed_dir"])
        # model.class_names: {1: "약이름", 2: "약이름", ...} (학습 때 쓴 categories 순서 그대로,
        # 체크포인트에 저장돼있어서 checkpoint만 넘겨서 만든 모델에서도 값이 채워짐)
        class_names = model.class_names

        results = []
        for image_path in image_paths:
            image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
            detections = rfdetr_detect(model, str(image_path), class_names, category_id_map)
            results.extend(detections_to_rows(image_id, detections))

        output_csv = args.output_csv or f"outputs/predictions/{config['output']['experiment_name']}.csv"
        save_predictions_csv(results, output_csv)
        print(f"{len(image_paths)}장 이미지에서 예측 {len(results)}건을 {output_csv}에 저장했습니다.")
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()