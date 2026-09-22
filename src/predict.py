"""학습된 YOLO 모델로 추론하는 스크립트.

사용 예:
    단일 이미지 추론
    python src/predict.py \
        --config configs/default.yaml \
        --checkpoint outputs/default2/weights/best.pt \
        --image path/to/image.jpg

    테스트 이미지 전체 추론 및 submission.csv 생성
    python src/predict.py \
        --config configs/default.yaml \
        --checkpoint outputs/default2/weights/best.pt \
        --image data/raw/sprint_ai_project1_data/test_images
"""

import argparse
import csv
import json
import os
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_annotations"
)



# ============================================================
# 2. EDA에서 제외했던 이미지
# ============================================================

EXCLUDE_IMAGE_IDS = {
    16, 193, 208, 239,
    783, 907, 1228,
    1258, 1267, 1383,
    1405, 1432
}


# ============================================================
# 3. Argument 설정
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    return parser.parse_args()


# ============================================================
# 4. Category ID 수집
# ============================================================

def collect_category_ids(annotation_dir):
    """
    Train annotation에서 category_id를 수집한다.

    preprocess.py와 동일한 기준으로
    EDA 이상 데이터를 제외한다.
    """

    category_ids = set()

    for root, dirs, files in os.walk(annotation_dir):

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

            valid_image_ids = {
                image["id"]
                for image in data.get("images", [])
                if image["id"] not in EXCLUDE_IMAGE_IDS
            }

            for ann in data.get("annotations", []):

                image_id = ann["image_id"]

                if image_id not in valid_image_ids:
                    continue

                bbox = ann.get("bbox")

                if not bbox or len(bbox) != 4:
                    continue

                x, y, w, h = bbox

                if (
                    w <= 0
                    or h <= 0
                    or x < 0
                    or y < 0
                ):
                    continue

                category_ids.add(
                    ann["category_id"]
                )

    return sorted(category_ids)


# ============================================================
# 5. YOLO class_id → category_id 매핑
# ============================================================

def create_class_to_category_mapping():

    category_ids = collect_category_ids(
        ANNOTATION_DIR
    )

    return {
        class_id: category_id
        for class_id, category_id
        in enumerate(category_ids)
    }


# ============================================================
# 6. Prediction
# ============================================================

def predict(
    model,
    image_path,
    class_to_category,
    conf=0.25
):
    """
    이미지 또는 이미지 폴더를 YOLO로 추론한다.

    반환값:
        submission 형식의 prediction rows
    """

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=True,
        stream=True
    )

    rows = []

    for result in results:

        image_path = Path(result.path)

        # 파일명이 image_id라고 가정
        image_id = int(
            image_path.stem
        )

        boxes = result.boxes

        if boxes is None:
            continue

        for i in range(len(boxes)):

            # YOLO class_id
            class_id = int(
                boxes.cls[i].item()
            )

            # YOLO class_id → category_id
            if class_id not in class_to_category:

                print(
                    f"경고: 알 수 없는 class_id "
                    f"{class_id}"
                )

                continue

            category_id = (
                class_to_category[class_id]
            )

            # XYXY 좌표
            x1, y1, x2, y2 = (
                boxes.xyxy[i]
                .cpu()
                .tolist()
            )

            # XYXY → XYWH
            bbox_x = x1
            bbox_y = y1
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Confidence
            score = float(
                boxes.conf[i].item()
            )

            rows.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": round(bbox_x, 2),
                "bbox_y": round(bbox_y, 2),
                "bbox_w": round(bbox_w, 2),
                "bbox_h": round(bbox_h, 2),
                "score": round(score, 6)
            })

    return rows


# ============================================================
# 7. Submission CSV 저장
# ============================================================

def save_submission(rows, conf):

    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score"
    ]
    submission_path = (
        PROJECT_ROOT / f"submission_conf{conf:.2f}.csv"
    )
    with open(
        submission_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for annotation_id, row in enumerate(
            rows,
            start=1
        ):

            writer.writerow({
                "annotation_id": annotation_id,
                **row
            })
    return submission_path

# ============================================================
# 8. Main
# ============================================================

def main():

    args = parse_args()

    print("=" * 60)
    print("YOLO Prediction 시작")
    print("=" * 60)

    # 경로 확인
    checkpoint_path = Path(
        args.checkpoint
    )

    image_path = Path(
        args.image
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다:\n"
            f"{checkpoint_path}"
        )

    if not image_path.exists():

        raise FileNotFoundError(
            f"이미지 또는 이미지 폴더를 찾을 수 없습니다:\n"
            f"{image_path}"
        )

    # Category mapping
    class_to_category = (
        create_class_to_category_mapping()
    )

    print("\n[Category Mapping]")
    print(
        "총 클래스 수:",
        len(class_to_category)
    )

    # 모델 로드
    print("\n[Model]")
    print(
        "Checkpoint:",
        checkpoint_path
    )

    model = YOLO(
        str(checkpoint_path)
    )

    # Prediction
    print("\n[Prediction]")
    print(
        "Input:",
        image_path
    )

    rows = predict(
        model=model,
        image_path=image_path,
        class_to_category=class_to_category,
        conf=args.conf
    )

    # Submission 저장
    submission_path = save_submission(
    rows,
    args.conf
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("Prediction 완료")
    print("=" * 60)

    print(
        "Submission:",
        submission_path
    )

    print(
        "총 예측 객체:",
        len(rows)
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()