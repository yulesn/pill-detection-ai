import csv
import json
import os
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_annotations"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "test_images"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "default2"
    / "weights"
    / "best.pt"
)

SUBMISSION_PATH = (
    PROJECT_ROOT
    / "submission.csv"
)


# ============================================================
# 2. EDA에서 제외했던 이미지
# ============================================================

EXCLUDE_IMAGE_IDS = {
    16, 193, 208, 239, 783, 907,
    1228, 1258, 1267, 1383, 1405, 1432
}


# ============================================================
# 3. Annotation 수집
# ============================================================

def collect_category_ids(annotation_dir):
    """
    train annotation에서 category_id를 수집한다.

    preprocess.py와 동일하게
    EDA 이상 데이터를 제외한 후
    category_id → YOLO class_id 매핑을 생성한다.
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

            # 이미지 ID를 먼저 확인
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

                # preprocess.py와 동일하게
                # 잘못된 BBox 제외
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
# 4. category_id ↔ YOLO class_id 매핑
# ============================================================

def create_class_to_category_mapping():
    """
    preprocess.py와 동일한 규칙으로
    YOLO class_id → 원본 category_id 매핑을 생성한다.
    """

    category_ids = collect_category_ids(
        ANNOTATION_DIR
    )

    class_to_category = {
        class_id: category_id
        for class_id, category_id
        in enumerate(category_ids)
    }

    return class_to_category


# ============================================================
# 5. Submission 생성
# ============================================================

def create_submission():

    print("=" * 60)
    print("Kaggle Submission 생성 시작")
    print("=" * 60)

    # --------------------------------------------------------
    # 모델 확인
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다:\n{MODEL_PATH}"
        )

    if not TEST_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Test 이미지 폴더를 찾을 수 없습니다:\n{TEST_IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Category mapping
    # --------------------------------------------------------

    class_to_category = (
        create_class_to_category_mapping()
    )

    print("\n[Category Mapping]")
    print(
        "총 클래스 수:",
        len(class_to_category)
    )

    for class_id, category_id in list(
        class_to_category.items()
    )[:10]:

        print(
            f"class_id: {class_id} "
            f"→ category_id: {category_id}"
        )

    # --------------------------------------------------------
    # 모델 로드
    # --------------------------------------------------------

    print("\n[Model]")
    print("모델:", MODEL_PATH)

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # Test 이미지 예측
    # --------------------------------------------------------

    print("\n[Prediction]")
    print("Test 이미지:", TEST_IMAGE_DIR)

    results = model.predict(
        source=str(TEST_IMAGE_DIR),
        conf=0.25,
        save=False,
        verbose=True,
        stream=True
    )

    # --------------------------------------------------------
    # CSV 데이터 생성
    # --------------------------------------------------------

    rows = []

    annotation_id = 1

    for result in results:

        # 이미지 파일명
        image_path = Path(
            result.path
        )

        # 예:
        # 984.png → 984
        image_id = int(
            image_path.stem
        )

        boxes = result.boxes

        if boxes is None:
            continue

        for i in range(
            len(boxes)
        ):

            # YOLO class_id
            class_id = int(
                boxes.cls[i].item()
            )

            # 원본 category_id
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

            # COCO 스타일 XYWH로 변환
            bbox_x = x1
            bbox_y = y1
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Confidence
            score = float(
                boxes.conf[i].item()
            )

            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": round(bbox_x, 2),
                "bbox_y": round(bbox_y, 2),
                "bbox_w": round(bbox_w, 2),
                "bbox_h": round(bbox_h, 2),
                "score": round(score, 6)
            })

            annotation_id += 1

    # --------------------------------------------------------
    # CSV 저장
    # --------------------------------------------------------

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

    with open(
        SUBMISSION_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Submission 생성 완료")
    print("=" * 60)

    print(
        "파일:",
        SUBMISSION_PATH
    )

    print(
        "총 예측 객체:",
        len(rows)
    )

    print(
        "마지막 annotation_id:",
        annotation_id - 1
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    create_submission()