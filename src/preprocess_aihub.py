import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 기존 데이터
ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_annotations"
)

# AI-Hub 원본
AIHUB_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "aihub"
    / "images"
)

AIHUB_LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "aihub"
    / "labels"
)

# AI-Hub 전처리 결과
AIHUB_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "aihub_processed"
)

AIHUB_PROCESSED_IMAGE_DIR = (
    AIHUB_PROCESSED_DIR
    / "images"
)

AIHUB_PROCESSED_LABEL_DIR = (
    AIHUB_PROCESSED_DIR
    / "labels"
)


# ============================================================
# 2. 설정
# ============================================================

MAX_IMAGES_PER_CLASS = 200
SEED = 42

random.seed(SEED)


# ============================================================
# 3. 기존 데이터의 56개 클래스 수집
# ============================================================

def collect_original_classes():

    category_ids = set()

    for json_path in ANNOTATION_DIR.rglob("*.json"):

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

        except Exception:
            continue

        for ann in data.get("annotations", []):

            category_id = ann.get("category_id")

            if category_id is not None:
                category_ids.add(
                    int(category_id)
                )

    category_ids = sorted(category_ids)

    category_to_class = {
        category_id: class_id
        for class_id, category_id
        in enumerate(category_ids)
    }

    return category_to_class


# ============================================================
# 4. AI-Hub 실제 이미지 수집
# ============================================================

def collect_existing_images():

    existing_images = {}

    duplicate_names = defaultdict(list)

    for image_path in AIHUB_IMAGE_DIR.rglob("*"):

        if not image_path.is_file():
            continue

        file_name = image_path.name

        duplicate_names[file_name].append(
            image_path
        )

        if file_name not in existing_images:
            existing_images[file_name] = image_path

    duplicate_count = sum(
        1
        for paths in duplicate_names.values()
        if len(paths) > 1
    )

    return existing_images, duplicate_count


# ============================================================
# 5. AI-Hub 데이터 수집
# ============================================================

def collect_aihub_data(category_to_class):

    """
    AI-Hub JSON을 기준으로 데이터를 수집합니다.

    핵심:
    - images[].drug_N → 기존 category_id로 변환
    - annotation은 해당 image_id에 연결
    - 같은 이미지가 여러 JSON에 등장하면
      각 JSON에서 다른 클래스 annotation을 누적
    """

    images_by_class = defaultdict(set)

    image_annotations = defaultdict(list)

    parse_errors = 0

    json_files = list(
        AIHUB_LABEL_DIR.rglob("*.json")
    )

    total_jsons = len(json_files)

    print(
        f"총 JSON 파일: "
        f"{total_jsons:,}개"
    )

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        # 진행률
        if index % 1000 == 0:

            print(
                f"JSON 처리 중... "
                f"{index:,} / {total_jsons:,}"
            )

        try:
            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

        except Exception:
            parse_errors += 1
            continue

        # ----------------------------------------------------
        # 이미지 정보
        # ----------------------------------------------------

        image_info = {}

        for img in data.get("images", []):

            image_id = img.get("id")
            file_name = img.get("file_name")
            drug_n = img.get("drug_N")

            if (
                image_id is None
                or not file_name
                or not drug_n
            ):
                continue

            # 예:
            # K-001900 → 1900

            try:
                category_id = int(
                    drug_n.split("-")[-1]
                )
            except Exception:
                continue

            # 기존 56개 클래스만 사용
            if category_id not in category_to_class:
                continue

            image_info[image_id] = {
                "file_name": file_name,
                "category_id": category_id,
                "width": img.get(
                    "width",
                    976
                ),
                "height": img.get(
                    "height",
                    1280
                ),
            }

            # 클래스별 이미지 목록
            images_by_class[
                category_id
            ].add(file_name)

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        for ann in data.get(
            "annotations",
            []
        ):

            image_id = ann.get(
                "image_id"
            )

            bbox = ann.get(
                "bbox"
            )

            if image_id not in image_info:
                continue

            if not bbox or len(bbox) != 4:
                continue

            info = image_info[
                image_id
            ]

            # 중요:
            # annotation.category_id를 사용하지 않고
            # 해당 JSON의 images[].drug_N으로
            # 실제 클래스 결정

            image_annotations[
                info["file_name"]
            ].append({

                "category_id": (
                    info["category_id"]
                ),

                "bbox": bbox,

                "width": info["width"],

                "height": info["height"]
            })

    return (
        images_by_class,
        image_annotations,
        parse_errors
    )


# ============================================================
# 6. Annotation 중복 제거
# ============================================================

def deduplicate_annotations(
    image_annotations
):

    deduplicated = defaultdict(list)

    for file_name, annotations in (
        image_annotations.items()
    ):

        seen = set()

        for ann in annotations:

            bbox = tuple(
                round(
                    float(value),
                    4
                )
                for value in ann["bbox"]
            )

            key = (
                ann["category_id"],
                bbox
            )

            if key in seen:
                continue

            seen.add(key)

            deduplicated[
                file_name
            ].append(ann)

    return deduplicated


# ============================================================
# 7. 클래스별 최대 200장 선택
# ============================================================

def select_images_by_class(
    images_by_class,
    existing_images,
    category_to_class
):

    selected_images_by_class = {}

    for category_id in sorted(
        category_to_class
    ):

        # 실제 이미지가 존재하는 것만
        available = [
            file_name
            for file_name
            in images_by_class[
                category_id
            ]
            if file_name in existing_images
        ]

        random.shuffle(
            available
        )

        selected_images_by_class[
            category_id
        ] = available[
            :MAX_IMAGES_PER_CLASS
        ]

    # 모든 클래스의 선택 이미지를 합치기
    selected_files = set()

    for images in (
        selected_images_by_class.values()
    ):

        selected_files.update(
            images
        )

    return (
        selected_images_by_class,
        selected_files
    )


# ============================================================
# 8. COCO → YOLO
# ============================================================

def coco_to_yolo(
    bbox,
    image_width,
    image_height
):

    x, y, w, h = map(
        float,
        bbox
    )

    x_center = (
        x + w / 2
    )

    y_center = (
        y + h / 2
    )

    x_center /= image_width
    y_center /= image_height

    w /= image_width
    h /= image_height

    return (
        x_center,
        y_center,
        w,
        h
    )


# ============================================================
# 9. YOLO 데이터셋 생성
# ============================================================

def create_processed_dataset(
    selected_files,
    existing_images,
    deduplicated_annotations,
    category_to_class
):

    # 기존 결과 삭제
    if AIHUB_PROCESSED_DIR.exists():

        shutil.rmtree(
            AIHUB_PROCESSED_DIR
        )

    AIHUB_PROCESSED_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AIHUB_PROCESSED_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    copied_images = 0
    created_labels = 0
    skipped_images = 0

    class_image_count = defaultdict(set)
    class_annotation_count = defaultdict(int)

    # --------------------------------------------------------
    # 이미지 처리
    # --------------------------------------------------------

    for file_name in sorted(
        selected_files
    ):

        if file_name not in existing_images:
            continue

        annotations = (
            deduplicated_annotations.get(
                file_name,
                []
            )
        )

        if not annotations:

            skipped_images += 1

            continue

        image_path = (
            existing_images[file_name]
        )

        destination_image = (
            AIHUB_PROCESSED_IMAGE_DIR
            / file_name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        label_path = (
            AIHUB_PROCESSED_LABEL_DIR
            / f"{Path(file_name).stem}.txt"
        )

        valid_lines = []

        # ----------------------------------------------------
        # Annotation 처리
        # ----------------------------------------------------

        for ann in annotations:

            category_id = (
                ann["category_id"]
            )

            if category_id not in category_to_class:
                continue

            image_width = float(
                ann["width"]
            )

            image_height = float(
                ann["height"]
            )

            (
                x_center,
                y_center,
                w,
                h
            ) = coco_to_yolo(
                ann["bbox"],
                image_width,
                image_height
            )

            # bbox 검증
            if w <= 0 or h <= 0:
                continue

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                continue

            class_id = (
                category_to_class[
                    category_id
                ]
            )

            valid_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

            class_image_count[
                category_id
            ].add(file_name)

            class_annotation_count[
                category_id
            ] += 1

        # ----------------------------------------------------
        # 유효 annotation 없음
        # ----------------------------------------------------

        if not valid_lines:

            destination_image.unlink()

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Label 저장
        # ----------------------------------------------------

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(valid_lines)
            )

        copied_images += 1
        created_labels += 1

    return (
        copied_images,
        created_labels,
        skipped_images,
        class_image_count,
        class_annotation_count
    )


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AI-Hub 데이터 전처리 시작")
    print("=" * 70)

    # --------------------------------------------------------
    # 기존 56개 클래스
    # --------------------------------------------------------

    category_to_class = (
        collect_original_classes()
    )

    print(
        f"\n기존 클래스 수: "
        f"{len(category_to_class)}"
    )

    if len(category_to_class) != 56:

        print(
            "⚠️ 기존 클래스가 56개가 아닙니다."
        )

        raise SystemExit

    # --------------------------------------------------------
    # 실제 AI-Hub 이미지
    # --------------------------------------------------------

    (
        existing_images,
        duplicate_count
    ) = collect_existing_images()

    print(
        f"AI-Hub 실제 이미지 파일: "
        f"{len(existing_images):,}개"
    )

    if duplicate_count:

        print(
            f"⚠️ 동일 파일명 중복: "
            f"{duplicate_count:,}개"
        )

    # --------------------------------------------------------
    # AI-Hub Annotation 수집
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AI-Hub Annotation 수집")
    print("=" * 70)

    (
        images_by_class,
        image_annotations,
        parse_errors
    ) = collect_aihub_data(
        category_to_class
    )

    print(
        f"\n기존 56개 클래스가 포함된 "
        f"AI-Hub 고유 이미지: "
        f"{len(image_annotations):,}개"
    )

    print(
        f"JSON 파싱 오류: "
        f"{parse_errors}개"
    )

    # --------------------------------------------------------
    # Annotation 중복 제거
    # --------------------------------------------------------

    deduplicated_annotations = (
        deduplicate_annotations(
            image_annotations
        )
    )

    # --------------------------------------------------------
    # 클래스별 이미지 선택
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"클래스별 이미지 선택 "
        f"(최대 {MAX_IMAGES_PER_CLASS}장)"
    )
    print("=" * 70)

    (
        selected_images_by_class,
        selected_files
    ) = select_images_by_class(
        images_by_class,
        existing_images,
        category_to_class
    )

    for category_id in sorted(
        category_to_class
    ):

        selected_count = len(
            selected_images_by_class[
                category_id
            ]
        )

        print(
            f"{category_id:>6} | "
            f"class "
            f"{category_to_class[category_id]:>2} | "
            f"{selected_count:>3}장"
        )

    print(
        f"\n선택된 전체 고유 이미지: "
        f"{len(selected_files):,}개"
    )

    # --------------------------------------------------------
    # YOLO 생성
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("YOLO 데이터셋 생성")
    print("=" * 70)

    (
        copied_images,
        created_labels,
        skipped_images,
        class_image_count,
        class_annotation_count
    ) = create_processed_dataset(
        selected_files,
        existing_images,
        deduplicated_annotations,
        category_to_class
    )

    # --------------------------------------------------------
    # 최종 결과
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AI-Hub 전처리 완료")
    print("=" * 70)

    print(
        f"매칭된 고유 이미지: "
        f"{len(image_annotations):,}개"
    )

    print(
        f"선택된 고유 이미지: "
        f"{len(selected_files):,}개"
    )

    print(
        f"복사된 이미지: "
        f"{copied_images:,}개"
    )

    print(
        f"생성된 Label: "
        f"{created_labels:,}개"
    )

    print(
        f"제외된 이미지: "
        f"{skipped_images:,}개"
    )

    print(
        f"JSON 파싱 오류: "
        f"{parse_errors:,}개"
    )

    # --------------------------------------------------------
    # 클래스별 결과
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("클래스별 결과")
    print("=" * 70)

    for category_id in sorted(
        category_to_class
    ):

        print(
            f"{category_id:>6} | "
            f"class "
            f"{category_to_class[category_id]:>2} | "
            f"이미지 "
            f"{len(class_image_count[category_id]):>3} | "
            f"annotation "
            f"{class_annotation_count[category_id]:>5}"
        )

    print("=" * 70)

    print(
        f"저장 위치: "
        f"{AIHUB_PROCESSED_DIR}"
    )

    print("=" * 70)