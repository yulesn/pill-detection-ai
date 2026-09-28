from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import random
import shutil
import yaml


# ============================================================
# AI-Hub 데이터 정제 + 1,000장 선별 + Combined Dataset 생성
# ============================================================
#
# 처리 흐름
#
# AI-Hub 원본 JSON / 이미지
#           ↓
# ① 56개 대회 클래스 매칭
#           ↓
# ② 이미지 존재 여부 검사
#           ↓
# ③ JSON / annotation 검증
#           ↓
# ④ BBox 이상치 제거
#           ↓
# ⑤ 동일 이미지 annotation 통합
#           ↓
# ⑥ 중복 annotation 제거
#           ↓
# ⑦ 클래스 불균형 기반 1,000장 선별
#           ↓
# ⑧ 기존 Train + AI-Hub 1,000장
#           ↓
# combined_augmented
#
# ============================================================


# ============================================================
# 1. 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 기존 대회 전처리 데이터
ORIGINAL_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

# AI-Hub 원본
AIHUB_DIR = (
    PROJECT_ROOT
    / "data"
    / "aihub"
)

AIHUB_IMAGE_DIR = (
    AIHUB_DIR
    / "images"
)

AIHUB_LABEL_DIR = (
    AIHUB_DIR
    / "labels"
)

# AI-Hub 기존 클래스 매핑 통계 파일
CLASS_MAPPING_FILE = (
    PROJECT_ROOT
    / "data"
    / "aihub_class_counts.txt"
)

# 최종 데이터셋
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "combined_augmented"
)

# 정제 결과 CSV
CLEANED_CSV = (
    OUTPUT_DIR
    / "cleaned_aihub.csv"
)

# 선택 결과 CSV
SELECTED_CSV = (
    OUTPUT_DIR
    / "selected_aihub.csv"
)

NUM_CLASSES = 56

TARGET_AIHUB_IMAGES = 1000

SEED = 42

RESET_OUTPUT = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# 한 이미지에서 같은 class가 지나치게 많은 경우
# 선별 점수에 최대 3개까지만 반영
MAX_OBJECTS_PER_CLASS_PER_IMAGE = 3

# 너무 작은 BBox 제거 기준
# 전체 이미지 면적 대비 BBox 면적 비율
MIN_BBOX_AREA_RATIO = 0.0005

# BBox가 이미지 밖으로 벗어나더라도
# 최대 5%까지만 허용
MAX_OUTSIDE_RATIO = 0.05


# ============================================================
# 2. 기존 Train 클래스 분포
# ============================================================

def count_original_train_classes():

    label_dir = (
        ORIGINAL_PROCESSED_DIR
        / "labels"
        / "train"
    )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"기존 Train label 폴더가 없습니다:\n{label_dir}"
        )

    counts = Counter()

    label_files = list(
        label_dir.glob("*.txt")
    )

    print("\n" + "=" * 70)
    print("기존 Train 클래스 분포")
    print("=" * 70)

    print(
        f"Train label 수: {len(label_files):,}"
    )

    for label_file in label_files:

        with open(
            label_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue

                if 0 <= class_id < NUM_CLASSES:
                    counts[class_id] += 1

    total = sum(counts.values())

    for class_id in range(NUM_CLASSES):

        count = counts[class_id]

        ratio = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"class_{class_id:02d} | "
            f"{count:4d}개 | "
            f"{ratio:5.2f}%"
        )

    print(
        f"\n전체 객체 수: {total:,}"
    )

    return counts


# ============================================================
# 3. 기존 56개 category_id → YOLO class_id
# ============================================================

def build_category_mapping():

    """
    data/aihub_class_counts.txt에서
    기존 대회 56개 category_id를 읽는다.

    파일 예시:

        3351 | 약품명 | 0개
        3483 | 약품명 | 0개
        1900 | 보령부스파정 5mg | 151개

    preprocess.py와 동일하게
    category_id를 숫자순으로 정렬한 뒤
    0부터 YOLO class_id를 부여한다.
    """

    print("\n" + "=" * 70)
    print("기존 대회 category_id → YOLO class_id 매핑")
    print("=" * 70)

    print(
        f"매핑 파일:\n{CLASS_MAPPING_FILE}"
    )

    if not CLASS_MAPPING_FILE.exists():

        raise FileNotFoundError(
            f"""
클래스 매핑 파일을 찾을 수 없습니다.

{CLASS_MAPPING_FILE}
"""
        )

    category_ids = []

    with open(
        CLASS_MAPPING_FILE,
        "r",
        encoding="utf-8-sig",
        errors="ignore",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if "|" not in line:
                continue

            # 예:
            # 3351 | 약품명 | 0개
            #
            # 첫 번째 | 앞부분만 사용
            first_part = (
                line.split("|", 1)[0]
                .strip()
            )

            if not first_part.isdigit():
                continue

            category_id = int(first_part)

            category_ids.append(
                category_id
            )

    print(
        f"\n읽은 category_id 수: "
        f"{len(category_ids)}"
    )

    # 중복 확인
    if len(category_ids) != len(set(category_ids)):

        duplicates = [
            category_id
            for category_id, count
            in Counter(category_ids).items()
            if count > 1
        ]

        raise RuntimeError(
            f"""
aihub_class_counts.txt에
중복 category_id가 있습니다.

중복:
{duplicates}
"""
        )

    # 숫자순 정렬
    category_ids = sorted(
        category_ids
    )

    # 반드시 56개
    if len(category_ids) != NUM_CLASSES:

        print(
            "\n읽은 category_id 목록:"
        )

        print(category_ids)

        raise RuntimeError(
            f"""
기존 대회 클래스 수가
{NUM_CLASSES}개가 아닙니다.

현재:
{len(category_ids)}개
"""
        )

    # category_id → YOLO class_id
    category_to_class = {
        category_id: class_id
        for class_id, category_id
        in enumerate(category_ids)
    }

    print(
        f"\n최종 매핑 클래스 수: "
        f"{len(category_to_class)}"
    )

    print("\ncategory_id → class_id")

    print("-" * 40)

    for category_id, class_id in (
        category_to_class.items()
    ):

        print(
            f"category_id {category_id:5d}"
            f" → class_id {class_id:2d}"
        )

    return category_to_class


# ============================================================
# 4. JSON 파일 수집
# ============================================================

def collect_json_files():

    if not AIHUB_LABEL_DIR.exists():

        raise FileNotFoundError(
            f"""
AI-Hub label 폴더가 없습니다.

{AIHUB_LABEL_DIR}
"""
        )

    json_files = list(
        AIHUB_LABEL_DIR.rglob("*.json")
    )

    print("\n" + "=" * 70)
    print("AI-Hub JSON 수집")
    print("=" * 70)

    print(
        f"JSON 파일: "
        f"{len(json_files):,}개"
    )

    return json_files


# ============================================================
# 5. 이미지 검색 인덱스
# ============================================================

def build_image_index():

    if not AIHUB_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"""
AI-Hub 이미지 폴더가 없습니다.

{AIHUB_IMAGE_DIR}
"""
        )

    print("\n" + "=" * 70)
    print("AI-Hub 이미지 인덱스 생성")
    print("=" * 70)

    image_index = defaultdict(list)

    image_files = [
        p
        for p in AIHUB_IMAGE_DIR.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    for image_path in image_files:

        image_index[
            image_path.name
        ].append(image_path)

    print(
        f"이미지 파일: "
        f"{len(image_files):,}장"
    )

    print(
        f"고유 파일명: "
        f"{len(image_index):,}개"
    )

    duplicate_names = sum(
        1
        for paths in image_index.values()
        if len(paths) > 1
    )

    print(
        f"동일 파일명 중복: "
        f"{duplicate_names:,}개"
    )

    return image_index


# ============================================================
# 6. 숫자 변환
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# 7. BBox 검증
# ============================================================

def validate_bbox(
    bbox,
    image_width,
    image_height,
):

    if not isinstance(
        bbox,
        list,
    ):

        return False, "bbox_not_list"

    if len(bbox) != 4:

        return False, "bbox_length"

    x = safe_float(bbox[0])
    y = safe_float(bbox[1])
    w = safe_float(bbox[2])
    h = safe_float(bbox[3])

    if None in (
        x,
        y,
        w,
        h,
    ):

        return False, "bbox_non_numeric"

    if w <= 0 or h <= 0:

        return False, "bbox_non_positive"

    if (
        image_width <= 0
        or image_height <= 0
    ):

        return False, "invalid_image_size"

    # COCO 기준
    # [x, y, width, height]

    x2 = x + w
    y2 = y + h

    # --------------------------------------------------------
    # 지나치게 작은 BBox
    # --------------------------------------------------------

    area_ratio = (
        w * h
        / (
            image_width
            * image_height
        )
    )

    if area_ratio < MIN_BBOX_AREA_RATIO:

        return False, "bbox_too_small"

    # --------------------------------------------------------
    # 이미지 바깥 영역 계산
    # --------------------------------------------------------

    outside_left = max(
        0,
        -x,
    )

    outside_top = max(
        0,
        -y,
    )

    outside_right = max(
        0,
        x2 - image_width,
    )

    outside_bottom = max(
        0,
        y2 - image_height,
    )

    outside_area = (
        outside_left * h
        + outside_right * h
        + outside_top * w
        + outside_bottom * w
    )

    bbox_area = w * h

    outside_ratio = (
        outside_area / bbox_area
        if bbox_area > 0
        else 1
    )

    if (
        outside_ratio
        > MAX_OUTSIDE_RATIO
    ):

        return False, "bbox_outside_image"

    # --------------------------------------------------------
    # 이미지 내부로 clipping
    # --------------------------------------------------------

    clipped_x1 = max(
        0,
        x,
    )

    clipped_y1 = max(
        0,
        y,
    )

    clipped_x2 = min(
        image_width,
        x2,
    )

    clipped_y2 = min(
        image_height,
        y2,
    )

    clipped_w = (
        clipped_x2
        - clipped_x1
    )

    clipped_h = (
        clipped_y2
        - clipped_y1
    )

    if (
        clipped_w <= 0
        or clipped_h <= 0
    ):

        return False, "bbox_no_overlap"

    return True, {
        "x": clipped_x1,
        "y": clipped_y1,
        "w": clipped_w,
        "h": clipped_h,
    }


# ============================================================
# 8. AI-Hub 이미지 경로 선택
# ============================================================

def choose_best_image(
    paths,
    json_path,
):

    if len(paths) == 1:

        return paths[0]

    # JSON 경로와 이미지 경로에서
    # 공통으로 포함되는 폴더명을 비교
    json_parts = set(
        json_path.parts
    )

    def score(path):

        path_parts = set(
            path.parts
        )

        return len(
            json_parts
            & path_parts
        )

    return max(
        paths,
        key=score,
    )


# ============================================================
# 9. AI-Hub JSON → 후보 데이터 정제
# ============================================================

def clean_aihub_candidates(
    json_files,
    image_index,
    category_to_class,
):

    print("\n" + "=" * 70)
    print("AI-Hub 데이터 정제 시작")
    print("=" * 70)

    # 동일 이미지가 여러 JSON에 나뉘어 있을 수 있음
    # → 이미지 단위로 annotation 통합

    image_records = {}

    stats = Counter()

    total_json = len(
        json_files
    )

    for index, json_path in enumerate(
        json_files,
        start=1,
    ):

        if (
            index % 1000 == 0
            or index == 1
            or index == total_json
        ):

            print(
                f"JSON 처리: "
                f"{index:,}/"
                f"{total_json:,}"
            )

        # ----------------------------------------------------
        # JSON 로드
        # ----------------------------------------------------

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

        except Exception:

            stats[
                "json_error"
            ] += 1

            continue

        # ----------------------------------------------------
        # images / annotations
        # ----------------------------------------------------

        images = data.get(
            "images",
            [],
        )

        annotations = data.get(
            "annotations",
            [],
        )

        if not images:

            stats[
                "no_images"
            ] += 1

            continue

        if not annotations:

            stats[
                "no_annotations"
            ] += 1

            continue

        # ----------------------------------------------------
        # image 정보
        # ----------------------------------------------------

        image_info_map = {}

        for image_info in images:

            file_name = (
                image_info.get(
                    "file_name"
                )
            )

            if not file_name:

                stats[
                    "filename_missing"
                ] += 1

                continue

            image_key = image_info.get(
                "id",
                file_name,
            )

            image_info_map[
                image_key
            ] = image_info

        # ----------------------------------------------------
        # annotation 처리
        # ----------------------------------------------------

        for annotation in annotations:

            # AI-Hub category_id는
            # Drug(1)이므로 클래스 매칭에는 사용하지 않음

            image_id = annotation.get(
                "image_id"
            )

            image_info = image_info_map.get(
                image_id
            )

            if image_info is None:

                stats[
                    "image_info_missing"
                ] += 1

                continue

            file_name = (
                image_info.get(
                    "file_name"
                )
            )

            if not file_name:

                stats[
                    "filename_missing"
                ] += 1

                continue

            # ------------------------------------------------
            # AI-Hub 실제 약품 종류
            # ------------------------------------------------

            drug_name = (
                image_info.get(
                    "drug_N"
                )
            )

            if not drug_name:

                drug_name = (
                    annotation.get(
                        "drug_N"
                    )
                )

            if not drug_name:

                stats[
                    "drug_name_missing"
                ] += 1

                continue

            # ------------------------------------------------
            # K-001900 → 1900
            # ------------------------------------------------

            try:

                category_id = int(
                    str(drug_name)
                    .replace(
                        "K-",
                        "",
                    )
                    .strip()
                )

            except ValueError:

                stats[
                    "drug_name_invalid"
                ] += 1

                continue

            # ------------------------------------------------
            # 기존 대회 category_id → YOLO class_id
            # ------------------------------------------------

            class_id = category_to_class.get(
                category_id
            )

            if class_id is None:

                stats[
                    "unmatched_class"
                ] += 1

                continue

            # ------------------------------------------------
            # 이미지 찾기
            # ------------------------------------------------

            matched_images = (
                image_index.get(
                    Path(
                        file_name
                    ).name,
                    [],
                )
            )

            if not matched_images:

                stats[
                    "image_missing"
                ] += 1

                continue

            image_path = choose_best_image(
                matched_images,
                json_path,
            )

            # ------------------------------------------------
            # 이미지 크기
            # ------------------------------------------------

            image_width = safe_float(
                image_info.get(
                    "width"
                )
            )

            image_height = safe_float(
                image_info.get(
                    "height"
                )
            )

            if (
                image_width is None
                or image_height is None
                or image_width <= 0
                or image_height <= 0
            ):

                stats[
                    "invalid_image_size"
                ] += 1

                continue

            # ------------------------------------------------
            # BBox
            # ------------------------------------------------

            bbox = annotation.get(
                "bbox"
            )

            valid, bbox_result = (
                validate_bbox(
                    bbox,
                    image_width,
                    image_height,
                )
            )

            if not valid:

                stats[
                    f"bbox_{bbox_result}"
                ] += 1

                continue

            # ------------------------------------------------
            # 이미지별 annotation 저장
            # ------------------------------------------------

            key = str(
                image_path.resolve()
            )

            if key not in image_records:

                image_records[key] = {

                    "image_path": image_path,

                    "width": image_width,

                    "height": image_height,

                    "annotations": [],
                }

            image_records[key][
                "annotations"
            ].append(
                {
                    "class_id": class_id,

                    "category_id": category_id,

                    "bbox": bbox_result,
                }
            )

            stats[
                "valid_annotations"
            ] += 1

    # ========================================================
    # 이미지 단위 정리
    # ========================================================

    candidates = []

    for record in image_records.values():

        annotations = record[
            "annotations"
        ]

        if not annotations:

            continue

        # ----------------------------------------------------
        # 동일 annotation 중복 제거
        # ----------------------------------------------------

        unique_annotations = []

        seen = set()

        for ann in annotations:

            bbox = ann[
                "bbox"
            ]

            key = (
                ann["class_id"],
                round(
                    bbox["x"],
                    3,
                ),
                round(
                    bbox["y"],
                    3,
                ),
                round(
                    bbox["w"],
                    3,
                ),
                round(
                    bbox["h"],
                    3,
                ),
            )

            if key in seen:

                stats[
                    "duplicate_annotations"
                ] += 1

                continue

            seen.add(key)

            unique_annotations.append(
                ann
            )

        if not unique_annotations:

            continue

        # ----------------------------------------------------
        # 이미지별 클래스 개수
        # ----------------------------------------------------

        class_counts = Counter()

        for ann in unique_annotations:

            class_counts[
                ann["class_id"]
            ] += 1

        candidates.append(
            {
                "image_path": record[
                    "image_path"
                ],

                "width": record[
                    "width"
                ],

                "height": record[
                    "height"
                ],

                "annotations": unique_annotations,

                "class_counts": class_counts,
            }
        )

    # ========================================================
    # 정제 통계
    # ========================================================

    print("\n" + "=" * 70)
    print("AI-Hub 정제 결과")
    print("=" * 70)

    print(
        f"최종 후보 이미지: "
        f"{len(candidates):,}장"
    )

    print(
        f"유효 annotation: "
        f"{stats['valid_annotations']:,}개"
    )

    print(
        f"JSON 오류: "
        f"{stats['json_error']:,}"
    )

    print(
        f"클래스 미매칭: "
        f"{stats['unmatched_class']:,}"
    )

    print(
        f"이미지 없음: "
        f"{stats['image_missing']:,}"
    )

    print(
        f"중복 annotation 제거: "
        f"{stats['duplicate_annotations']:,}"
    )

    print("\n[BBox 제거 내역]")

    bbox_reasons = [
        key
        for key in stats
        if key.startswith("bbox_")
    ]

    if bbox_reasons:

        for reason in sorted(
            bbox_reasons
        ):

            print(
                f"{reason:<35}"
                f"{stats[reason]:>8}"
            )

    else:

        print("제거된 BBox 없음")

    return candidates, stats


# ============================================================
# 10. 정제 결과 CSV
# ============================================================

def save_cleaned_csv(
    candidates,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CLEANED_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "image",
                "annotation_count",
                "classes",
            ]
        )

        for candidate in candidates:

            writer.writerow(
                [
                    candidate[
                        "image_path"
                    ].name,

                    len(
                        candidate[
                            "annotations"
                        ]
                    ),

                    ",".join(
                        str(class_id)
                        for class_id in sorted(
                            candidate[
                                "class_counts"
                            ]
                        )
                    ),
                ]
            )

    print(
        f"\n정제 결과 저장:\n"
        f"{CLEANED_CSV}"
    )


# ============================================================
# 11. 클래스 불균형 기반 1,000장 선별
# ============================================================

def select_balanced_subset(
    candidates,
    original_counts,
):

    print("\n" + "=" * 70)
    print("정제된 AI-Hub 후보에서 1,000장 선별")
    print("=" * 70)

    target_count = min(
        TARGET_AIHUB_IMAGES,
        len(candidates),
    )

    if target_count == 0:

        return []

    random.seed(
        SEED
    )

    # 기존 Train 클래스 개수부터 시작
    current_counts = Counter(
        original_counts
    )

    remaining = list(
        candidates
    )

    selected = []

    for step in range(
        target_count
    ):

        best_index = None

        best_score = -float(
            "inf"
        )

        for index, candidate in enumerate(
            remaining
        ):

            score = 0.0

            # ------------------------------------------------
            # 클래스별 희소성 점수
            # ------------------------------------------------

            for (
                class_id,
                object_count,
            ) in candidate[
                "class_counts"
            ].items():

                capped_count = min(
                    object_count,
                    MAX_OBJECTS_PER_CLASS_PER_IMAGE,
                )

                rarity_weight = (
                    1.0
                    / math.sqrt(
                        current_counts[
                            class_id
                        ]
                        + 1
                    )
                )

                score += (
                    capped_count
                    * rarity_weight
                )

            # ------------------------------------------------
            # annotation 너무 많은 이미지 완화
            # ------------------------------------------------

            annotation_count = len(
                candidate[
                    "annotations"
                ]
            )

            if annotation_count > 5:

                score *= (
                    5.0
                    / annotation_count
                )

            # 아주 작은 랜덤값
            # 동점일 때만 영향
            score += (
                random.random()
                * 1e-9
            )

            if score > best_score:

                best_score = score

                best_index = index

        # ----------------------------------------------------
        # 선택
        # ----------------------------------------------------

        selected_candidate = (
            remaining.pop(
                best_index
            )
        )

        selected.append(
            selected_candidate
        )

        # ----------------------------------------------------
        # 현재 클래스 분포 갱신
        # ----------------------------------------------------

        for (
            class_id,
            object_count,
        ) in selected_candidate[
            "class_counts"
        ].items():

            current_counts[
                class_id
            ] += object_count

        if (
            (step + 1) % 100 == 0
            or step + 1 == target_count
        ):

            print(
                f"선택 진행: "
                f"{step + 1:,}/"
                f"{target_count:,}"
            )

    print(
        f"\n최종 AI-Hub 선택: "
        f"{len(selected):,}장"
    )

    return selected


# ============================================================
# 12. 출력 폴더 준비
# ============================================================

def prepare_output_dir():

    if (
        OUTPUT_DIR.exists()
        and RESET_OUTPUT
    ):

        print(
            "\n기존 combined_augmented 삭제 중..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    (
        OUTPUT_DIR
        / "images"
        / "train"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "labels"
        / "train"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "images"
        / "val"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "labels"
        / "val"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 13. 기존 Train / Val 복사
# ============================================================

def copy_original_dataset():

    print("\n" + "=" * 70)
    print("기존 대회 데이터 복사")
    print("=" * 70)

    original_train_images = (
        ORIGINAL_PROCESSED_DIR
        / "images"
        / "train"
    )

    original_train_labels = (
        ORIGINAL_PROCESSED_DIR
        / "labels"
        / "train"
    )

    original_val_images = (
        ORIGINAL_PROCESSED_DIR
        / "images"
        / "val"
    )

    original_val_labels = (
        ORIGINAL_PROCESSED_DIR
        / "labels"
        / "val"
    )

    output_train_images = (
        OUTPUT_DIR
        / "images"
        / "train"
    )

    output_train_labels = (
        OUTPUT_DIR
        / "labels"
        / "train"
    )

    output_val_images = (
        OUTPUT_DIR
        / "images"
        / "val"
    )

    output_val_labels = (
        OUTPUT_DIR
        / "labels"
        / "val"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_count = 0

    for image_path in (
        original_train_images.iterdir()
    ):

        if not (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            continue

        label_path = (
            original_train_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            continue

        shutil.copy2(
            image_path,
            output_train_images
            / image_path.name,
        )

        shutil.copy2(
            label_path,
            output_train_labels
            / label_path.name,
        )

        train_count += 1

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_count = 0

    for image_path in (
        original_val_images.iterdir()
    ):

        if not (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            continue

        label_path = (
            original_val_labels
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            continue

        shutil.copy2(
            image_path,
            output_val_images
            / image_path.name,
        )

        shutil.copy2(
            label_path,
            output_val_labels
            / label_path.name,
        )

        val_count += 1

    print(
        f"기존 Train: "
        f"{train_count:,}장"
    )

    print(
        f"기존 Val: "
        f"{val_count:,}장"
    )


# ============================================================
# 14. 선택된 AI-Hub 데이터 → YOLO
# ============================================================

def copy_selected_aihub(
    selected_candidates,
):

    output_train_images = (
        OUTPUT_DIR
        / "images"
        / "train"
    )

    output_train_labels = (
        OUTPUT_DIR
        / "labels"
        / "train"
    )

    print("\n" + "=" * 70)
    print("선택된 AI-Hub 데이터 변환 / 복사")
    print("=" * 70)

    rows = []

    total_selected = len(
        selected_candidates
    )

    for index, candidate in enumerate(
        selected_candidates,
        start=1,
    ):

        source_image = candidate[
            "image_path"
        ]

        new_stem = (
            f"aihub_{index:04d}_"
            f"{source_image.stem}"
        )

        destination_image = (
            output_train_images
            / (
                f"{new_stem}"
                f"{source_image.suffix.lower()}"
            )
        )

        destination_label = (
            output_train_labels
            / f"{new_stem}.txt"
        )

        # ----------------------------------------------------
        # 이미지 복사
        # ----------------------------------------------------

        shutil.copy2(
            source_image,
            destination_image,
        )

        # ----------------------------------------------------
        # YOLO label 생성
        # ----------------------------------------------------

        with open(
            destination_label,
            "w",
            encoding="utf-8",
        ) as f:

            for annotation in candidate[
                "annotations"
            ]:

                bbox = annotation[
                    "bbox"
                ]

                x = bbox["x"]
                y = bbox["y"]
                w = bbox["w"]
                h = bbox["h"]

                image_width = candidate[
                    "width"
                ]

                image_height = candidate[
                    "height"
                ]

                # center
                xc = (
                    x + w / 2
                ) / image_width

                yc = (
                    y + h / 2
                ) / image_height

                # width / height
                nw = (
                    w / image_width
                )

                nh = (
                    h / image_height
                )

                f.write(
                    f"{annotation['class_id']} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{nw:.6f} "
                    f"{nh:.6f}\n"
                )

        rows.append(
            {
                "index": index,

                "output_image":
                    destination_image.name,

                "source_image":
                    source_image.name,

                "annotation_count":
                    len(
                        candidate[
                            "annotations"
                        ]
                    ),

                "classes": ",".join(
                    str(class_id)
                    for class_id in sorted(
                        candidate[
                            "class_counts"
                        ]
                    )
                ),
            }
        )

        if (
            index % 100 == 0
            or index == total_selected
        ):

            print(
                f"복사 진행: "
                f"{index:,}/"
                f"{total_selected:,}"
            )

    # --------------------------------------------------------
    # 선택 CSV
    # --------------------------------------------------------

    with open(
        SELECTED_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "output_image",
                "source_image",
                "annotation_count",
                "classes",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"\nAI-Hub 추가 완료: "
        f"{len(selected_candidates):,}장"
    )

    print(
        f"선택 CSV:\n"
        f"{SELECTED_CSV}"
    )


# ============================================================
# 15. data.yaml 생성
# ============================================================

def create_data_yaml():

    data = {
        "path": ".",

        "train":
            "images/train",

        "val":
            "images/val",

        "nc":
            NUM_CLASSES,

        "names": [
            f"class_{i}"
            for i in range(
                NUM_CLASSES
            )
        ],
    }

    yaml_path = (
        OUTPUT_DIR
        / "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print("\n" + "=" * 70)
    print("data.yaml 생성 완료")
    print("=" * 70)

    print(
        yaml_path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# 16. 최종 검증
# ============================================================

def validate_final_dataset(original_counts):

    print("\n" + "=" * 70)
    print("최종 데이터셋 검증")
    print("=" * 70)

    train_image_dir = (
        OUTPUT_DIR
        / "images"
        / "train"
    )

    train_label_dir = (
        OUTPUT_DIR
        / "labels"
        / "train"
    )

    val_image_dir = (
        OUTPUT_DIR
        / "images"
        / "val"
    )

    val_label_dir = (
        OUTPUT_DIR
        / "labels"
        / "val"
    )

    train_images = [
        p
        for p in train_image_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    train_labels = list(
        train_label_dir.glob("*.txt")
    )

    val_images = [
        p
        for p in val_image_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    val_labels = list(
        val_label_dir.glob("*.txt")
    )

    print(
        f"Train images : "
        f"{len(train_images):,}"
    )

    print(
        f"Train labels : "
        f"{len(train_labels):,}"
    )

    print(
        f"Val images   : "
        f"{len(val_images):,}"
    )

    print(
        f"Val labels   : "
        f"{len(val_labels):,}"
    )

    # --------------------------------------------------------
    # 이미지 / 라벨 매칭
    # --------------------------------------------------------

    train_image_stems = {
        p.stem
        for p in train_images
    }

    train_label_stems = {
        p.stem
        for p in train_labels
    }

    val_image_stems = {
        p.stem
        for p in val_images
    }

    val_label_stems = {
        p.stem
        for p in val_labels
    }

    print(
        "\nTrain 이미지/라벨 누락:"
        f" {len(train_image_stems - train_label_stems)}"
    )

    print(
        "Train 라벨/이미지 누락:"
        f" {len(train_label_stems - train_image_stems)}"
    )

    print(
        "Val 이미지/라벨 누락:"
        f" {len(val_image_stems - val_label_stems)}"
    )

    print(
        "Val 라벨/이미지 누락:"
        f" {len(val_label_stems - val_image_stems)}"
    )

    # --------------------------------------------------------
    # 클래스 분포
    # --------------------------------------------------------

    final_counts = Counter()

    for label_file in train_labels:

        with open(
            label_file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(
                        parts[0]
                    )

                except ValueError:
                    continue

                if 0 <= class_id < NUM_CLASSES:

                    final_counts[
                        class_id
                    ] += 1

    print("\n" + "=" * 70)
    print("최종 Train 클래스 분포")
    print("=" * 70)

    print(
        f"{'Class':<10}"
        f"{'Original':>10}"
        f"{'Final':>10}"
        f"{'Added':>10}"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        original = (
            original_counts[
                class_id
            ]
        )

        final = (
            final_counts[
                class_id
            ]
        )

        print(
            f"class_{class_id:<4}"
            f"{original:>10}"
            f"{final:>10}"
            f"{final - original:>10}"
        )

    total_original = sum(
        original_counts.values()
    )

    total_final = sum(
        final_counts.values()
    )

    print(
        f"\n원본 객체: "
        f"{total_original:,}"
    )

    print(
        f"최종 객체: "
        f"{total_final:,}"
    )

    if total_original > 0:

        print(
            f"증가 배율: "
            f"{total_final / total_original:.2f}배"
        )


# ============================================================
# 17. Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "AI-Hub 정제 + 1,000장 선별"
    )
    print("=" * 70)

    print(
        f"\n프로젝트:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nAI-Hub:\n"
        f"{AIHUB_DIR}"
    )

    print(
        f"\n출력:\n"
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # 1. 기존 Train 클래스 분포
    # --------------------------------------------------------

    original_counts = (
        count_original_train_classes()
    )

    # --------------------------------------------------------
    # 2. 기존 category_id → YOLO class_id 매핑
    # --------------------------------------------------------

    category_to_class = (
        build_category_mapping()
    )

    if len(
        category_to_class
    ) != NUM_CLASSES:

        raise RuntimeError(
            f"""
기존 대회 클래스 매핑 수가
{NUM_CLASSES}개가 아닙니다.

현재:
{len(category_to_class)}개
"""
        )

    print(
        f"\n✅ 기존 대회 클래스 매핑 확인: "
        f"{len(category_to_class)}개"
    )

    # --------------------------------------------------------
    # 3. AI-Hub JSON 수집
    # --------------------------------------------------------

    json_files = (
        collect_json_files()
    )

    if not json_files:

        raise RuntimeError(
            "AI-Hub JSON 파일이 없습니다."
        )

    # --------------------------------------------------------
    # 4. AI-Hub 이미지 인덱스
    # --------------------------------------------------------

    image_index = (
        build_image_index()
    )

    if not image_index:

        raise RuntimeError(
            "AI-Hub 이미지가 없습니다."
        )

    # --------------------------------------------------------
    # 5. AI-Hub 전체 정제
    # --------------------------------------------------------

    candidates, stats = (
        clean_aihub_candidates(
            json_files,
            image_index,
            category_to_class,
        )
    )

    if not candidates:

        raise RuntimeError(
            "\n정제 후 AI-Hub 후보가 없습니다."
        )

    # --------------------------------------------------------
    # 6. 정제 결과 저장
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_cleaned_csv(
        candidates
    )

    # --------------------------------------------------------
    # 7. 1,000장 선별
    # --------------------------------------------------------

    selected_candidates = (
        select_balanced_subset(
            candidates,
            original_counts,
        )
    )

    if not selected_candidates:

        raise RuntimeError(
            "\nAI-Hub 선택 결과가 없습니다."
        )

    # --------------------------------------------------------
    # 8. 출력 폴더 초기화
    # --------------------------------------------------------

    prepare_output_dir()

    # CSV 다시 저장
    save_cleaned_csv(
        candidates
    )

    # --------------------------------------------------------
    # 9. 기존 Train / Val 복사
    # --------------------------------------------------------

    copy_original_dataset()

    # --------------------------------------------------------
    # 10. AI-Hub 추가
    # --------------------------------------------------------

    copy_selected_aihub(
        selected_candidates
    )

    # --------------------------------------------------------
    # 11. data.yaml
    # --------------------------------------------------------

    create_data_yaml()

    # --------------------------------------------------------
    # 12. 최종 검증
    # --------------------------------------------------------

    validate_final_dataset(original_counts)

    # --------------------------------------------------------
    # 완료
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("✅ 전체 작업 완료")
    print("=" * 70)

    print(
        f"\n정제 후보: "
        f"{len(candidates):,}장"
    )

    print(
        f"최종 AI-Hub: "
        f"{len(selected_candidates):,}장"
    )

    print(
        f"\n최종 데이터셋:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "\nTrain = 기존 176장 + AI-Hub 선택 데이터"
    )

    print(
        "Val = 기존 Validation 44장 유지"
    )


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()