import shutil
from pathlib import Path
import yaml


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 기존 데이터
ORIGINAL_DIR = PROJECT_ROOT / "data" / "processed"

# AI-Hub 전처리 데이터
AIHUB_DIR = PROJECT_ROOT / "data" / "aihub_processed"

# 최종 통합 데이터
COMBINED_DIR = PROJECT_ROOT / "data" / "combined_processed"


# 기존 Train
ORIGINAL_TRAIN_IMAGES = ORIGINAL_DIR / "images" / "train"
ORIGINAL_TRAIN_LABELS = ORIGINAL_DIR / "labels" / "train"

# 기존 Val
ORIGINAL_VAL_IMAGES = ORIGINAL_DIR / "images" / "val"
ORIGINAL_VAL_LABELS = ORIGINAL_DIR / "labels" / "val"

# AI-Hub
AIHUB_IMAGES = AIHUB_DIR / "images"
AIHUB_LABELS = AIHUB_DIR / "labels"

# Combined Train
COMBINED_TRAIN_IMAGES = (
    COMBINED_DIR / "images" / "train"
)

COMBINED_TRAIN_LABELS = (
    COMBINED_DIR / "labels" / "train"
)

# Combined Val
COMBINED_VAL_IMAGES = (
    COMBINED_DIR / "images" / "val"
)

COMBINED_VAL_LABELS = (
    COMBINED_DIR / "labels" / "val"
)


# ============================================================
# 2. 클래스
# ============================================================

CLASS_NAMES = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
    "class_5",
    "class_6",
    "class_7",
    "class_8",
    "class_9",
    "class_10",
    "class_11",
    "class_12",
    "class_13",
    "class_14",
    "class_15",
    "class_16",
    "class_17",
    "class_18",
    "class_19",
    "class_20",
    "class_21",
    "class_22",
    "class_23",
    "class_24",
    "class_25",
    "class_26",
    "class_27",
    "class_28",
    "class_29",
    "class_30",
    "class_31",
    "class_32",
    "class_33",
    "class_34",
    "class_35",
    "class_36",
    "class_37",
    "class_38",
    "class_39",
    "class_40",
    "class_41",
    "class_42",
    "class_43",
    "class_44",
    "class_45",
    "class_46",
    "class_47",
    "class_48",
    "class_49",
    "class_50",
    "class_51",
    "class_52",
    "class_53",
    "class_54",
    "class_55",
]


# ============================================================
# 3. 기존 Combined 폴더 삭제
# ============================================================

if COMBINED_DIR.exists():
    print("기존 combined_processed 삭제 중...")
    shutil.rmtree(COMBINED_DIR)


# ============================================================
# 4. 폴더 생성
# ============================================================

COMBINED_TRAIN_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

COMBINED_TRAIN_LABELS.mkdir(
    parents=True,
    exist_ok=True
)

COMBINED_VAL_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

COMBINED_VAL_LABELS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. 기존 데이터 복사
# ============================================================

def copy_original_dataset(
    image_dir,
    label_dir,
    destination_image_dir,
    destination_label_dir,
    dataset_name
):

    copied_images = 0
    copied_labels = 0

    image_files = [
        p
        for p in image_dir.iterdir()
        if p.is_file()
    ]

    for image_path in image_files:

        destination_image = (
            destination_image_dir
            / image_path.name
        )

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        destination_label = (
            destination_label_dir
            / label_path.name
        )

        if destination_image.exists():
            raise FileExistsError(
                f"이미지 파일명 충돌: {image_path.name}"
            )

        shutil.copy2(
            image_path,
            destination_image
        )

        copied_images += 1

        if label_path.exists():

            shutil.copy2(
                label_path,
                destination_label
            )

            copied_labels += 1

        else:

            raise FileNotFoundError(
                f"Label 없음: {label_path}"
            )

    print(
        f"{dataset_name}: "
        f"이미지 {copied_images:,}개 / "
        f"Label {copied_labels:,}개"
    )

    return copied_images, copied_labels


# ============================================================
# 6. AI-Hub 데이터 이동
# ============================================================

def move_aihub_dataset():

    moved_images = 0
    moved_labels = 0

    image_files = [
        p
        for p in AIHUB_IMAGES.iterdir()
        if p.is_file()
    ]

    for image_path in image_files:

        destination_image = (
            COMBINED_TRAIN_IMAGES
            / image_path.name
        )

        label_path = (
            AIHUB_LABELS
            / f"{image_path.stem}.txt"
        )

        destination_label = (
            COMBINED_TRAIN_LABELS
            / label_path.name
        )

        # 파일명 충돌 검사
        if destination_image.exists():
            raise FileExistsError(
                f"AI-Hub 이미지 파일명 충돌: "
                f"{image_path.name}"
            )

        if destination_label.exists():
            raise FileExistsError(
                f"AI-Hub Label 파일명 충돌: "
                f"{label_path.name}"
            )

        # 이미지 이동
        shutil.move(
            str(image_path),
            str(destination_image)
        )

        moved_images += 1

        # Label 이동
        if label_path.exists():

            shutil.move(
                str(label_path),
                str(destination_label)
            )

            moved_labels += 1

        else:

            raise FileNotFoundError(
                f"AI-Hub Label 없음: {label_path}"
            )

    print(
        f"AI-Hub 이동: "
        f"이미지 {moved_images:,}개 / "
        f"Label {moved_labels:,}개"
    )

    return moved_images, moved_labels


# ============================================================
# 7. 기존 Train 복사
# ============================================================

print("=" * 70)
print("기존 Train 데이터 복사")
print("=" * 70)

original_train_images, original_train_labels = (
    copy_original_dataset(
        ORIGINAL_TRAIN_IMAGES,
        ORIGINAL_TRAIN_LABELS,
        COMBINED_TRAIN_IMAGES,
        COMBINED_TRAIN_LABELS,
        "기존 Train"
    )
)


# ============================================================
# 8. AI-Hub Train 이동
# ============================================================

print("\n" + "=" * 70)
print("AI-Hub 데이터 이동")
print("=" * 70)

aihub_images, aihub_labels = (
    move_aihub_dataset()
)


# ============================================================
# 9. 기존 Validation 복사
# ============================================================

print("\n" + "=" * 70)
print("기존 Validation 복사")
print("=" * 70)

original_val_images, original_val_labels = (
    copy_original_dataset(
        ORIGINAL_VAL_IMAGES,
        ORIGINAL_VAL_LABELS,
        COMBINED_VAL_IMAGES,
        COMBINED_VAL_LABELS,
        "기존 Validation"
    )
)


# ============================================================
# 10. data.yaml 생성
# ============================================================

data_yaml = {
    "path": str(COMBINED_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 56,
    "names": CLASS_NAMES,
}

data_yaml_path = COMBINED_DIR / "data.yaml"

with open(
    data_yaml_path,
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        data_yaml,
        f,
        allow_unicode=True,
        sort_keys=False
    )


# ============================================================
# 11. 최종 검증
# ============================================================

train_images = [
    p
    for p in COMBINED_TRAIN_IMAGES.iterdir()
    if p.is_file()
]

train_labels = [
    p
    for p in COMBINED_TRAIN_LABELS.iterdir()
    if p.is_file()
]

val_images = [
    p
    for p in COMBINED_VAL_IMAGES.iterdir()
    if p.is_file()
]

val_labels = [
    p
    for p in COMBINED_VAL_LABELS.iterdir()
    if p.is_file()
]


# ============================================================
# 12. 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("통합 데이터셋 생성 완료")
print("=" * 70)

print(
    f"기존 Train:      "
    f"{original_train_images:,} 이미지 / "
    f"{original_train_labels:,} Label"
)

print(
    f"AI-Hub:          "
    f"{aihub_images:,} 이미지 / "
    f"{aihub_labels:,} Label"
)

print(
    f"최종 Train:      "
    f"{len(train_images):,} 이미지 / "
    f"{len(train_labels):,} Label"
)

print(
    f"최종 Val:        "
    f"{len(val_images):,} 이미지 / "
    f"{len(val_labels):,} Label"
)

print(
    f"클래스 수:       "
    f"{len(CLASS_NAMES)}"
)

print(
    f"data.yaml:       "
    f"{data_yaml_path}"
)

print(
    f"저장 위치:       "
    f"{COMBINED_DIR}"
)

print("=" * 70)