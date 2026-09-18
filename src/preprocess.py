import os
import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


# ============================================================
# 1. 경로 설정
# ============================================================

# preprocess.py가 있는 프로젝트 기준으로 상위 폴더를 프로젝트 루트로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_annotations"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sprint_ai_project1_data"
    / "train_images"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 2. JSON Annotation 수집 및 이미지별 통합
# ============================================================

def collect_image_data(annotation_dir):
    """
    모든 JSON 파일의 images와 annotations 정보를
    image_id 기준으로 하나의 구조로 통합한다.
    """

    image_data = {}

    # annotation 폴더와 하위 폴더까지 모두 탐색
    for root, dirs, files in os.walk(annotation_dir):

        for file in files:

            # JSON 파일만 처리
            if not file.endswith(".json"):
                continue

            json_path = Path(root) / file

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ------------------------------------------------
            # images 정보 저장
            # ------------------------------------------------

            for image_info in data.get("images", []):

                image_id = image_info["id"]

                if image_id not in image_data:

                    image_data[image_id] = {
                        "file_name": image_info["file_name"],
                        "width": image_info["width"],
                        "height": image_info["height"],
                        "annotations": []
                    }

            # ------------------------------------------------
            # annotations 정보 저장
            # ------------------------------------------------

            for ann in data.get("annotations", []):

                image_id = ann["image_id"]

                if image_id in image_data:
                    image_data[image_id]["annotations"].append(ann)

    return image_data


# ============================================================
# 3. BBox 이상 여부 확인
# ============================================================

def check_invalid_bboxes(image_data):
    """
    이미지 영역을 벗어나거나 크기가 잘못된
    BBox를 확인한다.
    """

    invalid_bboxes = []

    for image_id, item in image_data.items():

        image_width = item["width"]
        image_height = item["height"]

        for ann in item["annotations"]:

            x, y, w, h = ann["bbox"]

            # BBox 이상 여부 확인
            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > image_width
                or y + h > image_height
            ):

                invalid_bboxes.append({
                    "image_id": image_id,
                    "file_name": item["file_name"],
                    "bbox": ann["bbox"],
                    "image_size": (
                        image_width,
                        image_height
                    ),
                    "category_id": ann["category_id"]
                })

    return invalid_bboxes


# ============================================================
# 4. Annotation이 2개인 이미지 확인
# ============================================================

def find_two_annotation_images(image_data):
    """
    Annotation이 정확히 2개인 이미지를 찾는다.
    """

    two_ann_images = []

    for image_id, item in image_data.items():

        if len(item["annotations"]) == 2:

            two_ann_images.append({
                "image_id": image_id,
                "file_name": item["file_name"],
                "annotations": item["annotations"]
            })

    return two_ann_images


# ============================================================
# 5. category_id → YOLO class_id 매핑
# ============================================================

def create_category_mapping(image_data):
    """
    COCO category_id를 YOLO class_id로 변환하기 위한
    매핑 정보를 생성한다.

    YOLO class_id는 0부터 시작한다.
    """

    category_ids = set()

    # 모든 annotation에서 category_id 수집
    for item in image_data.values():

        for ann in item["annotations"]:
            category_ids.add(ann["category_id"])

    # category_id를 정렬한 뒤 0부터 class_id 부여
    category_to_class = {
        category_id: class_id
        for class_id, category_id
        in enumerate(sorted(category_ids))
    }

    return category_to_class


# ============================================================
# 6. COCO BBox → YOLO BBox 변환
# ============================================================

def convert_bbox_to_yolo(
    bbox,
    image_width,
    image_height
):
    """
    COCO 형식

    [x, y, width, height]

    을 YOLO 형식

    [x_center, y_center, width, height]

    으로 변환하고 0~1 사이로 정규화한다.
    """

    x, y, w, h = bbox

    # 중심 좌표 계산
    x_center = x + w / 2
    y_center = y + h / 2

    # 0~1 사이로 정규화
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
# 7. YOLO Label 생성
# ============================================================
# JSON 파일에 들어있는 알약 위치 정보를 YOLO label 형식으로 변환한다.
def create_yolo_labels(
    image_data,
    category_to_class
):
    """
    COCO annotation을 YOLO label 형식으로 변환한다.
    """

    yolo_labels = {}

    for image_id, item in image_data.items():

        labels = []

        image_width = item["width"]
        image_height = item["height"]

        for ann in item["annotations"]:

            x, y, w, h = ann["bbox"]

            # -----------------------------------------------
            # 잘못된 BBox 제외
            # -----------------------------------------------

            if (
                w <= 0
                or h <= 0
                or x < 0
                or y < 0
                or x + w > image_width
                or y + h > image_height
            ):
                continue

            # -----------------------------------------------
            # category_id → YOLO class_id
            # -----------------------------------------------

            class_id = category_to_class[
                ann["category_id"]
            ]

            # -----------------------------------------------
            # COCO BBox → YOLO BBox
            # -----------------------------------------------

            (
                x_center,
                y_center,
                w,
                h
            ) = convert_bbox_to_yolo(
                ann["bbox"],
                image_width,
                image_height
            )

            # -----------------------------------------------
            # YOLO label 문자열 생성
            # -----------------------------------------------
            # YOLO label 형식: class_id x_center y_center width height
            # (1, 0.25, 0.4, 0.3, 0.4), (1, 0.75, 0.4, 0.3, 0.4), (2, 0.5, 0.5, 0.4, 0.4)
            # 형태로 저장

            labels.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

        yolo_labels[image_id] = labels

    return yolo_labels


# ============================================================
# 8. Train / Validation 분리
# ============================================================

def split_dataset(
    image_data,
    test_size=0.2,
    random_state=42
):
    """
    전체 이미지를 Train / Validation 데이터로 분리한다.
    """

    image_ids = sorted(image_data.keys())

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=test_size,
        random_state=random_state
    )

    return train_ids, val_ids


# ============================================================
# 9. processed 디렉토리 생성
# ============================================================

def create_processed_dirs(processed_dir):
    """
    YOLO 데이터셋 저장을 위한 디렉토리를 생성한다.
    기존 전처리 결과는 삭제하고 새로 생성한다.
    """

    for split in ["train", "val"]:
        image_dir = processed_dir / "images" / split
        label_dir = processed_dir / "labels" / split

        if image_dir.exists():
            shutil.rmtree(image_dir)

        if label_dir.exists():
            shutil.rmtree(label_dir)

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 10. 이미지 및 YOLO Label 저장
# ============================================================

def save_yolo_dataset(
    image_data,     #이미지 정보 + annotation
    yolo_labels,    # YOLO 형식으로 변환된 라벨
    train_ids,      # train에 들어갈 이미지 ID
    val_ids,        # val에 들어갈 이미지 ID
    image_dir,      # 원본 이미지가 있는 디렉토리
    processed_dir   # 전처리 결과를 저장할 디렉토리
):
    """
    Train / Val 이미지와 YOLO label을
    data/processed에 저장한다.
    """

    split_ids = {
        "train": train_ids,
        "val": val_ids
    }

    for split, ids in split_ids.items():

        for image_id in ids:

            item = image_data[image_id]

            # 원본 이미지 경로
            src_image = (
                image_dir
                / item["file_name"]
            )

            # 처리된 이미지 저장 경로
            dst_image = (
                processed_dir
                / "images"
                / split
                / item["file_name"]
            )

            # 이미지 복사
            shutil.copy2(
                src_image,
                dst_image
            )

            # YOLO label 파일명
            label_name = (
                Path(item["file_name"]).stem    #(abc.jpg → abc) + .txt
                + ".txt"
            )

            # YOLO label 저장 경로
            label_path = (                      #data/processed/labels/train/abc.txt
                processed_dir
                / "labels"
                / split
                / label_name
            )

            # YOLO annotation 작성
            labels = yolo_labels[image_id]

            label_path.write_text(
                "\n".join(labels),
                encoding="utf-8"
            )


# ============================================================
# 11. data.yaml 생성
# ============================================================

def create_data_yaml(
    processed_dir,
    num_classes
):
    """
    YOLO 학습에 사용할 data.yaml 파일을 생성한다.
    """
    
    yaml_path = (
        processed_dir
        / "data.yaml"
    )

    # 현재는 클래스 이름 정보가 없기 때문에
    # class_0 ~ class_55 형태로 생성
    names = [
        f"class_{i}"
        for i in range(num_classes)
    ]

    data_path = processed_dir.resolve().as_posix()

    yaml_text = f"""path: {data_path}
train: images/train
val: images/val

names:
"""

    for i, name in enumerate(names):

        yaml_text += (
            f"  {i}: {name}\n"
        )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8"
    )

    return yaml_path


# ============================================================
# 12. 전처리 결과 검증
# ============================================================

def validate_processed_dataset(
    processed_dir
):
    """
    Train / Val 이미지와 label의 개수를 확인하고
    이미지와 label의 파일명이 일치하는지 검증한다.
    """

    for split in ["train", "val"]:

        image_dir = (
            processed_dir
            / "images"
            / split
        )

        label_dir = (
            processed_dir
            / "labels"
            / split
        )

        images = sorted(
            image_dir.glob("*")
        )

        labels = sorted(
            label_dir.glob("*.txt")
        )

        image_stems = {
            p.stem
            for p in images
        }

        label_stems = {
            p.stem
            for p in labels
        }

        print(f"\n[{split}]")

        print(
            "이미지:",
            len(images)
        )

        print(
            "라벨:",
            len(labels)
        )

        print(
            "라벨 없는 이미지:",
            len(image_stems - label_stems)
        )

        print(
            "이미지만 없는 라벨:",
            len(label_stems - image_stems)
        )


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 60)
    print("YOLO 데이터셋 전처리 시작")
    print("=" * 60)

    # --------------------------------------------------------
    # 경로 확인
    # --------------------------------------------------------

    print("\n[경로]")

    print(
        "Annotation:",
        ANNOTATION_DIR
    )

    print(
        "Image:",
        IMAGE_DIR
    )

    print(
        "Processed:",
        PROCESSED_DIR
    )

    # --------------------------------------------------------
    # 경로 존재 여부 확인
    # --------------------------------------------------------

    if not ANNOTATION_DIR.exists():

        raise FileNotFoundError(
            "Annotation 디렉토리를 찾을 수 없습니다:\n"
            f"{ANNOTATION_DIR}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            "Image 디렉토리를 찾을 수 없습니다:\n"
            f"{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # 1. JSON annotation 수집
    # --------------------------------------------------------

    print("\n[1] JSON annotation 수집")

    image_data = collect_image_data(
        ANNOTATION_DIR
    )

    print(
        "총 이미지 수:",
        len(image_data)
    )

    total_annotations = sum(
        len(item["annotations"])
        for item in image_data.values()
    )

    print(
        "총 annotation 수:",
        total_annotations
    )

    # --------------------------------------------------------
    # 2. BBox 이상 여부 확인
    # --------------------------------------------------------

    print("\n[2] BBox 이상 여부 확인")

    invalid_bboxes = check_invalid_bboxes(
        image_data
    )

    print(
        "이상한 BBox 개수:",
        len(invalid_bboxes)
    )

    # 이상한 BBox가 있으면 정보 출력
    if invalid_bboxes:

        for bbox in invalid_bboxes:

            print(bbox)

    # --------------------------------------------------------
    # 3. Annotation이 2개인 이미지 확인
    # --------------------------------------------------------

    print(
        "\n[3] Annotation이 2개인 이미지 확인"
    )

    two_ann_images = (
        find_two_annotation_images(
            image_data
        )
    )

    print(
        "Annotation이 2개인 이미지:",
        len(two_ann_images)
    )

    for item in two_ann_images:

        print(
            f"image_id: {item['image_id']}, "
            f"file_name: {item['file_name']}"
        )
 
    # EDA에서 확인한 이상 데이터
    exclude_image_ids = {
        16, 193, 208, 239, 783, 907,
        1228, 1258, 1267, 1383, 1405, 1432
    }


    # 이상 데이터만 제외
    # 해당 ID가 있으면 삭제하고, 
    # 혹시 없으면 KeyError가 발생하지 않도록 pop()의 두 번째 인자를 None으로 설정
    for image_id in exclude_image_ids:
        image_data.pop(image_id, None)


    print(
        "EDA 이상 데이터 제외 후 남은 이미지 수:",
        len(image_data)
    )

    # --------------------------------------------------------
    # 4. Category mapping
    # --------------------------------------------------------

    print(
        "\n[4] category_id → YOLO class_id 매핑"
    )

    category_to_class = (
        create_category_mapping(
            image_data
        )
    )

    print(
        "총 클래스 수:",
        len(category_to_class)
    )

    for (
        category_id,
        class_id
    ) in list(
        category_to_class.items()
    )[:10]:

        print(
            f"category_id: {category_id} "
            f"→ class_id: {class_id}"
        )

    # --------------------------------------------------------
    # 5. YOLO Label 변환
    # --------------------------------------------------------

    print(
        "\n[5] COCO BBox → YOLO BBox 변환"
    )

    yolo_labels = create_yolo_labels(
        image_data,
        category_to_class
    )

    converted_annotations = sum(
        len(labels)
        for labels in yolo_labels.values()
    )

    print(
        "변환된 이미지 수:",
        len(yolo_labels)
    )

    print(
        "변환된 annotation 수:",
        converted_annotations
    )

    # --------------------------------------------------------
    # 6. Train / Val 분리
    # --------------------------------------------------------

    print(
        "\n[6] Train / Val 분리"
    )

    train_ids, val_ids = split_dataset(
        image_data,
        test_size=0.2,
        random_state=42
    )

    print(
        "전체 이미지 수:",
        len(image_data)
    )

    print(
        "Train 이미지 수:",
        len(train_ids)
    )

    print(
        "Val 이미지 수:",
        len(val_ids)
    )

    # Train과 Val에 중복 이미지가 없는지 확인
    overlap = len(
        set(train_ids) & set(val_ids)
    )

    print(
        "Train ∩ Val:",
        overlap
    )

    print(
        "Train + Val:",
        len(train_ids) + len(val_ids)
    )

    # --------------------------------------------------------
    # 7. processed 디렉토리 생성
    # --------------------------------------------------------

    print(
        "\n[7] processed 디렉토리 생성"
    )

    create_processed_dirs(
        PROCESSED_DIR
    )

    print(
        "processed 디렉토리 생성 완료"
    )

    # --------------------------------------------------------
    # 8. 이미지 / Label 저장
    # --------------------------------------------------------

    print(
        "\n[8] 이미지 및 YOLO Label 저장"
    )

    print("save train_ids:", len(train_ids))
    print("save val_ids:", len(val_ids))

    save_yolo_dataset(
        image_data=image_data,
        yolo_labels=yolo_labels,
        train_ids=train_ids,
        val_ids=val_ids,
        image_dir=IMAGE_DIR,
        processed_dir=PROCESSED_DIR
    )

    print(
        "YOLO 데이터셋 생성 완료"
    )
    # --------------------------------------------------------
    # 9. data.yaml 생성
    # --------------------------------------------------------

    print(
        "\n[9] data.yaml 생성"
    )

    yaml_path = create_data_yaml(
        PROCESSED_DIR,
        len(category_to_class)
    )

    print(
        "data.yaml 생성 완료"
    )

    print(
        "경로:",
        yaml_path
    )

    # --------------------------------------------------------
    # 10. 결과 검증
    # --------------------------------------------------------

    print(
        "\n[10] 전처리 결과 검증"
    )

    validate_processed_dataset(
        PROCESSED_DIR
    )

    print("\n" + "=" * 60)
    print("YOLO 데이터셋 전처리 완료")
    print("=" * 60)


# ============================================================
# 프로그램 실행
# ============================================================

if __name__ == "__main__":
    main()