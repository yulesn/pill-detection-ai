from pathlib import Path
from collections import Counter
import random
import csv
import statistics

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "train"
ORIGINAL_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "labels" / "train"

AUG_IMAGE_DIR = PROJECT_ROOT / "data" / "combined_augmented" / "images" / "train"
AUG_LABEL_DIR = PROJECT_ROOT / "data" / "combined_augmented" / "labels" / "train"

AIHUB_CSV = PROJECT_ROOT / "data" / "combined_augmented" / "selected_aihub.csv"

EDA_DIR = PROJECT_ROOT / "data" / "combined_augmented" / "eda"
VIS_DIR = EDA_DIR / "visualizations"

NUM_CLASSES = 56
SEED = 42
NUM_VISUALIZE = 20

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# 기본 함수
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def get_image_files(directory):
    if not directory.exists():
        return []

    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_label_files(directory):
    if not directory.exists():
        return []

    return [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ]


def read_yolo_label(label_path):
    """
    YOLO label:
    class_id x_center y_center width height
    """

    records = []

    if not label_path.exists():
        return records

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="cp949").splitlines()

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            records.append({
                "valid": False,
                "line_no": line_no,
                "raw": line,
            })
            continue

        try:
            class_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            records.append({
                "valid": True,
                "line_no": line_no,
                "class_id": class_id,
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
            })

        except ValueError:
            records.append({
                "valid": False,
                "line_no": line_no,
                "raw": line,
            })

    return records


# ============================================================
# 1. 이미지 / 라벨 매칭 검사
# ============================================================

def check_image_label_matching():

    print_section("1. 이미지 / 라벨 매칭 검사")

    image_files = get_image_files(AUG_IMAGE_DIR)
    label_files = get_label_files(AUG_LABEL_DIR)

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)

    print(f"Train images : {len(image_files)}")
    print(f"Train labels : {len(label_files)}")
    print(f"이미지-라벨 정상 매칭 : {len(image_stems & label_stems)}")

    print(f"라벨 없는 이미지 : {len(images_without_labels)}")
    print(f"이미지 없는 라벨 : {len(labels_without_images)}")

    if images_without_labels:
        print("\n[라벨 없는 이미지]")
        for x in images_without_labels[:20]:
            print(" ", x)

    if labels_without_images:
        print("\n[이미지 없는 라벨]")
        for x in labels_without_images[:20]:
            print(" ", x)


# ============================================================
# 2. 클래스 분포
# ============================================================

def count_classes(label_dir):

    counter = Counter()

    for label_file in get_label_files(label_dir):

        records = read_yolo_label(label_file)

        for record in records:

            if not record.get("valid", False):
                continue

            class_id = record["class_id"]

            if 0 <= class_id < NUM_CLASSES:
                counter[class_id] += 1

    return counter


def print_class_distribution():

    print_section("2. 클래스 분포")

    original = count_classes(ORIGINAL_LABEL_DIR)
    final = count_classes(AUG_LABEL_DIR)

    aihub = Counter()

    for label_file in get_label_files(AUG_LABEL_DIR):

        if not label_file.name.startswith("aihub_"):
            continue

        records = read_yolo_label(label_file)

        for record in records:

            if not record.get("valid", False):
                continue

            class_id = record["class_id"]

            if 0 <= class_id < NUM_CLASSES:
                aihub[class_id] += 1

    print(
        f"{'class':>5} "
        f"{'original':>10} "
        f"{'AI-Hub':>10} "
        f"{'final':>10} "
        f"{'increase':>10}"
    )

    print("-" * 55)

    rows = []

    for class_id in range(NUM_CLASSES):

        orig = original[class_id]
        add = aihub[class_id]
        fin = final[class_id]

        rows.append(
            (class_id, orig, add, fin, fin - orig)
        )

        print(
            f"{class_id:>5} "
            f"{orig:>10} "
            f"{add:>10} "
            f"{fin:>10} "
            f"{fin - orig:>10}"
        )

    total_original = sum(original.values())
    total_aihub = sum(aihub.values())
    total_final = sum(final.values())

    print("-" * 55)
    print(f"원본 총 객체 : {total_original}")
    print(f"AI-Hub 추가 객체 : {total_aihub}")
    print(f"최종 총 객체 : {total_final}")

    # CSV 저장
    output_csv = EDA_DIR / "class_distribution.csv"

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class_id",
            "original",
            "aihub",
            "final",
            "increase"
        ])

        writer.writerows(rows)

    print(f"\n저장: {output_csv}")


# ============================================================
# 3. 이미지당 객체 수
# ============================================================

def object_count_distribution():

    print_section("3. 이미지당 객체 수")

    counter = Counter()

    aihub_counter = Counter()

    for label_file in get_label_files(AUG_LABEL_DIR):

        records = read_yolo_label(label_file)

        valid_count = sum(
            1 for r in records
            if r.get("valid", False)
        )

        counter[valid_count] += 1

        if label_file.name.startswith("aihub_"):
            aihub_counter[valid_count] += 1

    print("[전체 Train]")

    for count in sorted(counter):
        print(
            f"객체 {count:>2}개 : "
            f"{counter[count]:>4}장"
        )

    print("\n[AI-Hub만]")

    for count in sorted(aihub_counter):
        print(
            f"객체 {count:>2}개 : "
            f"{aihub_counter[count]:>4}장"
        )


# ============================================================
# 4. BBox 통계
# ============================================================

def bbox_statistics():

    print_section("4. BBox 통계")

    widths = []
    heights = []
    areas = []

    invalid_boxes = []
    out_of_range = []

    total_objects = 0

    for label_file in get_label_files(AUG_LABEL_DIR):

        if not label_file.name.startswith("aihub_"):
            continue

        records = read_yolo_label(label_file)

        for record in records:

            if not record.get("valid", False):
                invalid_boxes.append(
                    (label_file.name, record)
                )
                continue

            total_objects += 1

            w = record["w"]
            h = record["h"]

            widths.append(w)
            heights.append(h)
            areas.append(w * h)

            if not (
                0 <= record["xc"] <= 1
                and 0 <= record["yc"] <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                out_of_range.append(
                    (label_file.name, record)
                )

    print(f"AI-Hub 객체 수 : {total_objects}")

    if not widths:
        print("BBox 데이터가 없습니다.")
        return

    def stats(values):
        return {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "max": max(values),
        }

    width_stats = stats(widths)
    height_stats = stats(heights)
    area_stats = stats(areas)

    print("\n[정규화 Width]")
    print(f"최소   : {width_stats['min']:.6f}")
    print(f"중앙값 : {width_stats['median']:.6f}")
    print(f"평균   : {width_stats['mean']:.6f}")
    print(f"최대   : {width_stats['max']:.6f}")

    print("\n[정규화 Height]")
    print(f"최소   : {height_stats['min']:.6f}")
    print(f"중앙값 : {height_stats['median']:.6f}")
    print(f"평균   : {height_stats['mean']:.6f}")
    print(f"최대   : {height_stats['max']:.6f}")

    print("\n[정규화 Area]")
    print(f"최소   : {area_stats['min']:.8f}")
    print(f"중앙값 : {area_stats['median']:.8f}")
    print(f"평균   : {area_stats['mean']:.8f}")
    print(f"최대   : {area_stats['max']:.8f}")

    print(f"\n잘못된 label 형식 : {len(invalid_boxes)}")
    print(f"범위 밖 bbox      : {len(out_of_range)}")

    if out_of_range:

        print("\n[범위 밖 bbox 최대 20개]")

        for filename, record in out_of_range[:20]:

            print(
                filename,
                record
            )


# ============================================================
# 5. 빈 라벨 검사
# ============================================================

def check_empty_labels():

    print_section("5. 빈 라벨 검사")

    empty_labels = []

    for label_file in get_label_files(AUG_LABEL_DIR):

        try:
            text = label_file.read_text(
                encoding="utf-8"
            ).strip()
        except UnicodeDecodeError:
            text = label_file.read_text(
                encoding="cp949"
            ).strip()

        if not text:
            empty_labels.append(label_file.name)

    print(f"빈 라벨 파일 : {len(empty_labels)}")

    if empty_labels:

        print("\n[빈 라벨]")
        for x in empty_labels[:30]:
            print(" ", x)


# ============================================================
# 6. AI-Hub CSV 검사
# ============================================================

def check_aihub_csv():

    print_section("6. AI-Hub 선택 데이터 검사")

    if not AIHUB_CSV.exists():

        print("selected_aihub.csv가 없습니다.")
        return

    with AIHUB_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    print(f"AI-Hub 선택 이미지 : {len(rows)}")

    output_names = [
        row["output_image"]
        for row in rows
    ]

    duplicates = [
        name
        for name, count
        in Counter(output_names).items()
        if count > 1
    ]

    print(f"중복 output_image : {len(duplicates)}")

    source_names = [
        row["source_image"]
        for row in rows
    ]

    source_duplicates = [
        name
        for name, count
        in Counter(source_names).items()
        if count > 1
    ]

    print(
        f"중복 source_image : "
        f"{len(source_duplicates)}"
    )

    annotation_counts = []

    for row in rows:

        try:
            annotation_counts.append(
                int(row["annotation_count"])
            )
        except (KeyError, ValueError):
            pass

    if annotation_counts:

        print("\n[AI-Hub 이미지당 annotation]")
        print(
            f"최소   : {min(annotation_counts)}"
        )
        print(
            f"중앙값 : {statistics.median(annotation_counts)}"
        )
        print(
            f"평균   : {statistics.mean(annotation_counts):.2f}"
        )
        print(
            f"최대   : {max(annotation_counts)}"
        )


# ============================================================
# 7. 랜덤 BBox 시각화
# ============================================================

def visualize_random_images():

    print_section("7. 랜덤 AI-Hub 이미지 BBox 시각화")

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = [
        p for p in get_image_files(AUG_IMAGE_DIR)
        if p.name.startswith("aihub_")
    ]

    if not images:

        print("AI-Hub 이미지가 없습니다.")
        return

    random.seed(SEED)

    sample_size = min(
        NUM_VISUALIZE,
        len(images)
    )

    selected = random.sample(
        images,
        sample_size
    )

    print(
        f"시각화 이미지 수 : "
        f"{sample_size}"
    )

    for index, image_path in enumerate(
        selected,
        start=1
    ):

        label_path = AUG_LABEL_DIR / (
            image_path.stem + ".txt"
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"[오류] {image_path.name}: {e}"
            )

            continue

        draw = ImageDraw.Draw(image)

        width, height = image.size

        records = read_yolo_label(
            label_path
        )

        for record in records:

            if not record.get("valid", False):
                continue

            class_id = record["class_id"]

            xc = record["xc"]
            yc = record["yc"]
            w = record["w"]
            h = record["h"]

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0),
                width=3
            )

            text = f"class {class_id}"

            # 텍스트 배경
            try:

                bbox = draw.textbbox(
                    (x1, y1),
                    text
                )

                draw.rectangle(
                    bbox,
                    fill=(255, 0, 0)
                )

            except Exception:
                pass

            draw.text(
                (x1, y1),
                text,
                fill=(255, 255, 255)
            )

        output_path = (
            VIS_DIR /
            f"{index:02d}_{image_path.name}"
        )

        image.save(
            output_path
        )

        print(
            f"[{index:02d}/{sample_size}] "
            f"{output_path.name}"
        )


# ============================================================
# 8. 종합 요약
# ============================================================

def print_summary():

    print_section("8. EDA 최종 요약")

    original_images = get_image_files(
        ORIGINAL_IMAGE_DIR
    )

    final_images = get_image_files(
        AUG_IMAGE_DIR
    )

    aihub_images = [
        p for p in final_images
        if p.name.startswith("aihub_")
    ]

    original_objects = sum(
        count_classes(
            ORIGINAL_LABEL_DIR
        ).values()
    )

    final_objects = sum(
        count_classes(
            AUG_LABEL_DIR
        ).values()
    )

    print(f"원본 Train 이미지 : {len(original_images)}")
    print(f"AI-Hub 추가 이미지 : {len(aihub_images)}")
    print(f"최종 Train 이미지 : {len(final_images)}")

    print()

    print(f"원본 Train 객체 : {original_objects}")
    print(
        f"AI-Hub 추가 객체 : "
        f"{final_objects - original_objects}"
    )
    print(f"최종 Train 객체 : {final_objects}")

    print()

    if original_objects > 0:

        print(
            "객체 수 증가 배율 : "
            f"{final_objects / original_objects:.2f}배"
        )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AI-Hub 보강 데이터셋 EDA")
    print("=" * 70)

    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"EDA_DIR      : {EDA_DIR}")

    EDA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    check_image_label_matching()

    print_class_distribution()

    object_count_distribution()

    bbox_statistics()

    check_empty_labels()

    check_aihub_csv()

    visualize_random_images()

    print_summary()

    print_section("EDA 완료")

    print(f"결과 폴더 : {EDA_DIR}")
    print(
        f"시각화 폴더 : {VIS_DIR}"
    )


if __name__ == "__main__":
    main()