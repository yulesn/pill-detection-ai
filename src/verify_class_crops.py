"""
클래스(category_id)별로 bbox 크롭을 모아 그리드 이미지로 저장하는 스크립트.
같은 클래스인데 모양이 확연히 다른 알약이 섞여 있으면 라벨 미스매치로 의심.
bbox 자체가 비정상(빈 배경만 잘림 등)이거나 IoU 중복이면 통째로 제외.

사용법:
    python verify_class_crops.py --category_id 2
    python verify_class_crops.py --all
    python verify_class_crops.py --locate 2 3
    python verify_class_crops.py --correct 2 3 5        # 라벨 교정: cat_id=2, index=3 -> cat_id=5로
    python verify_class_crops.py --exclude 2 102        # 통째로 제외 (cat_id + 그리드 #번호로)
    python verify_class_crops.py --exclude_ann 16       # 통째로 제외 (train_coco.json의 annotation id로)
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

COCO_PATH = Path("data/processed/train_coco.json")
IMAGE_DIR = Path("data/raw/sprint_ai_project1_data/train_images")
ANNOTATION_DIR = Path("data/raw/sprint_ai_project1_data/train_annotations")
OUTPUT_DIR = Path("outputs/class_review")
CORRECTIONS_PATH = Path("data/processed/label_corrections.json")
EXCLUDED_ANNOTATIONS_PATH = Path("data/processed/excluded_annotations.json")


def load_coco():
    with open(COCO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_crops_by_category(coco):
    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    crops_by_cat = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]

        file_name = image_id_to_file.get(image_id)
        if file_name is None:
            continue

        img = cv2.imread(str(IMAGE_DIR / file_name))
        if img is None:
            continue

        x, y, w, h = [int(v) for v in bbox]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops_by_cat[category_id].append({"crop": crop_rgb, "file_name": file_name, "bbox": bbox})

    return crops_by_cat, cat_id_to_name


def save_grid(category_id, items, cat_name):
    n = len(items)
    if n == 0:
        print(f"category_id={category_id}: 크롭 없음, 건너뜀")
        return

    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = [axes] if n == 1 else axes.flatten()

    for i, (ax, item) in enumerate(zip(axes, items)):
        ax.imshow(item["crop"])
        ax.set_title(f"#{i}", fontsize=11, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")

    fig.suptitle(f"category_id={category_id} ({cat_name})  -  n={n}", fontsize=12)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img_path = OUTPUT_DIR / f"class_{category_id}.png"
    fig.savefig(img_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    index_path = OUTPUT_DIR / f"class_{category_id}_index.json"
    index_data = [{"index": i, "file_name": it["file_name"], "bbox": it["bbox"]} for i, it in enumerate(items)]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)

    print(f"저장됨: {img_path}  (알약 {n}개, 매핑: {index_path.name})")


def find_source_json_path(file_name: str, target_bbox: list, tolerance: float = 3.0):
    candidates = list(ANNOTATION_DIR.rglob(f"{Path(file_name).stem}.json"))
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for anno in data.get("annotations", []):
            bbox = anno.get("bbox")
            if bbox and all(abs(a - b) <= tolerance for a, b in zip(bbox, target_bbox)):
                return path, candidates
    return None, candidates


def resolve_dl_name(category_id: int) -> str:
    coco = load_coco()
    for cat in coco["categories"]:
        if cat["id"] == category_id:
            return cat["name"]
    raise ValueError(f"category_id={category_id} 를 categories에서 못 찾음")


def get_index_entry(category_id: int, index: int):
    index_path = OUTPUT_DIR / f"class_{category_id}_index.json"
    if not index_path.exists():
        print(f"{index_path} 가 없음. 먼저 --category_id {category_id} 로 그리드를 생성해줘.")
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    entry = next((e for e in index_data if e["index"] == index), None)
    if entry is None:
        print(f"index={index} 를 못 찾음 (총 {len(index_data)}개, 0~{len(index_data) - 1} 범위)")
    return entry


def get_annotation_by_id(ann_id: int):
    coco = load_coco()
    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    for ann in coco["annotations"]:
        if ann["id"] == ann_id:
            return {"file_name": image_id_to_file[ann["image_id"]], "bbox": ann["bbox"], "category_id": ann["category_id"]}
    return None


def locate(category_id: int, index: int):
    entry = get_index_entry(category_id, index)
    if entry is None:
        return
    print(f"대상: file_name={entry['file_name']}, bbox={entry['bbox']}")
    source_path, candidates = find_source_json_path(entry["file_name"], entry["bbox"])
    if source_path:
        print(f"일치하는 원본 json 찾음: {source_path}")
    else:
        print("bbox가 정확히 일치하는 json을 못 찾음. 아래 후보들을 직접 확인해줘:")
        for c in candidates:
            print(f"  {c}")


def correct(wrong_category_id: int, index: int, correct_category_id: int):
    entry = get_index_entry(wrong_category_id, index)
    if entry is None:
        return

    source_path, candidates = find_source_json_path(entry["file_name"], entry["bbox"])
    if source_path is None:
        print("원본 json을 자동으로 못 찾음. 아래 후보를 직접 확인해줘:")
        for c in candidates:
            print(f"  {c}")
        return

    correct_name = resolve_dl_name(correct_category_id)

    corrections = []
    if CORRECTIONS_PATH.exists():
        with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            corrections = json.load(f)

    corrections.append({
        "original_json": str(source_path),
        "correct_dl_name": correct_name,
        "note": f"category_id={wrong_category_id} index={index} -> category_id={correct_category_id} ({correct_name})"
    })

    CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

    print(f"[교정] 추가됨 ({len(corrections)}번째 항목): {source_path.name} -> \"{correct_name}\"")


def _append_exclusion(source_path: Path, note: str):
    excluded = []
    if EXCLUDED_ANNOTATIONS_PATH.exists():
        with open(EXCLUDED_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            excluded = json.load(f)

    excluded.append({"original_json": str(source_path), "note": note})

    EXCLUDED_ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXCLUDED_ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(excluded, f, ensure_ascii=False, indent=2)

    print(f"[제외] 추가됨 ({len(excluded)}번째 항목): {source_path.name}")


def exclude(category_id: int, index: int):
    """cat_id + 그리드 #번호로 제외 (verify_class_crops 그리드에서 발견한 경우)"""
    entry = get_index_entry(category_id, index)
    if entry is None:
        return
    source_path, candidates = find_source_json_path(entry["file_name"], entry["bbox"])
    if source_path is None:
        print("원본 json을 자동으로 못 찾음. 아래 후보를 직접 확인해줘:")
        for c in candidates:
            print(f"  {c}")
        return
    _append_exclusion(source_path, f"category_id={category_id} index={index} - bbox 비정상(빈 배경 등)으로 제외")


def exclude_ann(ann_id: int):
    """train_coco.json의 annotation id로 직접 제외 (IoU 중복 검출 결과에서 발견한 경우)"""
    ann = get_annotation_by_id(ann_id)
    if ann is None:
        print(f"annotation id={ann_id} 를 train_coco.json에서 못 찾음")
        return
    source_path, candidates = find_source_json_path(ann["file_name"], ann["bbox"])
    if source_path is None:
        print("원본 json을 자동으로 못 찾음. 아래 후보를 직접 확인해줘:")
        for c in candidates:
            print(f"  {c}")
        return
    _append_exclusion(source_path, f"ann_id={ann_id} (category_id={ann['category_id']}) - IoU 중복으로 제외")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_id", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--locate", nargs=2, type=int, metavar=("CATEGORY_ID", "INDEX"))
    parser.add_argument("--correct", nargs=3, type=int, metavar=("WRONG_CAT_ID", "INDEX", "CORRECT_CAT_ID"))
    parser.add_argument("--exclude", nargs=2, type=int, metavar=("CATEGORY_ID", "INDEX"))
    parser.add_argument("--exclude_ann", type=int, metavar="ANN_ID")
    args = parser.parse_args()

    if args.correct:
        correct(args.correct[0], args.correct[1], args.correct[2])
        return
    if args.exclude:
        exclude(args.exclude[0], args.exclude[1])
        return
    if args.exclude_ann is not None:
        exclude_ann(args.exclude_ann)
        return
    if args.locate:
        locate(args.locate[0], args.locate[1])
        return

    coco = load_coco()
    crops_by_cat, cat_id_to_name = build_crops_by_category(coco)

    if args.all:
        print(f"총 {len(crops_by_cat)}개 클래스 처리 시작...")
        for category_id, items in sorted(crops_by_cat.items()):
            save_grid(category_id, items, cat_id_to_name.get(category_id, "unknown"))
        print("완료. outputs/class_review 폴더에서 하나씩 확인해줘.")
    elif args.category_id is not None:
        items = crops_by_cat.get(args.category_id, [])
        save_grid(args.category_id, items, cat_id_to_name.get(args.category_id, "unknown"))
    else:
        print("사용법: --category_id <id> / --all / --locate <id> <index> / --correct <원본id> <index> <수정id> / --exclude <id> <index> / --exclude_ann <ann_id>")


if __name__ == "__main__":
    main()