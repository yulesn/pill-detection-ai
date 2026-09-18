"""학습 데이터의 오류를 검출하고, train/val 파일을 실제로 나눠서 저장하는 스크립트.

annotation(bbox 범위 벗어남, 이미지 파일 손상/누락 등) 오류가 있는 이미지는
분할 대상에서 제외하고 별도로 기록한다. 나머지 이미지는 카테고리별 비율을
최대한 맞춰가며(근사 stratified) train/val로 나눠서 각각 별도 폴더에 복사한다.
PillDataset은 그대로 두고 data_dir만 바꿔서 두 개의 인스턴스(서로 다른
transform)를 만들면 되도록 하기 위함이다.

사용 예:
    python src/prepare_split.py --config configs/default.yaml --val-ratio 0.2

출력:
    data/processed/train/images/                      — train 이미지
    data/processed/train/annotations.json — train 표준 COCO annotation
    data/processed/val/images/                        — val 이미지
    data/processed/val/annotations.json   — val 표준 COCO annotation
    data/processed/splits/train_ids.json, val_ids.json     — 분할 결과 audit용 image_id 목록
    data/processed/splits/invalid_images.json              — 오류가 발견되어 제외한 이미지와 사유
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from utils import load_config, set_seed

DATASET_DIR_NAME = "sprint_ai_project1_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def load_annotations(annotation_dir: Path) -> tuple[dict, dict, dict]:
    """annotation_dir 아래 json들을 image_id 기준으로 모은다.

    return: (images: {image_id: info}, annotations: {image_id: [ann, ...]},
           categories: {cat_id: cat})
    """
    images: dict = {}
    annotations: dict = defaultdict(list)
    categories: dict = {}
    for annotation_file in sorted(annotation_dir.rglob("*.json")):
        with open(annotation_file, "r") as f:
            content = json.load(f)
        image = content["images"][0]
        ann = content["annotations"][0]
        category = content["categories"][0]
        image_id = image["id"]
        images.setdefault(image_id, image)
        annotations[image_id].append(ann)
        categories.setdefault(category["id"], category)
    return images, dict(annotations), categories


def is_match_image_and_ann(image_name, ann_size):
    """알약의 개수와 bbox 개수가 일치하는지 확인"""
    group_id = image_name.split('_')[0]
    category_ids = group_id.split('-')[1:]
    # 파일명에 포함된 알약의 개수가 추출된 bbox의 개수와 같은지 확인
    return True if len(category_ids) == ann_size else False



def compute_iou(box1, box2):
    """bbox 간 겹침 정도 IOU를 계산"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def validate_image(info: dict, anns: list, image_dir: Path) -> list[str]:
    """이미지 한 장에 대해 발견한 문제 목록을 반환한다 (없으면 빈 리스트)."""
    errors = []
    file_name = info["file_name"]
    image_path = image_dir / file_name
    if not image_path.exists():
        return [f"이미지 파일 없음: {file_name}"]

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError) as e:
        return [f"이미지를 열 수 없음: {e}"]

    if (width, height) != (info["width"], info["height"]):
        errors.append(
            f"json에 기록된 크기({info['width']}x{info['height']})와 "
            f"실제 이미지 크기({width}x{height})가 다름"
        )

    for i in range(len(anns)):
        bbox = anns[i]["bbox"]
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            errors.append(f"bbox 크기가 0 이하: {bbox} (annotation id {anns[i]['id']})")
        elif x < 0 or y < 0 or x + w > width or y + h > height:
            errors.append(
                f"bbox가 이미지 범위를 벗어남: {bbox} "
                f"(image {width}x{height}, annotation id {anns[i]['id']})"
            )
        for j in range(i+1,len(anns)):
            next_bbox = anns[j]["bbox"]
            iou = compute_iou(bbox, next_bbox)
            if iou > 0.5:
                errors.append(
                    f"bbox가 심하게 겹침: {bbox}, {next_bbox}" 
                    f"iou: ({iou})"
                )
    
    if not is_match_image_and_ann(file_name, len(anns)):
        errors.append(f"파일명({file_name})에 포함된 알약의 개수가 추출된 bbox의 개수({len(anns)})와 다름")

    return errors


def stratified_split(
    images: dict, annotations: dict, val_ratio: float, seed: int
) -> tuple[list, list]:
    """카테고리별 비율을 최대한 맞추는 근사 stratified split.

    이미지 한 장이 여러 카테고리(알약)를 포함할 수 있어 완벽한 stratification은
    불가능하므로, 이미지 수가 적은 카테고리부터 먼저 val 목표 수량을
    채우는 greedy 방식을 쓴다.
    """
    random.seed(seed)
    image_ids = list(images.keys())
    random.shuffle(image_ids)

    cat_to_images = defaultdict(set)
    for image_id in image_ids:
        for ann in annotations[image_id]:
            cat_to_images[ann["category_id"]].add(image_id)

    val_target = {
        cat_id: max(1, round(len(imgs) * val_ratio)) if len(imgs) >= 2 else 0
        for cat_id, imgs in cat_to_images.items()
    }
    val_count = defaultdict(int)
    assigned: dict = {}

    for cat_id in sorted(val_target, key=lambda c: len(cat_to_images[c])):
        candidates = [i for i in cat_to_images[cat_id] if i not in assigned]
        random.shuffle(candidates)
        need = val_target[cat_id] - val_count[cat_id]
        for image_id in candidates:
            if need <= 0:
                break
            assigned[image_id] = "val"
            for ann in annotations[image_id]:
                val_count[ann["category_id"]] += 1
            need -= 1

    remaining = [i for i in image_ids if i not in assigned]
    n_val_so_far = sum(1 for s in assigned.values() if s == "val")
    n_val_remaining = round(len(image_ids) * val_ratio) - n_val_so_far
    n_val_remaining = max(0, min(n_val_remaining, len(remaining)))
    for image_id in remaining[:n_val_remaining]:
        assigned[image_id] = "val"
    for image_id in remaining[n_val_remaining:]:
        assigned[image_id] = "train"

    train_ids = sorted(i for i, s in assigned.items() if s == "train")
    val_ids = sorted(i for i, s in assigned.items() if s == "val")
    return train_ids, val_ids


def build_cat_id_to_label(categories: dict) -> dict:
    """원본 category_id를 정렬 순서 기준 0-index 라벨로 매핑한다."""
    return {cat_id: i+1 for i, cat_id in enumerate(sorted(categories))}


def copy_split(
    images: dict,
    annotations: dict,
    categories: dict,
    cat_id_to_label: dict,
    ids: list,
    image_dir: Path,
    split_dir: Path,
) -> None:
    """ids에 해당하는 이미지를 복사하고, annotation은 표준 COCO 형식 단일
    파일로 저장한다. annotation의 category_id는 cat_id_to_label을 통해
    0부터 시작하는 라벨로 리매핑해서 저장한다.

    split_dir/images에 이미지를, split_dir/annotations.json에
    {"images": [...], "annotations": [...], "categories": [...]} 형태로 저장하므로,
    PillDataset(split_dir, ...)로 바로 읽을 수 있다.
    """
    dst_image_dir = split_dir / "images"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    dst_image_dir.mkdir(parents=True)

    coco_annotations = []
    used_category_ids = set()
    for image_id in ids:
        info = images[image_id]
        shutil.copy2(image_dir / info["file_name"], dst_image_dir / info["file_name"])
        for ann in annotations[image_id]:
            used_category_ids.add(ann["category_id"])
            coco_annotations.append(
                {**ann, "category_id": cat_id_to_label[ann["category_id"]]}
            )

    coco = {
        "images": [images[image_id] for image_id in ids],
        "annotations": coco_annotations,
        "categories": [
            {**categories[cat_id], "id": cat_id_to_label[cat_id]}
            for cat_id in sorted(used_category_ids)
        ],
    }
    with open(split_dir / "annotations.json", "w") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["train"]["seed"])

    data_dir = Path(config["data"]["raw_dir"]) / DATASET_DIR_NAME
    image_dir = data_dir / "train_images"
    annotation_dir = data_dir / "train_annotations"

    images, annotations, categories = load_annotations(annotation_dir)
    print(f"전체 이미지 {len(images)}개, 카테고리 {len(categories)}개 로드")

    invalid = {}
    for image_id, info in images.items():
        errors = validate_image(info, annotations[image_id], image_dir)
        if errors:
            invalid[image_id] = {"file_name": info["file_name"], "errors": errors}

    valid_images = {i: info for i, info in images.items() if i not in invalid}
    valid_annotations = {i: annotations[i] for i in valid_images}

    print(f"오류 이미지 {len(invalid)}개 발견 (분할 대상에서 제외)")
    for image_id, info in invalid.items():
        print(f"  - image_id={image_id} {info['file_name']}: {info['errors']}")

    train_ids, val_ids = stratified_split(
        valid_images,
        valid_annotations,
        val_ratio=args.val_ratio,
        seed=config["train"]["seed"],
    )

    valid_category_ids = {
        ann["category_id"]
        for anns in valid_annotations.values()
        for ann in anns
    }
    valid_categories = {cat_id: categories[cat_id] for cat_id in valid_category_ids}
    cat_id_to_label = build_cat_id_to_label(valid_categories)

    processed_dir = Path(config["data"]["processed_dir"])
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    copy_split(images, annotations, categories, cat_id_to_label, train_ids, image_dir, train_dir)
    copy_split(images, annotations, categories, cat_id_to_label, val_ids, image_dir, val_dir)

    report_dir = processed_dir / "splits"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_ids.json").write_text(json.dumps(train_ids, indent=2))
    (report_dir / "val_ids.json").write_text(json.dumps(val_ids, indent=2))
    (report_dir / "category_mapping.json").write_text(
        json.dumps(
            [
                {"label": label, "category_id": cat_id, "name": categories[cat_id]["name"]}
                for cat_id, label in sorted(cat_id_to_label.items(), key=lambda x: x[1])
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    (report_dir / "invalid_images.json").write_text(
        json.dumps(invalid, indent=2, ensure_ascii=False)
    )

    print(f"train {len(train_ids)}개 -> {train_dir}")
    print(f"val {len(val_ids)}개 -> {val_dir}")


if __name__ == "__main__":
    main()
