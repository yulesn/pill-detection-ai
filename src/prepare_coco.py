import json
import os
from pathlib import Path

from valid_data import collect_paths  # 경로 수집 로직 재사용 (기준 어긋남 방지)

EXCLUDED_PATH = "data/processed/excluded_images.json"
CORRECTIONS_PATH = "data/processed/label_corrections.json"
EXCLUDED_ANNOTATIONS_PATH = "data/processed/excluded_annotations.json"
OUTPUT_PATH = "data/processed/train_coco.json"


def load_corrections(path: str = CORRECTIONS_PATH) -> dict:
    """원본 json 경로 -> 올바른 dl_name 매핑."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        corrections = json.load(f)
    return {str(Path(c["original_json"])): c["correct_dl_name"] for c in corrections}


def load_excluded_annotations(path: str = EXCLUDED_ANNOTATIONS_PATH) -> set:
    """통째로 제외할 원본 json 경로 집합 (bbox 자체가 비정상인 경우)."""
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        excluded = json.load(f)
    return {str(Path(e["original_json"])) for e in excluded}


def main():
    img_paths, json_paths, valid_names = collect_paths()

    with open(EXCLUDED_PATH, "r", encoding="utf-8") as f:
        excluded_images = set(json.load(f))

    corrections = load_corrections()
    excluded_annotations = load_excluded_annotations()
    if corrections:
        print(f"라벨 교정 목록 {len(corrections)}건 로드됨: {CORRECTIONS_PATH}")
    if excluded_annotations:
        print(f"개별 알약 제외 목록 {len(excluded_annotations)}건 로드됨: {EXCLUDED_ANNOTATIONS_PATH}")

    images, annotations, categories = [], [], []
    category_name_to_id = {}
    image_id = 0
    annotation_id = 0
    skipped_out_of_bounds = 0
    corrected_count = 0
    excluded_annotation_count = 0

    for name in valid_names:
        if name in excluded_images:
            continue  # select_valid_images.py가 이미 "라벨 누락 의심"으로 판정한 이미지는 통째로 건너뜀

        with open(json_paths[name][0], "r", encoding="utf-8") as f:
            first_data = json.load(f)
        img_w = first_data["images"][0]["width"]
        img_h = first_data["images"][0]["height"]

        image_annotations = []
        for json_path in json_paths[name]:
            key = str(Path(json_path))

            if key in excluded_annotations:
                excluded_annotation_count += 1
                continue  # bbox 자체가 비정상인 알약 하나만 건너뜀 (같은 사진의 다른 알약은 그대로 사용)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pill_name = data["images"][0]["dl_name"]
            if key in corrections:
                corrected_name = corrections[key]
                if corrected_name != pill_name:
                    corrected_count += 1
                pill_name = corrected_name

            if pill_name not in category_name_to_id:
                category_name_to_id[pill_name] = len(category_name_to_id) + 1
                categories.append({"id": category_name_to_id[pill_name], "name": pill_name})
            category_id = category_name_to_id[pill_name]

            for anno in data["annotations"]:
                if "bbox" not in anno or len(anno["bbox"]) != 4:
                    continue
                x, y, w, h = anno["bbox"]
                if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                    skipped_out_of_bounds += 1
                    continue
                image_annotations.append({
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                })

        image_id += 1
        images.append({
            "id": image_id,
            "file_name": os.path.basename(img_paths[name]),
            "width": img_w,
            "height": img_h,
        })
        for anno in image_annotations:
            annotation_id += 1
            annotations.append({"id": annotation_id, "image_id": image_id, "iscrowd": 0, **anno})

    coco = {"images": images, "annotations": annotations, "categories": categories}

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"제외된 이미지(라벨 누락 의심): {len(excluded_images)}장")
    print(f"제외된 annotation(경계 이탈, 안전장치): {skipped_out_of_bounds}건")
    if corrections:
        print(f"실제로 적용된 라벨 교정: {corrected_count}건 (목록엔 {len(corrections)}건)")
    if excluded_annotations:
        print(f"실제로 제외된 개별 알약(bbox 비정상): {excluded_annotation_count}건 (목록엔 {len(excluded_annotations)}건)")
    print(f"최종: 이미지 {len(images)}장, annotation {len(annotations)}건, "
          f"클래스 {len(categories)}종 → {OUTPUT_PATH} 저장 완료")


if __name__ == "__main__":
    main()