# EDA에서 이상이 있든 데이터 8건(Bbox 누락 7건 + Bbox 이탈 1건) 제외를 위한 코드

import json
import os
from collections import defaultdict
from pathlib import Path

RAW_IMG_DIR = "data/raw/sprint_ai_project1_data/train_images/"
RAW_ANNOT_DIR = "data/raw/sprint_ai_project1_data/train_annotations/"
OUTPUT_PATH = "data/processed/excluded_images.json"
MIN_VALID_PILLS_PER_IMAGE = 3


def collect_paths():
    img_paths = {}
    json_paths = defaultdict(list)
    for root, _, files in os.walk(RAW_IMG_DIR):
        for file in files:
            if file.endswith(".png"):
                img_paths[os.path.splitext(file)[0]] = os.path.join(root, file)
    for root, _, files in os.walk(RAW_ANNOT_DIR):
        for file in files:
            if file.endswith(".json"):
                json_paths[os.path.splitext(file)[0]].append(os.path.join(root, file))
    return img_paths, json_paths, sorted(set(img_paths) & set(json_paths))


def main():
    img_paths, json_paths, valid_names = collect_paths()
    excluded = []

    for name in valid_names:
        with open(json_paths[name][0], "r", encoding="utf-8") as f:
            first_data = json.load(f)
        img_w, img_h = first_data["images"][0]["width"], first_data["images"][0]["height"]

        valid_count = 0
        for json_path in json_paths[name]:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for anno in data["annotations"]:
                if "bbox" not in anno or len(anno["bbox"]) != 4:
                    continue
                x, y, w, h = anno["bbox"]
                if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                    continue  
                valid_count += 1

        if valid_count < MIN_VALID_PILLS_PER_IMAGE:
            excluded.append(name)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(excluded, f, ensure_ascii=False, indent=2)
    print(f"제외 대상 이미지: {len(excluded)}장 → {OUTPUT_PATH} 저장 완료")


if __name__ == "__main__":
    main()