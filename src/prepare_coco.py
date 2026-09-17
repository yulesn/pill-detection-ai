import json
import os
from pathlib import Path

from valid_data import collect_paths  # 경로 수집 로직 재사용 (기준 어긋남 방지)

EXCLUDED_PATH = "data/processed/excluded_images.json"
OUTPUT_PATH = "data/processed/train_coco.json"


def main():
    img_paths, json_paths, valid_names = collect_paths()

    with open(EXCLUDED_PATH, "r", encoding="utf-8") as f:
        excluded = set(json.load(f))

    images, annotations, categories = [], [], []
    category_name_to_id = {}
    image_id = 0
    annotation_id = 0
    skipped_out_of_bounds = 0

    for name in valid_names:
        if name in excluded:
            continue  # select_valid_images.py가 이미 "라벨 누락 의심"으로 판정한 이미지는 통째로 건너뜀

        with open(json_paths[name][0], "r", encoding="utf-8") as f:
            first_data = json.load(f)
        img_w = first_data["images"][0]["width"]
        img_h = first_data["images"][0]["height"]

        image_annotations = []
        for json_path in json_paths[name]:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pill_name = data["images"][0]["dl_name"]
            if pill_name not in category_name_to_id:
                category_name_to_id[pill_name] = len(category_name_to_id) + 1
                categories.append({"id": category_name_to_id[pill_name], "name": pill_name})
            category_id = category_name_to_id[pill_name]

            for anno in data["annotations"]:
                if "bbox" not in anno or len(anno["bbox"]) != 4:
                    continue
                x, y, w, h = anno["bbox"]
                if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
                    # 지금 데이터엔 없지만, 나중에 새 데이터가 추가될 때를 대비한 안전장치
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

    print(f"제외된 이미지(라벨 누락 의심): {len(excluded)}장")
    print(f"제외된 annotation(경계 이탈, 안전장치): {skipped_out_of_bounds}건")
    print(f"최종: 이미지 {len(images)}장, annotation {len(annotations)}건, "
          f"클래스 {len(categories)}종 → {OUTPUT_PATH} 저장 완료")


if __name__ == "__main__":
    main()