import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

COCO_PATH = Path("data/processed/train_coco.json")


def compute_iou(box1, box2):
    """box = [x, y, w, h] 형식"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


def find_iou_duplicates(threshold: float = 0.5):
    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # 이미지별로 annotation 묶기
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    duplicates = []
    for image_id, anns in anns_by_image.items():
        # 같은 이미지 안의 모든 annotation 쌍(pair)을 비교
        for ann_a, ann_b in combinations(anns, 2):
            iou = compute_iou(ann_a["bbox"], ann_b["bbox"])
            if iou > threshold:
                duplicates.append({
                    "file_name": image_id_to_file[image_id],
                    "ann_id_a": ann_a["id"],
                    "ann_id_b": ann_b["id"],
                    "category_a": ann_a["category_id"],
                    "category_b": ann_b["category_id"],
                    "iou": round(iou, 3),
                })

    print(f"IoU > {threshold} 중복 쌍: {len(duplicates)}건\n")
    for d in duplicates:
        print(f"  {d['file_name']}  (ann {d['ann_id_a']} vs {d['ann_id_b']}, "
              f"cat {d['category_a']} vs {d['category_b']}, IoU={d['iou']})")

    return duplicates


if __name__ == "__main__":
    find_iou_duplicates()