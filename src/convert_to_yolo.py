"""COCO 포맷 데이터를 YOLO가 기대하는 디렉터리 구조로 변환한다.

data/processed/coco/{train,val}/images + annotations.json (prepare_split.py 결과) 을
data/processed/yolo/{train,val}/{images,labels} + data.yaml 로 변환한다.
이미지는 복사 대신 심볼릭 링크로 연결한다.

사용 전에 prepare_split.py 를 먼저 실행해서 data/processed/coco 를 준비해야 한다:
    python src/prepare_split.py --config configs/rfdetr.yaml \
        --image-dir data/augmented/raw/images --ann-dir data/augmented/raw/annotations
    python src/convert_to_yolo.py
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

SPLITS = ("train", "val")


def load_category_maps(processed_dir: Path) -> tuple[dict, dict]:
    """category_mapping.json(label/category_id/name)에서
    label(1..N) -> 0-index class_id, label -> "K-{코드}" 이름 매핑을 만든다.

    "K-{코드}" 포맷은 predict.py의 yolo_detect가
    label_map[idx].split('-')[1] 로 파싱하는 기존 계약을 그대로 쓰기 위함이다.
    """
    mapping_path = processed_dir / "splits" / "category_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_id_of = {entry["label"]: entry["label"] - 1 for entry in mapping}
    name_of = {entry["label"]: f"K-{entry['category_id']:06d}" for entry in mapping}
    return class_id_of, name_of


def convert_split(src_dir: Path, dst_dir: Path, class_id_of: dict) -> int:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_image_dir = dst_dir / "images"
    dst_label_dir = dst_dir / "labels"
    dst_image_dir.mkdir(parents=True)
    dst_label_dir.mkdir(parents=True)

    with open(src_dir / "annotations.json", "r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for image in coco["images"]:
        src_image = (src_dir / "images" / image["file_name"]).resolve()
        dst_image = dst_image_dir / image["file_name"]
        dst_image.symlink_to(src_image)

        w, h = image["width"], image["height"]
        lines = []
        for ann in anns_by_image.get(image["id"], []):
            x, y, bw, bh = ann["bbox"]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w, norm_h = bw / w, bh / h
            class_id = class_id_of[ann["category_id"]]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        stem = Path(image["file_name"]).stem
        (dst_label_dir / f"{stem}.txt").write_text("\n".join(lines))

    return len(coco["images"])


def save_data_yaml(output_dir: Path, class_id_of: dict, name_of: dict) -> None:
    names = {class_id_of[label]: name for label, name in sorted(name_of.items())}
    yaml_data = {
        # ultralytics는 상대경로일 때 로컬 datasets_dir 설정 기준으로 다시 붙여서
        # 엉뚱한 위치를 찾는 경우가 있어(머신마다 설정이 다름) 절대경로로 고정한다.
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed/coco")
    parser.add_argument("--output-dir", type=str, default="data/processed/yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    class_id_of, name_of = load_category_maps(processed_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        n = convert_split(processed_dir / split, output_dir / split, class_id_of)
        print(f"{split} -> {output_dir / split} ({n}장)")

    save_data_yaml(output_dir, class_id_of, name_of)
    print(f"data.yaml -> {output_dir / 'data.yaml'} (클래스 {len(name_of)}종)")


if __name__ == "__main__":
    main()
