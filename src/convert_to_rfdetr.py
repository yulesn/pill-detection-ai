"""COCO 포맷 데이터를 RF-DETR가 기대하는 디렉터리 구조로 변환한다.

data/processed/{train,val}/images + annotations.json 을
data/processed/rfdetr/{train,valid}/*.png + _annotations.coco.json 으로 변환한다
(RF-DETR는 Roboflow 스타일로 각 split 폴더에 이미지와 _annotations.coco.json이
함께 있어야 함). 이미지는 복사 대신 심볼릭 링크로 연결한다.

사용 예:
    python src/convert_to_rfdetr.py
"""

import argparse
import json
from pathlib import Path

SPLIT_MAP = {"train": "train", "val": "valid"}


def convert_split(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(src_dir / "annotations.json", "r", encoding="utf-8") as f:
        coco = json.load(f)

    for image in coco["images"]:
        src_image = (src_dir / "images" / image["file_name"]).resolve()
        dst_image = dst_dir / image["file_name"]
        if dst_image.exists() or dst_image.is_symlink():
            dst_image.unlink()
        dst_image.symlink_to(src_image)

    with open(dst_dir / "_annotations.coco.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="data/processed/rfdetr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    for src_split, dst_split in SPLIT_MAP.items():
        dst_dir = output_dir / dst_split
        convert_split(processed_dir / src_split, dst_dir)
        print(f"{src_split} -> {dst_dir}")


if __name__ == "__main__":
    main()
