"""
AI Hub에서 받은 zip을 압축 해제하면 K-코드별 하위 폴더(K-031998_json/ 등)로
나뉘어 있는데, 이걸 기존 프로젝트 구조처럼 한 폴더 안에 flat하게 모아주는 스크립트.

사용법:
    python src/flatten_aihub_data.py --src <압축푼 원본 폴더> --dst <목적지 폴더>

예시:
    # 라벨(json) 정리
    python src/flatten_aihub_data.py \
        --src data/raw/aihub_pill_raw/TL_15_단일 \
        --dst data/raw/aihub_pill/train_annotations

    # 이미지 정리
    python src/flatten_aihub_data.py \
        --src data/raw/aihub_pill_raw/TS_15_단일 \
        --dst data/raw/aihub_pill/train_images
"""

import argparse
import shutil
from pathlib import Path

def flatten(src_dir: str, dst_dir: str):
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"[ERROR] 원본 폴더가 없습니다: {src}")
        return

    count = 0
    skipped = 0

    # 하위 폴더가 몇 겹이든 상관없이 전체 파일을 재귀적으로 찾음
    for f in src.rglob("*"):
        if f.is_file():
            target = dst / f.name
            if target.exists():
                # 이름이 겹치면 건너뛰고 경고만 출력
                print(f"[SKIP] 이미 존재함: {f.name}")
                skipped += 1
                continue
            shutil.copy2(f, target)
            count += 1

    print(f"\n완료: {count}개 파일 복사, {skipped}개 건너뜀")
    print(f"목적지: {dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="압축 해제한 원본 폴더 경로")
    parser.add_argument("--dst", required=True, help="flat하게 모을 목적지 폴더 경로")
    args = parser.parse_args()
    flatten(args.src, args.dst)
