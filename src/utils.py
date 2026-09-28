"""여러 파일에서 공통으로 쓰는 유틸 함수 모음."""

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """재현성을 위해 random/numpy/torch의 seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """configs/*.yaml 파일을 읽어서 dict로 반환한다."""
    import yaml

    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_predictions_csv(results: list[dict], output_path: str | Path) -> None:
    """예측 결과를 annotation_id를 붙여 제출용 CSV로 저장한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for annotation_id, result in enumerate(results, start=1):
            writer.writerow({"annotation_id": annotation_id, **result})
