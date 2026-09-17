"""여러 파일에서 공통으로 쓰는 유틸 함수 모음."""

import random

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

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
