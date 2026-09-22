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


def load_model_weights(model, checkpoint_path: str, device) -> None:
    """체크포인트를 불러와 model에 적용한다.

    이 프로젝트의 train.py는 model.state_dict()를 그대로 저장하지만, 팀원에
    따라 {"model_state_dict": ..., "label_to_raw_id": ..., ...}처럼 메타데이터와
    함께 저장한 체크포인트도 있어서 두 형식을 모두 지원한다.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def load_config(config_path: str) -> dict:
    """configs/*.yaml 파일을 읽어서 dict로 반환한다."""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
