"""학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
"""

import argparse

from dataset import PillDataset
from model import build_model
from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["train"]["seed"])
    framework = config["model"]["framework"]

    if framework == "yolo":
        # TODO(YOLO 트랙): ultralytics가 학습 루프를 자체 제공하므로 커스텀 루프 불필요
        # model = build_model(config)
        # model.train(
        #     data="data/processed/yolo/data.yaml",
        #     epochs=config["train"]["epochs"],
        #     batch=config["train"]["batch_size"],
        # )
        raise NotImplementedError("YOLO 학습 루프를 구현해주세요.")

    if framework == "torchvision":
        # TODO(torchvision 트랙): PillDataset으로 train/val DataLoader 구성,
        # build_model(config)로 모델 생성, 커스텀 학습 루프 작성,
        # config["output"]["dir"]/config["output"]["experiment_name"] 아래에 체크포인트 저장
        raise NotImplementedError("torchvision 학습 루프를 구현해주세요.")

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
