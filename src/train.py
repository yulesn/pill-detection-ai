"""YOLO 모델 학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 설정 파일 로드
    config = load_config(args.config)

    # 랜덤 시드 설정
    set_seed(config["train"]["seed"])

    # 데이터 및 모델 설정
    data_dir = Path(config["data"]["processed_dir"])
    data_yaml = data_dir / "data.yaml"

    model_name = config["model"]["name"]

    # 학습 설정
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    learning_rate = config["train"]["learning_rate"]
    device = config["train"]["device"]

    # 출력 설정
    output_dir = config["output"]["dir"]
    experiment_name = config["output"]["experiment_name"]

    # YOLO 모델 생성
    model = YOLO(model_name)

    # 모델 학습
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        lr0=learning_rate,
        device=device,
        project=output_dir,
        name=experiment_name,
    )


if __name__ == "__main__":
    main()