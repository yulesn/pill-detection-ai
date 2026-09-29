"""학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
"""

import argparse
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights
from tqdm import tqdm

from dataset import PillDataset
from model import build_model
from utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def collate_fn(batch):  # 객체 검출 모델 학습을 위한 배치 데이터 묶는 함수 추가
    return tuple(zip(*batch))


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
        # 디바이스 설정
        if torch.backends.mps.is_available() and config['train']['device'] != 'cpu':
            device = torch.device('mps')
        else:
            device = torch.device(config["train"]['device'])

        print(f'사용 중인 디바이스 : {device}')

        # PillDataset / DataLoader 구성
        dataset = PillDataset(
            data_dir=config['data']['processed_dir'],
            img_size=512,
            is_train=True
        )

        data_loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )

        # build_model를 통한 모델 생성
        model = build_model(config)
        model.to(device)

        # 옵티마이저
        params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            params,
            lr=config["train"]["learning_rate"],
            weight_decay=0.0005
        )

        epochs = config["train"]["epochs"]

        # 학습루프
        print(f"=== RetinaNet 학습 시작 (총 {epochs} 에폭) ===")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            progress_bar = tqdm(
                data_loader,
                desc=f"Epoch [{epoch+1}/{epochs}]",
                leave=True
            )

            for images, targets in progress_bar:
                images = list(
                    image.to(device)
                    for image in images
                )

                targets = [
                    {
                        k: v.to(device)
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                # RetinaNet loss 계산
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()

                # 분류/박스 loss를 함께 확인
                progress_bar.set_postfix(
                    loss=f"{losses.item():.4f}",
                    cls=f"{loss_dict['classification'].item():.4f}",
                    box=f"{loss_dict['bbox_regression'].item():.4f}"
                )

            print(
                f"Epoch [{epoch+1}/{epochs}] 완료 | "
                f"평균 손실: "
                f"{epoch_loss / len(data_loader):.4f} | "
                f"LR: "
                f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        # 체크포인트 저장
        save_dir = os.path.join(
            config["output"]["dir"],
            config["output"]["experiment_name"]
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        model_save_path = os.path.join(
            save_dir,
            "retinanet_pill_model.pth"
        )

        torch.save(
            model.state_dict(),
            model_save_path
        )

        print(
            f"=== 학습 완료! 체크포인트 저장됨: "
            f"{model_save_path} ==="
        )

        return

    raise ValueError(
        f"지원하지 않는 framework입니다: {framework}"
    )


if __name__ == "__main__":
    main()