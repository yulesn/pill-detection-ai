"""학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
    # 체크포인트에서 이어서 학습
    python src/train.py --config configs/default.yaml --checkpoint outputs/checkpoints/default/last.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import PillDataset
from model import build_model
from utils import load_config, set_seed


train_transform = v2.Compose([
    v2.ToImage(),
    #밝기/대비/채도/색조를 무작위로 변화
    v2.RandomPhotometricDistort(p=0.5),     
    #캔버스를 최대 2배까지 확장하고 원본 이미지를 그 안 무작위 위치에 배치
    v2.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=0.3),     
    #원본의 일부를 무작위로 잘라내되, 남은 박스가 원래 박스와 충분히 겹치도록
    v2.RandomIoUCrop(),
    #좌우 반전
    v2.RandomHorizontalFlip(p=0.5),
    #크롭/줌 과정에서 화면 밖으로 밀려나거나 크기가 0에 가깝게 찌그러진 박스를 제거
    v2.SanitizeBoundingBoxes(),
    v2.ToDtype(torch.float32, scale=True),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="이어서 학습할 체크포인트(.pt) 경로"
    )
    return parser.parse_args()


def collate_fn(batch):
    """이미지당 알약 개수(N)가 달라 기본 collate로 배치를 쌓을 수 없으므로,
    (image, target) 튜플의 리스트를 그대로 둔다."""
    return tuple(zip(*batch))


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    return images, targets


def run_epoch(model, loader, device, optimizer=None) -> float:
    """optimizer가 주어지면 학습(역전파 O), 없으면 loss만 계산(역전파 X)."""
    total_loss = 0.0
    for images, targets in tqdm(loader, leave=False):
        images, targets = to_device(images, targets, device)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def train_torchvision(config: dict, checkpoint: str | None = None) -> None:
    device = torch.device(config["train"]["device"])
    processed_dir = Path(config["data"]["processed_dir"])

    train_dataset = PillDataset(processed_dir / "train", transform=train_transform)
    val_dataset = PillDataset(processed_dir / "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"체크포인트에서 이어서 학습합니다: {checkpoint}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config["train"]["learning_rate"], momentum=0.9, weight_decay=0.0005
    )
    # val_loss가 lr_patience epoch 동안 개선되지 않으면 lr을 lr_factor배로 줄인다.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["train"]["lr_factor"], patience=config["train"]["lr_patience"]
    )

    checkpoint_dir = (
        Path(config["output"]["dir"]) / "checkpoints" / config["output"]["experiment_name"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        train_loss = run_epoch(model, train_loader, device, optimizer)

        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, optimizer=None)

        scheduler.step(val_loss)

        print(
            f"[epoch {epoch}/{config['train']['epochs']}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")


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
        train_torchvision(config, checkpoint=args.checkpoint)
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
