"""학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
"""

import argparse
import csv
import os

from dataset import PillDataset
from model import build_model
from utils import load_config, set_seed


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["train"]["seed"])
    framework = config["model"]["framework"]
    exp_name = config["output"]["experiment_name"]

    if framework == "yolo":
        raise NotImplementedError("YOLO 학습 루프를 구현해주세요.")

    if framework == "torchvision":
        import torch
        from torch.amp import autocast, GradScaler
        from torch.utils.data import DataLoader

        data_root = config["data"]["raw_dir"] + "/sprint_ai_project1_data"

        train_dataset = PillDataset(data_root, train=True, subset="train")
        val_dataset = PillDataset(data_root, train=True, subset="val")
        print(f"학습 데이터: {len(train_dataset)}장 / 검증 데이터: {len(val_dataset)}장")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        device = torch.device(config["train"]["device"])
        model = build_model(config)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config["train"]["learning_rate"], momentum=0.9)
        scaler = GradScaler("cuda")

        # 이번 실험(exp_name)의 epoch별 loss를 기록할 로그 파일 준비
        log_dir = os.path.join(config["output"]["dir"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"kimgun_{exp_name}_losses.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

        best_val_loss = float("inf")

        for epoch in range(config["train"]["epochs"]):
            # ---- 학습 ----
            model.train()
            train_loss = 0.0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with autocast("cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # ---- 검증 ----
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with autocast("cuda"):
                        loss_dict = model(images, targets)
                        loss = sum(loss_dict.values())
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"[Epoch {epoch+1}/{config['train']['epochs']}] train_loss: {train_loss:.4f} / val_loss: {val_loss:.4f}")

            # 로그 파일에 이번 epoch 결과 한 줄 추가
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch + 1, round(train_loss, 4), round(val_loss, 4)])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dir = os.path.join(config["output"]["dir"], "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label_to_raw_id": train_dataset.label_to_raw_id,
                    "categories": train_dataset.categories,
                }, os.path.join(save_dir, f"kimgun_{exp_name}_best.pt"))
                print(f"  -> best 갱신, 저장됨 (val_loss: {best_val_loss:.4f})")
        return
    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()