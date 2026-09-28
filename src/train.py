"""학습 실행 스크립트.

사용 예:
    python src/train.py --config configs/default.yaml
"""

import argparse
import csv
import os
import time

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

    if framework == "yolo":
        data_yaml = config["data"]["yaml_path"]
        epochs = config["train"]["epochs"]
        batch_size = config["train"]["batch_size"]
        device = config["train"]["device"]

        output_dir = config["output"]["dir"]
        exp_name = config["output"]["experiment_name"]

        model = build_model(config)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            device=device,
            project=output_dir,
            name=exp_name,
            seed=config["train"]["seed"],
            save=True,
        )
        print(f"YOLO 학습 완료! 결과 저장 위치: {results.save_dir}")
        return

    if framework == "torchvision":
        import torch
        from torch.amp import autocast, GradScaler
        from torch.utils.data import DataLoader

        exp_name = config["output"]["experiment_name"]
        data_root = config["data"]["raw_dir"] + "/sprint_ai_project1_data"

        train_dataset = PillDataset(data_root, train=True, subset="train")
        val_dataset = PillDataset(data_root, train=True, subset="val")
        print(f"학습 데이터: {len(train_dataset)}장 / 검증 데이터: {len(val_dataset)}장", flush=True)

        expected_classes = len(train_dataset.label_to_raw_id) + 1   # 알약 클래스 수 + 배경
        if config["data"]["num_classes"] != expected_classes:
            raise SystemExit(
                f"config 의 num_classes({config['data']['num_classes']})가 "
                f"클래스 수 + 배경({expected_classes})과 다름. 설정 파일을 고쳐줘."
            )

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

        save_dir = os.path.join(config["output"]["dir"], "checkpoints")
        os.makedirs(save_dir, exist_ok=True)

        def make_checkpoint():
            return {
                "model_state_dict": model.state_dict(),
                "label_to_raw_id": train_dataset.label_to_raw_id,
                "categories": train_dataset.categories,
            }

        best_val_loss = float("inf")
        epochs = config["train"]["epochs"]
        n_iters = len(train_loader)

        for epoch in range(epochs):
            # ---- 학습 ----
            model.train()
            train_loss = 0.0
            t_epoch = time.time()
            for it, (images, targets) in enumerate(train_loader, start=1):
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

                if it % 100 == 0 or it == n_iters:
                    per_it = (time.time() - t_epoch) / it
                    msg = (f"  [epoch {epoch + 1}] {it}/{n_iters} iter | 최근 평균 loss {train_loss / it:.4f} | "
                           f"iter당 {per_it:.2f}초 | 이 epoch 남은 시간 약 {per_it * (n_iters - it) / 60:.1f}분")
                    if epoch == 0 and it == 100:
                        msg += f" | 전체 {epochs} epoch 예상 약 {per_it * n_iters * epochs / 3600:.1f}시간(검증 시간 제외)"
                    print(msg, flush=True)
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

            print(f"[Epoch {epoch+1}/{epochs}] train_loss: {train_loss:.4f} / val_loss: {val_loss:.4f} "
                  f"(epoch 소요 {(time.time() - t_epoch) / 60:.1f}분)", flush=True)

            # 로그 파일에 이번 epoch 결과 한 줄 추가
            with open(log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch + 1, round(train_loss, 4), round(val_loss, 4)])

            # 매 epoch 마지막 상태 저장 (중간에 멈춰도 여기까지는 예측에 쓸 수 있음)
            torch.save(make_checkpoint(), os.path.join(save_dir, f"kimgun_{exp_name}_last.pt"))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(make_checkpoint(), os.path.join(save_dir, f"kimgun_{exp_name}_best.pt"))
                print(f"  -> best 갱신, 저장됨 (val_loss: {best_val_loss:.4f})", flush=True)
        return

    if framework == "rfdetr":
        # 학습 전에 `python src/convert_to_rfdetr.py`로
        # data/processed/rfdetr/{train,valid}를 만들어둬야 한다.
        model = build_model(config)
        model.train(
            dataset_dir=config["data"]["rfdetr_dir"],
            epochs=config["train"]["epochs"],
            batch_size=config["train"]["batch_size"],
            lr=config["train"]["learning_rate"],
            output_dir=f"{config['output']['dir']}/{config['output']['experiment_name']}",
            run_test=False,
            device=config["train"]["device"],
        )
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()