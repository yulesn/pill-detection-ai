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
        import torch
        from torch.utils.data import DataLoader
        from dataset import PillDataset

        def collate_fn(batch):
            return tuple(zip(*batch))

        dataset = PillDataset(config["data"]["raw_dir"] + "/sprint_ai_project1_data", train=True)
        loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        device = torch.device(config["train"]["device"])
        model = build_model(config)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config["train"]["learning_rate"], momentum=0.9)

        model.train()
        for epoch in range(config["train"]["epochs"]):
            epoch_loss = 0.0
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"[Epoch {epoch+1}/{config['train']['epochs']}] loss: {epoch_loss / len(loader):.4f}")
        return
    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
