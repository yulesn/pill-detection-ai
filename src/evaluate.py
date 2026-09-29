"""학습된 RetinaNet 모델 평가 스크립트.

평가 항목:
- mAP@50
- mAP@50:95
- 평균 추론 시간
- FPS
- 전체 파라미터 수
- 학습 가능한 파라미터 수

주의:
현재 프로젝트에는 별도의 validation/test split이 없기 때문에
data/processed 전체 이미지를 평가에 사용한다.
"""

import argparse
import os
import time

import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

from dataset import PillDataset
from model import build_model
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=512
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.01,
        help="mAP 평가에 사용할 최소 confidence"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="추론 시간 측정 전에 수행할 warm-up 횟수"
    )

    return parser.parse_args()


def collate_fn(batch):
    """객체 탐지 모델용 batch 묶기."""
    return tuple(zip(*batch))


def get_device(config):
    """사용할 디바이스를 결정한다."""

    if (
        torch.backends.mps.is_available()
        and config["train"]["device"] != "cpu"
    ):
        return torch.device("mps")

    return torch.device(
        config["train"]["device"]
    )


def synchronize_device(device):
    """MPS/CUDA에서 정확한 시간 측정을 위해 동기화한다."""

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model):
    """모델 파라미터 수를 계산한다."""

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total_params, trainable_params


def load_image_tensor(image_path, img_size, device):
    """이미지를 학습과 동일하게 512x512로 resize하여 Tensor로 변환한다.

    RetinaNet 내부 transform에서 ImageNet mean/std 정규화를 수행하므로
    여기서는 별도의 Normalize를 적용하지 않는다.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    original_width, original_height = image.size

    resized_image = image.resize(
        (img_size, img_size),
        Image.Resampling.BILINEAR
    )

    image_tensor = F.to_tensor(
        resized_image
    )

    image_tensor = image_tensor.to(
        device
    )

    return (
        image_tensor,
        original_width,
        original_height
    )


def build_ground_truth(dataset, sample):
    """Dataset sample의 GT를 RetinaNet 평가 형식으로 변환한다."""

    boxes = torch.tensor(
        sample["boxes"],
        dtype=torch.float32
    )

    labels = torch.tensor(
        sample["labels"],
        dtype=torch.int64
    )

    return {
        "boxes": boxes,
        "labels": labels
    }


def main():

    args = parse_args()

    # ---------------------------------------------------------
    # 1. 설정 불러오기
    # ---------------------------------------------------------

    config = load_config(
        args.config
    )

    device = get_device(
        config
    )

    print()
    print("=" * 70)
    print("RetinaNet 모델 평가")
    print("=" * 70)

    print(
        f"사용 중인 디바이스: {device}"
    )

    # ---------------------------------------------------------
    # 2. Dataset 생성
    # ---------------------------------------------------------

    dataset = PillDataset(
        data_dir=config["data"]["processed_dir"],
        img_size=args.img_size,
        is_train=False
    )

    print()
    print("평가 Dataset")
    print(
        f"  Images       : {len(dataset)}"
    )
    print(
        f"  Drug classes : "
        f"{len(dataset.category_id_to_label)}"
    )

    # ---------------------------------------------------------
    # 3. 모델 생성
    # ---------------------------------------------------------

    model = build_model(
        config
    )

    # ---------------------------------------------------------
    # 4. 체크포인트 로드
    # ---------------------------------------------------------

    print()
    print(
        f"체크포인트 로드: "
        f"{args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(
        device
    )

    model.eval()

    # ---------------------------------------------------------
    # 5. mAP 계산을 위해 confidence threshold 낮추기
    # ---------------------------------------------------------

    # 기본 model.py에서는 score_thresh가 0.5이기 때문에
    # 낮은 confidence prediction을 평가 단계에서 확인할 수 있도록
    # threshold를 낮춘다.

    model.score_thresh = args.score_threshold

    # RetinaNet 내부 NMS threshold
    model.nms_thresh = 0.5

    # ---------------------------------------------------------
    # 6. 파라미터 수 계산
    # ---------------------------------------------------------

    total_params, trainable_params = (
        count_parameters(model)
    )

    print()
    print("[모델 파라미터]")

    print(
        f"  전체 파라미터 : "
        f"{total_params:,}"
    )

    print(
        f"  학습 가능 파라미터 : "
        f"{trainable_params:,}"
    )

    print(
        f"  전체 파라미터(M) : "
        f"{total_params / 1_000_000:.2f} M"
    )

    print(
        f"  모델 파일 크기 : "
        f"{os.path.getsize(args.checkpoint) / (1024 ** 2):.2f} MB"
    )

    # ---------------------------------------------------------
    # 7. TorchMetrics import
    # ---------------------------------------------------------

    try:
        from torchmetrics.detection.mean_ap import (
            MeanAveragePrecision
        )

    except ImportError:

        print()
        print(
            "torchmetrics가 설치되어 있지 않습니다."
        )
        print()
        print(
            "다음 명령어를 먼저 실행해주세요:"
        )
        print()
        print(
            "pip install torchmetrics"
        )

        return

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )

    # ---------------------------------------------------------
    # 8. Warm-up
    # ---------------------------------------------------------

    print()
    print(
        f"Warm-up 시작 "
        f"({args.warmup}회)"
    )

    if len(dataset) > 0:

        warmup_image_path = (
            dataset.samples[0]["img_path"]
        )

        warmup_tensor, _, _ = (
            load_image_tensor(
                warmup_image_path,
                args.img_size,
                device
            )
        )

        with torch.no_grad():

            for _ in range(
                args.warmup
            ):

                model(
                    [warmup_tensor]
                )

                synchronize_device(
                    device
                )

    print("Warm-up 완료")

    # ---------------------------------------------------------
    # 9. 전체 데이터 평가
    # ---------------------------------------------------------

    total_inference_time = 0.0

    total_gt_objects = 0
    total_pred_objects = 0

    inference_times = []

    print()
    print("=" * 70)
    print("전체 이미지 평가 시작")
    print("=" * 70)

    with torch.no_grad():

        for idx, sample in enumerate(
            dataset.samples
        ):

            image_path = sample["img_path"]

            # ---------------------------------------------
            # 이미지 로드
            # ---------------------------------------------

            image_tensor, _, _ = (
                load_image_tensor(
                    image_path,
                    args.img_size,
                    device
                )
            )

            # ---------------------------------------------
            # 추론 시간 측정
            # ---------------------------------------------

            synchronize_device(
                device
            )

            start_time = time.perf_counter()

            prediction = model(
                [image_tensor]
            )[0]

            synchronize_device(
                device
            )

            end_time = time.perf_counter()

            inference_time = (
                end_time - start_time
            )

            inference_times.append(
                inference_time
            )

            total_inference_time += (
                inference_time
            )

            # ---------------------------------------------
            # Prediction
            # ---------------------------------------------

            pred_boxes = (
                prediction["boxes"]
                .detach()
                .cpu()
            )

            pred_scores = (
                prediction["scores"]
                .detach()
                .cpu()
            )

            pred_labels = (
                prediction["labels"]
                .detach()
                .cpu()
            )

            # ---------------------------------------------
            # Ground Truth
            # ---------------------------------------------

            target = build_ground_truth(
                dataset,
                sample
            )

            # ---------------------------------------------
            # mAP에 사용할 prediction
            # ---------------------------------------------

            prediction_for_metric = {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels
            }

            metric.update(
                [prediction_for_metric],
                [target]
            )

            total_gt_objects += len(
                target["boxes"]
            )

            total_pred_objects += len(
                pred_boxes
            )

            # ---------------------------------------------
            # 진행 상황
            # ---------------------------------------------

            if (
                (idx + 1) % 20 == 0
                or idx == 0
                or idx + 1 == len(dataset)
            ):

                print(
                    f"[{idx + 1:3d}/"
                    f"{len(dataset):3d}] "
                    f"{os.path.basename(image_path)} | "
                    f"추론시간: "
                    f"{inference_time * 1000:.2f} ms | "
                    f"Prediction: "
                    f"{len(pred_boxes)}개"
                )

    # ---------------------------------------------------------
    # 10. mAP 계산
    # ---------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("mAP 계산 중...")
    print(
        "=" * 70
    )

    results = metric.compute()

    map_50 = float(
        results["map_50"]
    )

    map_50_95 = float(
        results["map"]
    )

    # ---------------------------------------------------------
    # 11. 추론 속도 계산
    # ---------------------------------------------------------

    num_images = len(
        inference_times
    )

    average_inference_time = (
        total_inference_time
        / num_images
    )

    fps = (
        1.0
        / average_inference_time
    )

    # ---------------------------------------------------------
    # 12. 최종 결과 출력
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("최종 평가 결과")
    print("=" * 70)

    print()
    print("[Detection Performance]")

    print(
        f"mAP@50       : "
        f"{map_50:.4f}"
    )

    print(
        f"mAP@50:95    : "
        f"{map_50_95:.4f}"
    )

    print()
    print("[Inference Speed]")

    print(
        f"평균 추론 시간 : "
        f"{average_inference_time * 1000:.2f} ms/image"
    )

    print(
        f"FPS            : "
        f"{fps:.2f}"
    )

    print()
    print("[Dataset]")

    print(
        f"평가 이미지 수 : "
        f"{num_images}"
    )

    print(
        f"GT 객체 수     : "
        f"{total_gt_objects}"
    )

    print(
        f"예측 객체 수   : "
        f"{total_pred_objects}"
    )

    print()
    print("[Model Size]")

    print(
        f"Parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Parameters(M)  : "
        f"{total_params / 1_000_000:.2f} M"
    )

    print(
        f"Checkpoint     : "
        f"{os.path.getsize(args.checkpoint) / (1024 ** 2):.2f} MB"
    )

    print()
    print("=" * 70)
    print("평가 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()