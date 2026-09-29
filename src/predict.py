"""학습된 RetinaNet 모델로 추론 + 시각화하는 스크립트.

단일 이미지 및 폴더 일괄 처리를 지원한다.

현재 데이터 구조에서는 PillDataset이
dl_mapping_code -> 모델 label(1~56) 매핑을 만들기 때문에
별도의 pill_cat_map.json을 사용하지 않는다.

학습 시 이미지가 512x512로 리사이즈되고
RetinaNet 내부에서 ImageNet mean/std로 정규화되므로
추론에서도 별도의 Normalize를 적용하지 않는다.

GT(정답, 초록)와 예측(빨강) 박스를 함께 그리고,
confidence와 검출 개수를 출력한다.

score_threshold와 iou_threshold(NMS)는 실행 시 인자로 조정 가능하다.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms import functional as F

from dataset import PillDataset
from model import build_model
from utils import load_config


def get_font(size=24):
    """한글을 지원하는 시스템 폰트를 찾아서 반환한다."""

    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def load_gt_boxes(processed_dir, image_path, dataset):
    """현재 AI Hub 데이터 구조에서 GT 박스와 모델 label을 가져온다.

    하나의 실제 이미지에 여러 JSON이 존재할 수 있기 때문에
    해당 file_name을 가진 모든 JSON을 찾아 annotation을 합친다.

    각 JSON의 images[0]["dl_mapping_code"]가
    해당 annotation의 실제 약품 class이다.
    """

    boxes = []
    labels = []

    image_name = Path(image_path).name

    # train_images_ 접두사를 제거해서 JSON의 file_name과 비교한다.
    if image_name.startswith("train_images_"):
        target_file_name = image_name[len("train_images_"):]
    else:
        target_file_name = image_name

    processed_dir = Path(processed_dir)

    # 현재 processed 데이터의 모든 JSON 탐색
    json_files = sorted(processed_dir.rglob("*.json"))

    for json_path in json_files:

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        images = data.get("images", [])

        if not images:
            continue

        image_info = images[0]

        json_file_name = image_info.get("file_name")

        if json_file_name != target_file_name:
            continue

        # 실제 약품 코드
        drug_code = image_info.get("dl_mapping_code")

        if drug_code is None:
            continue

        # 현재 dataset.py와 동일한 label mapping 사용
        label = dataset.category_id_to_label.get(drug_code)

        if label is None:
            continue

        annotations = data.get("annotations", [])

        for ann in annotations:

            bbox = ann.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox

            # AI Hub bbox: [x, y, width, height]
            if w <= 0 or h <= 0:
                continue

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            # 원본 annotation 좌표 오류 보정
            if (
                drug_code == "K-018357"
                and bbox == [6567, 625, 311, 315]
            ):
                x1 = 657
                y1 = 625
                x2 = 657 + 311
                y2 = 625 + 315

            boxes.append(
                [x1, y1, x2, y2]
            )

            labels.append(label)

    return boxes, labels


def draw_label(draw, xy, text, color, font, max_width=512):
    """글씨 뒤에 배경 박스를 깔아서 잘 보이게 한다."""

    x, y = xy

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if x + text_w > max_width - 10:
        x = max(
            5,
            max_width - text_w - 15
        )

    if y < 5:
        y = xy[1] + 35

    pad = 3

    draw.rectangle(
        [
            x - pad,
            y - pad,
            x + text_w + pad,
            y + text_h + pad
        ],
        fill=color
    )

    draw.text(
        (x, y),
        text,
        fill="white",
        font=font
    )


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
        "--image",
        type=str,
        required=True,
        help="단일 이미지 파일 경로 또는 이미지 폴더 경로"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="폴더 입력 시 처리할 이미지 개수"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="학습 시 사용한 리사이즈 크기"
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.2,
        help="이 값 이상인 예측만 표시"
    )

    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.4,
        help="NMS에서 겹침 판정 기준 IoU"
    )

    return parser.parse_args()


def process_single_image(
    image_path,
    model,
    config,
    dataset,
    args,
    device
):
    """이미지 하나를 추론하고 결과 이미지를 저장한다."""

    image_path = Path(image_path)

    print()
    print("=" * 60)
    print(f"이미지 추론: {image_path.name}")
    print("=" * 60)

    # 원본 이미지 읽기
    image = Image.open(
        image_path
    ).convert("RGB")

    orig_w, orig_h = image.size

    print(
        f"원본 이미지 크기: "
        f"{orig_w} x {orig_h}"
    )

    # 학습과 동일하게 512x512 resize
    resized_image = image.resize(
        (args.img_size, args.img_size),
        Image.Resampling.BILINEAR
    )

    # PIL -> Tensor
    img_tensor = F.to_tensor(
        resized_image
    )

    # RetinaNet 입력용 Tensor 변환
    # RetinaNet 내부에서 ImageNet mean/std 정규화를 적용하기 때문에 추론 단계에서는 별도로 Normalize하지 않음.
    img_tensor = img_tensor.to(device)

    # RetinaNet 추론
    with torch.no_grad():
        prediction = model(
            [img_tensor]
        )[0]

    pred_boxes = prediction["boxes"].detach().cpu()
    pred_scores = prediction["scores"].detach().cpu()
    pred_labels = prediction["labels"].detach().cpu()

    print(
        f"모델 원본 출력: "
        f"{len(pred_boxes)}개"
    )

    # NMS
    if len(pred_boxes) > 0:

        keep_idx = nms(
            pred_boxes,
            pred_scores,
            iou_threshold=args.iou_threshold
        )

        pred_boxes = pred_boxes[keep_idx]
        pred_scores = pred_scores[keep_idx]
        pred_labels = pred_labels[keep_idx]

    # 512x512 bbox를 원본 이미지 크기로 복원
    scale_x = orig_w / args.img_size
    scale_y = orig_h / args.img_size

    valid_boxes = []
    valid_scores = []
    valid_labels = []

    if len(pred_boxes) > 0:

        pred_boxes = pred_boxes.clone()

        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y

        for box, score, label in zip(
            pred_boxes,
            pred_scores,
            pred_labels
        ):

            score_value = float(
                score.item()
            )

            # confidence threshold
            if score_value < args.score_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()

            width = x2 - x1
            height = y2 - y1

            # 너무 작은 박스 제거
            if width < 10 or height < 10:
                continue

            # 이미지 전체를 거의 덮는 이상한 박스 제거
            if (
                width > orig_w * 0.95
                and height > orig_h * 0.95
            ):
                continue

            # 이미지 범위 안으로 보정
            x1 = max(
                0,
                min(x1, orig_w - 1)
            )

            y1 = max(
                0,
                min(y1, orig_h - 1)
            )

            x2 = max(
                0,
                min(x2, orig_w - 1)
            )

            y2 = max(
                0,
                min(y2, orig_h - 1)
            )

            valid_boxes.append(
                [x1, y1, x2, y2]
            )

            valid_scores.append(
                score
            )

            valid_labels.append(
                label
            )

    # GT 가져오기
    gt_boxes, gt_labels = load_gt_boxes(
        config["data"]["processed_dir"],
        image_path,
        dataset
    )

    print(
        f"GT 객체 수: {len(gt_boxes)}"
    )

    # 결과 이미지 생성
    draw_image = image.copy()

    draw = ImageDraw.Draw(
        draw_image
    )

    font = get_font(24)

    # GT - 초록색 박스
    print()
    print("[GT]")

    for box, label in zip(
        gt_boxes,
        gt_labels
    ):

        x1, y1, x2, y2 = box

        drug_name = dataset.label_to_name.get(
            label,
            f"Unknown_{label}"
        )

        print(
            f"  {drug_name} "
            f"(label={label}, "
            f"bbox=["
            f"{int(x1)}, "
            f"{int(y1)}, "
            f"{int(x2)}, "
            f"{int(y2)}])"
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="lime",
            width=4
        )

        label_y = (
            y1 - 32
            if y1 - 32 > 5
            else y1 + 5
        )

        draw_label(
            draw,
            (x1, label_y),
            f"GT: {drug_name}",
            "green",
            font,
            orig_w
        )

    # Prediction - 빨간색 박스
    print()
    print("[Prediction]")

    for box, score, label in zip(
        valid_boxes,
        valid_scores,
        valid_labels
    ):

        x1, y1, x2, y2 = box

        model_label = int(
            label.item()
        )

        drug_name = dataset.label_to_name.get(
            model_label,
            f"Unknown_label_{model_label}"
        )

        score_value = float(
            score.item()
        )

        print(
            f"  {drug_name} "
            f"(label={model_label}, "
            f"confidence={score_value:.3f}, "
            f"bbox=["
            f"{int(x1)}, "
            f"{int(y1)}, "
            f"{int(x2)}, "
            f"{int(y2)}])"
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=4
        )

        draw_label(
            draw,
            (
                x1,
                y2 + 5
            ),
            (
                f"Pred: "
                f"{drug_name} "
                f"({score_value:.2f})"
            ),
            "red",
            font,
            orig_w
        )

    # 결과 저장
    out_dir = (
        Path(config["output"]["dir"])
        / config["output"]["experiment_name"]
        / "predictions"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_path = (
        out_dir
        / f"{image_path.stem}_result.jpg"
    )

    draw_image.save(
        result_path
    )

    # 최종 결과 출력
    print()

    print(
        f"총 검출 객체 수: "
        f"{len(valid_boxes)}"
    )

    print(
        f"결과 저장 완료 -> "
        f"{result_path}"
    )


def main() -> None:

    args = parse_args()

    config = load_config(
        args.config
    )

    # 디바이스 설정
    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print(
        f"사용 중인 디바이스: "
        f"{device}"
    )

    # 현재 Dataset 생성
    dataset = PillDataset(
        data_dir=config["data"]["processed_dir"],
        img_size=args.img_size,
        is_train=False
    )

    print()

    print("현재 Dataset 정보")

    print(
        f"  Images       : "
        f"{len(dataset)}"
    )

    print(
        f"  Drug classes : "
        f"{len(dataset.label_to_name)}"
    )

    # 모델 생성
    # 현재 YAML의 num_classes = 56 사용
    model = build_model(
        config
    )


    # 학습된 checkpoint 로드
    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: "
            f"{checkpoint_path}"
        )

    print()

    print(
        f"체크포인트 로드: "
        f"{checkpoint_path}"
    )

    try:

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

    except TypeError:

        state_dict = torch.load(
            checkpoint_path,
            map_location=device
        )

    model.load_state_dict(
        state_dict
    )

    # 추론 threshold 설정
    # 학습 때의 0.5보다 낮은 threshold로 설정해서 epoch 모델의 초기 예측도 확인할 수 있게 함.
    model.score_thresh = 0.01

    model.nms_thresh = args.iou_threshold

    model.to(device)
    model.eval()

    print(
        "모델 로드 완료"
    )

    # 입력 이미지 확인
    input_path = Path(
        args.image
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"이미지 경로를 찾을 수 없습니다: "
            f"{input_path}"
        )

    if input_path.is_dir():

        valid_exts = {
            ".png",
            ".jpg",
            ".jpeg"
        }

        image_files = sorted(
            [
                f
                for f in input_path.iterdir()
                if f.suffix.lower()
                in valid_exts
            ]
        )[:args.num_images]

        print()

        print(
            f"총 {len(image_files)}개 "
            f"이미지 일괄 추론 시작..."
        )

    else:

        image_files = [
            input_path
        ]

    # 이미지별 추론
    for img_file in image_files:

        process_single_image(
            img_file,
            model,
            config,
            dataset,
            args,
            device
        )

    print()

    print(
        "모든 추론 작업이 완료되었습니다!"
    )


if __name__ == "__main__":
    main()