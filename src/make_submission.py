"""학습된 모델로 추론 + 시각화하는 스크립트 (단일 파일 및 폴더 일괄 처리 지원).

학습 시 이미지를 512x512로 리사이즈했고, 카테고리 라벨도 원본 category_id를
1부터 순차 재배정한 매핑(pill_cat_map.json)을 썼으므로, 추론 시 동일하게
리사이즈 + 역매핑을 적용해야 한다.

GT(정답, 초록)와 예측(빨강) 박스를 함께 그리고, confidence/개수를 출력한다.
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

from model import build_model
from utils import load_config


def get_font(size=24):
    """한글을 지원하는 시스템 폰트를 찾아서 반환한다."""
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS 대체
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Colab/Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # 최후 대체(한글 미지원)
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def build_category_id_to_name(ann_dir: str) -> dict:
    """annotations 폴더(하위 폴더 포함)의 모든 JSON에서 category_id -> 한글 이름 매핑을 만든다."""
    id_to_name = {}
    if not os.path.exists(ann_dir):
        return id_to_name
    for root, _, files in os.walk(ann_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            with open(os.path.join(root, f), "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for cat in data.get("categories", []):
                    id_to_name[cat["id"]] = cat["name"]
    return id_to_name


def load_gt_boxes(ann_dir, image_id):
    """하위 폴더 전체에서 image_id와 파일명이 일치하는 JSON을 모두 찾아 GT를 합친다."""
    boxes, labels = [], []
    target_filename = f"{image_id}.json"

    if not os.path.exists(ann_dir):
        return boxes, labels

    for root, _, files in os.walk(ann_dir):
        if target_filename not in files:
            continue
        with open(os.path.join(root, target_filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            anns = data if isinstance(data, list) else data.get("annotations", [data])

            for ann in anns:
                bbox = ann.get("bbox", ann.get("bounding_box", None))
                if bbox and len(bbox) == 4:
                    x, y, w, h = bbox
                    if w > 0 and h > 0:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(int(ann.get("category_id", ann.get("label", 1))))
    return boxes, labels


def draw_label(draw, xy, text, color, font, max_width=512):
    """글씨 뒤에 배경 박스를 깔아서 잘 보이게 하고, 이미지 밖으로 튀어나가지 않도록 보정"""
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    if x + text_w > max_width - 10:
        x = max(5, max_width - text_w - 15)
    
    if y < 5:
        y = xy[1] + 35
    
    pad = 3
    draw.rectangle([x - pad, y - pad, x + text_w + pad, y + text_h + pad], fill=color)
    draw.text((x, y), text, fill="white", font=font)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cat_map", type=str, default="pill_cat_map.json",
                        help="학습 시 저장된 category_id -> 모델라벨 매핑 파일")
    parser.add_argument("--image", type=str, required=True, help="단일 이미지 파일 경로 또는 이미지 폴더 경로")
    parser.add_argument("--num_images", type=int, default=10, help="폴더 입력 시 처리할 이미지 개수")
    parser.add_argument("--img_size", type=int, default=512,
                        help="학습 시 사용한 리사이즈 크기 (학습 코드와 동일하게 512)")
    parser.add_argument("--score_threshold", type=float, default=0.2,
                        help="이 값 이상인 예측만 표시 (confidence threshold)")
    parser.add_argument("--iou_threshold", type=float, default=0.4,
                        help="NMS에서 겹침 판정 기준 IoU")
    return parser.parse_args()


def process_single_image(image_path, model, config, id_to_name, label_to_catid, train_ann_dir, args, device):
    image_path = Path(image_path)
    image_id = image_path.stem

    data_dir = Path(config["data"]["raw_dir"]) / "sprint_ai_project1_data"
    if "test" in str(image_path):
        ann_dir = data_dir / "test_annotations"
    else:
        ann_dir = train_ann_dir

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    resized_image = image.resize((args.img_size, args.img_size), Image.Resampling.BILINEAR)
    img_tensor = F.to_tensor(resized_image).to(device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    pred_boxes = prediction["boxes"].cpu()
    pred_scores = prediction["scores"].cpu()
    pred_labels = prediction["labels"].cpu()

    if len(pred_boxes) > 0:
        keep_idx = nms(pred_boxes, pred_scores, iou_threshold=args.iou_threshold)
        pred_boxes = pred_boxes[keep_idx]
        pred_scores = pred_scores[keep_idx]
        pred_labels = pred_labels[keep_idx]

    scale_x = orig_w / args.img_size
    scale_y = orig_h / args.img_size
    valid_boxes, valid_scores, valid_labels = [], [], []
    
    if len(pred_boxes) > 0:
        pred_boxes = pred_boxes.clone()
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y

        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            if w < 10 or h < 10 or (w > orig_w * 0.95 and h > orig_h * 0.95):
                continue
            if score.item() < args.score_threshold:
                continue
            valid_boxes.append(box)
            valid_scores.append(score)
            valid_labels.append(label)

    gt_boxes, gt_labels = load_gt_boxes(str(ann_dir), image_id)

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    font = get_font(24)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        name = id_to_name.get(label) or id_to_name.get(str(label), f"Unknown_{label}")
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        label_y = y1 - 32 if y1 - 32 > 5 else y1 + 5
        draw_label(draw, (x1, label_y), f"GT: {name}", "green", font, orig_w)

    for box, score, label in zip(valid_boxes, valid_scores, valid_labels):
        x1, y1, x2, y2 = box.tolist()
        model_label = int(label.item())
        orig_cat_id = label_to_catid.get(model_label)
        name = id_to_name.get(orig_cat_id) or id_to_name.get(str(orig_cat_id), f"Unknown_model_label_{model_label}")
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw_label(draw, (x1, y2 + 5), f"Pred: {name} ({score.item():.2f})", "red", font, orig_w)

    sub_folder = "test_predictions" if "test" in str(image_path) else "val_predictions"
    out_dir = Path(config["output"]["dir"]) / config["output"]["experiment_name"] / sub_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{image_id}_result.jpg"
    draw_image.save(result_path)
    print(f"[{image_path.name}] 탐지: {len(valid_boxes)}개 | 저장 완료 -> {result_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"사용 중인 디바이스: {device}")

    if not os.path.exists(args.cat_map):
        raise FileNotFoundError(f"카테고리 매핑 파일을 찾을 수 없습니다: {args.cat_map}")

    with open(args.cat_map, "r", encoding="utf-8") as f:
        cat_map = json.load(f)
    label_to_catid = {int(v): int(k) for k, v in cat_map.items()}
    config["data"]["num_classes"] = len(cat_map) + 1

    model = build_model(config)
    try:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.score_thresh = 0.01
    model.nms_thresh = args.iou_threshold
    model.to(device)
    model.eval()

    data_dir = Path(config["data"]["raw_dir"]) / "sprint_ai_project1_data"
    train_ann_dir = data_dir / "train_annotations"
    id_to_name = build_category_id_to_name(str(train_ann_dir))

    input_path = Path(args.image)
    if input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg"}
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in valid_exts])[:args.num_images]
        print(f"\n총 {len(image_files)}개 이미지 일괄 추론 시작...")
    else:
        image_files = [input_path]

    for img_file in image_files:
        process_single_image(img_file, model, config, id_to_name, label_to_catid, train_ann_dir, args, device)

    print("\n모든 추론 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()