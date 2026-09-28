"""학습된 모델로 추론하는 스크립트.

사용 예:
python src/predict.py --config configs/default.yaml --checkpoint outputs/default/checkpoints/best.pt --test_dir data/raw/sprint_ai_project1_data/test_images --output_csv outputs/predictions/submission.csv

torchvision(Faster R-CNN) 예:
python src/predict.py --config configs/fasterrcnn_t5.yaml --checkpoint outputs/checkpoints/kimgun_fasterrcnn_t5_best.pt --test_dir data/raw/sprint_ai_project1_data/test_images --output_csv outputs/predictions/fasterrcnn_t5_submission.csv
"""
import os
import argparse
from pathlib import Path

import pandas as pd

from utils import load_config


def box_iou(a, b):
    """두 박스 (x1, y1, x2, y2)의 IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(dets, iou_thr):
    """같은 클래스끼리 IoU가 iou_thr 보다 크게 겹치면 점수 높은 것만 남긴다 (YOLO 의 iou= 옵션과 같은 방식).
    dets: [(score, label, (x1, y1, x2, y2)), ...]  ->  점수 내림차순 목록"""
    kept = []
    for det in sorted(dets, key=lambda d: d[0], reverse=True):
        if all(k[1] != det[1] or box_iou(k[2], det[2]) <= iou_thr for k in kept):
            kept.append(det)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/predictions/submission.csv")
    # 아래 3개는 torchvision 분기에서만 쓴다 (YOLO 분기는 아래 코드의 기존 값 그대로)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.1)
    parser.add_argument("--max_det", type=int, default=4)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    framework = config["model"]["framework"]
    
    if framework == "yolo":
        from ultralytics import YOLO

        data_file_path = config['data']['yaml_path']
        data_file = load_config(data_file_path)
        model = YOLO(args.checkpoint)
        results = model.predict(source=args.test_dir, conf=0.25, save=False, iou=0.1, max_det=4)
        label_map = data_file['names']  # 모델 예측 결과를 데이터 제출 형식에 맞춤
        results_list = []
        annotation_counter = 1

        for result in results:
            # image_id 추출
            file_name = os.path.basename(result.path) 
            image_id = os.path.splitext(file_name)[0]
            
            try:
                image_id = int(image_id)
            except ValueError:
                pass

            # BBox 파싱
            for box in result.boxes:
                category_idx = int(box.cls[0].item())    # tensor 객체를 가져와 int로 변환
                category_id = int(label_map[category_idx].split('-')[1])
                score = round(float(box.conf[0].item()), 3)

                # xyxy -> xywh 변환
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                bbox_x = round(x_min, 1)
                bbox_y = round(y_min, 1)
                bbox_w = round((x_max - x_min), 1)
                bbox_h = round((y_max - y_min), 1)

                results_list.append({
                    'annotation_id': annotation_counter,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox_x': bbox_x,
                    'bbox_y': bbox_y,
                    'bbox_w': bbox_w,
                    'bbox_h': bbox_h,
                    'score': score
                })
                annotation_counter += 1
                
        # DataFrame 변환 및 csv로 저장
        df_sub = pd.DataFrame(results_list)

        # 컬럼 순서 명시적 지정
        columns_order = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
        df_sub = df_sub[columns_order]

        df_sub.to_csv(args.output_csv, index=False)
        print(f'[{args.output_csv}]파일 생성 완료')
        print(f'총 {len(df_sub)}개의 Bounding Box 감지 결과가 작성되었습니다.')
        return

    if framework == "torchvision":
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        from model import build_model

        device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
        model = build_model(config)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)  # 우리가 학습 때 저장한 파일
        model.load_state_dict(checkpoint["model_state_dict"])
        label_to_raw_id = checkpoint["label_to_raw_id"]   # 모델 라벨(1~N) -> 대회 category_id
        model.to(device)
        model.eval()

        image_files = sorted(Path(args.test_dir).rglob("*.png"))
        results_list = []
        annotation_counter = 1

        with torch.no_grad():
            for i, image_file in enumerate(image_files, start=1):
                image = Image.open(image_file).convert("RGB")
                output = model([TF.to_tensor(image).to(device)])[0]
                boxes = output["boxes"].cpu().tolist()
                scores = output["scores"].cpu().tolist()
                labels = output["labels"].cpu().tolist()

                # 점수 기준 -> 같은 클래스끼리 겹침 제거 -> 점수 높은 순으로 max_det 개
                dets = [(s, int(l), b) for b, s, l in zip(boxes, scores, labels) if s >= args.conf]
                dets = nms_per_class(dets, args.iou)[:args.max_det]

                image_id = os.path.splitext(image_file.name)[0]
                try:
                    image_id = int(image_id)
                except ValueError:
                    pass

                for score, label, (x_min, y_min, x_max, y_max) in dets:
                    results_list.append({
                        'annotation_id': annotation_counter,
                        'image_id': image_id,
                        'category_id': int(label_to_raw_id[label]),
                        'bbox_x': round(x_min, 1),
                        'bbox_y': round(y_min, 1),
                        'bbox_w': round(x_max - x_min, 1),
                        'bbox_h': round(y_max - y_min, 1),
                        'score': round(float(score), 3),
                    })
                    annotation_counter += 1

                if i % 100 == 0 or i == len(image_files):
                    print(f"  {i}/{len(image_files)}장 예측 완료", flush=True)

        columns_order = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
        df_sub = pd.DataFrame(results_list, columns=columns_order)
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df_sub.to_csv(args.output_csv, index=False)
        print(f'[{args.output_csv}]파일 생성 완료')
        print(f'총 {len(df_sub)}개의 Bounding Box 감지 결과가 작성되었습니다.')
        return

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()