"""학습된 모델로 추론하는 스크립트.

사용 예:
    python src/predict.py --config configs/default.yaml --checkpoint outputs/default/checkpoints/best.pt --image path/to/image.jpg
"""
import os
import argparse
import pandas as pd
from ultralytics import YOLO

from model import build_model
from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="outputs/predictions/submission.csv")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    framework = config["model"]["framework"]

    if framework == "yolo":
        model = YOLO(args.checkpoint)
        results = model.predict(source=args.test_dir, conf=0.25, save=False)

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
                category_id = int(box.cls[0].item()) + 1    # tensor 객체를 가져와 int로 변환
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
        # TODO(torchvision 트랙): args.checkpoint 로드, 이미지 전처리,
        # 커스텀 추론 후 (클래스, bbox) 리스트로 출력
        raise NotImplementedError("torchvision 추론 로직을 구현해주세요.")

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


if __name__ == "__main__":
    main()
