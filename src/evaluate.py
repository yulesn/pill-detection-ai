# src/evaluate.py

"""학습된 모델을 Validation 데이터셋으로 검증하고 성능을 평가하는 스크립트.

사용 예:
    python src/evaluate.py --config configs/default.yaml --checkpoint outputs/exp1_yolo11n4/weights/best.pt
"""
  
import argparse
from ultralytics import YOLO
from utils import load_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best.pt')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print('학습된 모델을 검증 모드로 불러옵니다.')
    model = YOLO(args.checkpoint)

    data_yaml = config['data']['yaml_path']

    print('Validation 데이터셋 평가를 시작합니다.')
    metrics = model.val(data=data_yaml, split='val')    # mAP 등 다양한 객체 탐지 평가지표를 자동으로 계산

    print('\n === Validation 평가 결과 ===')
    print(f'mAP50-95 : {metrics.box.map:.4f}')
    print(f'mAP50    : {metrics.box.map50:.4f}')
    print(f'mAP75    : {metrics.box.map75:.4f}')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall   : {metrics.box.mr:.4f}')

if __name__ == '__main__':
    main()
