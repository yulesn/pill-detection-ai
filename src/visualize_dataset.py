import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import PillDataset

import matplotlib.font_manager as fm
plt.rc('font', family='AppleGothic')
# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def visualize_samples(num_samples=4):
    """PillDataset에서 샘플을 불러와 이미지와 바운딩 박스를 시각화한다."""
    data_dir = "data/processed"
    dataset = PillDataset(data_dir=data_dir, is_train=False)
    
    print(f"총 유효 샘플 수: {len(dataset)}개")
    print(f"클래스 개수: {len(dataset.category_id_to_label)}개")
    
    if len(dataset) == 0:
        print("[ERROR] 시각화할 샘플이 없습니다.")
        return

    num_samples = min(num_samples, len(dataset))
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    
    # [안전 장치] num_samples가 1이거나 axes가 리스트가 아닐 때 발생하는 UnboundLocalError 방지
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(num_samples):
        image_tensor, target = dataset[i]
        
        # Tensor를 numpy 이미지로 변환 (C, H, W -> H, W, C)
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # 정규화(Normalize) 역변환 수행하여 원래 색상 복원
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)

        ax = axes[i]
        ax.imshow(image_np)
        
        boxes = target['boxes']
        labels = target['labels']
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            
            # 바운딩 박스 사각형 그리기 (빨간색)
            rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # 클래스 이름 텍스트 표시
            label_idx = label.item()
            label_name = dataset.label_to_name.get(label_idx, str(label_idx))
            ax.text(x1, max(0, y1 - 5), label_name, color='white', fontsize=8, backgroundcolor='red')

        ax.axis('off')
        ax.set_title(f"Sample {i+1}", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_samples()


# 이미지와 JSON 라벨 파일을 누락없이 찾아내도록 매칭 로직 검증 ---> 데이터가 화면에 올바르게 그려지는지 확인하기 위한 시각화 툴
# 알부멘테이션 및 텐서 변환 역과정 처리 ---> 텐서로 변환되고 정규화된 이미지를 다시 시각화용 numpy 배열로 되돌려 화면에 정상 출력함
# 바운딩 박스 및 클래스 이름 매핑 출력 ---> 데이터셋이 부여한 순차적 레이블과 실제 박스 좌표가 이미지 위에 정확한 위치에 그려지는지 직관적으로 검증함