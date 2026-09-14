"""여러 파일에서 공통으로 쓰는 유틸 함수 모음."""

import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 한글 폰트 설정 (맥 기본 내장 폰트)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def visualize_samples(dataset, num_samples=4):
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(6 * num_samples, 6))

    for ax, idx in zip(axes, indices):
        image, target = dataset[idx]
        ax.imshow(image)
        img_width, img_height = image.size

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            full_name = dataset.label_to_name[label.item()]
            short_name = full_name.split('(')[0].strip()

            if x1 > img_width / 2:
                text_x, ha = x2, 'right'
            else:
                text_x, ha = x1, 'left'

            offset = 25
            if y1 - offset > 0:
                text_y, va = y1 - offset, 'bottom'
            else:
                text_y, va = y2 + offset, 'top'

            ax.text(
                text_x, text_y, short_name,
                color='red', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                clip_on=True, ha=ha, va=va,
            )

        ax.axis('off')

    plt.subplots_adjust(wspace=0.3)
    plt.show()


# 어노테이션 무결성 검증
def check_annotation_integrity(dataset):
    errors = []
    total_anns = 0

    for idx in range(len(dataset)):
        img_path = dataset.img_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = dataset.ann_map.get(base_name)

        if json_path is None:   
            errors.append(f'[{base_name}] JSON 파일 누락 !')
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_info = data.get('images', [{}])[0]
        img_w = img_info.get('width', 0)
        img_h = img_info.get('height', 0)

        for i, ann in enumerate(data.get('annotations', [])):
            total_anns += 1
            x,y,w,h = ann['bbox']

                # 크기 오류 검사 (0 이하)
            if w <= 0 or h <= 0:
                errors.append(f'[{base_name}] 어노테이션 #{i}: 크기가 0 이하 (w={w}, h={h})')

                # 이미지 범위를 벗어나는지 검사
            if x < 0 or y < 0 or (x+w) > img_w + 2 or (y+h) > img_h + 2:
                errors.append(f"[{base_name}] 어노테이션 #{i}: 이미지 범위 이탈 (bbox: {x,y,w,h}, img_size: {img_w}x{img_h})")

    print('==================== 어노테이션 무결성 검증 결과 ====================')
    print(f'총 검사한 어노테이션 수: {total_anns}개')

    if errors:
        print(f'발견된 에러 / 경고: {len(errors)}건')
        for err in errors[:10]: # 에러 상위 10개만 출력
            print(f' - {err}')
        if len(errors) > 10:
            print(f' ,,, 외 {len(errors) -10}건 생략함')

    else:
        print('발견된 어노테이션 에러 없음 !')
    return errors