import matplotlib.pyplot as plt 

import json
import glob
import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import os
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
    


def load_data(path):
    train_image_path = os.path.join(path, 'train_images')
    train_ann_path = os.path.join(path, 'train_annotations')
    test_image_path = os.path.join(path, 'test_images')

    train_images = [obj for obj in os.listdir(train_image_path) if obj.endswith(('png','jpg','jpeg'))]
    test_images = [obj for obj in os.listdir(test_image_path) if obj.endswith(('png','jpg','jpeg'))]
    train_anns = []
    for root, dirs, files in os.walk(train_ann_path):
        for file in files:
            if file.endswith('.json'):
                train_anns.append(file)
    # 정렬
    train_images = sorted(train_images)
    train_anns = sorted(train_anns)
    test_images = sorted(test_images)

    return train_image_path, train_ann_path, test_image_path,\
           train_images, train_anns, test_images


def define_mapping_dict(train_images, train_image_path, train_ann_path):
    '''
    {알약 종류:idx} 형태의 mapping dict 생성
    '''
    label_list = []

    for image in train_images:
        image_path = os.path.join(train_image_path, image)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # 이미지 파일명에서 조합 ID 추출
        group_id = image_name.split('_')[0]
        dir_name = f'{group_id}_json'

        ann_path = os.path.join(train_ann_path, dir_name)
        for ann_dir in os.listdir(ann_path):
            label_list.append(ann_dir)

    label_list = sorted(list(set(label_list)))
    label_map = {label:idx for idx, label in enumerate(label_list)}

    return label_map        


def get_image_informations(train_images, annotation_path) -> list:
    '''
    image_informations = [
    {'image': image_name, 'bboxes': boxes, 'labels': labels, 'image_size':image_size},
    {'image': image_name, 'bboxes': boxes, 'labels': labels, 'image_size':image_size},
    ...
    ]
    '''
    image_informations = []

    for image in train_images:
        image_name = image.split('.')[0]
        # 이미지 파일명에서 조합 ID 추출
        group_id = image_name.split('_')[0]
        folder_name = f'{group_id}_json'
        search_path = os.path.join(annotation_path, folder_name, '*', f'{image_name}.json')
        matched_json_files = glob.glob(search_path)

        boxes = []
        labels = []
        # 찾은 JSON파일들을 읽어서 Bounding Box 및 Class 정보 병합
        for json_file in matched_json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            boxes.append(data['annotations'][0]['bbox'])
            labels.append(data['images'][0]['dl_mapping_code'])
            image_size = [data['images'][0]['width'], data['images'][0]['height']]

        image_informations.append({'image': image_name, 'bboxes': boxes, 'labels': labels, 'image_size':image_size})

    return image_informations


def is_match_image_and_ann(image_informations):
    '''알약의 개수와 bbox 개수가 일치하는지 확인'''
    unmatch_pill_ann_images = []
    for image_info in image_informations:
        group_id = image_info['image'].split('_')[0]
        ann_folder_name = f'{group_id}_json'
        # 폴더명에 포함된 알약의 개수가 추출된 bbox의 개수와 같은지 확인
        if len(ann_folder_name.split('-')[1:]) != len(image_info['bboxes']):    
            unmatch_pill_ann_images.append(image_info['image'])

    return unmatch_pill_ann_images 

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def verify_bbox(image_informations):
    '''
    1. Bounding Box가 이미지에서 벗어나지 않았는지 확인
    2. 겹치는 Bounding Box가 있는지 확인(IOU > 0.5)
    '''
    bbox_out_of_bounds_images = []
    overlaped_images = []
    for image_info in image_informations:
        image_w, image_h = image_info['image_size']
        for bbox in image_info['bboxes']:
            x_min, y_min, w, h = bbox
            # Verify bounding box out of Bounds
            if x_min<0 or (x_min+w)>image_w or y_min<0 or (y_min+h)>image_h:
                bbox_out_of_bounds_images.append(image_info['image'])

        # 2. 겹치는 Bounding Box가 있는지 확인(IOU > 0.5)
        bboxes = image_info['bboxes']
        IOU_THRESHOLD = 0.5  
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                iou = compute_iou(tuple(bboxes[i]), tuple(bboxes[j]))
                if iou >= IOU_THRESHOLD:
                    overlaped_images.append(image_info['image'])

    return bbox_out_of_bounds_images, overlaped_images                   



def visualize_corrupted_images(train_image_path, image_informations):
    unmatch_pill_ann_images = is_match_image_and_ann(image_informations)  
    bbox_out_of_bounds_images, overlaped_images = verify_bbox(image_informations)
    corrupted_images = unmatch_pill_ann_images + bbox_out_of_bounds_images + overlaped_images
    corrupted_images = list(set(corrupted_images))  # 중복제거
    corrupted_image_informations = [img_info for img_info in image_informations if img_info['image'] in corrupted_images]

    # 시각화
    fig, axes = plt.subplots(1, len(corrupted_images), figsize=(5,5))
    
    for idx, img_info in enumerate(corrupted_image_informations):
        img_path = os.path.join(train_image_path, f'{img_info["image"]}.png')
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #bbox
        box_color = (0, 255, 0)
        text_color = (255, 255, 255)
        thickness = 3

        for bbox, label in zip(img_info['bboxes'], img_info['labels']):
            x, y, w, h = bbox

            x_max = int(x + w)
            y_max = int(y + h)
            x_min, y_min = int(x), int(y)

            # 사각형 그리기
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), box_color, thickness)
            # 텍스트
            text = str(label)
            # 텍스트 크기 측정
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # 텍스트 가독성을 위한 배경 상자
            cv2.rectangle(img_rgb, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), box_color, -1)
            cv2.putText(img_rgb, text, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)        

        axes[idx].imshow(img_rgb)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
        

def verify_and_delete_corrupted_images(image_informations):
    '''오염된 이미지 삭제'''
    unmatch_pill_ann_images = is_match_image_and_ann(image_informations)
    bbox_out_of_bounds_images, overlaped_images = verify_bbox(image_informations)
    corrupted_images = unmatch_pill_ann_images + bbox_out_of_bounds_images + overlaped_images
    corrupted_images = list(set(corrupted_images))
    verified_image_informations = [img_info
                                   for img_info in image_informations
                                   if img_info['image'] not in corrupted_images]

    print(f'원래 이미지 개수: {len(image_informations)}')
    print(f'알약과 ann개수가 맞지 않는 이미지 개수: {len(unmatch_pill_ann_images)}개: {unmatch_pill_ann_images}')
    print(f'bbox 오류 이미지 개수: {len(bbox_out_of_bounds_images)}개:{bbox_out_of_bounds_images}')
    print(f'IOU 0.5이상 이미지 개수: {len(overlaped_images)}개: {overlaped_images}')
    print(f'삭제 후 이미지 개수: {len(verified_image_informations)}')
    return verified_image_informations


def convert_ann_file_to_txt(project_path, image_informations, label_map) -> None:
    '''
    data/processed/train_val_labels/
    image_name.txt
    class_id    x_center    y_center    norm_w  norm_h
    class_id    x_center    y_center    norm_w  norm_h
    ...

    '''
    ann_path = os.path.join(project_path, 'data/processed/train_val_labels')
    os.makedirs(ann_path, exist_ok=True)

    for image_info in image_informations:
    
        ann_lines = []  # text파일에 들어갈 라인
        image_name = image_info['image']
        image_w, image_h = image_info['image_size']
        # bbox unpacking
        for bbox, label in zip(image_info['bboxes'], image_info['labels']):
            class_id = label_map[label]
            x_min, y_min, w, h = bbox
            # Normalization
            x_center = ((x_min+w) / 2) / image_w    
            y_center = ((y_min+h) / 2) / image_h
            norm_w = w / image_w
            norm_h = h / image_h

            ann_lines.append(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}')

        with open(f'{os.path.join(ann_path, image_name)}.txt', 'w') as f:
            f.write('\n'.join(ann_lines))


def move_files(image_dir, ann_dir, save_dir, image_list, split_type):
    '''
    이미지, 라벨을 원본 폴더에서 전처리 폴더로 이동
    '''
    for img_name in image_list:
        ann_name = f'{os.path.splitext(img_name)[0]}.txt'

        # 이미지 이동
        src_img = os.path.join(image_dir, img_name)
        dst_img = os.path.join(save_dir, split_type, 'images', img_name)

        # 라벨 이동
        src_label = os.path.join(ann_dir, ann_name)
        dst_label = os.path.join(save_dir, split_type, 'labels', ann_name)
        
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
        
def split_data(project_path, image_informations):
    '''
    train과 validation용 데이터 분리
    dir구조:  
        data/processed/
    ├── train/
    │   ├── images/   # image1.jpg, image2.jpg ...
    │   └── labels/   # image1.txt, image2.txt ... 
    └── val/
        ├── images/
        └── labels/
    '''
    image_dir = os.path.join(project_path, 'data/raw/sprint_ai_project1_data/train_images')
    save_dir = os.path.join(project_path, 'data/processed')
    ann_dir = os.path.join(save_dir, 'train_val_labels')
    all_images = [f'{img_info["image"]}.png' for img_info in image_informations]

    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    # 파일 복사
    move_files(image_dir, ann_dir, save_dir, train_images, 'train')
    move_files(image_dir, ann_dir, save_dir, val_images, 'val')
    




def main() -> None:
    my_project_path = 'c://Bootcamp/project/pill-detection-ai' # 내 local 환경 절대경로
    data_path = os.path.join(my_project_path, 'data/raw/sprint_ai_project1_data')

    # 데이터 로드
    train_image_path, train_ann_path, test_image_path,\
    train_images, train_anns, test_images = load_data(data_path)
    # annotation용 mapping_dict 정의
    label_map = define_mapping_dict(train_images, train_image_path, train_ann_path)    
    # 이미지 정보 추출
    image_informations = get_image_informations(train_images, train_ann_path)
    # # 오염된 이미지 시각화
    # visualize_corrupted_images(train_image_path, image_informations)
    # 데이터 검증 후 삭제
    verified_image_informations = verify_and_delete_corrupted_images(image_informations)
    # annotation.txt파일 생성
    convert_ann_file_to_txt(my_project_path, verified_image_informations, label_map)
    # train과 validation 데이터 분리 후 저장
    split_data(my_project_path, verified_image_informations)


if __name__ == '__main__':
    main()