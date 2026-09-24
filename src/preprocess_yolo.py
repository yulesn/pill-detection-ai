# src/preprocess_yolo.py
# AI Hub 데이터셋으로 증강된 데이터를 전처리하는 스크립트

import matplotlib.pyplot as plt 

import json
import glob
import cv2
import os
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from pathlib import Path

# ====================================================================================
# annotation 로드
# ====================================================================================
def findout_decoding_error(json_file):
    '''
    Decoding 에러 검출
    UnicodeDecodeError 발생 시에는 encoding 방식을 cp949로 바꾸어 한 번 더 시도
    그 외 예상치 못한 오류 발생 시 출력
    '''
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except UnicodeDecodeError: 
        try:
            with open(json_file, 'r', encoding='cp949') as f:   # encoding 방식이 cp949로 작성된 경우도 있으니 다시 시도
                data = json.load(f)
            return data
        except:
            return
    except json.JSONDecodeError:
        return

    except Exception as e:
        print(f'[예상치 못한 오류 발생] {json_file}: {e}')
        return

def parse_annotations(ann_path):
    rows = []

    for json_file in sorted(ann_path.rglob('*.json')):
        
        data = findout_decoding_error(json_file)

        if data:    # data가 None이 아닌 경우
            file_name = data['images'][0]['file_name']
            drug_N = data['images'][0]['drug_N']
            bbox = data['annotations'][0]['bbox']

            if file_name and drug_N and len(bbox)==4:  # annotation 정보가 정상적이지 않은 경우 건너뛰기   
                print(bbox)
                x, y, w, h = bbox

                row = {'file_name': file_name,
                        'drug_N': drug_N,
                        'bbox_x': x,
                        'bbox_y': y,
                        'bbox_w': w,
                        'bbox_h': h
                }
                rows.append(row)
    annotations = pd.DataFrame(rows)
    return annotations
 

def load_annotations(data_path, annotation_path):
    '''
    데이터를 로드하여 annotation 파일을 저장하는 함수
    annotation은 필요한 부분만 파싱(file_name: 이미지파일 이름
                                 drug_N: 알약이름
                                 bbox_x: bbox x_min좌표
                                 bbox_y: bbox y_min좌표
                                 bbox_w: bbox width
                                 bbox_h: bbox height
                                 )
    param: data_path: 데이터 경로
    param: annotation_path: annotation 정보가 담긴 pkl파일이 저장될 경로 
    '''
    ann_path = Path(data_path) / 'annotations' 
    annotations = parse_annotations(ann_path)

    # pkl파일로 저장
    annotations.to_pickle(annotation_path)


def define_mapping_dict(annotation_path):
    '''{알약 종류:idx} 형태의 mapping dict 생성'''
    annotations = pd.read_pickle(annotation_path)
    unique_drug_N = sorted(np.unique(annotations['drug_N']))
    label_map = {drug_N:idx for idx,drug_N in enumerate(unique_drug_N)}
    return annotations, label_map 
# ====================================================================================
# 데이터 오류 검출
# ====================================================================================

def delete_unmatched_ann(annotations):
    '''알약의 개수와 bbox 개수가 일치하는지 확인'''
    annotations['pill_cnt_in_name'] = (     # 이미지 이름을 이용하여 알약 개수 count(ex.K-001900-016548-019607-033009)
        annotations['file_name'].str.split('_').str[0].str.count('-')     
    )

    annotations['ann_cnts'] = annotations.groupby('file_name')['file_name'].transform('count')
    parsed_anns = annotations[annotations['pill_cnt_in_name'] == annotations['ann_cnts']]

    print(f'이미지 속 알약의 개수가 annotation 파일 개수와 일치하지 않는 이미지: {len(annotations)-len(parsed_anns)}')
    return parsed_anns


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

def verify_bbox(annotations):
    '''
    1. Bounding Box가 이미지에서 벗어나지 않았는지 확인
    2. 겹치는 Bounding Box가 있는지 확인(IOU > 0.5)
    '''
    bbox_error_anns = annotations[
        (annotations['bbox_x'] < 0) |
        ((annotations['bbox_x']+annotations['bbox_w']) > 976) |
        (annotations['bbox_y'] < 0) |
        ((annotations['bbox_y']+annotations['bbox_h']) > 1280)
    ]
    print(f'Bounding Box가 이미지의 크기를 벗어나는 이미지: {len(bbox_error_anns)}')
    # 오류 이미지 제거
    parsed_anns = annotations[~annotations['file_name'].isin(bbox_error_anns['file_name'])]    
    
    # 2. 겹치는 Bounding Box가 있는지 확인(IOU > 0.5)
    IOU_THRESHOLD = 0.5
    parsed_anns['bbox'] = list(zip(parsed_anns['bbox_x'],
                                  parsed_anns['bbox_y'],
                                  parsed_anns['bbox_w'],
                                  parsed_anns['bbox_h']))
    overlapped_images = set()
    for file_name, bboxes in parsed_anns.groupby('file_name')['bbox']:
        bbox_list = bboxes.tolist()
        for i in range(len(bbox_list)):
            for j in range(i+1, len(bbox_list)):
                iou = compute_iou(bbox_list[i], bbox_list[j])
                if iou > IOU_THRESHOLD:
                    overlapped_images.add(file_name)
                    break
    print(f'Bounding Box가 심하게 겹쳐진 이미지: {len(overlapped_images)}')
    # Bbox가 심하게 겹쳐진 이미지 삭제
    parsed_anns = parsed_anns[~parsed_anns['file_name'].isin(overlapped_images)]

    return parsed_anns


def convert_ann_to_txt(data_path, annotations, label_map) -> None:
    '''
    data/processed/train_val_labels/
    image_name.txt
    class_id    x_center    y_center    norm_w  norm_h
    class_id    x_center    y_center    norm_w  norm_h
    ...

    '''
    ann_path = Path(data_path) / 'train_val_anns' 

    if ann_path.exists() == True:   # 폴더가 이미 존재하는 경우 삭제 후 재생성
        shutil.rmtree(ann_path)

    os.makedirs(ann_path)
    
    for file_name, group in annotations.groupby('file_name')[['bbox','drug_N']]:
        ann_lines = []  # text파일에 들어갈 라인
        image_name = file_name.split('.')[0]
        image_w, image_h = 976, 1280
        # bbox unpacking
        for bbox, drug_N in zip(group['bbox'], group['drug_N']):
            class_id = label_map[drug_N]
            x_min, y_min, w, h = bbox
            # Normalization
            x_center = (x_min + (w/2)) / image_w    
            y_center = (y_min + (h/2)) / image_h
            norm_w = w / image_w
            norm_h = h / image_h

            ann_lines.append(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}')

        with open(f'{os.path.join(ann_path, image_name)}.txt', 'w') as f:
            f.write('\n'.join(ann_lines))

def is_exist_image(data_path, annotations):
    '''annotations['file_name'] 이미지가 실제로 존재하는지 확인하는 함수'''
    image_path = Path(data_path) / 'images'
    exist_images = {img_f.name for img_f in image_path.rglob('*.png')}

    error_images = [img_f for img_f in annotations['file_name'] if img_f not in exist_images]

    print(f'annotation의 이미지가 실제로 존재하지 않는 경우: {len(error_images)}')
    # 삭제
    parsed_anns = annotations[~annotations['file_name'].isin(error_images)]

    return parsed_anns


def move_files(image_dir, ann_dir, save_dir, image_list, split_type):
    '''
    이미지, 라벨을 원본 폴더에서 전처리 폴더로 이동
    '''
    for img_name in image_list:
        ann_name = f'{os.path.splitext(img_name)[0]}.txt'

        # 이미지 이동
        src_img = next(image_dir.rglob(img_name))
        dst_img = os.path.join(save_dir, split_type, 'images', img_name)

        # 라벨 이동
        src_label = os.path.join(ann_dir, ann_name)
        dst_label = os.path.join(save_dir, split_type, 'labels', ann_name)
        
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
        
def split_data(data_path, annotations):
    '''
    train과 validation용 데이터 분리
    dir구조:  
        data/augmented/processed/
    ├── train/
    │   ├── images/   # image1.png, image2.png ...
    │   └── labels/   # image1.txt, image2.txt ... 
    └── val/
        ├── images/
        └── labels/
    '''
    image_dir = Path(data_path) / 'images'
    save_dir = Path(data_path) / 'processed'
    ann_dir = Path(data_path) / 'train_val_anns'
    all_images = annotations['file_name'].unique().tolist()

    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # 폴더가 이미 존재할 경우 삭제 후 재생성
    if save_dir.exists():
        shutil.rmtree(save_dir)
        
    # 파일 복사
    move_files(image_dir, ann_dir, save_dir, train_images, 'train')
    move_files(image_dir, ann_dir, save_dir, val_images, 'val')
    
def save_data_yaml(label_map):
    '''configs/data.yaml파일 생성'''
    names_dict = {idx:code for code, idx in label_map.items()}

    yaml_data = {
        'path': 'data/augmented',
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(names_dict),
        'names': names_dict
    }

    with open('configs/data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)



def main() -> None:
    PROJECT_PATH = 'c://Bootcamp/project/pill-detection-ai' # 내 local 환경 절대경로
    DATA_PATH = 'G://내 드라이브/colab_notebooks/project/pill_detection_ai/data/augmented'
    ANNOTATION_PATH = Path(__file__).resolve().parent.parent / 'data' / 'augmented' / 'annotations.pkl'

    # 데이터 로드(최초 1회)
    # load_annotations(DATA_PATH, ANNOTATION_PATH)
    
    
    # yolo annotatinos파일을 위한 mapping dict 정의
    annotations, label_map = define_mapping_dict(ANNOTATION_PATH)
    # 알약의 개수와 bbox 개수가 일치하지 않는 행 삭제
    parsed_annotations = delete_unmatched_ann(annotations)
    # Bbox 오류 이미지 삭제
    parsed_annotations = verify_bbox(parsed_annotations)
    # 실제로 이미지가 존재하지 않는 경우 삭제
    preprocessed_annotations = is_exist_image(DATA_PATH, parsed_annotations)

    # annotation .txt형태로 저장(최초 1회)   
    # convert_ann_to_txt(DATA_PATH, preprocessed_annotations, label_map)

    # train/val 데이터 스플릿
    split_data(DATA_PATH, preprocessed_annotations)    
    # 데이터 정보를 data_yaml 파일로 저장
    save_data_yaml(label_map)


if __name__ == '__main__':
    main()