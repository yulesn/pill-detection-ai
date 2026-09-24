# src/download_additional_data.py

"""=============================================================================
[AI Hub 데이터셋 다운로드 가이드]

1. 환경 요구사항 (Windows 사용자 필수):
   - 본 다운로드 방식은 Linux/macOS 환경에 최적화된 aihubshell을 사용합니다.
   - Windows 사용자는 WSL 설치 또는 Gib bash 터미널 이용(Linux / MacOS 사용자의 경우 생략)
     
- WSL2 (Ubuntu) 사용 방법:
      - PowerShell(관리자 권한) 실행 후: wsl --install
      - Ubuntu 
      - 설치 완료 후 우분투 진입: wsl -d Ubuntu
        - 비밀번호 설정 등 계정 생성
              
3. 공통 다운로드 절차
    - 1. aihubshell 다운로드
        curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do
    - 2. 실행권한 부여
        chmod +x /usr/bin/aihubshell
    - 3. 데이터를 저장할 폴더로 이동
        cd ./data/raw
    - 4. 본인 API KEY 입력 하여 다운로드 진행(API Key 발급 방법: aihub.or.kr -> AI 데이터찾기 -> AI 허브 오픈 API -> API Key 발급)
        aihubshell -mode d -datasetkey 576 -filekey 66065,66067,66068,66069,66070,66071,66072,66154,66156,66157,66158,66159,66160,66161 -aihubapikey 'MY_API_KEY'
============================================================================="""

# 아래 코드는 {data/raw/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터} 생성 후 시행
# 데이터셋 압축 해제 및 병합

import os
from pathlib import Path
import zipfile
import shutil



def add_original_train_data(src_path, dst_img_path, dst_ann_path):
    '''기존 train data 추가'''
    image_path = src_path / 'train_images'
    ann_path = src_path / 'train_annotations'

    shutil.copytree(image_path, dst_img_path, dirs_exist_ok=True)
    shutil.copytree(ann_path, dst_ann_path, dirs_exist_ok=True)    
    
def merge_data(img_dir_path, ann_dir_path, dst_path):
    '''
    AI Hub 데이터셋의 압축을 해제하고 merge하는 함수
    
    param: img_dir_path: 압축된 이미지 파일 경로
    param: ann_dir_path: 압축된 ann 파일 경로
    param: dst_path: 파일 저장될 위치 경로
    '''
    img_files = os.listdir(img_dir_path)
    ann_files = os.listdir(ann_dir_path)
    # dst path 지정
    dst_img_path = os.path.join(dst_path, 'images')
    dst_ann_path = os.path.join(dst_path, 'annotations')
    

    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)  # 디렉토리가 이미 존재하는 경우 삭제

    os.makedirs(dst_path)
    os.makedirs(dst_img_path)
    os.makedirs(dst_ann_path)

    # image
    for img_file in img_files:
        img_path = os.path.join(img_dir_path, img_file)

        with zipfile.ZipFile(img_path, 'r') as zip_ref:
            for member in zip_ref.infolist():

                file_stem = Path(member.filename).stem
                if file_stem.endswith('_index'):    # 내부구조 
                    continue

                target_path = Path(dst_img_path) / member.filename

                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        zip_ref.open(member) as source,
                        open(target_path, 'wb') as target,
                    ):
                        target.write(source.read())
    print('이미지 파일 압축해제 완료')

    # annotation
    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir_path, ann_file)

        with zipfile.ZipFile(ann_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                 
                target_path = Path(dst_ann_path) / member.filename

                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        zip_ref.open(member) as source,
                        open(target_path, 'wb') as target,
                    ):
                        target.write(source.read())
    print('annotation 파일 압축해제 완료')

    # 기존 train 데이터 추가
    add_original_train_data(src_path=Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'sprint_ai_project1_data',
                            dst_img_path=dst_img_path,
                            dst_ann_path=dst_ann_path)
    print('train 파일 추가 완료')
    



def main() -> None:
    PROJECT_PATH = Path(__file__).resolve().parent.parent
    IMAGE_PATH =  PROJECT_PATH / 'data' / 'raw' /\
              '166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터' / '01.데이터' / '1.Training' / '원천데이터' / '경구약제조합_5000종'
    ANN_PATH =  PROJECT_PATH / 'data' / 'raw' /\
              '166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터' / '01.데이터' / '1.Training' / '라벨링데이터' / '경구약제조합_5000종'
    
    DST_PATH = 'G://내 드라이브/colab_notebooks/project/pill_detection_ai/data/augmented'


    merge_data(IMAGE_PATH, ANN_PATH, DST_PATH)


if __name__ == '__main__': 
    main()



