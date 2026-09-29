# pill-detection-ai

알약 사진 한 장을 입력받아, 사진 속 최대 4개 알약의 종류(클래스)와 위치(bounding box)를 출력하는 객체 검출 모델.

## 팀 소개

| 이름 | 담당 |
|---|---|
| 신유정 | Github 관리, 리팩토링, RF-DETR, 앙상블 |
| 김건 | Faster R-CNN, 성능 고도화, 회의록 작성, 중간 발표자료 취합 |
| 김예원 | RetinaNet, 최종 발표자료 작성 |
| 원재민 | YOLO, 모델 학습 관련 자료 작성 |
| 이주엽 | YOLO, 데이터 관련 자료 작성 |

## 기술 스택

- 언어: Python 3.11
- 프레임워크: PyTorch 2.5.1, Ultralytics (YOLO), RF-DETR, torchvision (Faster R-CNN)
- 데이터: OpenCV, Albumentations, pycocotools, ensemble-boxes (WBF)
- 실험 환경: Colab, 로컬 GPU 서버
- 협업: GitHub, Notion

## 프로젝트 개요

- 기간: 2026.09.10 ~ 2026.09.29
- 과제: 이미지 속 알약(최대 4개)의 클래스와 bbox 검출
- 데이터: 대회 train 232장(56종) + AI Hub 추가 데이터(선택) → 정제 후 train 7,954장 / val 2,451장(118종, AI Hub 포함 시), test 842장, COCO 포맷
- 평가 지표: mAP@[0.75:0.95]
- 최종 결과: 0.61364

## 모델 후보

| 모델 | 선정 여부 | 비고 |
|---|---|---|
| YOLO11n (118종) | 채택 | RF-DETR와 성능이 비슷하며 체크포인트가 약 20배 가벼움 (5.6MB vs 122MB) |
| RF-DETR nano | 미채택 | YOLO와의 차이가 작고 모델이 무거움 |
| YOLO11n + RF-DETR (WBF) | 미채택 | 두 모델이 서로 보완할 오류가 없어 두 단독 모델보다 모두 점수가 낮음 |
| Faster R-CNN (ResNet50-FPN v2) | 미채택 | 베이스라인 |

## 로직

### 아키텍처

```
테스트 이미지 ─┬→ YOLO11n (최종 제출) ────────────┐
              ├→ RF-DETR nano ────────────────────┤
              ├→ Faster R-CNN ────────────────────┼→ category_id 매핑 → 제출 CSV
              └→ YOLO11n + RF-DETR → WBF (실험) ──┘
```

### 파이프라인

```
대회 원본 (232장, 56종) ──┐
AI Hub 추가 데이터 (선택) ┴→ 검증·정제 → train/val 분할 (결과: COCO JSON) ─┬→ 그대로 사용        → Faster R-CNN (초기)
                                                                           ├→ YOLO 포맷 변환     → YOLO11n
                                                                           └→ Roboflow 포맷 변환 → RF-DETR nano
```

1. 데이터 선택: 대회 원본에 AI Hub 추가 데이터를 넣을지 `prepare_split.py` 옵션으로 선택 (넣으면 56종 → 118종)
2. 데이터 정제: bbox 형식 오류, 중복 박스, 이미지 밖 박스, 파일명의 알약 수와 박스 수 불일치 등 불량 이미지 제외 (AI Hub 포함 시 330장)
3. 분할: train/val 8:2, 모든 클래스가 val에 들어가도록 클래스 비율을 맞춘 근사 stratified split. 결과는 COCO JSON으로 저장
4. 포맷 변환: Faster R-CNN은 COCO JSON을 그대로 쓰고, YOLO와 RF-DETR용으로는 각각 YOLO 포맷과 Roboflow 포맷으로 변환
5. 학습: 세 모델 모두 COCO 사전학습 가중치에서 fine-tuning하고, val 성능이 가장 좋은 체크포인트를 사용

   | 모델 | 체크포인트 |
   |---|---|
   | YOLO11n (최종) | `best.pt` |
   | RF-DETR nano | EMA 가중치 중 val 최고 성능 (`checkpoint_best_ema.pth`) |
   | Faster R-CNN | `best.pt` |

6. 추론: 모델별로 예측하고 후처리

   | 모델 | 후처리 |
   |---|---|
   | YOLO11n (최종) | conf 0.25 이상, NMS, 이미지당 최대 4개 |
   | RF-DETR nano | threshold 0.5 이상 (1:1 매칭 구조라 NMS 없음) |
   | Faster R-CNN | conf 0.25 이상 → 클래스별 NMS → 점수 상위 4개 |
   | 앙상블 (실험) | YOLO11n과 RF-DETR 예측을 WBF로 병합 (IoU 0.5, 가중치 1:1) |

7. 제출: 모델마다 다른 클래스 번호를 대회 `category_id`로 매핑하고, 박스를 xyxy → xywh로 변환해 CSV로 저장
   - YOLO11n: `data.yaml`의 클래스명(`K-000250` 등)에서 코드 추출
   - RF-DETR: `category_mapping.json`으로 클래스명 → `category_id`
   - Faster R-CNN: 체크포인트에 저장된 `label_to_raw_id`

### 데이터 포맷

세 포맷은 같은 라벨을 담고, 파일을 나누는 방식과 좌표 표기만 다르다.

| 포맷 | 사용 모델 | 라벨 파일 | 박스 좌표 | 폴더 구조 |
|---|---|---|---|---|
| COCO JSON | Faster R-CNN | split마다 `annotations.json` 1개 | `[x, y, w, h]`, 픽셀, 왼쪽 위 기준 | `train/images/` + `train/annotations.json` |
| YOLO | YOLO11n | 이미지마다 `.txt` 1개 + `data.yaml` | `class cx cy w h`, 0~1 정규화, 중심 기준 | `train/images/` + `train/labels/` |
| Roboflow COCO | RF-DETR | split마다 `_annotations.coco.json` 1개 | COCO와 동일 | 이미지와 JSON이 한 폴더에: `train/`, `valid/` |

Roboflow 포맷은 내용이 COCO JSON과 같고 폴더 구조만 다르다. 그래서 `convert_to_rfdetr.py`는 라벨을 바꾸지 않고 파일 위치와 이름만 맞춘다 (`val` → `valid`, 이미지는 심볼릭 링크).

## 핵심 로직

- **AI Hub 데이터 추가**: 대회 데이터가 232장뿐이라 같은 형식의 AI Hub 경구약제 이미지를 더해 클래스를 56종 → 118종으로 늘림
- **train/val 분할 통일**: 한 번 분할한 COCO JSON을 모델별 포맷으로 변환해, 세 모델을 같은 train/val에서 공정하게 비교함
- **앙상블 검증 후 단일 모델 선택**: YOLO+RF-DETR WBF 앙상블을 실험했지만, 두 모델이 테스트 842장 모두에서 같은 알약을 같은 위치(박스 3,229개 전부, IoU 0.94 이상)에서 찾아 보완 효과가 없었고, WBF가 박스와 점수를 평균 내면서 오히려 두 단독 모델보다 모두 낮아짐. 예측이 거의 같으므로 훨씬 가볍고 빠른 YOLO11n 단독을 선택

## 실험 결과

| 실험 | 데이터 | 모델 | 점수 |
|---|---|---|---|
| baseline | 232장 / 56종 | Faster R-CNN | (점수) |
| YOLO | 232장 / 56종 | YOLO11n | (점수) |
| **데이터 확장 (최종)** | 7,954장 / 118종 | YOLO11n | (점수) |
| 데이터 확장 | 7,954장 / 118종 | RF-DETR nano | (점수) |
| 앙상블 | 7,954장 / 118종 | YOLO11n + RF-DETR nano (WBF) | (점수) |

## 환경 설정

```bash
git clone https://github.com/yulesn/pill-detection-ai.git
cd pill-detection-ai
conda env create -f environment.yml
conda activate pill-detection
```

`environment.yml`은 CPU/MPS(맥) 기준이다. GPU 서버에서는 같은 PyTorch 버전(2.5.1)의 CUDA 빌드로 다시 설치한다.

데이터는 용량 문제로 저장소에 포함하지 않는다. 대회 데이터는 `python src/download_data.py`로 받고(Kaggle 토큰 필요, [data/README.md](data/README.md) 참고), AI Hub 데이터는 `src/download_additional_data.py`의 안내를 따른다. `data/` 아래 구조는 다음과 같다.

```
data/
├── raw/
│   ├── sprint_ai_project1_data/   # 대회 원본 (train_annotations, test_images)
│   └── 166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터/   # AI Hub 원본
├── augmented/raw/                 # 대회 원본 + AI Hub 합친 결과 (images, annotations)
└── processed/
    ├── coco/      # prepare_split.py 결과 (train / val / splits)
    ├── yolo/      # convert_to_yolo.py 결과
    └── rfdetr/    # convert_to_rfdetr.py 결과
```

데이터 준비:

```bash
# AI Hub 추가 데이터 포함 (--image-dir/--ann-dir 생략 시 대회 원본만 사용)
python src/prepare_split.py --config configs/rfdetr.yaml \
  --image-dir data/augmented/raw/images --ann-dir data/augmented/raw/annotations
python src/convert_to_yolo.py
python src/convert_to_rfdetr.py
```

학습과 추론:

```bash
# 학습 (최종 모델: YOLO11n, data/processed/yolo/data.yaml 사용)
python src/train.py --config configs/exp2_yolo11n.yaml

# 추론 (제출 CSV 생성)
python src/predict.py --config configs/exp2_yolo11n.yaml --checkpoint outputs/checkpoints/yolo_jy_best.pt --test_dir data/raw/sprint_ai_project1_data/test_images

# (참고) RF-DETR 학습, 앙상블 추론
python src/train.py --config configs/rfdetr.yaml
python src/ensemble_predict.py --config configs/ensemble.yaml --test_dir data/raw/sprint_ai_project1_data/test_images
```

## 관련 문서

- [최종 보고서](https://github.com/yulesn/pill-detection-ai/blob/main/%5BAI%5D%20%EC%B4%88%EA%B8%89%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%204%ED%8C%80%20%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf?raw=true)
- [협업 일지](https://app.notion.com/p/3d7232ca6c7b801081e2e1820e421042?v=3d7232ca6c7b803f9a1d000c3bdfb74b&source=copy_link)
- [Notion](https://app.notion.com/p/AI-3d7232ca6c7b8097a511d1e2535751d4?source=copy_link)
