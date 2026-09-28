"""모델 생성 코드.

config["model"]["framework"] 값(yolo / torchvision)에 따라 분기한다.
각 분기 안은 해당 트랙 담당자가 채운다. train.py, predict.py는
이 함수를 통해서만 모델을 가져오므로 다른 파일은 수정할 필요가 없다.
"""


def build_model(config: dict):
    """config["model"]["framework"]/["name"] 값에 따라 모델을 생성해서 반환한다."""
    framework = config["model"]["framework"]
    model_name = config["model"]["name"]

    if framework is None or model_name is None:
        raise ValueError(
            "config['model']['framework']/['name']이 설정되지 않았습니다. "
            "팀에서 사용할 프레임워크/모델을 정한 뒤 configs/*.yaml에 채워주세요."
        )

    if framework == "yolo":
        from ultralytics import YOLO

        if not model_name.endswith('.pt'):
            model_name = f'{model_name}.pt'
        return YOLO(model_name)

    if framework == "torchvision":
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        num_classes = config["data"]["num_classes"]
        weights = "DEFAULT" if config["model"]["pretrained"] else None
        model = torchvision.models.detection.__dict__[config["model"]["name"]](weights=weights)

        # 사전학습 모델의 분류기 마지막 층을, 우리 클래스 개수(57)에 맞게 새로 교체
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")