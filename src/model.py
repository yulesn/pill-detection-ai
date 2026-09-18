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
        # TODO(YOLO 트랙): ultralytics 모델 생성
        # from ultralytics import YOLO
        # return YOLO(f"{model_name}.pt")
        raise NotImplementedError("YOLO 분기를 구현해주세요.")

    if framework == "torchvision":
        import torchvision

        model = torchvision.models.detection.__dict__[model_name](
            pretrained=config["model"]["pretrained"]
        )
        num_classes = config["data"]["num_classes"]
        if num_classes is not None:
            _replace_classification_head(model, num_classes)
        return model

    raise ValueError(f"지원하지 않는 framework입니다: {framework}")


def _replace_classification_head(model, num_classes: int) -> None:
    """COCO 사전학습 head를 데이터셋 클래스 수(num_classes, 배경 포함)에 맞게 교체한다."""
    if hasattr(model, "roi_heads"):
        # Faster R-CNN 계열
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #elif hasattr(model, "head") and hasattr(model.head, "classification_head"):   
    else:
        raise NotImplementedError(
            f"{type(model).__name__}의 분류 head 교체 로직이 구현되어 있지 않습니다."
        )
