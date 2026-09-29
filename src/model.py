"""모델 생성 코드.

config["model"]["framework"] 값(yolo / torchvision)에 따라 분기한다.
각 분기 안은 해당 트랙 담당자가 채운다. train.py, predict.py는
이 함수를 통해서만 모델을 가져오므로 다른 파일은 수정할 필요가 없다.
"""

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_V2_Weights


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
        pretrained = config["model"]["pretrained"]
        num_classes = config["data"].get("num_classes", 2)

        # RetinaNet v2 모델 명시적 처리 및 안전한 가중치/클래스 수 바인딩
        if model_name == "retinanet_resnet50_fpn_v2":

            weights = (
                RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                if pretrained
                else None
            )

            # NOTE: 신뢰도 임계값(score_thresh)은 0.5 이상으로 설정합니다.
            model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=weights,
                score_thresh=0.5
            )

            # 기존 COCO classification head를 약품 class 수에 맞게 변경
            in_features = (
                model.head.classification_head.cls_logits.in_channels
            )

            model.head.classification_head.num_classes = num_classes

            num_anchors = (
                model.head.classification_head.num_anchors
            )

            model.head.classification_head.cls_logits = torch.nn.Conv2d(
                in_features,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1
            )

            # 새 classification head 가중치 초기화
            torch.nn.init.normal_(
                model.head.classification_head.cls_logits.weight,
                std=0.01
            )

            torch.nn.init.constant_(
                model.head.classification_head.cls_logits.bias,
                0
            )

            return model

        # 기타 torchvision 디텍션 모델 동적 지원
        if model_name in torchvision.models.detection.__dict__:
            model_fn = torchvision.models.detection.__dict__[model_name]
            weights_backbone = "DEFAULT" if pretrained else None

            return model_fn(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
                score_thresh=0.2,
                nms_thresh=0.3,
            )

        raise ValueError(
            f"지원하지 않는 torchvision 모델 이름입니다: {model_name}"
        )

    raise ValueError(
        f"지원하지 않는 framework입니다: {framework}"
    )