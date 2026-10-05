"""Transfer-learning classifiers with one stable output index per global class."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B2_Weights,
    ResNet50_Weights,
    efficientnet_b2,
    resnet50,
)


class MarineClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        architecture: str = "resnet50",
        pretrained: bool = True,
        train_last_block: bool = True,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            input_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(input_features, num_classes)
            self.image_size = 224
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if train_last_block:
                for parameter in self.backbone.layer4.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.fc.parameters():
                parameter.requires_grad = True
        elif self.architecture == "efficientnet_b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b2(weights=weights)
            input_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(input_features, num_classes)
            self.image_size = 260
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            if train_last_block:
                for parameter in self.backbone.features[-1].parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.classifier[-1].parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(
                "Unsupported architecture; choose 'resnet50' or 'efficientnet_b2'"
            )

    def train(self, mode: bool = True) -> "MarineClassifier":
        super().train(mode)
        if mode:
            if self.architecture == "resnet50":
                for layer in (
                    self.backbone.conv1,
                    self.backbone.bn1,
                    self.backbone.layer1,
                    self.backbone.layer2,
                    self.backbone.layer3,
                ):
                    layer.eval()
            else:
                for layer in self.backbone.features[:-1]:
                    layer.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    @property
    def cam_target_layer(self) -> nn.Module:
        if self.architecture == "resnet50":
            return self.backbone.layer4[-1]
        return self.backbone.features[-1]


def create_model(
    num_classes: int,
    pretrained: bool = True,
    train_layer4: bool = True,
    architecture: str = "resnet50",
) -> MarineClassifier:
    """Create a model while retaining backward compatibility with train_layer4."""
    return MarineClassifier(
        num_classes,
        architecture=architecture,
        pretrained=pretrained,
        train_last_block=train_layer4,
    )
