"""Image decoding, augmentation, normalization, and PyTorch dataset wrappers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from marine_plastic.dataset import ImageRecord


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = IMAGE_SIZE) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


class MultiLabelImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[ImageRecord],
        classes: list[str],
        active_classes: list[str],
        transform: transforms.Compose,
        mask_classes: list[str] | None = None,
    ) -> None:
        self.records = records
        self.classes = classes
        self.active_classes = active_classes
        self.transform = transform
        self.active_indices = [classes.index(name) for name in active_classes]
        self.mask_indices = [
            active_classes.index(name)
            for name in (mask_classes if mask_classes is not None else active_classes)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        image_tensor = self.transform(image)
        target = torch.tensor(
            [record.labels[class_index] for class_index in self.active_indices],
            dtype=torch.float32,
        )
        mask = torch.zeros(len(self.active_classes), dtype=torch.float32)
        mask[self.mask_indices] = 1.0
        return image_tensor, target, mask


def transform_uploaded_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    _, evaluation_transform = build_transforms(image_size)
    return evaluation_transform(image.convert("RGB")).unsqueeze(0)
