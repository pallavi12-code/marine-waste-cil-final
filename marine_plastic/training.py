"""Validation-first backbone selection and incremental experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from marine_plastic.dataset import ImageRecord
from marine_plastic.evaluation import (
    add_retention_metrics,
    evaluate_model,
    save_confusion_matrices,
)
from marine_plastic.memory_replay import ExemplarMemory
from marine_plastic.models import MarineClassifier, create_model
from marine_plastic.preprocessing import MultiLabelImageDataset, build_transforms


ARCHITECTURES = ("resnet50", "efficientnet_b2")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _leaf_datasets(dataset: Dataset[Any]) -> list[MultiLabelImageDataset]:
    if isinstance(dataset, MultiLabelImageDataset):
        return [dataset]
    if isinstance(dataset, ConcatDataset):
        return [leaf for child in dataset.datasets for leaf in _leaf_datasets(child)]
    raise TypeError(f"Unsupported training dataset type: {type(dataset).__name__}")


def _balanced_weights(dataset: Dataset[Any]) -> torch.DoubleTensor:
    leaves = _leaf_datasets(dataset)
    supervised_records = [
        (record, leaf.mask_indices)
        for leaf in leaves
        for record in leaf.records
    ]
    if not supervised_records:
        raise ValueError("A training stage has no labeled examples")
    active = sorted(
        {index for _, supervised in supervised_records for index in supervised}
    )
    if not active:
        raise ValueError("No supervised classes were selected for this training stage")
    positive_counts = {
        index: sum(
            record.labels[index]
            for record, supervised in supervised_records
            if index in supervised
        )
        for index in active
    }
    weights: list[float] = []
    for record, supervised in supervised_records:
        supervised_positive = [
            1.0 / positive_counts[index]
            for index in supervised
            if positive_counts.get(index, 0) and record.labels[index]
        ]
        weights.append(
            float(np.mean(supervised_positive)) if supervised_positive else 1e-8
        )
    return torch.as_tensor(weights, dtype=torch.double)


def _fit(
    model: MarineClassifier,
    dataset: Dataset[Any],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    if len(dataset) == 0:
        raise ValueError("A training stage has no matching labeled images")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        _balanced_weights(dataset),
        num_samples=len(dataset),
        replacement=True,
        generator=loader_generator,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, targets, masks in loader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(images), targets)
            per_example = (losses * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1.0)
            loss = per_example.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        print(f"      epoch {epoch + 1}/{epochs}: balanced masked BCE={total_loss / len(loader):.4f}")


def _save_checkpoint(
    model: MarineClassifier,
    classes: list[str],
    active_classes: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes,
            "active_classes": active_classes,
            "global_class_mapping": classes,
            "image_size": model.image_size,
            "model_name": model.architecture,
        },
        path,
    )


def _evaluate_stage(
    model: MarineClassifier,
    records: list[ImageRecord],
    all_classes: list[str],
    seen_classes: list[str],
    previous_classes: list[str],
    transform: Any,
    device: torch.device,
    batch_size: int,
    best_recall: dict[str, float],
    metrics_dir: Path,
    experiment: str,
    stage_number: int,
) -> dict[str, Any]:
    metrics, y_true, predictions = evaluate_model(
        model, records, all_classes, seen_classes, transform, device, batch_size,
        model_classes=all_classes,
    )
    add_retention_metrics(metrics, seen_classes, previous_classes, best_recall)
    metrics["stage"] = stage_number
    metrics["architecture"] = model.architecture
    save_confusion_matrices(
        y_true, predictions, seen_classes, metrics_dir, experiment, stage_number
    )
    retention = metrics["old_class_retention"]
    retention_text = "n/a" if retention is None else f"{retention:.3f}"
    print(
        f"      val accuracy={metrics['accuracy']:.3f} macro-F1={metrics['macro_f1']:.3f} "
        f"old recall={retention_text} forgetting={metrics['forgetting']:.3f}"
    )
    return metrics


def train_static_baselines(
    train_records: list[ImageRecord],
    validation_records: list[ImageRecord],
    classes: list[str],
    output_dir: Path,
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 16,
    seed: int = 42,
    pretrained: bool = True,
    train_last_block: bool = True,
    learning_rate: float = 1e-4,
    architectures: tuple[str, ...] = ARCHITECTURES,
) -> tuple[Path, str, dict[str, dict[str, Any]]]:
    """Train and select architecture exclusively on validation macro-F1."""
    baseline_metrics: dict[str, dict[str, Any]] = {}
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "confusion_matrices"
    for architecture in architectures:
        print(f"\nStatic baseline: {architecture}")
        set_seed(seed)
        model = create_model(
            len(classes), pretrained=pretrained,
            train_layer4=train_last_block, architecture=architecture,
        ).to(device)
        train_transform, evaluation_transform = build_transforms(model.image_size)
        dataset = MultiLabelImageDataset(
            train_records, classes, classes, train_transform, classes
        )
        _fit(model, dataset, device, epochs, batch_size, learning_rate, seed)
        metric, truth, predictions = evaluate_model(
            model, validation_records, classes, classes, evaluation_transform,
            device, batch_size, model_classes=classes,
        )
        metric["architecture"] = architecture
        metric["split"] = "validation"
        baseline_metrics[architecture] = metric
        save_confusion_matrices(
            truth, predictions, classes, metrics_dir, f"baseline_{architecture}", 0
        )
        _save_checkpoint(
            model, classes, classes, models_dir / f"{architecture}_static.pt"
        )
        print(
            f"      validation macro-F1={metric['macro_f1']:.4f} "
            f"accuracy={metric['accuracy']:.4f}"
        )

    selected_architecture = max(
        architectures, key=lambda name: baseline_metrics[name]["macro_f1"]
    )
    summary = {
        "selection_metric": "validation_macro_f1",
        "selected_architecture": selected_architecture,
        "architectures": baseline_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"\nSelected {selected_architecture} by validation Macro-F1 "
        f"({baseline_metrics[selected_architecture]['macro_f1']:.4f}); "
        "the test split was not used for model selection."
    )
    return models_dir.joinpath(f"{selected_architecture}_static.pt"), selected_architecture, baseline_metrics


def load_checkpoint_model(
    checkpoint_path: Path,
    architecture: str,
    classes: list[str],
    device: torch.device,
) -> MarineClassifier:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["global_class_mapping"] != classes:
        raise ValueError("Checkpoint global class mapping does not match this dataset")
    model = create_model(
        len(classes), pretrained=False, train_layer4=False, architecture=architecture
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def run_incremental_experiments(
    train_records: list[ImageRecord],
    validation_records: list[ImageRecord],
    classes: list[str],
    stages: list[list[str]],
    selected_static_model: MarineClassifier,
    architecture: str,
    output_dir: Path,
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 16,
    memory_capacity: int = 100,
    seed: int = 42,
    pretrained: bool = True,
    train_last_block: bool = True,
    learning_rate: float = 1e-4,
) -> dict[str, list[dict[str, Any]]]:
    metrics_dir = output_dir / "confusion_matrices"
    metrics: dict[str, list[dict[str, Any]]] = {
        "static_all_classes": [],
        "naive_sequential": [],
        "memory_replay": [],
    }
    _, eval_transform = build_transforms(selected_static_model.image_size)
    seen: list[str] = []
    previous: list[str] = []
    static_best: dict[str, float] = {}
    print("\nExperiment: static all-class baseline")
    for stage_index, stage_classes in enumerate(stages, start=1):
        previous = seen.copy()
        seen.extend(stage_classes)
        ordered_seen = [name for name in classes if name in seen]
        result = _evaluate_stage(
            selected_static_model, validation_records, classes, ordered_seen, previous,
            eval_transform, device, batch_size, static_best, metrics_dir,
            "static", stage_index,
        )
        metrics["static_all_classes"].append(result)
    _save_checkpoint(
        selected_static_model, classes, classes,
        output_dir / "models" / "selected_static.pt",
    )

    for experiment_name, use_memory in (
        ("naive_sequential", False),
        ("memory_replay", True),
    ):
        print(f"\nExperiment: {experiment_name} ({architecture})")
        set_seed(seed)
        model: MarineClassifier | None = None
        seen = []
        best_recall: dict[str, float] = {}
        memory = ExemplarMemory.empty(memory_capacity)
        for stage_index, new_classes in enumerate(stages, start=1):
            previous = seen.copy()
            seen.extend(new_classes)
            ordered_seen = [name for name in classes if name in seen]
            print(f"    stage {stage_index}: introducing {', '.join(new_classes)}")
            stage_records = [
                record for record in train_records
                if any(record.labels[classes.index(name)] for name in new_classes)
            ]
            if model is None:
                model = create_model(
                    len(classes), pretrained=pretrained,
                    train_layer4=train_last_block, architecture=architecture,
                ).to(device)
            train_transform, _ = build_transforms(model.image_size)
            new_dataset = MultiLabelImageDataset(
                stage_records, classes, classes, train_transform, new_classes
            )
            datasets: list[Dataset[Any]] = [new_dataset]
            if use_memory and memory.records:
                replay_dataset = MultiLabelImageDataset(
                    memory.records, classes, classes, train_transform, ordered_seen
                )
                datasets.append(replay_dataset)
                print(f"      replaying {len(memory.records)} exemplars")
            stage_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            _fit(
                model, stage_dataset, device, epochs, batch_size, learning_rate,
                seed + stage_index,
            )
            result = _evaluate_stage(
                model, validation_records, classes, ordered_seen, previous,
                eval_transform, device, batch_size, best_recall, metrics_dir,
                "replay" if use_memory else "naive", stage_index,
            )
            metrics[experiment_name].append(result)
            _save_checkpoint(
                model, classes, classes,
                output_dir / "models" / f"{experiment_name}_stage_{stage_index}.pt",
            )
            if use_memory:
                memory.update(train_records, ordered_seen, classes, seed + stage_index)
                memory.save(output_dir / "memory" / f"stage_{stage_index}.json")
        if model is None:
            raise RuntimeError("No incremental stages were generated")
        if use_memory:
            _save_checkpoint(
                model, classes, classes, output_dir / "models" / "proposed_final.pt"
            )
    return metrics
