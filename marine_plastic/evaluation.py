"""Global-class multilabel metrics, retention, forgetting, and reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from marine_plastic.dataset import ImageRecord
from marine_plastic.preprocessing import MultiLabelImageDataset


def evaluate_model(
    model: torch.nn.Module,
    records: list[ImageRecord],
    all_classes: list[str],
    active_classes: list[str],
    transform: Any,
    device: torch.device,
    batch_size: int,
    model_classes: list[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate only seen labels, in the order of their single global mapping."""
    if not active_classes:
        raise ValueError("At least one active class is required for evaluation")
    if len(set(all_classes)) != len(all_classes):
        raise ValueError("The global class mapping contains duplicate class names")
    model_classes = model_classes or all_classes
    if any(name not in all_classes for name in active_classes):
        raise ValueError("Active classes must come from the global class mapping")
    if any(name not in model_classes for name in active_classes):
        raise ValueError("The model is missing an output for an active class")

    active_indices = [all_classes.index(name) for name in active_classes]
    model_indices = [model_classes.index(name) for name in active_classes]
    dataset = MultiLabelImageDataset(
        records, all_classes, all_classes, transform, mask_classes=active_classes
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets, _ in loader:
            logits = model(images.to(device))
            probability_batches.append(torch.sigmoid(logits[:, model_indices]).cpu().numpy())
            true_batches.append(targets[:, active_indices].numpy())
    y_true = np.concatenate(true_batches, axis=0).astype(np.int32)
    probabilities = np.concatenate(probability_batches, axis=0)
    predictions = (probabilities >= 0.5).astype(np.int32)
    per_class_precision = np.asarray(
        [
            precision_score(y_true[:, index], predictions[:, index], zero_division=0)
            for index in range(len(active_classes))
        ],
        dtype=float,
    )
    per_class_recall = np.asarray(
        [
            recall_score(y_true[:, index], predictions[:, index], zero_division=0)
            for index in range(len(active_classes))
        ],
        dtype=float,
    )
    per_class_f1 = np.asarray(
        [
            f1_score(y_true[:, index], predictions[:, index], zero_division=0)
            for index in range(len(active_classes))
        ],
        dtype=float,
    )
    matrices = [
        confusion_matrix(y_true[:, index], predictions[:, index], labels=[0, 1])
        for index in range(len(active_classes))
    ]

    metrics: dict[str, Any] = {
        "samples": int(y_true.shape[0]),
        "classes": list(active_classes),
        "global_class_mapping": list(all_classes),
        "threshold": 0.5,
        "accuracy": float(np.mean(y_true == predictions)),
        "exact_match_accuracy": float(np.all(y_true == predictions, axis=1).mean()),
        "precision": float(np.mean(per_class_precision)),
        "recall": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "f1": float(np.mean(per_class_f1)),
        "per_class_precision": {
            name: float(value) for name, value in zip(active_classes, per_class_precision)
        },
        "per_class_recall": {
            name: float(value) for name, value in zip(active_classes, per_class_recall)
        },
        "per_class_f1": {
            name: float(value) for name, value in zip(active_classes, per_class_f1)
        },
        "confusion_matrices": {
            name: matrix.tolist() for name, matrix in zip(active_classes, matrices)
        },
    }
    return metrics, y_true, predictions


def add_retention_metrics(
    metrics: dict[str, Any],
    seen_classes: list[str],
    previous_classes: list[str],
    best_recall: dict[str, float],
) -> dict[str, float]:
    """Record old-label macro recall and mean absolute forgetting (both [0, 1])."""
    class_recall = metrics["per_class_recall"]
    old_classes = [name for name in previous_classes if name in class_recall]
    if old_classes:
        metrics["old_class_retention"] = float(
            np.mean([class_recall[name] for name in old_classes])
        )
        metrics["forgetting"] = float(
            np.mean(
                [max(0.0, best_recall[name] - class_recall[name]) for name in old_classes]
            )
        )
    else:
        metrics["old_class_retention"] = None
        metrics["forgetting"] = 0.0

    for name in seen_classes:
        if name in class_recall:
            best_recall[name] = max(best_recall.get(name, class_recall[name]), class_recall[name])
    metrics["old_classes"] = old_classes
    return best_recall


def save_confusion_matrices(
    y_true: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
    output_dir: Path,
    experiment: str,
    stage_number: int,
) -> None:
    if y_true.shape != predictions.shape or y_true.shape[1] != len(classes):
        raise ValueError("Confusion matrix labels must match the evaluated class columns")
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = [
        confusion_matrix(y_true[:, index], predictions[:, index], labels=[0, 1])
        for index in range(len(classes))
    ]
    csv_path = output_dir / f"{experiment}_stage_{stage_number}_confusion.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(
            ["class", "true_negative", "false_positive", "false_negative", "true_positive"]
        )
        for name, matrix in zip(classes, matrices):
            tn, fp, fn, tp = matrix.ravel()
            writer.writerow([name, int(tn), int(fp), int(fn), int(tp)])

    figure, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4), squeeze=False)
    for axis, name, matrix in zip(axes[0], classes, matrices):
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / f"{experiment}_stage_{stage_number}_confusion.png", dpi=140)
    plt.close(figure)


def save_metrics(metrics_by_experiment: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_by_experiment, indent=2), encoding="utf-8"
    )
    fields = [
        "experiment", "architecture", "stage", "accuracy", "exact_match_accuracy",
        "precision", "recall", "macro_f1", "old_class_retention", "forgetting", "samples",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for experiment, stages in metrics_by_experiment.items():
            for metric in stages:
                writer.writerow(
                    {
                        "experiment": experiment,
                        **{field: metric.get(field) for field in fields[1:]},
                    }
                )


def save_comparison_summary(
    metrics_by_experiment: dict[str, list[dict[str, Any]]], output_dir: Path
) -> dict[str, Any]:
    naive = metrics_by_experiment["naive_sequential"][-1]
    replay = metrics_by_experiment["memory_replay"][-1]
    summary = {
        "final_stage": int(replay["stage"]),
        "naive_forgetting": float(naive["forgetting"]),
        "memory_replay_forgetting": float(replay["forgetting"]),
        "forgetting_delta_replay_minus_naive": float(
            replay["forgetting"] - naive["forgetting"]
        ),
        "replay_reduces_forgetting": bool(replay["forgetting"] < naive["forgetting"]),
        "naive_old_class_retention": naive["old_class_retention"],
        "memory_replay_old_class_retention": replay["old_class_retention"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plot_metric_graphs(
    metrics_by_experiment: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series = (
        ("accuracy", "Label-wise accuracy", "accuracy_by_stage.png"),
        ("macro_f1", "Macro F1-score", "f1_by_stage.png"),
        ("old_class_retention", "Old-class macro recall", "retention_by_stage.png"),
        ("forgetting", "Mean forgetting", "forgetting_by_stage.png"),
    )
    stage_count = max(len(values) for values in metrics_by_experiment.values())
    for key, title, filename in series:
        figure, axis = plt.subplots(figsize=(8, 5))
        for experiment, stages in metrics_by_experiment.items():
            values = [
                np.nan if stage.get(key) is None else float(stage[key])
                for stage in stages
            ]
            axis.plot(range(1, len(values) + 1), values, marker="o", label=experiment)
        axis.set(title=title, xlabel="Incremental stage", ylabel=title)
        axis.set_xticks(range(1, stage_count + 1))
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)
