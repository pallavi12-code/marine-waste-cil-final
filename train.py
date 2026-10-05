"""Create balanced dataset splits, choose a static baseline, then run incremental experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from marine_plastic.dataset import (
    duplicate_paths,
    inspect_dataset,
    load_records,
    print_dataset_report,
    records_without_corruption,
    split_records,
    write_dataset_report,
    write_split_manifest,
)
from marine_plastic.evaluation import (
    plot_metric_graphs,
    save_comparison_summary,
    save_confusion_matrices,
    save_metrics,
)
from marine_plastic.explainability import save_grad_cam
from marine_plastic.preprocessing import build_transforms
from marine_plastic.training import (
    ARCHITECTURES,
    choose_device,
    load_checkpoint_model,
    run_incremental_experiments,
    set_seed,
    train_static_baselines,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "marine debris images dataset",
        help="Folder containing train/ and valid/ with _classes.csv files",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--phase",
        choices=("all", "baseline", "incremental"),
        default="all",
        help="Train both static architectures first, then run the selected incremental experiments",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per training phase")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--memory-size", type=int, default=100, help="Maximum exemplar count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-last-block",
        action="store_true",
        help="Train only the classifier head; by default the final feature block is fine-tuned",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize backbones randomly instead of using ImageNet weights",
    )
    parsed = parser.parse_args(arguments)
    if parsed.epochs < 1 or parsed.batch_size < 1 or parsed.memory_size < 1:
        parser.error("epochs, batch-size, and memory-size must be positive")
    return parsed


def _load_data(args: argparse.Namespace) -> tuple[dict, dict, list, list, list]:
    report = inspect_dataset(args.data_root, args.seed)
    print_dataset_report(report)
    output_dir = args.output_dir.expanduser().resolve()
    report_path = write_dataset_report(report, output_dir)
    print(f"  Saved report: {report_path}")

    classes = report["classes"]
    duplicates = duplicate_paths(report)
    records = (
        load_records(args.data_root, "train", classes)
        + load_records(args.data_root, "valid", classes)
    )
    records = records_without_corruption(records)
    records = [record for record in records if str(record.path) not in duplicates]
    splits = split_records(records, (0.7, 0.15, 0.15), args.seed)
    write_split_manifest(splits, output_dir)
    for split_name, split_records_list in splits.items():
        counts = {
            name: sum(record.labels[index] for record in split_records_list)
            for index, name in enumerate(classes)
        }
        print(
            f"  {split_name}: {len(split_records_list)} images; "
            + ", ".join(f"{name}={count}" for name, count in counts.items())
        )
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("Integrity filtering left an empty train/validation/test split")
    return report, splits, classes, list(report["incremental_stages"].values()), output_dir


def _evaluate_selected_on_test(
    model: torch.nn.Module,
    architecture: str,
    classes: list[str],
    test_records: list,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    experiment_name: str | None = None,
) -> dict:
    from marine_plastic.evaluation import evaluate_model

    _, evaluation_transform = build_transforms(model.image_size)
    metrics, truth, predictions = evaluate_model(
        model, test_records, classes, classes, evaluation_transform, device, batch_size,
        model_classes=classes,
    )
    metrics["architecture"] = architecture
    metrics["split"] = "test"
    test_dir = output_dir / "confusion_matrices"
    save_confusion_matrices(
        truth,
        predictions,
        classes,
        test_dir,
        experiment_name or f"test_{architecture}",
        3,
    )
    return metrics


def _save_gradcam_examples(
    model: torch.nn.Module,
    classes: list[str],
    validation_records: list,
    output_dir: Path,
    device: torch.device,
) -> None:
    for class_index, class_name in enumerate(classes):
        example = next(
            (record for record in validation_records if record.labels[class_index]),
            None,
        )
        if example is None:
            raise ValueError(f"Validation split contains no positive examples of {class_name!r}")
        safe_name = "".join(char if char.isalnum() else "_" for char in class_name)
        save_grad_cam(
            model,
            example.path,
            class_index,
            device,
            output_dir / "gradcam" / f"{safe_name}.png",
        )


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_arguments(arguments)
    args.data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    set_seed(args.seed)
    report, splits, classes, stages, output_dir = _load_data(args)
    device = choose_device()
    print(f"\nUsing device: {device}")

    comparison_path = output_dir / "baseline_comparison.json"
    if args.phase in ("all", "baseline"):
        selected_path, selected_architecture, _ = train_static_baselines(
            splits["train"],
            splits["validation"],
            classes,
            output_dir,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            pretrained=not args.no_pretrained,
            train_last_block=not args.freeze_last_block,
            learning_rate=args.learning_rate,
            architectures=ARCHITECTURES,
        )
        selected_model = load_checkpoint_model(
            selected_path, selected_architecture, classes, device
        )
        test_metrics = _evaluate_selected_on_test(
            selected_model, selected_architecture, classes, splits["test"],
            output_dir, device, args.batch_size,
        )
        (output_dir / "baseline_test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
        _save_gradcam_examples(
            selected_model, classes, splits["validation"], output_dir, device
        )
        print(
            f"Static baseline test: accuracy={test_metrics['accuracy']:.4f}, "
            f"macro-F1={test_metrics['macro_f1']:.4f}"
        )
        if args.phase == "baseline":
            print(
                "\nStatic baseline phase complete. Review baseline_comparison.json "
                "and baseline_test_metrics.json before starting --phase incremental."
            )
            return
    else:
        if not comparison_path.is_file():
            raise FileNotFoundError(
                "Static baselines are required first. Run `python train.py --phase baseline`."
            )
        summary = json.loads(comparison_path.read_text(encoding="utf-8"))
        selected_architecture = summary["selected_architecture"]
        selected_path = output_dir / "models" / f"{selected_architecture}_static.pt"
        selected_model = load_checkpoint_model(
            selected_path, selected_architecture, classes, device
        )
        if not (output_dir / "baseline_test_metrics.json").is_file():
            raise FileNotFoundError(
                "Held-out static baseline evaluation is missing; rerun --phase baseline."
            )
        del summary

    if args.phase == "baseline":
        return

    metrics = run_incremental_experiments(
        splits["train"],
        splits["validation"],
        classes,
        stages,
        selected_model,
        selected_architecture,
        output_dir,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        memory_capacity=args.memory_size,
        seed=args.seed,
        pretrained=not args.no_pretrained,
        train_last_block=not args.freeze_last_block,
        learning_rate=args.learning_rate,
    )

    test_metrics_by_method = {
        "static_all_classes": _evaluate_selected_on_test(
            selected_model, selected_architecture, classes, splits["test"],
            output_dir, device, args.batch_size, "test_static_all_classes",
        )
    }
    for method, checkpoint_name in (
        ("naive_sequential", "naive_sequential_stage_3.pt"),
        ("memory_replay", "proposed_final.pt"),
    ):
        model = load_checkpoint_model(
            output_dir / "models" / checkpoint_name,
            selected_architecture, classes, device,
        )
        test_metrics_by_method[method] = _evaluate_selected_on_test(
            model, selected_architecture, classes, splits["test"],
            output_dir, device, args.batch_size, f"test_{method}",
        )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics_by_method, indent=2), encoding="utf-8"
    )

    save_metrics(metrics, output_dir)
    comparison = save_comparison_summary(metrics, output_dir)
    plot_metric_graphs(metrics, output_dir / "graphs")
    print(
        "\nFinal validation comparison: "
        f"forgetting naive={comparison['naive_forgetting']:.3f}, "
        f"replay={comparison['memory_replay_forgetting']:.3f}; "
        f"old-class recall naive={comparison['naive_old_class_retention']}, "
        f"replay={comparison['memory_replay_old_class_retention']}"
    )
    print(
        f"All stages complete using {selected_architecture}. Validation selected the model; "
        f"held-out test metrics saved under {output_dir}."
    )


if __name__ == "__main__":
    main()
