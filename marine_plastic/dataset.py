"""Dataset discovery, annotation parsing, integrity checks, and staged labels."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    labels: tuple[int, ...]
    split: str


def _read_annotations(split_dir: Path) -> tuple[list[str], list[ImageRecord]]:
    annotation_file = split_dir / "_classes.csv"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Required annotation file not found: {annotation_file}")

    with annotation_file.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        if "filename" not in columns or len(columns) < 2:
            raise ValueError(f"{annotation_file} must have filename and class columns")
        classes = [column for column in columns if column != "filename"]
        records: list[ImageRecord] = []
        seen_files: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Empty filename in {annotation_file}:{row_number}")
            if filename in seen_files:
                raise ValueError(f"Duplicate annotation row for {filename} in {annotation_file}")
            seen_files.add(filename)
            image_path = (split_dir / filename).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image referenced by CSV does not exist: {image_path}")
            values: list[int] = []
            for class_name in classes:
                try:
                    value = int(row[class_name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid label for {class_name!r} at {annotation_file}:{row_number}"
                    ) from error
                if value not in (0, 1):
                    raise ValueError(
                        f"Labels must be binary; found {value} at {annotation_file}:{row_number}"
                    )
                values.append(value)
            records.append(ImageRecord(image_path, tuple(values), split_dir.name))
    return classes, records


def split_classes(
    classes: list[str], seed: int, positive_counts: dict[str, int] | None = None
) -> list[list[str]]:
    """Partition labels into three stages with similar positive-label support."""
    if not classes:
        raise ValueError("No classes were found in the training annotations")
    rng = random.Random(seed)
    shuffled = sorted(classes)
    rng.shuffle(shuffled)
    tie_order = {name: index for index, name in enumerate(shuffled)}
    shuffled.sort(
        key=lambda name: (-(positive_counts or {}).get(name, 0), tie_order[name])
    )
    stages: list[list[str]] = [[], [], []]
    stage_support = [0, 0, 0]
    for class_name in shuffled:
        stage_index = min(range(3), key=lambda index: (stage_support[index], len(stages[index])))
        stages[stage_index].append(class_name)
        stage_support[stage_index] += (positive_counts or {}).get(class_name, 0)
    return [stage for stage in stages if stage]


def split_records(
    records: list[ImageRecord],
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, list[ImageRecord]]:
    """Iteratively stratify multilabel records into train, validation, and test."""
    if not records:
        raise ValueError("Cannot split an empty dataset")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("Provide three positive split fractions")
    if not abs(sum(fractions) - 1.0) < 1e-8:
        raise ValueError("Split fractions must sum to 1")

    rng = random.Random(seed)
    sample_count = len(records)
    requested = [fraction * sample_count for fraction in fractions]
    capacities = [int(value) for value in requested]
    for index in sorted(
        range(3), key=lambda item: requested[item] - capacities[item], reverse=True
    )[: sample_count - sum(capacities)]:
        capacities[index] += 1

    labels = [list(record.labels) for record in records]
    label_count = len(labels[0])
    if any(len(row) != label_count for row in labels):
        raise ValueError("All records must use the same global label mapping")
    remaining = set(range(sample_count))
    assignments: list[list[int]] = [[], [], []]
    desired = [
        [sum(row[label] for row in labels) * fraction for label in range(label_count)]
        for fraction in fractions
    ]
    tie_break = [rng.random() for _ in range(sample_count)]

    while remaining:
        positive_counts = [
            sum(labels[index][label] for index in remaining)
            for label in range(label_count)
        ]
        active_labels = [label for label, count in enumerate(positive_counts) if count]
        if active_labels:
            rare_label = min(active_labels, key=lambda label: positive_counts[label])
            sample_index = min(
                (index for index in remaining if labels[index][rare_label]),
                key=lambda index: tie_break[index],
            )
            available = [
                split for split in range(3)
                if len(assignments[split]) < capacities[split]
            ]
            selected_split = max(
                available,
                key=lambda split: (
                    desired[split][rare_label],
                    capacities[split] - len(assignments[split]),
                    rng.random(),
                ),
            )
        else:
            sample_index = min(remaining, key=lambda index: tie_break[index])
            available = [
                split for split in range(3)
                if len(assignments[split]) < capacities[split]
            ]
            selected_split = max(
                available,
                key=lambda split: (
                    capacities[split] - len(assignments[split]),
                    rng.random(),
                ),
            )
        sample_labels = [label for label, value in enumerate(labels[sample_index]) if value]
        assignments[selected_split].append(sample_index)
        remaining.remove(sample_index)
        for label in sample_labels:
            desired[selected_split][label] -= 1.0

    names = ("train", "validation", "test")
    result = {
        name: [records[index] for index in sorted(indices)]
        for name, indices in zip(names, assignments)
    }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_dataset(data_root: Path, seed: int) -> dict[str, Any]:
    """Validate the dataset and write an auditable report before model training."""
    data_root = data_root.expanduser().resolve()
    train_classes, train_records = _read_annotations(data_root / "train")
    valid_classes, valid_records = _read_annotations(data_root / "valid")
    if train_classes != valid_classes:
        raise ValueError("Training and validation CSVs must have the same classes in the same order")

    corrupt_images: list[str] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    annotated_paths = {str(record.path.resolve()) for record in train_records + valid_records}
    all_image_paths = sorted(
        path
        for split in ("train", "valid")
        for path in (data_root / split).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    unannotated_images = [str(path) for path in all_image_paths if str(path.resolve()) not in annotated_paths]
    for image_path in all_image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            corrupt_images.append(f"{image_path}: {error}")
            continue
        hashes[_sha256(image_path)].append(str(image_path))

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_path_set = {path for group in duplicates for path in group}
    corrupt_path_set = {entry.split(": ", maxsplit=1)[0] for entry in corrupt_images}
    all_records = [
        record for record in train_records + valid_records
        if str(record.path) not in duplicate_path_set | corrupt_path_set
    ]
    data_splits = split_records(all_records, (0.7, 0.15, 0.15), seed)
    all_class_counts = {
        class_name: sum(record.labels[index] for record in all_records)
        for index, class_name in enumerate(train_classes)
    }
    stages = split_classes(train_classes, seed, all_class_counts)
    distribution = {
        split: {
            class_name: sum(record.labels[index] for record in records)
            for index, class_name in enumerate(train_classes)
        }
        for split, records in data_splits.items()
    }
    reports = {
        "dataset_root": str(data_root),
        "splits": {
            split: {
                "images": len(records),
                "label_combinations": len({record.labels for record in records}),
            }
            for split, records in data_splits.items()
        },
        "classes": train_classes,
        "class_distribution": distribution,
        "split_strategy": {
            "algorithm": "iterative multilabel stratification",
            "fractions": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "seed": seed,
            "source_files": ["train/_classes.csv", "valid/_classes.csv"],
        },
        "incremental_stages": {f"stage_{index + 1}": stage for index, stage in enumerate(stages)},
        "corrupt_images": corrupt_images,
        "corrupt_image_count": len(corrupt_images),
        "unannotated_images": unannotated_images,
        "unannotated_image_count": len(unannotated_images),
        "exact_duplicate_groups": duplicates,
        "exact_duplicate_group_count": len(duplicates),
        "label_format": "multi-label binary vectors from _classes.csv",
    }
    return reports


def write_split_manifest(
    splits: dict[str, list[ImageRecord]], output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        split: [
            {"path": str(record.path), "labels": list(record.labels)}
            for record in records
        ]
        for split, records in splits.items()
    }
    manifest_path = output_dir / "data_splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def write_dataset_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def print_dataset_report(report: dict[str, Any]) -> None:
    print("\nDataset inspection (completed before training)")
    print(f"  Root: {report['dataset_root']}")
    for split, details in report["splits"].items():
        print(
            f"  {split}: {details['images']} images, "
            f"{details['label_combinations']} distinct multi-label combinations"
        )
    print("  Positive labels by split:")
    for split, counts in report["class_distribution"].items():
        print(f"    {split}: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    print(f"  Corrupt images: {report['corrupt_image_count']}")
    print(f"  Exact duplicate groups: {report['exact_duplicate_group_count']}")
    print(f"  Images without CSV labels: {report['unannotated_image_count']}")
    print("  Automatic class stages:")
    for stage, classes in report["incremental_stages"].items():
        print(f"    {stage}: {', '.join(classes)}")


def load_records(data_root: Path, split: str, classes: list[str]) -> list[ImageRecord]:
    found_classes, records = _read_annotations(data_root / split)
    if found_classes != classes:
        raise ValueError(f"{split} class columns do not match the inspected dataset report")
    return records


def records_without_corruption(records: list[ImageRecord]) -> list[ImageRecord]:
    usable: list[ImageRecord] = []
    for record in records:
        try:
            with Image.open(record.path) as image:
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
        usable.append(record)
    return usable


def duplicate_paths(report: dict[str, Any]) -> set[str]:
    """Return paths in duplicate groups so train/validation leakage can be avoided."""
    return {path for group in report["exact_duplicate_groups"] for path in group}
