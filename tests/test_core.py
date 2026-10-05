import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from marine_plastic.dataset import ImageRecord, _read_annotations, split_classes, split_records
from marine_plastic.evaluation import add_retention_metrics, evaluate_model
from marine_plastic.memory_replay import ExemplarMemory


class IncrementalLearningCoreTests(unittest.TestCase):
    def test_annotations_preserve_dataset_classes_and_column_order(self):
        with TemporaryDirectory() as directory:
            split_dir = Path(directory) / "train"
            split_dir.mkdir()
            Image.new("RGB", (8, 8)).save(split_dir / "sample.png")
            with (split_dir / "_classes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.writer(target)
                writer.writerow(
                    ["filename", "can", "foam", "plastic", "plastic bottle", "unknow"]
                )
                writer.writerow(["sample.png", 1, 0, 1, 0, 1])

            classes, records = _read_annotations(split_dir)

        self.assertEqual(classes, ["can", "foam", "plastic", "plastic bottle", "unknow"])
        self.assertEqual(records[0].labels, (1, 0, 1, 0, 1))

    def test_class_split_is_deterministic_and_complete(self):
        classes = ["can", "foam", "plastic", "plastic bottle", "unknow"]
        stages = split_classes(classes, seed=42)

        self.assertEqual(stages, split_classes(classes, seed=42))
        self.assertEqual(sorted(classes), sorted(name for stage in stages for name in stage))
        self.assertEqual(3, len(stages))

    def test_multilabel_splits_are_reproducible_and_preserve_rare_labels(self):
        records = [
            ImageRecord(
                Path(f"sample_{index}.jpg"),
                (int(index < 30), int(index % 2 == 0), int(index % 3 == 0)),
                "train",
            )
            for index in range(100)
        ]
        first = split_records(records, (0.7, 0.15, 0.15), seed=42)
        second = split_records(records, (0.7, 0.15, 0.15), seed=42)

        self.assertEqual([len(first[name]) for name in ("train", "validation", "test")], [70, 15, 15])
        self.assertEqual(
            [[record.path for record in first[name]] for name in first],
            [[record.path for record in second[name]] for name in second],
        )
        for split_name in ("validation", "test"):
            self.assertTrue(any(record.labels[0] for record in first[split_name]))

    def test_memory_respects_capacity_and_represents_seen_classes(self):
        classes = ["can", "foam", "plastic"]
        records = [
            ImageRecord(Path(f"sample_{index}.jpg"), tuple(int(i == index % 3) for i in range(3)), "train")
            for index in range(15)
        ]
        memory = ExemplarMemory.empty(capacity=6)
        memory.update(records, classes, classes, seed=42)

        self.assertEqual(6, len(memory.records))
        represented = {
            class_index
            for record in memory.records
            for class_index, value in enumerate(record.labels)
            if value
        }
        self.assertEqual({0, 1, 2}, represented)

    def test_retention_and_forgetting_use_old_classes_only(self):
        metrics = {"per_class_recall": {"can": 0.4, "foam": 0.8, "plastic": 0.2}}
        best_recall = {"can": 0.8, "foam": 0.8}

        updated = add_retention_metrics(
            metrics,
            seen_classes=["can", "foam", "plastic"],
            previous_classes=["can", "foam"],
            best_recall=best_recall,
        )

        self.assertAlmostEqual(0.6, metrics["old_class_retention"])
        self.assertAlmostEqual(0.2, metrics["forgetting"])
        self.assertEqual(0.2, updated["plastic"])
        self.assertLessEqual(metrics["old_class_retention"], 1.0)

    def test_evaluation_uses_global_indices_and_reports_all_active_labels(self):
        class FixedModel(torch.nn.Module):
            def forward(self, images):
                logits = torch.tensor([4.0, -4.0, 4.0], device=images.device)
                return logits.expand(images.shape[0], -1)

        with TemporaryDirectory() as directory:
            records = []
            for index, labels in enumerate(((1, 0, 1), (0, 1, 0))):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", (8, 8), (index * 30, 20, 20)).save(path)
                records.append(ImageRecord(path, labels, "test"))

            transform = lambda image: torch.zeros(3, 8, 8)
            metrics, truth, predictions = evaluate_model(
                FixedModel(),
                records,
                ["a", "b", "c"],
                ["c", "a"],
                transform,
                torch.device("cpu"),
                batch_size=2,
                model_classes=["a", "b", "c"],
            )
            single_class_metrics, single_truth, single_predictions = evaluate_model(
                FixedModel(),
                records,
                ["a", "b", "c"],
                ["c"],
                transform,
                torch.device("cpu"),
                batch_size=2,
                model_classes=["a", "b", "c"],
            )

        self.assertEqual(metrics["classes"], ["c", "a"])
        self.assertEqual(list(metrics["confusion_matrices"]), ["c", "a"])
        self.assertEqual(truth.shape, (2, 2))
        self.assertEqual(predictions.shape, (2, 2))
        self.assertAlmostEqual(0.5, metrics["accuracy"])
        self.assertAlmostEqual(0.5, metrics["exact_match_accuracy"])
        self.assertEqual(single_truth.shape, (2, 1))
        self.assertEqual(single_predictions.shape, (2, 1))
        self.assertAlmostEqual(0.5, single_class_metrics["accuracy"])
        self.assertAlmostEqual(0.5, single_class_metrics["precision"])
        self.assertAlmostEqual(1.0, single_class_metrics["recall"])
        self.assertAlmostEqual(2.0 / 3.0, single_class_metrics["macro_f1"])
        self.assertEqual(
            single_class_metrics["confusion_matrices"]["c"], [[0, 1], [0, 1]]
        )


if __name__ == "__main__":
    unittest.main()
