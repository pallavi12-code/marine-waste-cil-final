"""Fixed-capacity, class-balanced exemplar memory for rehearsal."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from marine_plastic.dataset import ImageRecord


@dataclass
class ExemplarMemory:
    capacity: int
    records: list[ImageRecord]

    @classmethod
    def empty(cls, capacity: int) -> "ExemplarMemory":
        if capacity < 1:
            raise ValueError("Memory capacity must be at least 1")
        return cls(capacity=capacity, records=[])

    def update(
        self,
        candidates: list[ImageRecord],
        seen_classes: list[str],
        all_classes: list[str],
        seed: int,
    ) -> None:
        if not seen_classes:
            raise ValueError("Cannot select exemplars before any classes are learned")
        quota = max(1, self.capacity // len(seen_classes))
        selected: dict[str, ImageRecord] = {}
        rng = random.Random(seed)
        class_pools: list[list[ImageRecord]] = []
        for class_name in seen_classes:
            class_index = all_classes.index(class_name)
            class_candidates = [
                record for record in candidates if record.labels[class_index] == 1
            ]
            class_candidates.sort(key=lambda record: str(record.path))
            rng.shuffle(class_candidates)
            class_pools.append(class_candidates)

        for class_candidates in class_pools:
            added = 0
            for record in class_candidates:
                if str(record.path) not in selected:
                    selected[str(record.path)] = record
                    added += 1
                if added >= quota:
                    break

        remaining = [
            record
            for record in candidates
            if any(record.labels[all_classes.index(name)] for name in seen_classes)
            and str(record.path) not in selected
        ]
        rng.shuffle(remaining)
        for record in remaining:
            selected[str(record.path)] = record
            if len(selected) >= self.capacity:
                break
        exemplars = list(selected.values())[: self.capacity]
        rng.shuffle(exemplars)
        self.records = exemplars

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "size": len(self.records),
            "exemplars": [
                {"path": str(record.path), "labels": list(record.labels), "split": record.split}
                for record in self.records
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
