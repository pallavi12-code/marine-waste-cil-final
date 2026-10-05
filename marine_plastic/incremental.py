"""Small helpers for stage-wise class-incremental protocols."""

from __future__ import annotations

from marine_plastic.dataset import ImageRecord


def samples_for_classes(
    records: list[ImageRecord], classes: list[str], selected_classes: list[str]
) -> list[ImageRecord]:
    indices = [classes.index(name) for name in selected_classes]
    return [record for record in records if any(record.labels[index] for index in indices)]

