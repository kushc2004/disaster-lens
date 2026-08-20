from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .schemas import DisasterSample


@dataclass(frozen=True)
class Split:
    train: tuple[DisasterSample, ...]
    val: tuple[DisasterSample, ...]
    test: tuple[DisasterSample, ...]

    def validate(self) -> None:
        groups = (self.train, self.val, self.test)
        event_sets = [{sample.event_id for sample in group} for group in groups]
        if event_sets[0] & event_sets[1] or event_sets[0] & event_sets[2] or event_sets[1] & event_sets[2]:
            raise ValueError("Event leakage detected between split partitions")
        tile_sets = [{sample.tile_id for sample in group} for group in groups]
        if tile_sets[0] & tile_sets[1] or tile_sets[0] & tile_sets[2] or tile_sets[1] & tile_sets[2]:
            raise ValueError("Tile leakage detected between split partitions")


def _select(samples: Sequence[DisasterSample], events: Iterable[str]) -> tuple[DisasterSample, ...]:
    event_set = set(events)
    return tuple(sample for sample in samples if sample.event_id in event_set)


def event_holdout_split(samples: Sequence[DisasterSample], *, train_events: Iterable[str], val_events: Iterable[str], test_events: Iterable[str]) -> Split:
    requested = [set(train_events), set(val_events), set(test_events)]
    if any(not event_set for event_set in requested):
        raise ValueError("Event-held-out split requires non-empty train, val, and test event sets")
    known = {sample.event_id for sample in samples}
    unknown = set().union(*requested).difference(known)
    if unknown:
        raise ValueError(f"Split references unknown event IDs: {sorted(unknown)}")
    split = Split(_select(samples, requested[0]), _select(samples, requested[1]), _select(samples, requested[2]))
    split.validate()
    return split


def official_split(samples: Sequence[DisasterSample], *, split_root: Path) -> Split:
    split_root = Path(split_root)
    ids: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        path = split_root / f"{name}_set.txt"
        if not path.exists():
            raise FileNotFoundError(f"Official BRIGHT split file missing: {path}")
        ids[name] = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    duplicate_ids = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    if duplicate_ids:
        raise ValueError(f"Official split has duplicate tile IDs: {sorted(duplicate_ids)[:5]}")
    by_id = {sample.tile_id: sample for sample in samples}
    unknown = set().union(*ids.values()).difference(by_id)
    if unknown:
        raise ValueError(f"Official split references tiles absent from manifest: {sorted(unknown)[:5]}")
    # Standard benchmark can share events by design; only tile-level leakage is forbidden.
    return Split(tuple(by_id[x] for x in sorted(ids["train"])), tuple(by_id[x] for x in sorted(ids["val"])), tuple(by_id[x] for x in sorted(ids["test"])))


def standard_tile_split(
    samples: Sequence[DisasterSample],
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    """Build a deterministic event-stratified tile-level standard split.

    Official BRIGHT split files take precedence. This fallback is only for
    official distributions that omit them. Events may occur in several
    partitions, but a tile can occur in exactly one partition.
    """
    if not samples:
        raise ValueError("Cannot split an empty BRIGHT manifest")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("train_fraction and val_fraction must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than one")
    if len({sample.tile_id for sample in samples}) != len(samples):
        raise ValueError("Standard split candidates must have unique tile IDs")

    by_event: dict[str, list[DisasterSample]] = {}
    for sample in samples:
        by_event.setdefault(sample.event_id, []).append(sample)
    partitions: dict[str, list[DisasterSample]] = {"train": [], "val": [], "test": []}
    for event_id, group in sorted(by_event.items()):
        token = hashlib.sha256(f"{seed}:{event_id}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(token, "big"))
        ordered = [group[index] for index in rng.permutation(len(group))]
        count = len(ordered)
        if count >= 3:
            train_count = min(count - 2, max(1, round(count * train_fraction)))
            val_count = min(count - train_count - 1, max(1, round(count * val_fraction)))
        else:
            train_count, val_count = 1, 0
        partitions["train"].extend(ordered[:train_count])
        partitions["val"].extend(ordered[train_count : train_count + val_count])
        partitions["test"].extend(ordered[train_count + val_count :])

    for target in ("val", "test"):
        if not partitions[target]:
            donor = max((name for name in partitions if name != target), key=lambda name: len(partitions[name]))
            if len(partitions[donor]) <= 1:
                raise ValueError("At least three BRIGHT tiles are required for a standard split")
            partitions[target].append(partitions[donor].pop())
    return Split(
        tuple(sorted(partitions["train"], key=lambda item: item.tile_id)),
        tuple(sorted(partitions["val"], key=lambda item: item.tile_id)),
        tuple(sorted(partitions["test"], key=lambda item: item.tile_id)),
    )
