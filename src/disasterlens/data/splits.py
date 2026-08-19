from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

