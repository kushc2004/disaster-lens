"""Event-balanced sampling for cross-disaster training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from .schemas import DisasterSample


class EventBalancedSampler(Sampler[int]):
    """Sample events uniformly, then a tile uniformly within that event.

    ``samples_per_epoch`` caps dominant events without discarding their tiles
    permanently.  Calling :meth:`set_epoch` gives deterministic but different
    draws each epoch.
    """

    def __init__(self, samples: Sequence[DisasterSample], *, samples_per_epoch: int | None = None, seed: int = 42) -> None:
        if not samples:
            raise ValueError("EventBalancedSampler requires samples")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            grouped[sample.event_id].append(index)
        self.groups = dict(sorted(grouped.items()))
        self.events = tuple(self.groups)
        self.samples_per_epoch = samples_per_epoch or len(samples)
        if self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.samples_per_epoch):
            event = self.events[rng.randrange(len(self.events))]
            indices = self.groups[event]
            yield indices[rng.randrange(len(indices))]
