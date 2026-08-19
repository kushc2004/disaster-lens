from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class LabelValidationError(ValueError):
    """Raised when a dataset mask does not match its declared schema."""


@dataclass(frozen=True)
class LabelSchema:
    name: str
    classes: dict[int, str]
    ignore_ids: frozenset[int] = frozenset({255})

    def validate(self, values: np.ndarray | Iterable[int]) -> set[int]:
        observed = {int(value) for value in np.unique(values)}
        unknown = observed.difference(self.classes).difference(self.ignore_ids)
        if unknown:
            raise LabelValidationError(
                f"{self.name} observed unknown label IDs {sorted(unknown)}; "
                f"allowed={sorted(self.classes)}, ignore={sorted(self.ignore_ids)}"
            )
        return observed


BRIGHT_V1 = LabelSchema(
    name="bright_v1",
    classes={0: "background", 1: "intact", 2: "damaged", 3: "destroyed"},
)


@dataclass(frozen=True)
class DisasterSample:
    event_id: str
    disaster_type: str
    tile_id: str
    pre_optical: Path | None = None
    post_optical: Path | None = None
    pre_sar: Path | None = None
    post_sar: Path | None = None
    label: Path | None = None
    crs: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self, *, root: Path | None = None) -> dict[str, Any]:
        record = asdict(self)
        for key in ("pre_optical", "post_optical", "pre_sar", "post_sar", "label"):
            path = getattr(self, key)
            if path is not None:
                record[key] = str(path.relative_to(root) if root and path.is_relative_to(root) else path)
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any], *, root: Path | None = None) -> "DisasterSample":
        data = dict(record)
        for key in ("pre_optical", "post_optical", "pre_sar", "post_sar", "label"):
            if data.get(key) is not None:
                path = Path(data[key])
                data[key] = root / path if root and not path.is_absolute() else path
        if data.get("bounds") is not None:
            data["bounds"] = tuple(data["bounds"])
        return cls(**data)

