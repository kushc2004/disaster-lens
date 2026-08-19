"""Small, dependency-light configuration loader for the M0--M1 tooling."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_VALUE = re.compile(r"^\$\{oc\.env:([^,}]+),?([^}]*)\}$")


def _resolve(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    if isinstance(value, str) and (match := _ENV_VALUE.match(value)):
        name, default = match.groups()
        return os.environ.get(name, default)
    return value


def load_yaml(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or []:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key in {"data", "split"}:
            continue
        cursor = config
        pieces = key.split(".")
        for piece in pieces[:-1]:
            cursor = cursor.setdefault(piece, {})
        cursor[pieces[-1]] = yaml.safe_load(value)
    config = _resolve(config)
    # Accept the Hydra-style dataset.root override shown in the implementation
    # specification while retaining a compact standalone YAML configuration.
    if isinstance(config.get("dataset"), dict) and "root" in config["dataset"]:
        config["root"] = config["dataset"]["root"]
    return config
