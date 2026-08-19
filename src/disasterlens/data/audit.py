"""Dataset inspection and audit outputs for BRIGHT."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .manifest import build_bright_manifest
from .schemas import DisasterSample, LabelSchema


def _summary(path: Path) -> dict[str, object]:
    import rasterio

    with rasterio.open(path) as source:
        return {"shape": (source.height, source.width), "bands": source.count,
                "crs": str(source.crs) if source.crs else None,
                "bounds": tuple(round(value, 6) for value in source.bounds)}


def _stats(channels: int) -> dict[str, np.ndarray]:
    return {"count": np.zeros(channels), "sum": np.zeros(channels), "sum_sq": np.zeros(channels),
            "min": np.full(channels, np.inf), "max": np.full(channels, -np.inf)}


def _update_stats(path: Path, channels: int, totals: dict[str, np.ndarray]) -> None:
    """Read all pixels in blocks so the audit can also detect unreadable rasters."""
    import rasterio

    with rasterio.open(path) as source:
        if source.count < channels:
            raise ValueError(f"{path} has {source.count} bands; expected at least {channels}")
        for _, window in source.block_windows(1):
            values = source.read(indexes=list(range(1, channels + 1)), window=window).astype(np.float64)
            flat = values.reshape(channels, -1)
            totals["count"] += flat.shape[1]
            totals["sum"] += flat.sum(axis=1)
            totals["sum_sq"] += np.square(flat).sum(axis=1)
            totals["min"] = np.minimum(totals["min"], flat.min(axis=1))
            totals["max"] = np.maximum(totals["max"], flat.max(axis=1))


def _serialise_stats(totals: dict[str, np.ndarray]) -> dict[str, list[float] | list[int]]:
    mean = totals["sum"] / totals["count"]
    variance = np.maximum(totals["sum_sq"] / totals["count"] - np.square(mean), 1e-12)
    return {"mean": [float(value) for value in mean], "std": [float(value) for value in np.sqrt(variance)],
            "min": [float(value) for value in totals["min"]], "max": [float(value) for value in totals["max"]],
            "count": [int(value) for value in totals["count"]]}


def audit_bright(root: str | Path, schema: LabelSchema, output_dir: str | Path,
                 normalization_path: str | Path | None = None) -> list[DisasterSample]:
    """Audit BRIGHT files and produce the M1 report and required figures."""
    import matplotlib.pyplot as plt
    import rasterio

    samples = build_bright_manifest(Path(root))
    output_dir = Path(output_dir)
    figures, reports = output_dir / "figures", output_dir / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    events: Counter[str] = Counter()
    classes: Counter[int] = Counter()
    alignment_issues: list[str] = []
    modalities: Counter[str] = Counter()
    optical_stats, sar_stats = _stats(3), _stats(1)

    for sample in samples:
        assert sample.pre_optical and sample.post_sar and sample.label
        pre, post, target = _summary(sample.pre_optical), _summary(sample.post_sar), _summary(sample.label)
        events[sample.event_id] += 1
        modalities[f"pre-event: {pre['bands']} band"] += 1
        modalities[f"post-event: {post['bands']} band"] += 1
        _update_stats(sample.pre_optical, 3, optical_stats)
        _update_stats(sample.post_sar, 1, sar_stats)
        if pre["shape"] != post["shape"] or pre["shape"] != target["shape"]:
            alignment_issues.append(f"{sample.tile_id}: pixel dimensions differ")
        if pre["crs"] != post["crs"] or pre["crs"] != target["crs"]:
            alignment_issues.append(f"{sample.tile_id}: CRS differs")
        if pre["bounds"] != post["bounds"] or pre["bounds"] != target["bounds"]:
            alignment_issues.append(f"{sample.tile_id}: bounds differ")
        with rasterio.open(sample.label) as source:
            mask = source.read(1)
        schema.validate(mask)
        values, counts = np.unique(mask, return_counts=True)
        classes.update({int(value): int(count) for value, count in zip(values, counts)})
    if alignment_issues:
        raise ValueError("BRIGHT modality alignment audit failed:\n" + "\n".join(alignment_issues))
    normalization = {"pre_optical": _serialise_stats(optical_stats), "post_sar": _serialise_stats(sar_stats)}
    if normalization_path:
        target_path = Path(normalization_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")

    names, counts = zip(*sorted(events.items()))
    figure, axis = plt.subplots(figsize=(max(6, len(names) * .65), 4))
    axis.bar(names, counts, color="#2979b8"); axis.set_ylabel("tiles"); axis.set_title("BRIGHT tiles per event")
    axis.tick_params(axis="x", rotation=45, labelsize=8); figure.tight_layout()
    figure.savefig(figures / "event_distribution.png", dpi=160); plt.close(figure)
    ids = sorted(classes)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([schema.classes.get(code, f"unknown:{code}") for code in ids], [classes[code] for code in ids], color="#e57c23")
    axis.set_ylabel("pixels"); axis.set_title("BRIGHT damage-mask distribution"); figure.tight_layout()
    figure.savefig(figures / "class_distribution.png", dpi=160); plt.close(figure)

    example = samples[0]
    assert example.pre_optical and example.post_sar and example.label
    with rasterio.open(example.pre_optical) as source: pre_image = source.read()[:3]
    with rasterio.open(example.post_sar) as source: post_image = source.read(1)
    with rasterio.open(example.label) as source: mask = source.read(1)
    display = np.moveaxis(pre_image, 0, -1).astype(np.float32)
    display = np.clip(display / (np.percentile(display, 99) or 1), 0, 1)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title, kwargs in zip(axes, (display, post_image, mask), ("pre-event optical", "post-event SAR", "damage mask"), ({}, {"cmap": "gray"}, {"cmap": "viridis"})):
        axis.imshow(image, **kwargs); axis.set_title(title); axis.set_axis_off()
    figure.suptitle(example.tile_id); figure.tight_layout()
    figure.savefig(figures / "modality_examples.png", dpi=160); plt.close(figure)

    class_rows = "\n".join(f"| {code} | {schema.classes.get(code, 'ignore')} | {classes[code]:,} |" for code in ids)
    modality_rows = "\n".join(f"| {name} | {count:,} |" for name, count in sorted(modalities.items()))
    event_rows = "\n".join(f"- `{event}`: {count:,} tiles" for event, count in sorted(events.items()))
    stat_rows = "\n".join(
        f"| {name} | {', '.join(f'{value:.6g}' for value in values['mean'])} | {', '.join(f'{value:.6g}' for value in values['std'])} | {', '.join(f'{value:.6g}' for value in values['min'])} | {', '.join(f'{value:.6g}' for value in values['max'])} |"
        for name, values in normalization.items()
    )
    stats_note = f"- Normalization statistics: `{Path(normalization_path).resolve()}`\n" if normalization_path else ""
    (reports / "bright_data_audit.md").write_text(
        f"# BRIGHT data audit\n\n- Dataset root: `{Path(root).resolve()}`\n- Tiles: {len(samples):,}\n- Events: {len(events):,}\n- Layout: `pre-event`, `post-event`, `target`\n- Alignment: passed (CRS, bounds, and pixel dimensions)\n{stats_note}\n## Modalities\n\n| Modality | Assets |\n| --- | ---: |\n{modality_rows}\n\n## Intensity statistics\n\n| Modality | Mean | Std | Min | Max |\n| --- | --- | --- | --- | --- |\n{stat_rows}\n\n## Mask values\n\n| Code | Class | Pixels |\n| --- | --- | ---: |\n{class_rows}\n\n## Events\n\n{event_rows}\n", encoding="utf-8")
    return samples
