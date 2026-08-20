"""Required, explicitly labelled evaluation figures from saved real predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


CLASS_NAMES = ("background", "intact", "damaged", "destroyed")
CLASS_COLORS = ("#202020", "#4daf4a", "#ffbf00", "#d73027")


def _robust_image(values: np.ndarray) -> np.ndarray:
    image = np.asarray(values, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if not len(finite):
        return np.zeros_like(image)
    low, high = np.quantile(finite, [0.02, 0.98])
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _save_image(image: np.ndarray, path: Path, title: str, *, cmap: str | None = None) -> None:
    figure, axis = plt.subplots(figsize=(7, 7))
    artist = axis.imshow(image, cmap=cmap)
    axis.set_title(title)
    axis.set_axis_off()
    if cmap is not None:
        figure.colorbar(
            artist,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="Normalized intensity",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_class_map(mask: np.ndarray, path: Path, title: str) -> None:
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
    display = np.ma.masked_where(mask == 255, mask)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(display, cmap=cmap, norm=norm)
    axis.set_title(title)
    axis.set_axis_off()
    legend = [
        Patch(facecolor=color, label=name.title())
        for name, color in zip(CLASS_NAMES, CLASS_COLORS, strict=True)
    ]
    legend.append(
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Ignored/unlabelled")
    )
    axis.legend(handles=legend, title="Damage class", loc="lower left", framealpha=0.9)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_confusion(matrix: np.ndarray, path: Path) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals > 0,
    )
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    artist = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(4):
        for column in range(4):
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2%}\n(n={int(matrix[row, column]):,})",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "black",
                fontsize=8,
            )
    axis.set_xticks(
        range(4), [name.title() for name in CLASS_NAMES], rotation=30, ha="right"
    )
    axis.set_yticks(range(4), [name.title() for name in CLASS_NAMES])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    axis.set_title("Row-normalized confusion matrix")
    figure.colorbar(artist, ax=axis, label="Fraction within ground-truth class")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_per_event(rows: list[dict[str, Any]], path: Path) -> None:
    event_ids = [str(row["event_id"]) for row in rows]
    locations = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(9.0, 0.7 * len(rows)), 6.5))
    axis.bar(
        locations - width / 2,
        [float(row["macro_f1"]) for row in rows],
        width,
        label="Macro F1",
    )
    axis.bar(
        locations + width / 2,
        [float(row["miou"]) for row in rows],
        width,
        label="mIoU",
    )
    axis.set_xticks(locations, event_ids, rotation=45, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_xlabel("BRIGHT event")
    axis.set_title("Per-event segmentation performance")
    axis.legend(title="Metric")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_evaluation_figures(
    result: dict[str, Any], predictions_dir: Path, output_dir: Path
) -> None:
    """Write the spec-required evaluation figures from a persisted example."""
    example: Path | None = None
    for path in sorted(predictions_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if "pre_optical" in payload.files and "post_sar" in payload.files:
                example = path
                break
    if example is None:
        raise FileNotFoundError(
            f"No prediction bundle with visual inputs exists below {predictions_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(example, allow_pickle=False) as payload:
        optical = payload["pre_optical"]
        sar = payload["post_sar"]
        logits = payload["logits"]
        target = payload["mask"]
    rgb = np.moveaxis(optical[:3], 0, -1)
    rgb = np.stack(
        [_robust_image(rgb[..., channel]) for channel in range(3)], axis=-1
    )
    sar_display = _robust_image(sar[0] if sar.ndim == 3 else sar)
    probabilities = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    prediction = probabilities.argmax(axis=0)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(axis=0)
    entropy /= np.log(probabilities.shape[0])

    _save_image(
        rgb,
        output_dir / "pre_event_optical.png",
        "Pre-event optical (normalized RGB)",
    )
    _save_image(
        sar_display,
        output_dir / "post_event_sar.png",
        "Post-event SAR (normalized intensity)",
        cmap="gray",
    )
    _save_class_map(
        target, output_dir / "ground_truth.png", "Ground-truth damage classes"
    )
    _save_class_map(
        prediction,
        output_dir / "predicted_damage_map.png",
        "Estimated damage classes",
    )

    figure, axis = plt.subplots(figsize=(7, 7))
    artist = axis.imshow(entropy, cmap="magma", vmin=0.0, vmax=1.0)
    axis.set_title("Normalized predictive entropy")
    axis.set_axis_off()
    figure.colorbar(
        artist,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Model uncertainty (0–1)",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "entropy_map.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    _save_confusion(
        np.asarray(result["pooled"]["confusion_matrix"]),
        output_dir / "confusion_matrix.png",
    )
    _save_per_event(result["per_event"], output_dir / "per_event_metrics.png")
