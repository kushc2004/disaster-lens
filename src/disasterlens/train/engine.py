from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from disasterlens.eval import confusion_matrix, metrics_from_confusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, sar = batch["images"]["pre_optical"], batch["images"]["post_sar"]
    if pre is None or sar is None:
        raise ValueError("M2 early-fusion baseline requires pre_optical and post_sar")
    return pre.to(device, non_blocking=True), sar.to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: Any, criterion: nn.Module, device: torch.device) -> dict[str, float | list[float] | list[list[int]]]:
    model.eval()
    total_loss, examples, confusion = 0.0, 0, None
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        pre, sar, mask = _move(batch, device)
        logits = model(pre, sar)
        total_loss += float(criterion(logits, mask)) * mask.shape[0]
        examples += mask.shape[0]
        matrix = confusion_matrix(logits.cpu(), mask.cpu())
        confusion = matrix if confusion is None else confusion + matrix
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            print(f"[evaluation] batch {batch_index}/{total_batches}", flush=True)
    if not examples or confusion is None:
        raise ValueError("Evaluation loader is empty")
    metrics = metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / examples
    return metrics


@dataclass
class Trainer:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: torch.device
    checkpoint_dir: Path
    amp: bool = True

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.device.type == "cuda")

    def fit(self, train_loader: Any, val_loader: Any, *, epochs: int, scheduler: Any | None = None) -> list[dict[str, Any]]:
        best, history = float("-inf"), []
        for epoch in range(1, epochs + 1):
            print(f"[training] epoch {epoch}/{epochs} started ({len(train_loader)} training batches)", flush=True)
            transform = getattr(train_loader.dataset, "transform", None)
            if transform is not None and hasattr(transform, "set_epoch"):
                transform.set_epoch(epoch)
            self.model.train()
            train_loss, examples = 0.0, 0
            for batch in train_loader:
                pre, sar, mask = _move(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp and self.device.type == "cuda"):
                    logits = self.model(pre, sar)
                    loss = self.criterion(logits, mask)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                train_loss += float(loss.detach()) * mask.shape[0]
                examples += mask.shape[0]
            if not examples:
                raise ValueError("Training loader is empty")
            if scheduler is not None:
                scheduler.step()
            print(f"[training] epoch {epoch}/{epochs} training complete; validating", flush=True)
            validation = evaluate_epoch(self.model, val_loader, self.criterion, self.device)
            record = {"epoch": epoch, "train_loss": train_loss / examples, **{f"val_{key}": value for key, value in validation.items()}}
            history.append(record)
            metric = float(validation["f1_damage"])
            state = {"epoch": epoch, "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "metric": metric}
            if metric > best:
                best = metric
                torch.save(state, self.checkpoint_dir / "best.pt")
            torch.save(state, self.checkpoint_dir / "last.pt")
            (self.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            print(f"[training] epoch {epoch}/{epochs} complete: {json.dumps(record, sort_keys=True)}", flush=True)
        return history
