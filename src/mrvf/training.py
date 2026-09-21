"""Loss and training loop."""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig

log = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """``prefer`` may be 'cpu', 'cuda' or None (CUDA if available)."""
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def release_gpu_memory(*_local_refs) -> None:
    """Hand cached CUDA memory back to the driver.

    Training keeps the whole dataset on the GPU and PyTorch caches those blocks. Dictionary matching then needs
    a ~1.5 GB score matrix per batch, which on a 6 GB card spills into system memory and slows down ~10x.
    (The arguments only exist so callers can drop their last references before this runs.)
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy and PyTorch. ``None`` leaves everything unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedMAELoss(nn.Module):
    """Per-parameter weighted mean absolute error on scaled targets."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        return (torch.abs(pred - target) * self.w.unsqueeze(0)).sum(dim=1).mean()


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    ckpt_path: str | Path,
    device: torch.device,
):
    """Train with AdamW + ReduceLROnPlateau + early stopping; keep the best-validation checkpoint.

    Returns ``(model_with_best_weights, history)`` where history has ``train_loss`` and ``val_loss``.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = WeightedMAELoss(cfg.param_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-7)

    x_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_v = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=cfg.batch_size, shuffle=True)
    best_val, patience_cnt = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}

    log.info("Training up to %d epochs (patience=%d)", cfg.epochs, cfg.patience)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        tloss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(x_t)

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(x_v), y_v).item()

        scheduler.step(vloss)
        history["train_loss"].append(tloss)
        history["val_loss"].append(vloss)

        if vloss < best_val:
            best_val, patience_cnt = vloss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch <= 5:
            log.info(
                "  Epoch %4d  train=%.5f  val=%.5f  best=%.5f  lr=%.2e  (%.1fs)",
                epoch, tloss, vloss, best_val, optimizer.param_groups[0]["lr"], time.time() - t0,
            )
        if patience_cnt >= cfg.patience:
            log.info("Early stop at epoch %d", epoch)
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info("Best val=%.6f  loaded from %s", best_val, ckpt_path)
    release_gpu_memory(x_t, y_t, x_v, y_v, loader)
    return model, history
