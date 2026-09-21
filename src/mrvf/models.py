"""Network architectures.

Attribute names (``conv``, ``fc1``, ``film1`` ...) are part of the checkpoint format: they must not be
renamed or the shipped ``.pt`` files will no longer load.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Clamp01(nn.Module):
    """Constrain outputs to the scaled parameter range [0, 1]."""

    def forward(self, x):
        return x.clamp(0.0, 1.0)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: ``y = gamma(cond) * x + beta(cond)``, initialised to identity."""

    def __init__(self, feature_dim: int, cond_in: int = 3, cond_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_in, cond_hidden),
            nn.ReLU(),
            nn.Linear(cond_hidden, 2 * feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        bias = torch.zeros(2 * feature_dim)
        bias[:feature_dim] = 1.0  # gamma starts at 1
        self.net[-1].bias.data.copy_(bias)
        self.feature_dim = feature_dim

    def forward(self, x, cond):
        p = self.net(cond)
        return p[:, : self.feature_dim] * x + p[:, self.feature_dim :]


def _conv_backbone(dropout: float) -> nn.Sequential:
    # 40 -> 20 -> 10 -> 5 samples, 256 channels: 1280 features out
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(dropout),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128), nn.ReLU(),
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm1d(256), nn.ReLU(),
    )


class TripleRegimeModel(nn.Module):
    """Conv1D backbone on the 40 echoes, FiLM-conditioned on the three R2* features.

    Input  ``(B, 43)`` = 40 L2-normalised echoes + feat_A + feat_B + feat_C
    Output ``(B, 4)``  = SO2, CBV, R, T2, each scaled to [0, 1]
    """

    def __init__(self, n_outputs: int = 4, dropout: float = 0.05, film_cond_hidden: int = 32):
        super().__init__()
        self.conv = _conv_backbone(dropout)
        self.fc1 = nn.Linear(1280, 512); self.bn1 = nn.BatchNorm1d(512)
        self.film1 = FiLMLayer(512, cond_in=3, cond_hidden=film_cond_hidden)
        self.fc2 = nn.Linear(512, 256); self.bn2 = nn.BatchNorm1d(256)
        self.film2 = FiLMLayer(256, cond_in=3, cond_hidden=film_cond_hidden)
        self.fc3 = nn.Linear(256, 128); self.bn3 = nn.BatchNorm1d(128)
        self.film3 = FiLMLayer(128, cond_in=3, cond_hidden=film_cond_hidden)
        self.fc_out = nn.Linear(128, n_outputs)
        self.out_act = Clamp01()
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        echo = x[:, :40].unsqueeze(1)
        cond = x[:, 40:43]
        c = self.conv(echo).flatten(1)
        h = self.drop(self.relu(self.bn1(self.film1(self.fc1(c), cond))))
        h = self.drop(self.relu(self.bn2(self.film2(self.fc2(h), cond))))
        h = self.drop(self.relu(self.bn3(self.film3(self.fc3(h), cond))))
        return self.out_act(self.fc_out(h))


class Conv1DModel(nn.Module):
    """DL-4p baseline: Conv1D + MLP on the 40 L2-normalised echoes (no R2* features)."""

    def __init__(self, n_outputs: int = 4, dropout: float = 0.3, in_dim: int = 40):
        super().__init__()
        self.conv_path = _conv_backbone(dropout)
        extra = in_dim - 40
        self.mlp = nn.Sequential(
            nn.Linear(1280 + extra, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )
        self.out_act = Clamp01()

    def forward(self, x):
        c = self.conv_path(x[:, :40].unsqueeze(1)).flatten(1)
        return self.out_act(self.mlp(torch.cat([c, x[:, 40:]], dim=1)))


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load weights from either a bare state_dict or a ``{'model_state_dict': ...}`` bundle; set eval mode."""
    try:
        ckpt = torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        # Some bundles (e.g. the DL-4p checkpoint) also store NumPy arrays, which the safe loader rejects.
        # Newer PyTorch versions default to weights_only=True, so fall back explicitly. Only load files you trust.
        log.warning("%s holds non-tensor objects; loading with weights_only=False", path)
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model
