"""Figures for the training and evaluation stages.

Import this module lazily: it pulls in matplotlib, which the numerical code does not need.
Every function returns the figure and, if ``out_dir`` is given, also saves ``<name>.png`` and ``.pdf``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import PARAM_KEYS, PARAM_LABELS, PARAM_SCALE, PARAM_UNITS

COLORS = {
    "DM": "#E65100",            # deep orange
    "DL-4p": "#1565C0",         # deep blue
    "Triple-NF": "#AD1457",     # deep pink
    "Triple-Noisy": "#6A1B9A",  # deep purple
}
METHOD_STYLES = {
    "DM": dict(color=COLORS["DM"], marker="D", ls=":", lw=2, ms=6),
    "DL-4p": dict(color=COLORS["DL-4p"], marker="o", ls="--", lw=2, ms=6),
    "Triple-NF": dict(color=COLORS["Triple-NF"], marker="s", ls="-.", lw=2, ms=6),
    "Triple-Noisy": dict(color=COLORS["Triple-Noisy"], marker="^", ls="-", lw=2, ms=6),
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_fig(fig, out_dir: str | Path | None, name: str, formats=("pdf", "png")) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def plot_training_curve(history: dict, title: str, color_val: str, out_dir=None, name="training_curve"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=COLORS["DL-4p"], lw=1.5, label="Train")
    ax.plot(epochs, history["val_loss"], color=color_val, lw=1.5, ls="--", label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted MAE"); ax.set_title(title, fontweight="bold")
    ax.legend(); _clean_axes(ax); fig.tight_layout()
    save_fig(fig, out_dir, name)
    return fig


def plot_scatter(y_true_raw, y_pred_raw, title: str, color: str, out_dir=None, name="scatter", n_plot=15000):
    """Predicted vs ground truth for each parameter (display units), random subset of ``n_plot`` points."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), gridspec_kw={"wspace": 0.38})
    n = min(n_plot, len(y_pred_raw))
    idx = np.random.default_rng(42).choice(len(y_pred_raw), n, replace=False)
    for i, ax in enumerate(axes):
        t, p = y_true_raw[idx, i] * PARAM_SCALE[i], y_pred_raw[idx, i] * PARAM_SCALE[i]
        ax.scatter(t, p, s=1.5, alpha=0.10, color=color, rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.6)
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.sum() > 10:
            r2 = np.corrcoef(t[ok], p[ok])[0, 1] ** 2
            rmse = np.sqrt(np.mean((t[ok] - p[ok]) ** 2))
            ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=8, fontweight="bold")
            ax.text(0.05, 0.83, f"RMSE = {rmse:.2f}", transform=ax.transAxes, fontsize=7.5, color="#333")
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
        ax.set_title(f"{PARAM_LABELS[i]} {PARAM_UNITS[i]}", fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=10, y=1.02); fig.tight_layout()
    save_fig(fig, out_dir, name)
    return fig


def plot_rmse_vs_snr(results: dict, snr_levels, title: str, out_dir=None, name="rmse_vs_snr"):
    """``results`` is ``{method: {param: [rmse per SNR]}}``. One panel per parameter."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), gridspec_kw={"wspace": 0.38})
    x = np.arange(len(snr_levels))
    for ax, key, label, unit in zip(axes, PARAM_KEYS, PARAM_LABELS, PARAM_UNITS):
        for method, curves in results.items():
            ax.plot(x, curves[key], label=method, **METHOD_STYLES.get(method, {}))
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in snr_levels])
        ax.set_xlabel("SNR"); ax.set_ylabel(f"RMSE {unit}"); ax.set_title(label, fontweight="bold")
        _clean_axes(ax)
    axes[0].legend(fontsize=7.5, framealpha=0.9, loc="upper right")
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.02, ha="left"); fig.tight_layout()
    save_fig(fig, out_dir, name)
    return fig


def plot_ablation_heatmap(ablation: dict, title: str, out_dir=None, name="ablation_heatmap"):
    labels = list(ablation)
    matrix = np.array([ablation[c] for c in labels])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if matrix[i, j] < matrix.max() * 0.75 else "white")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{n} {u}" for n, u in zip(PARAM_LABELS, PARAM_UNITS)])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("RMSE (physical units)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, name)
    return fig
