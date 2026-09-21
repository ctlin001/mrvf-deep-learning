"""High-level entry points: train a triple-regime model, evaluate methods across SNR.

Both the command line (``scripts/run_pipeline.py``) and the notebooks call these two functions, so they
cannot drift apart.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import PARAM_KEYS, PipelineConfig
from .data import build_mixed_snr_dataset, build_noise_free_dataset, load_timing
from .dictionary_matching import DictionaryMatcher
from .evaluation import (
    batched_predict,
    dl_method,
    dm_method,
    evaluate_across_snr,
    feature_ablation,
    paired_error_tests,
    regression_metrics,
)
from .io import load_mat
from .models import Conv1DModel, TripleRegimeModel, load_checkpoint
from .preprocessing import params_inverse
from .training import get_device, set_seed, train_model

log = logging.getLogger(__name__)

VARIANTS = {
    # name: (checkpoint stem, human title)
    "noisy": ("triple_regime", "Triple-Regime (A+B+C), mixed-SNR training"),
    "noise-free": ("triple_nf", "Triple-Regime (A+B+C), noise-free training"),
}


def _dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_training(
    cfg: PipelineConfig,
    variant: str,
    output_dir: str | Path,
    device: torch.device | None = None,
    make_figures: bool = True,
) -> dict:
    """Train one triple-regime model and write every artefact into ``output_dir``.

    Writes ``models/<stem>_best.pt``, ``models/<stem>_final.pt``, ``training_history.json``,
    ``test_results.json``, ``per_snr_rmse.json``, ``ablation_results.json`` and (optionally) ``figures/``.
    Returns ``{'model', 'history', 'results', 'per_snr', 'ablation'}``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    stem, title = VARIANTS[variant]
    output_dir = Path(output_dir)
    device = device or get_device()
    mins4, maxs4 = cfg.params.mins_array[:4], cfg.params.maxs_array[:4]

    set_seed(cfg.train.torch_seed)
    timing = load_timing(cfg)

    log.info("Building %s dataset", variant)
    data = (build_mixed_snr_dataset if variant == "noisy" else build_noise_free_dataset)(cfg, timing)

    model = TripleRegimeModel(n_outputs=4, dropout=cfg.train.dropout).to(device)
    log.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    best_path = output_dir / "models" / f"{stem}_best.pt"
    model, history = train_model(model, data.x_train, data.y_train, data.x_val, data.y_val, cfg.train, best_path, device)
    _dump(history, output_dir / "training_history.json")

    # Held-out test set
    y_pred_raw = params_inverse(batched_predict(model, data.x_test, 4096, device), mins4, maxs4)
    results = regression_metrics(y_pred_raw, data.y_test_raw)
    _dump(results, output_dir / "test_results.json")

    # RMSE per SNR level over every dictionary entry (as the original training notebooks did)
    per_snr, _ = evaluate_across_snr({"model": dl_method(model, "triple", cfg, device)}, cfg, timing)
    per_snr = per_snr["model"]
    _dump(per_snr, output_dir / "per_snr_rmse.json")

    ablation = feature_ablation(model, data.x_test, data.y_test_raw, cfg, device)
    _dump(ablation, output_dir / "ablation_results.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": json.loads(json.dumps(asdict(cfg), default=str)),
            "results": results,
            "per_snr_rmse": per_snr,
            "ablation": ablation,
            "input_dim": 43,
            "input_layout": "40 L2-norm echoes + feat_A + feat_B + feat_C",
            "feature_scaling": asdict(cfg.scaling),
        },
        output_dir / "models" / f"{stem}_final.pt",
    )

    if make_figures:
        from . import plotting

        plotting.apply_style()
        fig_dir = output_dir / "figures"
        color = plotting.COLORS["Triple-Noisy" if variant == "noisy" else "Triple-NF"]
        plotting.plot_training_curve(history, f"{title}: training curve", color, fig_dir, "training_curve")
        plotting.plot_scatter(data.y_test_raw, y_pred_raw, f"{title}: predicted vs ground truth", color, fig_dir, "scatter")
        plotting.plot_rmse_vs_snr({title: per_snr}, cfg.train.snr_levels, f"{title}: RMSE vs SNR", fig_dir, "rmse_vs_snr")
        plotting.plot_ablation_heatmap(ablation, f"{title}: feature-ablation RMSE", fig_dir, "ablation_heatmap")

    return {"model": model, "history": history, "results": results, "per_snr": per_snr, "ablation": ablation}


def run_evaluation(
    cfg: PipelineConfig,
    checkpoints: dict[str, str | Path],
    output_dir: str | Path,
    device: torch.device | None = None,
    test_frac: float = 0.15,
    with_stats: bool = False,
    make_figures: bool = True,
    dl_batch_size: int = 4096,
    dm_batch_size: int = 256,
) -> dict:
    """RMSE vs SNR for DM and any of the checkpoints ``DL-4p``, ``Triple-NF``, ``Triple-Noisy``.

    All methods are scored on the same held-out rows (seeded split of the dictionary). Writes
    ``ablation_rmse.json`` (and ``mae_stats.json`` if ``with_stats``) into ``output_dir``.
    """
    known = {"DL-4p", "Triple-NF", "Triple-Noisy"}
    unknown = set(checkpoints) - known
    if unknown:
        raise ValueError(f"Unknown checkpoint names {sorted(unknown)}; expected a subset of {sorted(known)}")
    output_dir = Path(output_dir)
    device = device or get_device()
    timing = load_timing(cfg)

    log.info("Loading noise-free dictionary for DM")
    matcher = DictionaryMatcher.from_raw(
        load_mat(cfg.paths.noise_free_path, cfg.paths.signal_key),
        load_mat(cfg.paths.params_path, cfg.paths.param_key)[:, :4],
        cfg.params.mins_array, cfg.params.maxs_array, device,
    )
    methods = {"DM": dm_method(matcher, dm_batch_size)}
    if "DL-4p" in checkpoints:
        m = load_checkpoint(Conv1DModel(n_outputs=4, dropout=0.3, in_dim=40).to(device), checkpoints["DL-4p"], device)
        methods["DL-4p"] = dl_method(m, "signal", cfg, device, dl_batch_size)
    for name in ("Triple-NF", "Triple-Noisy"):
        if name in checkpoints:
            m = load_checkpoint(TripleRegimeModel(n_outputs=4, dropout=cfg.train.dropout).to(device), checkpoints[name], device)
            methods[name] = dl_method(m, "triple", cfg, device, dl_batch_size)

    results, errors = evaluate_across_snr(
        methods, cfg, timing, test_frac=test_frac, collect_errors=with_stats
    )
    _dump(results, output_dir / "ablation_rmse.json")

    out = {"results": results}
    if with_stats:
        ref = "Triple-Noisy" if "Triple-Noisy" in methods else next(reversed(methods))
        stats = paired_error_tests(errors, ref, cfg.train.snr_levels)
        _dump({str(k): v for k, v in stats.items()}, output_dir / "mae_stats.json")
        out["stats"] = stats
    if make_figures:
        from . import plotting

        plotting.apply_style()
        plotting.plot_rmse_vs_snr(
            results, cfg.train.snr_levels, "RMSE vs SNR: " + " vs ".join(methods), output_dir / "figures", "rmse_vs_snr"
        )
    return out


def compare_results(new: dict, reference_path: str | Path) -> dict:
    """Largest absolute and relative difference between ``new`` and a saved ``ablation_rmse.json``."""
    with open(reference_path) as f:
        ref = json.load(f)
    worst_abs, worst_rel, n = 0.0, 0.0, 0
    for method in new:
        for p in PARAM_KEYS:
            for a, b in zip(new[method][p], ref[method][p]):
                worst_abs = max(worst_abs, abs(a - b))
                worst_rel = max(worst_rel, abs(a - b) / max(abs(b), 1e-12))
                n += 1
    return {"n_values": n, "max_abs_diff": worst_abs, "max_rel_diff": worst_rel}
