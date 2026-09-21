"""Inference, error metrics, RMSE-vs-SNR evaluation and feature ablation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .config import PARAM_KEYS, PARAM_SCALE, PARAM_UNITS, PipelineConfig
from .dictionary_matching import DictionaryMatcher
from .features import EchoTiming, build_triple_input
from .io import load_mat
from .preprocessing import euclidean_norm, finite_mask, in_range_mask, params_inverse

log = logging.getLogger(__name__)


# ── inference & metrics ─────────────────────────────────────────────────────────────────────────
def batched_predict(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Run ``model`` on ``x`` in batches (eval mode, no grad). Returns scaled predictions (N, 4)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32).to(device).contiguous()
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def rmse_per_parameter(pred_raw: np.ndarray, true_raw: np.ndarray) -> dict[str, float]:
    """RMSE of each parameter in display units (%, %, µm, ms)."""
    return {
        name: float(np.sqrt(np.mean((pred_raw[:, i] * sc - true_raw[:, i] * sc) ** 2)))
        for i, (name, sc) in enumerate(zip(PARAM_KEYS, PARAM_SCALE))
    }


def regression_metrics(pred_raw: np.ndarray, true_raw: np.ndarray) -> dict[str, dict]:
    """RMSE, bias, Pearson r and R² per parameter in display units."""
    out = {}
    for i, (name, unit, sc) in enumerate(zip(PARAM_KEYS, PARAM_UNITS, PARAM_SCALE)):
        pred, true = pred_raw[:, i] * sc, true_raw[:, i] * sc
        diff = pred - true
        out[name] = {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "bias": float(np.mean(diff)),
            "r": float(np.corrcoef(true, pred)[0, 1]),
            "r2": float(r2_score(true, pred)),
            "unit": unit,
        }
    return out


def feature_ablation(
    model: torch.nn.Module,
    x_test: np.ndarray,
    y_test_raw: np.ndarray,
    cfg: PipelineConfig,
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, list[float]]:
    """RMSE (display units, one per parameter) when feat_A/B/C are neutralised to 0.5, alone and in combination."""
    layout = {  # column index of each feature in the 43-dim input
        "A": 40, "B": 41, "C": 42,
    }
    conditions = {
        "Full (A+B+C)": [],
        "Ablate feat_A (→0.5)": ["A"],
        "Ablate feat_B (→0.5)": ["B"],
        "Ablate feat_C (→0.5)": ["C"],
        "Ablate feat_A+B (→0.5)": ["A", "B"],
        "Ablate feat_A+C (→0.5)": ["A", "C"],
        "Ablate feat_B+C (→0.5)": ["B", "C"],
        "Ablate all (→0.5)": ["A", "B", "C"],
    }
    mins, maxs = cfg.params.mins_array[:4], cfg.params.maxs_array[:4]
    results: dict[str, list[float]] = {}
    for label, ablated in conditions.items():
        x = x_test.copy()
        for f in ablated:
            x[:, layout[f]] = 0.5
        pred_raw = params_inverse(batched_predict(model, x, batch_size, device), mins, maxs)
        results[label] = [
            float(np.sqrt(np.mean((pred_raw[:, j] * sc - y_test_raw[:, j] * sc) ** 2)))
            for j, sc in enumerate(PARAM_SCALE)
        ]
    return results


# ── RMSE vs SNR ─────────────────────────────────────────────────────────────────────────────────
@dataclass
class SnrTestSet:
    """One SNR level's evaluation rows, prepared once and shared by every method."""

    sig_raw: np.ndarray   # (N, 40) noisy signal, un-normalised
    sig_norm: np.ndarray  # (N, 40) L2-normalised
    x_triple: np.ndarray  # (N, 43) triple-regime input
    params: np.ndarray    # (N, 4)  ground truth, physical units


# A method maps a SnrTestSet to predictions (N, 4) in physical units.
Method = Callable[[SnrTestSet], np.ndarray]


def dm_method(matcher: DictionaryMatcher, batch_size: int = 256) -> Method:
    return lambda ts: matcher.predict(ts.sig_norm, batch_size=batch_size)


def dl_method(
    model: torch.nn.Module,
    input_kind: str,
    cfg: PipelineConfig,
    device: torch.device,
    batch_size: int = 4096,
) -> Method:
    """``input_kind`` is 'signal' (40 L2-norm echoes, DL-4p) or 'triple' (43-dim, triple-regime)."""
    if input_kind not in ("signal", "triple"):
        raise ValueError(f"input_kind must be 'signal' or 'triple', got {input_kind!r}")
    mins, maxs = cfg.params.mins_array, cfg.params.maxs_array

    def predict(ts: SnrTestSet) -> np.ndarray:
        x = ts.sig_norm if input_kind == "signal" else ts.x_triple
        return params_inverse(batched_predict(model, x, batch_size, device), mins, maxs)

    return predict


def shared_test_indices(n_total: int, test_frac: float, seed: int) -> np.ndarray:
    """Indices of the held-out test rows (same seeded split for every SNR level)."""
    _, test_idx = train_test_split(np.arange(n_total), test_size=test_frac, random_state=seed)
    return test_idx


def prepare_snr_test_set(
    sig_all: np.ndarray,
    params_all: np.ndarray,
    rows: np.ndarray | None,
    cfg: PipelineConfig,
    timing: EchoTiming,
) -> SnrTestSet:
    """Select ``rows`` (``None`` = every dictionary entry), drop out-of-range / non-finite entries."""
    sig = sig_all if rows is None else sig_all[rows]
    par = params_all if rows is None else params_all[rows]
    keep = (
        in_range_mask(par, cfg.params.mins_array, cfg.params.maxs_array)
        & finite_mask(sig)
        & finite_mask(par)
    )
    sig, par = sig[keep], par[keep]
    x_tri = build_triple_input(sig, timing, cfg.geometry, cfg.params, cfg.scaling)
    ok = finite_mask(x_tri)
    if not ok.all():
        log.warning("Dropping %d rows with non-finite triple-regime features", (~ok).sum())
    return SnrTestSet(sig[ok], euclidean_norm(sig[ok]), x_tri[ok], par[ok])


def evaluate_across_snr(
    methods: dict[str, Method],
    cfg: PipelineConfig,
    timing: EchoTiming,
    snr_levels: tuple[int, ...] | None = None,
    test_frac: float | None = None,
    collect_errors: bool = False,
):
    """RMSE of every method at every SNR level, on identical rows.

    Parameters
    ----------
    test_frac : if given, evaluate on the seeded held-out split of the dictionary (the shared test set).
                If ``None``, evaluate on every dictionary entry (what the training notebooks' ``per_snr_rmse`` did).
    collect_errors : also return per-sample absolute errors (display units) for significance testing.

    Returns
    -------
    results : ``{method: {param: [rmse at each SNR]}}``
    errors  : ``{snr: {method: (N, 4) abs errors}}`` if ``collect_errors`` else ``None``
    """
    snr_levels = snr_levels or cfg.train.snr_levels
    params_all = load_mat(cfg.paths.params_path, cfg.paths.param_key)[:, :4]
    rows = None
    if test_frac is not None:
        rows = shared_test_indices(len(params_all), test_frac, cfg.train.split_seed)
        log.info("Shared test set: %s of %s entries", f"{len(rows):,}", f"{len(params_all):,}")

    results = {m: {p: [] for p in PARAM_KEYS} for m in methods}
    errors: dict | None = {snr: {} for snr in snr_levels} if collect_errors else None
    scale = np.array(PARAM_SCALE)

    for snr in snr_levels:
        log.info("SNR = %s", snr)
        sig_all = load_mat(cfg.paths.noisy_path(snr), cfg.paths.signal_key)
        ts = prepare_snr_test_set(sig_all, params_all, rows, cfg, timing)
        del sig_all
        log.info("  %s evaluation rows", f"{len(ts.params):,}")
        for name, predict in methods.items():
            pred = predict(ts)
            rmse = rmse_per_parameter(pred, ts.params)
            for p in PARAM_KEYS:
                results[name][p].append(rmse[p])
            if errors is not None:
                errors[snr][name] = np.abs(pred - ts.params) * scale
            log.info("  %-13s %s", name, "  ".join(f"{p}={rmse[p]:.3f}" for p in PARAM_KEYS))
    return results, errors


def paired_error_tests(errors: dict, reference: str, snr_levels, alpha_family: int | None = None):
    """One-sided Wilcoxon signed-rank test that each other method's |error| exceeds ``reference``'s.

    Returns ``{snr: {method: {param: {'mean_ref', 'mean_other', 'p', 'p_bonferroni'}}}}``. Errors are used
    exactly as computed: no method's errors are rescaled.
    """
    from scipy.stats import wilcoxon

    methods = [m for m in errors[snr_levels[0]] if m != reference]
    n_tests = alpha_family or len(methods) * len(snr_levels) * len(PARAM_KEYS)
    out: dict = {}
    for snr in snr_levels:
        out[snr] = {}
        for m in methods:
            out[snr][m] = {}
            for j, p in enumerate(PARAM_KEYS):
                e_ref, e_oth = errors[snr][reference][:, j], errors[snr][m][:, j]
                _, pval = wilcoxon(e_oth, e_ref, alternative="greater")
                out[snr][m][p] = {
                    "mean_ref": float(e_ref.mean()),
                    "mean_other": float(e_oth.mean()),
                    "p": float(pval),
                    "p_bonferroni": float(min(1.0, pval * n_tests)),
                }
    return out
