"""Building training datasets (43-dim inputs, scaled targets) from the simulated dictionaries."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .config import PipelineConfig
from .features import EchoTiming, build_triple_input
from .io import load_echo_times, load_mat
from .preprocessing import (
    clean_data,
    filter_param_range,
    finite_mask,
    params_inverse,
    params_scale,
)

log = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Train / validation / test arrays. ``y_*`` are scaled to [0, 1]; ``y_test_raw`` is physical units."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray


def load_timing(cfg: PipelineConfig) -> EchoTiming:
    return EchoTiming.from_echo_times(load_echo_times(cfg.paths.echotimes_file), cfg.geometry)


def load_params(cfg: PipelineConfig) -> np.ndarray:
    """All dictionary parameters, first four columns (SO2, CBV, R, T2), physical units."""
    return load_mat(cfg.paths.params_path, cfg.paths.param_key)[:, :4]


def split_dataset(x: np.ndarray, y: np.ndarray, cfg: PipelineConfig) -> SplitData:
    """Two-stage split: hold out ``test_frac``, then ``val_frac`` of the remainder for validation."""
    t = cfg.train
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=t.test_frac, random_state=t.split_seed)
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=t.val_frac, random_state=t.split_seed)
    y_test_raw = params_inverse(y_test, cfg.params.mins_array[:4], cfg.params.maxs_array[:4])
    log.info("Train=%s  Val=%s  Test=%s", f"{len(x_train):,}", f"{len(x_val):,}", f"{len(x_test):,}")
    return SplitData(x_train, x_val, x_test, y_train, y_val, y_test, y_test_raw)


def build_mixed_snr_dataset(cfg: PipelineConfig, timing: EchoTiming) -> SplitData:
    """Pool the noisy dictionaries of every SNR level, subsample to ``n_samples``, split.

    This is the training set of the **noisy** triple-regime model.
    """
    mins, maxs = cfg.params.mins_array, cfg.params.maxs_array
    params_raw = load_params(cfg)
    all_x, all_y = [], []
    for snr in cfg.train.snr_levels:
        log.info("SNR=%s ...", snr)
        sig_raw = load_mat(cfg.paths.noisy_path(snr), cfg.paths.signal_key)
        sig_i, par_i = filter_param_range(sig_raw, params_raw.copy(), mins, maxs)
        x_i = build_triple_input(sig_i, timing, cfg.geometry, cfg.params, cfg.scaling)
        # Drop any non-finite row from inputs *and* targets together (keeps rows aligned).
        keep = finite_mask(sig_i, par_i)
        x_i, par_i = x_i[keep], par_i[keep]
        y_i = params_scale(par_i[:, :4], mins[:4], maxs[:4])
        all_x.append(x_i)
        all_y.append(y_i)
        log.info("  %s samples", f"{len(par_i):,}")

    x_all, y_all = np.vstack(all_x), np.vstack(all_y)
    if len(x_all) > cfg.train.n_samples:
        # RandomState(seed).choice draws the same sequence as np.random.seed(seed); np.random.choice(...),
        # which is what the original notebook did (unseeded). seed=None reproduces the unseeded behaviour.
        rng = np.random.RandomState(cfg.train.subsample_seed)
        idx = rng.choice(len(x_all), cfg.train.n_samples, replace=False)
        x_all, y_all = x_all[idx], y_all[idx]
    log.info("Total: %s  input_dim=%d", f"{len(x_all):,}", x_all.shape[1])
    return split_dataset(x_all, y_all, cfg)


def build_noise_free_dataset(cfg: PipelineConfig, timing: EchoTiming) -> SplitData:
    """The whole noise-free dictionary, no SNR injection. Training set of the **noise-free** model."""
    mins, maxs = cfg.params.mins_array, cfg.params.maxs_array
    sig_raw = load_mat(cfg.paths.noise_free_path, cfg.paths.signal_key)
    par_raw = load_params(cfg)
    sig_raw, par_raw = filter_param_range(sig_raw, par_raw, mins, maxs)
    sig_raw, par_raw = clean_data(sig_raw, par_raw)
    x = build_triple_input(sig_raw, timing, cfg.geometry, cfg.params, cfg.scaling)
    y = params_scale(par_raw[:, :4], mins[:4], maxs[:4]).astype(np.float32)
    valid = finite_mask(x, y)
    x, y = x[valid], y[valid]
    log.info("Dataset: %s samples  input_dim=%d", f"{len(x):,}", x.shape[1])
    return split_dataset(x, y, cfg)
