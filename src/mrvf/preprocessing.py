"""Signal and parameter preprocessing shared by training and evaluation."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def euclidean_norm(data: np.ndarray) -> np.ndarray:
    """|signal| divided by its L2 norm along the echo axis (float32)."""
    data = np.abs(data).astype(np.float32)
    return data / np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)


def params_scale(p: np.ndarray, mins, maxs) -> np.ndarray:
    """Physical units -> [0, 1]."""
    return ((p - mins) / (maxs - mins)).astype(np.float32)


def params_inverse(p: np.ndarray, mins, maxs) -> np.ndarray:
    """[0, 1] -> physical units."""
    return (p * (maxs - mins) + mins).astype(np.float32)


def in_range_mask(params: np.ndarray, mins, maxs) -> np.ndarray:
    """True for rows whose parameters all lie inside [mins, maxs]."""
    mask = np.ones(len(params), dtype=bool)
    for i in range(min(params.shape[1], len(mins))):
        mask &= (params[:, i] >= mins[i]) & (params[:, i] <= maxs[i])
    return mask


def filter_param_range(signals: np.ndarray, params: np.ndarray, mins, maxs):
    """Drop dictionary entries whose parameters fall outside the modelled range."""
    mask = in_range_mask(params, mins, maxs)
    if (~mask).sum():
        log.info("Filtered %d out-of-range entries (%.1f%%)", (~mask).sum(), (~mask).mean() * 100)
    return signals[mask], params[mask]


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """True for rows where every array is finite."""
    valid = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        valid &= np.all(np.isfinite(a), axis=1)
    return valid


def clean_data(signals: np.ndarray, params: np.ndarray):
    """Drop rows containing NaN/inf in either the signals or the parameters."""
    valid = finite_mask(signals, params)
    if (~valid).sum():
        log.info("Removed %d non-finite entries", (~valid).sum())
    return signals[valid], params[valid]
