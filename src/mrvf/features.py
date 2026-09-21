"""Triple-regime R2* features derived from the three segments of the GESFIDE signal.

* **Part A** (FID, echoes 0-13):            ``R2*_A = R2 + R2'``
* **Part B** (rephasing, echoes 14-29):     ``R2*_B = R2 - R2'``  (can be negative)
* **Part C** (post spin-echo, echoes 30-39): ``R2*_C = R2 + R2'``, fitted on time *relative to the
  spin echo* so that its intercept is the spin-echo amplitude.

Each is the negative slope of an ordinary-least-squares fit of ``log|S|`` against time.
The network input is the 40 L2-normalised echoes followed by the three scaled features (43 values).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FeatureScaling, GesfideGeometry, ParamSpace
from .preprocessing import euclidean_norm


@dataclass(frozen=True)
class EchoTiming:
    """Echo times (seconds) of each GESFIDE segment."""

    t_a: np.ndarray
    t_b: np.ndarray
    t_c_rel: np.ndarray  # Part C times relative to the spin echo

    @classmethod
    def from_echo_times(cls, echo_times_s: np.ndarray, geometry: GesfideGeometry) -> "EchoTiming":
        se = geometry.se_echo
        t_c = echo_times_s[se:]
        t_se = echo_times_s[se - 1]
        return cls(t_a=echo_times_s[: geometry.n_fid], t_b=echo_times_s[geometry.n_fid : se], t_c_rel=t_c - t_se)


def ols_slope(t_vec: np.ndarray, sig_mat: np.ndarray) -> np.ndarray:
    """OLS slope of ``log|S|`` versus ``t`` for each row of ``sig_mat``. Returns shape (N,)."""
    log_s = np.log(np.maximum(np.abs(sig_mat), 1e-9)).astype(np.float64)
    t = t_vec.astype(np.float64)
    t_c = t - t.mean()
    log_sm = log_s - log_s.mean(axis=1, keepdims=True)
    return (log_sm * t_c[None, :]).sum(axis=1) / (t_c**2).sum()


def compute_triple_regime_features(
    sig_raw: np.ndarray, timing: EchoTiming, geometry: GesfideGeometry, params: ParamSpace
):
    """R2*_A, R2*_B, R2*_C (s^-1) from the **un-normalised** signal.

    A and C are clipped below at ``1 / T2_max``; B is left signed.
    """
    n_fid, se = geometry.n_fid, geometry.se_echo
    lower = params.r2_lower_bound
    r2_a = np.maximum(-ols_slope(timing.t_a, sig_raw[:, :n_fid]), lower).astype(np.float32)
    r2_b = (-ols_slope(timing.t_b, sig_raw[:, n_fid:se])).astype(np.float32)
    r2_c = np.maximum(-ols_slope(timing.t_c_rel, sig_raw[:, se:]), lower).astype(np.float32)
    return r2_a, r2_b, r2_c


def scale_triple_features(r2_a, r2_b, r2_c, scaling: FeatureScaling):
    """Min-max scale the three features to [0, 1] (clipped)."""

    def scale(x, lo, hi):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return scale(r2_a, *scaling.a), scale(r2_b, *scaling.b), scale(r2_c, *scaling.c)


def build_triple_input(
    sig_raw: np.ndarray,
    timing: EchoTiming,
    geometry: GesfideGeometry,
    params: ParamSpace,
    scaling: FeatureScaling,
) -> np.ndarray:
    """Raw signal (N, 40) -> model input (N, 43) = 40 L2-normalised echoes + feat_A/B/C."""
    r2_a, r2_b, r2_c = compute_triple_regime_features(sig_raw, timing, geometry, params)
    f_a, f_b, f_c = scale_triple_features(r2_a, r2_b, r2_c, scaling)
    sig_norm = euclidean_norm(sig_raw)
    return np.concatenate([sig_norm, f_a[:, None], f_b[:, None], f_c[:, None]], axis=1).astype(np.float32)
