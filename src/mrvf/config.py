"""Typed configuration for the MRvF pipeline.

Every value here is copied from the CONFIG dictionaries of the original notebooks
(``final_TripleRegime_ABC_Train`` and ``final_ablation_rmse_vs_snr``). Nothing in the
library reads module-level state: functions take these objects as explicit arguments.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

# Parameter order everywhere in the package: SO2, CBV, R, T2.
PARAM_KEYS: tuple[str, ...] = ("SO2", "CBV", "R", "T2")
PARAM_LABELS: tuple[str, ...] = ("SO₂", "CBV", "R", "T2")
PARAM_UNITS: tuple[str, ...] = ("(%)", "(%)", "(µm)", "(ms)")
# Multiply a parameter in SI-ish units (fraction, fraction, metres, seconds) to get display units.
PARAM_SCALE: tuple[float, ...] = (100, 100, 1e6, 1000)


@dataclass(frozen=True)
class ParamSpace:
    """Physical range of the four estimated parameters (SO2, CBV, R [m], T2 [s])."""

    mins: tuple[float, ...] = (0.0, 0.0025, 1.0e-6, 0.050)
    maxs: tuple[float, ...] = (1.0, 0.15, 25.0e-6, 0.200)

    @property
    def mins_array(self) -> np.ndarray:
        return np.array(self.mins)

    @property
    def maxs_array(self) -> np.ndarray:
        return np.array(self.maxs)

    @property
    def r2_lower_bound(self) -> float:
        """Lower clip for R2* features: 1 / T2_max  (~5 s^-1)."""
        return 1.0 / self.maxs[3]


@dataclass(frozen=True)
class GesfideGeometry:
    """How the 40 GESFIDE echoes are split: Part A (FID), B (rephasing), C (post spin-echo)."""

    n_fid: int = 14
    n_rephas: int = 16
    n_postse: int = 10

    @property
    def se_echo(self) -> int:
        """Index of the first Part-C echo (the spin echo is the last Part-B echo)."""
        return self.n_fid + self.n_rephas

    @property
    def n_echoes(self) -> int:
        return self.n_fid + self.n_rephas + self.n_postse


@dataclass(frozen=True)
class FeatureScaling:
    """Min-max ranges (s^-1) used to scale R2*_A, R2*_B, R2*_C into [0, 1]."""

    a: tuple[float, float] = (2.0, 55.0)
    b: tuple[float, float] = (-30.0, 22.0)  # R2*_B = R2 - R2' can be negative
    c: tuple[float, float] = (2.0, 55.0)


@dataclass(frozen=True)
class TrainConfig:
    snr_levels: tuple[int, ...] = (20, 50, 100, 150)
    n_samples: int = 1_600_000
    test_frac: float = 0.15
    val_frac: float = 0.15
    batch_size: int = 16384
    epochs: int = 200
    patience: int = 25
    lr: float = 5e-5
    dropout: float = 0.05
    param_weights: tuple[float, ...] = (1.0, 12.0, 8.0, 1.0)
    split_seed: int = 42  # train/val/test split (fixed in the original notebooks)
    subsample_seed: int | None = 42  # original: unseeded. None reproduces that.
    torch_seed: int | None = 42  # original: unseeded. None reproduces that.


@dataclass(frozen=True)
class DataPaths:
    """Location of the simulated dictionaries. Relative paths resolve against the CWD."""

    dict_dir: Path = Path("../subsamples/subsamples_v3")
    params_file: str = "QuasiRand_par_t2_200.mat"
    noise_free_file: str = "QuasiRand_t2_200.mat"
    noisy_pattern: str = "QuasiRand_t2_snr{snr}.mat"
    echotimes_file: Path = Path("../echotimes.mat")
    signal_key: str = "Dico40_save"
    param_key: str = "par_save"

    @property
    def params_path(self) -> Path:
        return Path(self.dict_dir) / self.params_file

    @property
    def noise_free_path(self) -> Path:
        return Path(self.dict_dir) / self.noise_free_file

    def noisy_path(self, snr: int) -> Path:
        return Path(self.dict_dir) / self.noisy_pattern.format(snr=snr)


@dataclass(frozen=True)
class PipelineConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    params: ParamSpace = field(default_factory=ParamSpace)
    geometry: GesfideGeometry = field(default_factory=GesfideGeometry)
    scaling: FeatureScaling = field(default_factory=FeatureScaling)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce(cls: type, values: dict[str, Any]):
    """Build a dataclass from a TOML table, turning lists into tuples and str into Path."""
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        if isinstance(value, list):
            value = tuple(value)
        if known[key].type in ("Path", Path) or key in ("dict_dir", "echotimes_file"):
            value = Path(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a TOML file (see ``configs/default.toml``); ``None`` gives the built-in defaults."""
    if path is None:
        return PipelineConfig()
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return PipelineConfig(
        paths=_coerce(DataPaths, raw.get("paths", {})),
        params=_coerce(ParamSpace, raw.get("params", {})),
        geometry=_coerce(GesfideGeometry, raw.get("geometry", {})),
        scaling=_coerce(FeatureScaling, raw.get("scaling", {})),
        train=_coerce(TrainConfig, raw.get("train", {})),
    )


def with_overrides(cfg: PipelineConfig, **train_overrides: Any) -> PipelineConfig:
    """Return a copy of ``cfg`` with some TrainConfig fields replaced."""
    clean = {k: v for k, v in train_overrides.items() if v is not None}
    return replace(cfg, train=replace(cfg.train, **clean))
