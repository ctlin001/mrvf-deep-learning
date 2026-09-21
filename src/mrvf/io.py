"""Reading dictionaries and echo times from MATLAB files."""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

log = logging.getLogger(__name__)


def load_mat(path: str | Path, key: str) -> np.ndarray:
    """Load one array from a ``.mat`` file as float32, handling both v5 and v7.3 (HDF5) files.

    If ``key`` is missing the first non-private variable is used (as in the original notebooks).
    v7.3 files are stored transposed, so 2-D data is transposed back.
    """
    path = str(path)
    try:
        mat = sio.loadmat(path)
        if key in mat:
            return np.array(mat[key], dtype=np.float32)
        candidates = [k for k in mat if not k.startswith("_")]
        log.warning('Key "%s" not found in %s, using "%s"', key, path, candidates[0])
        return np.array(mat[candidates[0]], dtype=np.float32)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            data = f[key][()] if key in f else f[next(k for k in f if not k.startswith("#"))][()]
            if data.ndim >= 2:
                data = data.T
            return np.array(data, dtype=np.float32)


def load_echo_times(path: str | Path, key: str = "Echotimes") -> np.ndarray:
    """Echo times in **seconds** (the file stores milliseconds)."""
    return sio.loadmat(str(path))[key].flatten() / 1000.0
