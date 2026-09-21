"""Inner-product dictionary matching (the DM baseline)."""
from __future__ import annotations

import numpy as np
import torch

from .preprocessing import clean_data, euclidean_norm, filter_param_range


class DictionaryMatcher:
    """Match L2-normalised query signals to the best (max inner-product) dictionary entry.

    Parameters
    ----------
    dict_norm : (N, 40) L2-normalised dictionary signals.
    dict_params : (N, 4) parameters of each entry, in physical units.
    """

    def __init__(self, dict_norm: np.ndarray, dict_params: np.ndarray, device: torch.device):
        self.device = device
        self.params = dict_params
        self._dict = torch.tensor(dict_norm, dtype=torch.float32).to(device)

    @classmethod
    def from_raw(cls, signals: np.ndarray, params: np.ndarray, mins, maxs, device: torch.device):
        """Filter to the modelled parameter range, drop non-finite rows, L2-normalise, move to ``device``."""
        signals, params = filter_param_range(signals, params, mins, maxs)
        signals, params = clean_data(signals, params)
        return cls(euclidean_norm(signals), params, device)

    def __len__(self) -> int:
        return len(self.params)

    def predict(self, query_norm: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Return the matched parameters, shape (M, 4), physical units."""
        m = len(query_norm)
        preds = np.empty((m, 4), dtype=np.float32)
        q_gpu = torch.tensor(query_norm, dtype=torch.float32).to(self.device)
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            scores = torch.mm(q_gpu[start:end], self._dict.T)
            idx = scores.argmax(dim=1).cpu().numpy()
            preds[start:end] = self.params[idx]
        return preds
