"""Fast unit tests for the mrvf library. They need no data files and no GPU.

Run from the repository root:   python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from mrvf import features, models, preprocessing
from mrvf.config import GesfideGeometry, ParamSpace, PipelineConfig, load_config
from mrvf.dictionary_matching import DictionaryMatcher
from mrvf.evaluation import rmse_per_parameter, shared_test_indices

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[1]


class TestPreprocessing(unittest.TestCase):
    def test_euclidean_norm_gives_unit_rows(self):
        x = np.random.default_rng(0).normal(size=(50, 40)).astype(np.float32)
        norms = np.linalg.norm(preprocessing.euclidean_norm(x), axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_scale_then_inverse_round_trips(self):
        space = ParamSpace()
        p = np.array([[0.6, 0.03, 8e-6, 0.09]])
        scaled = preprocessing.params_scale(p, space.mins_array, space.maxs_array)
        self.assertTrue(((scaled >= 0) & (scaled <= 1)).all())
        back = preprocessing.params_inverse(scaled, space.mins_array, space.maxs_array)
        np.testing.assert_allclose(back, p, rtol=1e-5)

    def test_filter_and_clean_drop_bad_rows(self):
        space = ParamSpace()
        sig = np.ones((3, 40), dtype=np.float32)
        par = np.array([[0.5, 0.05, 5e-6, 0.1], [2.0, 0.05, 5e-6, 0.1], [0.5, 0.05, 5e-6, 0.1]])
        _, kept = preprocessing.filter_param_range(sig, par, space.mins_array, space.maxs_array)
        self.assertEqual(len(kept), 2)  # the SO2 = 2.0 row is out of range

        sig[2, 7] = np.nan  # a non-finite signal value
        _, kept = preprocessing.clean_data(sig, par)
        self.assertEqual(len(kept), 2)  # that row is dropped, out-of-range rows are not this function's job


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.geo = GesfideGeometry()
        echo = np.linspace(0.004, 0.16, self.geo.n_echoes)  # seconds
        self.timing = features.EchoTiming.from_echo_times(echo, self.geo)

    def test_segment_sizes(self):
        self.assertEqual(len(self.timing.t_a), 14)
        self.assertEqual(len(self.timing.t_b), 16)
        self.assertEqual(len(self.timing.t_c_rel), 10)
        self.assertEqual(self.geo.se_echo, 30)

    def test_ols_slope_recovers_exponential_rate(self):
        t = np.linspace(0, 0.1, 14)
        sig = np.exp(-30.0 * t)[None, :]
        self.assertAlmostEqual(float(features.ols_slope(t, sig)[0]), -30.0, places=4)

    def test_r2star_features_recover_known_rates(self):
        rate_a, rate_b, rate_c = 30.0, 12.0, 25.0
        sig = np.concatenate(
            [np.exp(-rate_a * self.timing.t_a), np.exp(-rate_b * self.timing.t_b), np.exp(-rate_c * self.timing.t_c_rel)]
        )[None, :]
        a, b, c = features.compute_triple_regime_features(sig, self.timing, self.geo, ParamSpace())
        self.assertAlmostEqual(float(a[0]), rate_a, places=2)
        self.assertAlmostEqual(float(b[0]), rate_b, places=2)
        self.assertAlmostEqual(float(c[0]), rate_c, places=2)

    def test_build_triple_input_layout(self):
        sig = np.abs(np.random.default_rng(1).normal(size=(6, 40))).astype(np.float32) + 0.1
        cfg = PipelineConfig()
        x = features.build_triple_input(sig, self.timing, cfg.geometry, cfg.params, cfg.scaling)
        self.assertEqual(x.shape, (6, 43))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(x[:, :40], axis=1), 1.0, rtol=1e-5)
        self.assertTrue(((x[:, 40:] >= 0) & (x[:, 40:] <= 1)).all())  # features are clipped to [0, 1]


class TestModels(unittest.TestCase):
    def test_triple_regime_output_shape_and_range(self):
        out = models.TripleRegimeModel().eval()(torch.rand(8, 43))
        self.assertEqual(tuple(out.shape), (8, 4))
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_conv1d_baseline_output_shape(self):
        self.assertEqual(tuple(models.Conv1DModel().eval()(torch.rand(8, 40)).shape), (8, 4))

    def test_checkpoint_key_names_are_stable(self):
        """Renaming a layer would silently break every shipped .pt file."""
        triple = set(models.TripleRegimeModel().state_dict())
        for key in ("conv.0.weight", "fc1.weight", "film1.net.0.weight", "film3.net.2.bias", "fc_out.weight"):
            self.assertIn(key, triple)
        base = set(models.Conv1DModel().state_dict())
        for key in ("conv_path.0.weight", "mlp.0.weight", "mlp.16.weight"):
            self.assertIn(key, base)

    def test_load_checkpoint_round_trip(self):
        torch.manual_seed(0)
        src = models.TripleRegimeModel()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            torch.save({"model_state_dict": src.state_dict()}, path)  # bundle format
            dst = models.load_checkpoint(models.TripleRegimeModel(), path, CPU)
        x = torch.rand(4, 43)
        self.assertTrue(torch.equal(src.eval()(x), dst(x)))


    def test_load_checkpoint_with_numpy_metadata(self):
        """Bundles that also store NumPy arrays (like the shipped DL-4p file) must still load on new PyTorch."""
        torch.manual_seed(1)
        src = models.Conv1DModel()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            torch.save({"model_state_dict": src.state_dict(), "scaler": np.arange(3.0)}, path)
            dst = models.load_checkpoint(models.Conv1DModel(), path, CPU)
        x = torch.rand(4, 40)
        self.assertTrue(torch.equal(src.eval()(x), dst(x)))


class TestDictionaryMatching(unittest.TestCase):
    def test_exact_query_returns_its_own_parameters(self):
        rng = np.random.default_rng(3)
        d = preprocessing.euclidean_norm(np.abs(rng.normal(size=(200, 40))) + 0.1)
        params = rng.uniform(size=(200, 4)).astype(np.float32)
        matcher = DictionaryMatcher(d, params, CPU)
        np.testing.assert_array_equal(matcher.predict(d[[5, 77, 150]], batch_size=2), params[[5, 77, 150]])


class TestEvaluation(unittest.TestCase):
    def test_rmse_is_in_display_units(self):
        true = np.zeros((10, 4))
        pred = np.tile([0.01, 0.01, 1e-6, 0.001], (10, 1))  # 1 %, 1 %, 1 µm, 1 ms
        r = rmse_per_parameter(pred, true)
        for name in ("SO2", "CBV", "R", "T2"):
            self.assertAlmostEqual(r[name], 1.0, places=4)

    def test_shared_test_split_is_deterministic(self):
        a, b = shared_test_indices(1000, 0.15, 42), shared_test_indices(1000, 0.15, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 150)


class TestConfig(unittest.TestCase):
    def test_default_toml_matches_builtin_defaults(self):
        """configs/default.toml and the dataclass defaults must not drift apart."""
        self.assertEqual(load_config(ROOT / "configs/default.toml"), PipelineConfig())


if __name__ == "__main__":
    unittest.main()
