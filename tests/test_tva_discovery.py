# -*- coding: utf-8 -*-
"""
Tests for TVA structure discovery (autots/evaluator/tva/discovery.py).

Torch-free by design — these run in CI. The single most valuable test here is
test_pure_confounder_yields_no_edges: pairwise Granger (the approach this
module replaces) fails it catastrophically on co-trending panels.

Run with:  python -m pytest tests/test_tva_discovery.py -v
"""
import unittest

import numpy as np
import pandas as pd

from autots.datasets.synthetic import make_svar_panel
from autots.datasets import generate_synthetic_daily_data
from autots.evaluator.tva.discovery import (
    discover_structure,
    forecast_factors,
    match_factors,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# smaller subsample count keeps CI fast while remaining stable
FAST_CONFIG = {'n_subsamples': 30}


def _edge_set(result, families=('lasso',)):
    return {
        (e['source'], e['target'])
        for e in result['edges']
        if e['family'] in families
    }


def _make_pure_confounder_panel(n_series=16, n_obs=500, seed=7) -> pd.DataFrame:
    """N series = shared factor + iid noise. NO true edges exist."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_obs)
    factor = np.cumsum(rng.normal(0, 0.2, n_obs)) + 0.01 * t
    data = {}
    for i in range(n_series):
        loading = rng.uniform(0.5, 1.5)
        data[f'conf_{i}'] = 10 + loading * factor + rng.normal(0, 0.5, n_obs)
    return pd.DataFrame(
        data, index=pd.date_range('2020-01-01', periods=n_obs, freq='D')
    )


class TestDiscoveryDeterminism(unittest.TestCase):
    def test_discovery_deterministic(self):
        df, *_ = make_svar_panel(n_series=10, n_obs=400, seed=3)
        r1 = discover_structure(df, config=FAST_CONFIG, seed=42)
        r2 = discover_structure(df, config=FAST_CONFIG, seed=42)
        self.assertEqual(r1['edges'], r2['edges'])
        np.testing.assert_allclose(r1['loadings'], r2['loadings'])
        np.testing.assert_allclose(r1['adjacency'], r2['adjacency'])


class TestPureConfounder(unittest.TestCase):
    def test_pure_confounder_yields_no_edges(self):
        """Series driven only by a shared factor must produce ~no edges."""
        df = _make_pure_confounder_panel(n_series=16, n_obs=500)
        result = discover_structure(df, config=FAST_CONFIG, seed=42)
        lasso_edges = _edge_set(result, families=('lasso',))
        # allow at most 5% of N spurious edges
        self.assertLessEqual(
            len(lasso_edges),
            max(1, int(0.05 * df.shape[1])),
            f"spurious edges on a pure-confounder panel: {sorted(lasso_edges)}",
        )


class TestSVARRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.true_adj, cls.true_lags, cls.true_loadings = make_svar_panel(
            n_series=12, n_obs=600, n_factors=1, edge_density=0.2, seed=11
        )
        cls.result = discover_structure(cls.df, config=FAST_CONFIG, seed=42)
        cls.true_edges = {
            (f'svar_{j}', f'svar_{i}')
            for j in range(12)
            for i in range(12)
            if cls.true_adj[j, i] != 0
        }

    def test_svar_recovery(self):
        found = _edge_set(self.result, families=('lasso',))
        self.assertGreater(len(self.true_edges), 3, "generator made too few edges")
        tp = len(found & self.true_edges)
        recall = tp / len(self.true_edges)
        precision = tp / max(len(found), 1)
        self.assertGreaterEqual(
            recall, 0.6, f"recall {recall:.2f}; found={sorted(found)}"
        )
        self.assertGreaterEqual(
            precision, 0.7, f"precision {precision:.2f}; found={sorted(found)}"
        )
        # direction accuracy: among true undirected pairs recovered, the
        # direction must match the generator
        reversed_edges = {(b, a) for a, b in self.true_edges}
        directed_hits = len(found & self.true_edges)
        directed_misses = len(found & reversed_edges)
        if directed_hits + directed_misses > 0:
            direction_accuracy = directed_hits / (directed_hits + directed_misses)
            self.assertGreaterEqual(direction_accuracy, 0.8)

    def test_lag_recovery(self):
        lag_errors = []
        for e in self.result['edges']:
            if e['family'] != 'lasso':
                continue
            j = int(e['source'].split('_')[1])
            i = int(e['target'].split('_')[1])
            if self.true_lags[j, i] > 0:
                lag_errors.append(abs(e['lag'] - self.true_lags[j, i]))
        if lag_errors:
            self.assertLessEqual(float(np.median(lag_errors)), 1.0)

    def test_edges_have_required_fields(self):
        for e in self.result['edges']:
            for key in (
                'source', 'target', 'lag', 'sign', 'weight', 'family', 'stability',
                'delta_mse',
            ):
                self.assertIn(key, e)


class TestFactorRecovery(unittest.TestCase):
    def test_factor_recovery(self):
        """Recover the true number of factors and the loading pattern."""
        rng = np.random.default_rng(5)
        n_obs, n_series = 500, 12
        t = np.arange(n_obs)
        f1 = np.cumsum(rng.normal(0, 0.3, n_obs)) + 0.02 * t
        f2 = np.cumsum(rng.normal(0, 0.3, n_obs)) - 0.015 * t
        true_loadings = np.zeros((n_series, 2))
        data = {}
        for i in range(n_series):
            if i < 6:
                true_loadings[i, 0] = rng.uniform(0.8, 1.2)
            else:
                true_loadings[i, 1] = rng.uniform(0.8, 1.2)
            signal = true_loadings[i, 0] * f1 + true_loadings[i, 1] * f2
            data[f'fs_{i}'] = 20 + signal + rng.normal(0, 0.3, n_obs)
        df = pd.DataFrame(
            data, index=pd.date_range('2020-01-01', periods=n_obs, freq='D')
        )
        result = discover_structure(df, config=FAST_CONFIG, seed=42)
        loadings = result['loadings']
        self.assertEqual(loadings.shape[1], 2, "wrong number of factors")
        # Procrustes-style alignment: each true group should map dominantly to
        # one distinct discovered factor
        group1 = np.abs(loadings[:6]).mean(axis=0)
        group2 = np.abs(loadings[6:]).mean(axis=0)
        self.assertNotEqual(int(group1.argmax()), int(group2.argmax()))
        # communities follow the factor split
        communities = result['communities']
        self.assertEqual(len(set(communities[:6])), 1)
        self.assertEqual(len(set(communities[6:])), 1)

    def test_forecast_factors_shape(self):
        factors = np.cumsum(np.random.default_rng(0).normal(0, 1, (200, 3)), axis=0)
        fc = forecast_factors(factors, horizon=14)
        self.assertEqual(fc.shape, (14, 3))


class TestScaleInvariance(unittest.TestCase):
    def test_scale_invariance(self):
        df, *_ = make_svar_panel(n_series=10, n_obs=400, seed=13)
        r1 = discover_structure(df, config=FAST_CONFIG, seed=42)
        df_scaled = df.copy()
        df_scaled[df.columns[0]] = df_scaled[df.columns[0]] * 1000.0
        r2 = discover_structure(df_scaled, config=FAST_CONFIG, seed=42)
        self.assertEqual(
            _edge_set(r1, families=('lasso',)),
            _edge_set(r2, families=('lasso',)),
        )


class TestSeedVariance(unittest.TestCase):
    def test_seed_variance(self):
        """Topology must be essentially seed-invariant (Jaccard >= 0.9)."""
        df, *_ = make_svar_panel(n_series=12, n_obs=600, edge_density=0.2, seed=11)
        edge_sets = []
        for seed in (1, 2, 3):
            result = discover_structure(df, config=FAST_CONFIG, seed=seed)
            edge_sets.append(_edge_set(result, families=('lasso',)))
        for a in range(len(edge_sets)):
            for b in range(a + 1, len(edge_sets)):
                union = edge_sets[a] | edge_sets[b]
                if not union:
                    continue
                jaccard = len(edge_sets[a] & edge_sets[b]) / len(union)
                self.assertGreaterEqual(
                    jaccard,
                    0.9,
                    f"seed instability: {edge_sets[a]} vs {edge_sets[b]}",
                )


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestGraphChangesForecast(unittest.TestCase):
    def test_graph_changes_forecast(self):
        """The graph must be load-bearing in the architecture: with the
        mixers active, A vs A = 0 changes the network's trend forecast.

        This is a wiring check on the network itself (the mixers are given
        non-trivial weights, as training would); whether the graph HELPS is
        the Phase-4 ablation's question, not a unit test's.
        """
        from autots.evaluator.tva.trend_network import CompositeTrendNetworkV2

        torch.manual_seed(0)
        edges = [
            {'source': 0, 'target': 1, 'lag': 2, 'sign': 1, 'weight': 0.9,
             'family': 'lasso'},
            {'source': 2, 'target': 3, 'lag': 1, 'sign': -1, 'weight': 0.7,
             'family': 'lasso'},
            {'source': 4, 'target': 1, 'lag': 0, 'sign': 1, 'weight': 0.5,
             'family': 'event'},
        ]
        loadings = np.random.default_rng(0).normal(size=(6, 2)).astype(np.float32)
        net = CompositeTrendNetworkV2(
            n_series=6,
            window_size=40,
            forecast_horizon=8,
            d_token=16,
            n_meso=4,
            n_global=2,
            n_prototypes=2,
            n_heads=2,
            discovery={'edges': edges, 'loadings': loadings},
        )
        with torch.no_grad():
            # give the zero-initialized mixers and correction heads the
            # non-trivial weights a trained network would have (P1-3 zero-init
            # means the untrained output is the damped baseline only)
            torch.nn.init.normal_(net.graph_mix.weight, std=0.1)
            torch.nn.init.normal_(net.lift_mix.weight, std=0.1)
            torch.nn.init.normal_(net.decoder.forecast_head.weight, std=0.1)
            torch.nn.init.normal_(net.responder.forecast_head.weight, std=0.1)

        x = torch.randn(2, 6, 40)
        with torch.no_grad():
            fc_with = net(x)['trend_forecast'].clone()
            net.graph_learner.edge_prior.zero_()
            net.graph_learner.edge_delta.zero_()
            fc_without = net(x)['trend_forecast']
        denom = fc_with.abs().mean().item() + 1e-9
        rel_diff = (fc_with - fc_without).abs().mean().item() / denom
        self.assertGreater(rel_diff, 1e-3)


def _factor_panel(strength=0.85, seed=42, n_days=900, n_series=15, **kwargs):
    """Latent-factor panel from the generator, plus its ground truth."""
    gen = generate_synthetic_daily_data(
        n_days=n_days,
        n_series=n_series,
        random_seed=seed,
        n_latent_factors=3,
        factor_strength=strength,
        series_type_override='standard',
        noise_level=0.05,
        **kwargs,
    )
    data = gen.get_data()
    trends = pd.DataFrame(
        {name: gen.get_components(name)['trend'] for name in data.columns},
        index=data.index,
    )
    return gen, data, trends


def _recovery_scores(gen, panel, config=None):
    """Run discovery on `panel` and score it against the generator's truth."""
    truth = gen.get_true_factors()
    result = discover_structure(panel, config=config or FAST_CONFIG, seed=42)
    matched = match_factors(truth['factors'], result['factors'])

    true_loadings = truth['loadings'].reindex(panel.columns).values
    nonzero = np.abs(true_loadings).max(axis=1) > 0
    true_dominant = np.abs(true_loadings).argmax(axis=1)
    inverse = {est: true for true, est in matched['assignment'].items()}
    mapped = np.array([inverse.get(int(c), -1) for c in result['communities']])
    accuracy = (
        float((mapped[nonzero] == true_dominant[nonzero]).mean())
        if nonzero.any()
        else 0.0
    )
    return {
        'n_factors': int(result['loadings'].shape[1]),
        'mean_abs_corr': matched['mean_abs_corr'],
        'community_accuracy': accuracy,
    }


class TestGeneratorFactorRecovery(unittest.TestCase):
    """Discovery against SyntheticDailyGenerator latent-factor ground truth.

    Thresholds were calibrated empirically on these fixed seeds and set
    ~0.1 below the observed values, per this suite's convention.
    """

    def test_generator_factor_recovery_oracle(self):
        """On true trend components, factors and communities are recovered."""
        gen, _data, trends = _factor_panel(strength=0.85, seed=42)
        scores = _recovery_scores(gen, trends)
        self.assertEqual(scores['n_factors'], 3, "should find all three factors")
        self.assertGreaterEqual(scores['mean_abs_corr'], 0.8)
        self.assertGreaterEqual(scores['community_accuracy'], 0.7)

    def test_generator_factor_recovery_raw_is_much_weaker(self):
        """Raw daily data is NOT a substitute for a decomposed trend.

        A piecewise-linear trend contributes almost no variance to first
        differences (its diff is a step function that only moves at
        changepoints), so observation noise swamps the factor signal in the
        differenced panel discovery operates on. This is the decomposition
        bottleneck made measurable: it guards against anyone concluding that
        discovery can be fed raw series.
        """
        gen, data, trends = _factor_panel(strength=0.85, seed=42)
        raw = _recovery_scores(gen, data)
        oracle = _recovery_scores(gen, trends)
        self.assertLess(raw['mean_abs_corr'], 0.35)
        self.assertGreater(oracle['mean_abs_corr'] - raw['mean_abs_corr'], 0.4)

    def test_generator_factor_recovery_low_strength_degrades(self):
        """Weak factor structure scores below strong factor structure."""
        gen_strong, _d, trends_strong = _factor_panel(strength=0.85, seed=42)
        gen_weak, _d2, trends_weak = _factor_panel(strength=0.3, seed=42)
        strong = _recovery_scores(gen_strong, trends_strong)
        weak = _recovery_scores(gen_weak, trends_weak)
        self.assertGreater(strong['mean_abs_corr'], weak['mean_abs_corr'])
        self.assertGreaterEqual(
            strong['community_accuracy'], weak['community_accuracy']
        )


class TestFactorHPSmoothing(unittest.TestCase):
    """The opt-in `factor_hp_lambda` pre-smoothing and its measured tradeoff."""

    def test_hp_smoothing_off_by_default(self):
        from autots.evaluator.tva.discovery import DEFAULT_DISCOVERY_CONFIG

        self.assertIsNone(DEFAULT_DISCOVERY_CONFIG['factor_hp_lambda'])

    def test_hp_smoothing_rescues_noise_dominated_input(self):
        """On raw daily data, pre-smoothing lifts recovery by a wide margin."""
        gen, data, _trends = _factor_panel(strength=0.85, seed=42)
        off = _recovery_scores(gen, data)
        smoothed_config = dict(FAST_CONFIG, factor_hp_lambda=1e8)
        on = _recovery_scores(gen, data, config=smoothed_config)
        self.assertLess(off['mean_abs_corr'], 0.35)
        self.assertGreater(on['mean_abs_corr'], 0.45)

    def test_hp_smoothing_costs_accuracy_on_clean_input(self):
        """It is a tradeoff, not a free win — it hurts an already-clean panel.

        Guards against anyone promoting it to a global default without
        re-measuring the clean-input and edge-precision costs.
        """
        gen, _data, trends = _factor_panel(strength=0.85, seed=42)
        off = _recovery_scores(gen, trends)
        smoothed_config = dict(FAST_CONFIG, factor_hp_lambda=1e8)
        on = _recovery_scores(gen, trends, config=smoothed_config)
        self.assertGreater(off['mean_abs_corr'], on['mean_abs_corr'])


class TestTorchFreeMode(unittest.TestCase):
    """trend_network='none' must never touch torch.

    Regression test: the torch-free mode shipped raising
    ``NameError: name 'torch' is not defined`` on any machine without torch,
    because device selection in ``__init__`` and the training-tensor
    construction in ``fit()`` both ran before the 'none' early return.
    """

    @staticmethod
    def _panel():
        rng = np.random.default_rng(0)
        index = pd.date_range('2021-01-01', periods=400, freq='D')
        return pd.DataFrame(
            {
                f'tf_{i}': 50 + np.cumsum(rng.normal(0.02, 0.5, 400))
                for i in range(5)
            },
            index=index,
        )

    def test_none_mode_without_torch(self):
        from unittest import mock

        import autots.evaluator.tva.tva as tva_module

        df = self._panel()
        with mock.patch.object(tva_module, 'HAS_TORCH', False):
            saved = tva_module.__dict__.pop('torch', None)
            try:
                model = tva_module.TVA(trend_network='none', verbose=0)
                model.fit(df)
                forecast = model.predict(14)
            finally:
                if saved is not None:
                    tva_module.torch = saved
        values = forecast['trend'] if isinstance(forecast, dict) else forecast
        self.assertEqual(np.asarray(values).shape, (14, df.shape[1]))

    def test_network_modes_require_torch(self):
        from unittest import mock

        import autots.evaluator.tva.tva as tva_module

        with mock.patch.object(tva_module, 'HAS_TORCH', False):
            with self.assertRaises(ImportError):
                tva_module.TVA(trend_network='v2')


class TestMatchFactors(unittest.TestCase):
    def test_match_factors_identity(self):
        """Permuted, sign-flipped, rescaled factors recover at corr ~= 1."""
        rng = np.random.default_rng(0)
        factors = np.cumsum(rng.normal(0, 1, (400, 3)), axis=0)
        permuted = factors[:, [2, 0, 1]] * np.array([-1.0, 1.0, 2.5])
        matched = match_factors(factors, permuted)
        self.assertEqual(matched['assignment'], {0: 1, 1: 2, 2: 0})
        self.assertEqual(matched['signs'], {0: 1, 1: 1, 2: -1})
        self.assertGreater(matched['mean_abs_corr'], 0.999)

    def test_match_factors_penalizes_missing_factors(self):
        """Recovering only 2 of 3 factors cannot score a perfect match."""
        rng = np.random.default_rng(1)
        factors = np.cumsum(rng.normal(0, 1, (300, 3)), axis=0)
        matched = match_factors(factors, factors[:, :2])
        self.assertEqual(len(matched['assignment']), 2)
        self.assertAlmostEqual(matched['mean_abs_corr'], 2.0 / 3.0, places=3)

    def test_match_factors_handles_empty(self):
        matched = match_factors(np.zeros((0, 2)), np.zeros((0, 2)))
        self.assertEqual(matched['mean_abs_corr'], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
