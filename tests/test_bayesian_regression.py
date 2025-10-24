import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from autots.models.sklearn import retrieve_regressor
from autots.tools.bayesian_regression import BayesianMultiOutputRegression


def _make_data(n_samples=80, n_features=4, n_outputs=2, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_features, n_outputs))
    Y = X @ coef + rng.normal(scale=noise, size=(n_samples, n_outputs))
    return X, Y, coef


class TestBayesianRegression(unittest.TestCase):
    def test_multi_output_shapes(self):
        X, Y, coef = _make_data()
        model = BayesianMultiOutputRegression(alpha=0.5, wishart_prior_scale=0.2)
        model.fit(X, Y)

        self.assertEqual(model.coef_mean_.shape, coef.shape)
        self.assertEqual(model.coef_std_.shape, coef.shape)
        self.assertTrue(np.all(np.isfinite(model.coef_mean_)))
        self.assertTrue(np.all(model.coef_std_ >= 0))

        preds, pred_std = model.predict(X[:5], return_std=True)
        self.assertEqual(preds.shape, (5, Y.shape[1]))
        self.assertEqual(pred_std.shape, (5, Y.shape[1]))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(pred_std > 0))

        lower, upper = model.coefficient_interval()
        self.assertEqual(lower.shape, coef.shape)
        self.assertTrue(np.all(lower <= upper))

        samples = model.sample_posterior(3)
        self.assertEqual(samples.shape, (3,) + coef.shape)

    def test_single_output(self):
        X, Y, _ = _make_data(n_outputs=1, noise=0.05, seed=1)
        y = Y.ravel()
        model = BayesianMultiOutputRegression(alpha=1.0, wishart_prior_scale=0.1)
        model.fit(X, y)

        preds = model.predict(X[:4])
        self.assertEqual(preds.shape, (4,))
        preds, pred_std = model.predict(X[:4], return_std=True)
        self.assertEqual(preds.shape, (4,))
        self.assertEqual(pred_std.shape, (4,))
        self.assertTrue(np.all(pred_std > 0))

    def test_retrieve_regressor(self):
        X, Y, _ = _make_data(n_outputs=1, noise=0.2, seed=2)
        y = Y.ravel()
        reg = retrieve_regressor(
            {
                "model": "BayesianMultiOutputRegression",
                "model_params": {"alpha": 0.3, "wishart_prior_scale": 0.5},
            },
            multioutput=False,
        )
        reg.fit(X, y)
        preds = reg.predict(X[:3])
        self.assertEqual(preds.shape, (3,))


if __name__ == "__main__":
    unittest.main()
