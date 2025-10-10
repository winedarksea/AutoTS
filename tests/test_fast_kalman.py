import os
import random
import unittest

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from autots.tools.fast_kalman import KalmanFilter
from autots.tools.fast_kalman_params import new_kalman_params


class TestFastKalman(unittest.TestCase):
    def setUp(self):
        self.kf = KalmanFilter(
            state_transition=[[1, 1], [0, 1]],
            process_noise=np.diag([0.1, 0.01]),
            observation_model=[[1, 0]],
            observation_noise=1.0,
        )

    def test_smooth_produces_expected_shapes(self):
        data = np.linspace(0.0, 4.0, num=5).reshape(1, -1)
        result = self.kf.smooth(data, covariances=False)

        self.assertEqual(result.observations.mean.shape, data.shape)
        self.assertEqual(result.states.mean.shape[:2], (data.shape[0], data.shape[1]))
        self.assertTrue(np.all(np.isfinite(result.observations.mean)))

    def test_predict_observations_length(self):
        data = np.linspace(0.0, 4.0, num=5).reshape(1, -1)
        forecast_horizon = 3
        result = self.kf.predict(data, forecast_horizon, covariances=False)

        self.assertEqual(result.observations.mean.shape, (1, forecast_horizon))
        self.assertTrue(np.all(np.isfinite(result.observations.mean)))


class TestFastKalmanParams(unittest.TestCase):
    def test_new_kalman_params_structure(self):
        random.seed(1)
        params = new_kalman_params()

        self.assertIn("state_transition", params)
        self.assertIn("process_noise", params)
        self.assertIn("observation_model", params)
        self.assertIn("observation_noise", params)
        self.assertIn("em_iter", params)

        transition = np.asarray(params["state_transition"])
        process_noise = np.asarray(params["process_noise"])
        observation_model = np.asarray(params["observation_model"])

        self.assertEqual(transition.shape, process_noise.shape)
        self.assertEqual(observation_model.shape[-1], transition.shape[-1])
        self.assertGreaterEqual(process_noise.shape[0], 1)

    def test_ucm_random_walk_parameters_present(self):
        random.seed(30)
        params = new_kalman_params()

        self.assertEqual(params["model_name"], "ucm_random_walk_drift_ar1")
        self.assertEqual(params["level"], "random walk with drift")
        self.assertEqual(params["cov_type"], "opg")
        self.assertEqual(params["autoregressive"], 1)


if __name__ == "__main__":
    unittest.main()
