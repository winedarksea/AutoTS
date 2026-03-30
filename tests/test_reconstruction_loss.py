import unittest
import pandas as pd
import numpy as np
from autots.evaluator.feature_detector import ReconstructionLoss


class TestReconstructionLoss(unittest.TestCase):
    def setUp(self):
        self.loss_calculator = ReconstructionLoss()
        self.index = pd.date_range(start="2023-01-01", periods=100, freq="D")
        self.observed = pd.Series(np.random.randn(100), index=self.index)

    def test_noise_whiteness_penalty(self):
        # Test with white noise
        white_noise = pd.Series(np.random.normal(0, 1, 100), index=self.index)
        penalty_white = self.loss_calculator._noise_whiteness_penalty(white_noise)
        self.assertLess(penalty_white, 1.0, "White noise should have low penalty")

        # Test with correlated noise
        correlated_noise = white_noise.rolling(window=3).mean().fillna(0)
        penalty_corr = self.loss_calculator._noise_whiteness_penalty(correlated_noise)
        self.assertGreater(
            penalty_corr, penalty_white, "Correlated noise should have higher penalty"
        )