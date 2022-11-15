import numpy as np
import unittest
from autots import load_linear
from autots.tools.percentile import nan_quantile, nan_percentile
import timeit


class TestImpute(unittest.TestCase):

    def test_percentile(self):
        print("Starting test_percentile")
        df = load_linear(long=False, shape=(300, 10000), introduce_nan=0.2, introduce_random=100)
        arr = df.to_numpy()

        old_func = np.nanpercentile(arr, q=range(0, 100), axis=0)
        arr = df.to_numpy()
        new_func = nan_percentile(arr, q=range(0, 100))
        self.assertTrue(np.allclose(new_func, old_func))

        arr = df.to_numpy().copy()
        res1 = nan_quantile(arr, q=0.5).round(3)
        res2 = df.quantile(0.5).values.round(3)
        self.assertTrue(
            (res1 == res2).all()
        )
        self.assertTrue(
            np.allclose(nan_quantile(arr, q=0.5), np.nanquantile(arr, 0.5, axis=0))
        )

        start_time = timeit.default_timer()
        res1 = nan_percentile(arr, q=[10, 25, 50, 75, 90])
        runtime_custom = timeit.default_timer() - start_time

        start_time = timeit.default_timer()
        res2 = np.nanpercentile(arr, q=[10, 25, 50, 75, 90], axis=0)
        runtime_np = timeit.default_timer() - start_time

        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(
            runtime_custom < runtime_np,
            "Failed to assert custom percentile was faster than numpy percentile. Rerun may fix."
        )
