import numpy as np
import unittest
from autots import load_linear
from autots.tools.percentile import nan_quantile, nan_percentile
import timeit
import warnings


class TestImpute(unittest.TestCase):
    def _assert_all_nan_slices(self, quantile):
        values = np.array(
            [[1.0, np.nan, 9.0], [2.0, np.nan, np.nan],
             [np.nan, np.nan, 6.0], [4.0, np.nan, 3.0],
             [8.0, np.nan, 2.0]]
        )
        for method, numpy_method in [
            ("linear", "linear"), ("nearest", "nearest"),
            ("lowest", "lower"), ("highest", "higher"),
        ]:
            for array in [values, np.stack([values, values[:, ::-1]], axis=1)]:
                for percentile in [50, [0, 50, 100]]:
                    with self.subTest(method=method, shape=array.shape, q=percentile):
                        original = array.copy()
                        keyword = ("method" if np.lib.NumpyVersion(np.__version__) >= "1.22.0"
                                   else "interpolation")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            expected = np.nanpercentile(
                                array, percentile, axis=0, **{keyword: numpy_method}
                            )
                        if quantile:
                            q = (np.asarray(percentile) / 100).tolist()
                            actual = nan_quantile(array, q=q, method=method)
                        else:
                            actual = nan_percentile(array, q=percentile, method=method)
                        np.testing.assert_allclose(actual, expected, equal_nan=True)
                        np.testing.assert_array_equal(array, original)

    def test_percentile_all_nan_slices(self):
        self._assert_all_nan_slices(quantile=False)

    def test_quantile_all_nan_slices(self):
        self._assert_all_nan_slices(quantile=True)

    def test_percentile(self):
        print("Starting test_percentile")
        df = load_linear(
            long=False, shape=(300, 10000), introduce_nan=0.2, introduce_random=100
        )
        arr = df.to_numpy()

        old_func = np.nanpercentile(arr, q=range(0, 100), axis=0)
        arr = df.to_numpy()
        new_func = nan_percentile(arr, q=range(0, 100))
        self.assertTrue(np.allclose(new_func, old_func))

        arr = df.to_numpy().copy()
        res1 = nan_quantile(arr, q=0.5).round(3)
        res2 = df.quantile(0.5).values.round(3)
        self.assertTrue((res1 == res2).all())
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
            "Failed to assert custom percentile was faster than numpy percentile. Rerun may fix.",
        )
