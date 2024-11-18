"""Fill NA."""

import numpy as np
import pandas as pd
from autots.tools.seasonal import seasonal_independent_match

try:
    from sklearn.impute import KNNImputer

    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa
    except Exception:
        pass
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import IterativeImputer
except Exception:
    pass


def fill_zero(df):
    """Fill NaN with zero."""
    df = df.fillna(0)
    return df


def fill_one(df):
    """Fill NaN with zero."""
    return df.fillna(1)


def fillna_np(array, values):
    if np.isnan(array.sum()):
        array = np.nan_to_num(array) + np.isnan(array) * values
    return array


def fill_forward_alt(df):
    """Fill NaN with previous values."""
    # this is faster if only some columns have NaN
    df2 = df.copy()
    for i in df2.columns[df2.isnull().any(axis=0)]:
        df2[i] = df2[i].ffill().bfill().fillna(0)
    return df2


def fill_forward(df):
    """Fill NaN with previous values."""
    df = df.ffill()
    return df.bfill().fillna(0)


def fill_mean_old(df):
    """Fill NaN with mean."""
    df = df.fillna(df.mean().fillna(0).to_dict())
    return df


def fill_mean(df):
    arr = np.array(df)
    arr = np.nan_to_num(arr) + np.isnan(arr) * np.nan_to_num(np.nanmean(arr, axis=0))
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def fill_median_old(df):
    """Fill NaN with median."""
    df = df.fillna(df.median().fillna(0).to_dict())
    return df


def fill_median(df):
    """Fill nan with median values. Does not work with non-numeric types."""
    arr = np.array(df)
    arr = np.nan_to_num(arr) + pd.isna(arr) * np.nan_to_num(np.nanmedian(arr, axis=0))
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def rolling_mean(df, window: int = 10):
    """Fill NaN with mean of last window values."""
    df = fill_forward(df.fillna(df.rolling(window=window, min_periods=1).mean()))
    return df


def biased_ffill(df, mean_weight: float = 1):
    """Fill NaN with average of last value and mean."""
    df_mean = fill_mean(df)
    df_ffill = fill_forward(df)
    df = ((df_mean * mean_weight) + df_ffill) / (1 + mean_weight)
    return df


def fake_date_fill_old(df, back_method: str = 'slice'):
    """
    Return a dataframe where na values are removed and values shifted forward.

    Warnings:
        Thus, values will have incorrect timestamps!

    Args:
        back_method (str): how to deal with tails left by shifting NaN
            - 'bfill' -back fill the last value
            - 'slice' - drop any rows above threshold where half are nan, then bfill remainder
            - 'slice_all' - drop any rows with any na
            - 'keepna' - keep the lagging na
    """
    df2 = df.sort_index(ascending=False).copy()
    df2 = df2.apply(lambda x: pd.Series(x.dropna().values))
    df2 = df2.sort_index(ascending=False)
    df2.index = df.index[-df2.shape[0] :]
    # df2 = df2.dropna(how='all', axis=0)
    if df2.empty:
        df2 = df.fillna(0)

    if back_method == 'bfill':
        return fill_forward(df2)
    elif back_method == 'slice':
        # cut until half of columns are not NaN then backfill
        thresh = int(df.shape[1] * 0.5)
        thresh = thresh if thresh > 1 else 1
        df3 = df2.dropna(thresh=thresh, axis=0)
        if df3.empty or df3.shape[0] < 8:
            return fill_forward(df2)
        else:
            return fill_forward(df3)
    elif back_method == 'slice_all':
        return df2.dropna(how="any", axis=0)
    elif back_method == 'keepna':
        return df2
    else:
        print('back_method not recognized in fake_date_fill')
        return df2


def fake_date_fill(df, back_method: str = 'slice'):
    """Numpy vectorized version.
    Return a dataframe where na values are removed and values shifted forward.

    Warnings:
        Thus, values will have incorrect timestamps!

    Args:
        back_method (str): how to deal with tails left by shifting NaN
            - 'bfill' -back fill the last value
            - 'slice' - drop any rows above threshold where half are nan, then bfill remainder
            - 'slice_all' - drop any rows with any na
            - 'keepna' - keep the lagging na
    """
    arr = np.array(df)
    indices = np.indices(arr.shape)
    indices0 = indices[0]
    # basically takes advantage of indices always positive to make then
    # conditionally negative to float up in sort
    to_sort = np.where(np.isnan(arr), -indices0, indices0)
    df2 = pd.DataFrame(
        arr[np.abs(np.sort(to_sort, axis=0)), indices[1]],
        index=df.index,
        columns=df.columns,
    )
    if df2.empty:
        df2 = df.fillna(0)

    if back_method == 'bfill':
        return fill_forward(df2)
    elif back_method == 'slice':
        # cut until half of columns are not NaN then backfill
        thresh = int(df.shape[1] * 0.5)
        thresh = thresh if thresh > 1 else 1
        df3 = df2.dropna(thresh=thresh, axis=0)
        if df3.empty or df3.shape[0] < 8:
            return fill_forward(df2)
        else:
            return fill_forward(df3)
    elif back_method == 'slice_all':
        return df2.dropna(how="any", axis=0)
    elif back_method == 'keepna':
        return df2
    else:
        raise ValueError('back_method not recognized in fake_date_fill')


df_interpolate = {
    'linear': 0.1,
    'time': 0.1,
    # 'pad': 0.1,  # deprecated for reasons unknown
    'nearest': 0.1,
    'zero': 0.1,
    'quadratic': 0.1,
    'cubic': 0.1,
    'barycentric': 0.01,  # this parallelizes and is noticeably slower
    'piecewise_polynomial': 0.01,
    'spline': 0.01,  # can fail sometimes
    'pchip': 0.1,
    'akima': 0.1,
    # these seem to cause more harm than good usually
    'polynomial': 0.0,
    'krogh': 0.0,
    'cubicspline': 0.0,
    'from_derivatives': 0.0,
    'slinear': 0.0,
}
df_interpolate_full = list(df_interpolate.keys())


def FillNA(df, method: str = 'ffill', window: int = 10):
    """Fill NA values using different methods.

    Args:
        method (str):
            'ffill' - fill most recent non-na value forward until another non-na value is reached
            'zero' - fill with zero. Useful for sales and other data where NA does usually mean $0.
            'mean' - fill all missing values with the series' overall average value
            'median' - fill all missing values with the series' overall median value
            'rolling mean' - fill with last n (window) values
            'ffill mean biased' - simple avg of ffill and mean
            'fake date' - shifts forward data over nan, thus values will have incorrect timestamps
            also most `method` values of pd.DataFrame.interpolate()
        window (int): length of rolling windows for filling na, for rolling methods
    """
    if isinstance(method, (int, float)):
        return df.fillna(method)

    method = str(method).replace(" ", "_")

    if method == 'zero':
        return fill_zero(df)

    elif method == 'ffill':
        return fill_forward(df)

    elif method == 'mean':
        return fill_mean(df)

    elif method == 'median':
        return fill_median(df)

    elif method == 'rolling_mean':
        return rolling_mean(df, window=window)

    elif method == 'rolling_mean_24':
        return rolling_mean(df, window=24)

    elif method == 'ffill_mean_biased':
        return biased_ffill(df)

    elif method == 'fake_date':
        return fake_date_fill(df, back_method='slice')

    elif method == 'fake_date_slice':
        return fake_date_fill(df, back_method='slice_all')

    elif method in df_interpolate_full:
        df = df.interpolate(method=method, order=5).bfill()
        if df.isnull().values.any():
            df = fill_forward(df)
        return df

    elif method == 'IterativeImputer':
        cols = df.columns
        indx = df.index

        df = IterativeImputer(random_state=0, max_iter=100).fit_transform(df)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
            df.index = indx
            df.columns = cols
        return df

    elif method == 'IterativeImputerExtraTrees':
        cols = df.columns
        indx = df.index

        df_local = IterativeImputer(
            ExtraTreesRegressor(n_estimators=10, random_state=0),
            random_state=0,
            max_iter=100,
        ).fit_transform(df)
        if not isinstance(df_local, pd.DataFrame):
            df_local = pd.DataFrame(df_local)
            df_local.index = indx
            df_local.columns = cols
        return df_local

    elif method == 'KNNImputer':
        cols = df.columns
        indx = df.index

        df_knn = KNNImputer(n_neighbors=5).fit_transform(df)
        if not isinstance(df_knn, pd.DataFrame):
            df_knn = pd.DataFrame(df_knn)
            df_knn.index = indx
            df_knn.columns = cols
        return df_knn

    elif method == 'SeasonalityMotifImputer':
        s_imputer = SeasonalityMotifImputer(
            k=3,
            datepart_method="common_fourier",
            distance_metric="canberra",
            linear_mixed=False,
        )
        return s_imputer.impute(df)  # .rename(lambda x: str(x) + "_motif", axis=1)

    elif method == 'SimpleSeasonalityMotifImputer':
        s_imputer = SimpleSeasonalityMotifImputer(
            datepart_method="common_fourier",
            distance_metric="canberra",
            linear_mixed=False,
        )
        return s_imputer.impute(df)  # .rename(lambda x: str(x) + "_motif", axis=1)

    elif method == 'SeasonalityMotifImputer1K':
        s_imputer = SimpleSeasonalityMotifImputer(
            datepart_method="common_fourier",
            distance_metric="mae",
            linear_mixed=False,
        )
        return s_imputer.impute(df)  # .rename(lambda x: str(x) + "_motif", axis=1)

    elif method == 'SeasonalityMotifImputerLinMix':
        s_imputer = SimpleSeasonalityMotifImputer(
            datepart_method="common_fourier",
            distance_metric="canberra",
            linear_mixed=True,
        )
        return s_imputer.impute(df)  # .rename(lambda x: str(x) + "_motif", axis=1)

    elif method == 'DatepartRegressionImputer':
        # circular import
        from autots.tools.transform import DatepartRegressionTransformer

        imputer = DatepartRegressionTransformer(
            datepart_method="common_fourier",
            holiday_country=["US"],
            holiday_countries_used=True,
            regression_model={
                "model": 'RandomForest',
                "model_params": {
                    'n_estimators': 150,
                    'min_samples_leaf': 1,
                    'bootstrap': False,
                },
            },
        )
        imputer.fit(df)
        return imputer.impute(df)

    elif method is None or method in ['None', 'null']:
        return df

    elif method == 'one':
        return fill_one(df)

    else:
        print(f"FillNA method `{str(method)}` not known, returning original")
        return df


class SeasonalityMotifImputer(object):
    def __init__(
        self,
        k: int = 3,
        datepart_method: str = "simple_2",
        distance_metric: str = "canberra",
        linear_mixed: bool = False,
    ):
        """Shares arg params with SeasonalityMotif model with which it has much in common.

        Args:
            k (int): n neighbors. More is smoother, fewer is most accurate, usually
            datepart_method (str): standard date part methods accepted
            distance_metirc (str): same as seaonality motif, ie 'mae', 'canberra'
            linear_mixed (bool): if True, take simple average of this and linear interpolation
        """
        self.k = k
        self.datepart_method = datepart_method
        self.distance_metric = distance_metric
        self.linear_mixed = linear_mixed

    def impute(self, df):
        """Infer missing values on input df."""
        test, scores = seasonal_independent_match(
            DTindex=df.index,
            DTindex_future=df.index,
            k=df.shape[0] - 1,  # not really used here
            datepart_method=self.datepart_method,
            distance_metric=self.distance_metric,
        )
        full_dist = np.argsort(scores)
        full_nan_mask = ~np.isnan(df.to_numpy())

        brdcst_mask = np.broadcast_to(
            full_nan_mask[..., None], full_nan_mask.shape + (df.shape[0],)
        ).T
        brdcst_mask = np.moveaxis(
            np.broadcast_to(
                full_nan_mask[..., None], full_nan_mask.shape + (df.shape[0],)
            ),
            0,
            0,
        )
        # brdcst = np.array(np.broadcast_to(full_dist[...,None],full_dist.shape+(df.shape[1],)))  # .reshape(brdcst_mask.shape)
        # this uses WAY too much memory
        brdcst = np.moveaxis(
            np.broadcast_to(full_dist[..., None], full_dist.shape + (df.shape[1],)),
            -1,
            1,
        )
        del full_dist

        # mask_positive = (np.cumsum(~brdcst_mask, axis=-1) <= k) & ~brdcst_mask  # True = keeps
        # mask_negative = (np.cumsum(~brdcst_mask, axis=-1) > k) | brdcst_mask  # True = don't keep
        mask_negative = np.cumsum(brdcst_mask, axis=-1) > self.k  # True = don't keep

        # test = np.ma.masked_array(brdcst.T, mask_negative)
        # temp = np.take(df.to_numpy()[..., None], brdcst)
        # arrd = np.take(df.to_numpy().T, brdcst).T

        arrd = np.take_along_axis(
            np.broadcast_to(df.to_numpy()[..., None], df.shape + (df.shape[0],)),
            brdcst,
            axis=0,
        )
        arrd_mask = np.isnan(arrd)
        mask_negative = np.cumsum(~arrd_mask, axis=-1) > self.k  # True = don't keep
        arrd[arrd_mask] = 0
        temp = np.ma.masked_array(arrd, mask_negative)
        test = (temp.sum(axis=2) / self.k).data
        self.df_impt = pd.DataFrame(test, index=df.index, columns=df.columns)
        if self.linear_mixed:
            self.df_impt = self.df_impt - 0.5 * (
                self.df_impt.rolling(14, min_periods=1, center=True).mean()
                - df.interpolate("linear")
            )

        # col = "US__sv_feed_interface"
        # pd.concat([df.loc[:, col], df_impt.loc[:, col + "_imputed"]], axis=1).plot()

        return df.where(full_nan_mask, self.df_impt)


class SimpleSeasonalityMotifImputer(object):
    def __init__(
        self,
        datepart_method: str = "simple_2",
        distance_metric: str = "canberra",
        linear_mixed: bool = False,
        max_iter: int = 100,
    ):
        """Shares arg params with SeasonalityMotif model with which it has much in common.
        Only takes the nearest one non-nan neighbor.
        This isn't quite as fast as the other version but doesn't explode into terabytes of memory at scale, either.

        Args:
            datepart_method (str): standard date part methods accepted
            distance_metirc (str): same as seaonality motif, ie 'mae', 'canberra'
            linear_mixed (bool): if True, take simple average of this and linear interpolation
        """
        self.datepart_method = datepart_method
        self.distance_metric = distance_metric
        self.linear_mixed = linear_mixed
        self.max_iter = max_iter

    def impute(self, df):
        """Infer missing values on input df."""
        # discard rows where all series are NaN
        all_nan = df.isnull().all(axis=1)
        test, scores = seasonal_independent_match(
            DTindex=df.index,
            DTindex_future=df.index,
            k=df.shape[0] - 1,  # not really used here
            datepart_method=self.datepart_method,
            distance_metric=self.distance_metric,
            full_sort=True,
            nan_array=all_nan,
        )
        count = 0
        arr = df.to_numpy()
        while count < self.max_iter:
            current_fill = arr[test[:, count]]
            arr = np.where(np.isnan(arr), current_fill, arr)
            if not np.isnan(np.min(arr)):
                break
            # df.update(df.where(df.notna(), current_fill), overwrite=False)
            count += 1
        else:
            # if iters run out, do a basic nan fill
            arr = np.nan_to_num(arr)

        return pd.DataFrame(arr, index=df.index, columns=df.columns)


# accuracy test (not necessarily a test of "best")
if False:
    from autots import load_daily
    from autots.tools.transform import na_probs

    df_daily = load_daily(long=False)
    start = -400
    end = -300
    test = df_daily.iloc[start:end].copy()
    df_daily.iloc[start:end] = np.nan
    impute_mape = na_probs.copy()
    impute_mape = {**impute_mape, **df_interpolate}
    for key in impute_mape.keys():
        df_imputed = FillNA(df_daily, method=key, window=10)
        impute_mape[key] = (
            (df_imputed.iloc[start:end] - test).abs().mean() / df_daily.mean()
        ).mean()
    impute_mape
