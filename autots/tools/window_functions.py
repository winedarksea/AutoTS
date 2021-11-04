import numpy as np
import pandas as pd


def window_maker(
    df,
    window_size: int = 10,
    input_dim: str = 'univariate',
    normalize_window: bool = False,
    shuffle: bool = False,
    output_dim: str = 'forecast_length',
    forecast_length: int = 1,
    max_windows: int = 5000,
    regression_type: str = None,
    future_regressor=None,
    random_seed: int = 1234,
):
    """Convert a dataset into slices with history and y forecast.

    Args:
        df (pd.DataFrame): `wide` format df with sorted index
        window_size (int): length of history to use for X window
        input_dim (str): univariate or multivariate. If multivariate, all series in single X row
        shuffle (bool): (deprecated)
        output_dim (str): 'forecast_length' or '1step' where 1 step is basically forecast_length=1
        forecast_length (int): number of periods ahead that will be forecast
        max_windows (int): a cap on total number of windows to generate. If exceeded, random of this int are selected.
        regression_type (str): None or "user" if to try to concat regressor to windows
        future_regressor (pd.DataFrame): values of regressor if used
        random_seed (int): a consistent random

    Returns:
        X, Y
    """
    if output_dim == '1step':
        forecast_length = 1
    phrase_n = forecast_length + window_size
    try:
        if input_dim == "multivariate":
            raise ValueError("input_dim=`multivariate` not supported this way.")
        x = np.lib.stride_tricks.sliding_window_view(df.to_numpy(), phrase_n, axis=0)
        x = x.reshape(-1, x.shape[-1])
        Y = x[:, window_size:]
        if Y.ndim > 1:
            if Y.shape[1] == 1:
                Y = Y.ravel()
        X = x[:, :window_size]
        r_arr = None
        if max_windows is not None:
            X_size = x.shape[0]
            if max_windows < X_size:
                r_arr = np.random.default_rng(random_seed).integers(
                    0, X_size, size=max_windows
                )
                Y = Y[r_arr]
                X = X[r_arr]
        if normalize_window:
            div_sum = np.nansum(X, axis=1).reshape(-1, 1)
            X = X / np.where(div_sum == 0, 1, div_sum)
        # regressors
        if str(regression_type).lower() == "user":
            shape_1 = df.shape[1] if df.ndim > 1 else 1
            if isinstance(future_regressor, pd.DataFrame):
                regr_arr = np.repeat(
                    future_regressor.reindex(df.index).to_numpy()[(phrase_n - 1) :],
                    shape_1,
                    axis=0,
                )
                if r_arr is not None:
                    regr_arr = regr_arr[r_arr]
                X = np.concatenate([X, regr_arr], axis=1)

    except Exception:
        if str(regression_type).lower() == "user":
            if input_dim == "multivariate":
                raise ValueError(
                    "input_dim=`multivariate` and regression_type=`user` cannot be combined."
                )
            else:
                raise ValueError(
                    "WindowRegression regression_type='user' requires numpy >= 1.20"
                )
        max_pos_wind = df.shape[0] - phrase_n + 1
        max_pos_wind = max_windows if max_pos_wind > max_windows else max_pos_wind
        if max_pos_wind == max_windows:
            numbers = np.random.default_rng(random_seed).choice(
                (df.shape[0] - phrase_n), size=max_pos_wind, replace=False
            )
            if not shuffle:
                numbers = np.sort(numbers)
        else:
            numbers = np.array(range(max_pos_wind))
            if shuffle:
                np.random.shuffle(numbers)

        X = pd.DataFrame()
        Y = pd.DataFrame()
        for z in numbers:
            if input_dim == 'univariate':
                rand_slice = df.iloc[
                    z : (z + phrase_n),
                ]
                rand_slice = (
                    rand_slice.reset_index(drop=True)
                    .transpose()
                    .set_index(np.repeat(z, (df.shape[1],)), append=True)
                )
                cX = rand_slice.iloc[:, 0:(window_size)]
                cY = rand_slice.iloc[:, window_size:]
            else:
                cX = df.iloc[
                    z : (z + window_size),
                ]
                cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
                cY = df.iloc[
                    (z + window_size) : (z + phrase_n),
                ]
                cY = pd.DataFrame(cY.stack().reset_index(drop=True)).transpose()
            X = pd.concat([X, cX], axis=0)
            Y = pd.concat([Y, cY], axis=0)
        if normalize_window:
            X = X.div(X.sum(axis=1), axis=0)
        X.columns = [str(x) for x in range(len(X.columns))]

    return X, Y


def last_window(
    df,
    window_size: int = 10,
    input_dim: str = 'univariate',
    normalize_window: bool = False,
):
    """Pandas based function to provide the last window of window_maker."""
    z = df.shape[0] - window_size
    shape_1 = df.shape[1] if df.ndim > 1 else 1
    if input_dim == 'univariate':
        cX = df.iloc[
            z : (z + window_size),
        ]
        cX = (
            cX.reset_index(drop=True)
            .transpose()
            .set_index(np.repeat(z, (shape_1,)), append=True)
        )
    else:
        cX = df.iloc[
            z : (z + window_size),
        ]
        cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
    if normalize_window:
        cX = cX.div(cX.sum(axis=1), axis=0)

    return cX


def window_id_maker(
    window_size: int,
    max_steps: int,
    start_index: int = 0,
    stride_size: int = 1,
    skip_size: int = 1,
):
    """Create indices for array of multiple window slices of data

    Args:
        window_size (int): length of time history to include
        max_steps (int): the maximum number of windows to create
        start_index (int): if to not start at the first point, start at this point
        stride_size (int): number of skips between each window start point
        skip_size (int): number of skips between each obs in a window (downsamples)

    Returns:
        np.array with 3D shape (num windows, window_length, num columns/series), 2D array if only 1D `array` provided)
    """
    window_idxs = (
        start_index
        + np.expand_dims(np.arange(window_size, step=skip_size), 0)
        + np.expand_dims(np.arange(max_steps + 1, step=stride_size), 0).T
    )

    return window_idxs


def window_maker_2(
    array,
    window_size: int,
    max_steps: int = None,
    start_index: int = 0,
    stride_size: int = 1,
    skip_size: int = 1,
):
    """Create array of multiple window slices of data
    Note that this returns a different orientation than window_maker_3

    Args:
        array (np.array): source of historic information of shape (num_obs, num_series)
        window_size (int): length of time history to include
        max_steps (int): the maximum number of windows to create
        start_index (int): if to not start at the first point, start at this point
        stride_size (int): number of skips between each window start point
        skip_size (int): number of skips between each obs in a window (downsamples)

    Returns:
        np.array with 3D shape (num windows, window_length, num columns/series), 2D array if only 1D `array` provided)
    """
    if max_steps is None:
        max_steps = array.shape[0] - window_size

    window_idxs = window_id_maker(
        window_size=window_size,
        start_index=start_index,
        max_steps=max_steps,
        stride_size=stride_size,
        skip_size=skip_size,
    )

    return array[window_idxs]


def window_maker_3(array, window_size: int, **kwargs):
    """stride tricks version of window. About 40% faster than window_maker_2
    Note that this returns a different orientation than window_maker_2

    Args:
        array (np.array): in shape of (num_obs, num_series)
        window_size (int): length of slice of history
        **kwargs passed to np.lib.stride_tricks.sliding_window_view

    Returns:
        np.array with 3D shape (num windows, num columns/series, window_length), 2D array if only 1D `array` provided)
    """
    x = np.lib.stride_tricks.sliding_window_view(array, window_size, axis=0, **kwargs)
    return x


def retrieve_closest_indices(
    df,
    num_indices,
    forecast_length,
    window_size: int = 10,
    distance_metric: str = "braycurtis",
    stride_size: int = 1,
    start_index: int = None,
    include_differenced: bool = False,
    include_last: bool = True,
    verbose: int = 0,
):
    """Find next indicies closest to the final segment of forecast_length

    Args:
        df (pd.DataFrame): source data in wide format
        num_indices (int): number of indices to return
        forecast_length (int): length of forecast
        window_size (int): length of comparison
        distance_metric (str): distance measure from scipy and nan_euclidean
        stride_size (int): length of spacing between windows
        start_index (int): index to begin creation of windows from
        include_difference (bool): if True, also compare on differences
    """
    array = df.to_numpy()
    index = df.index
    tlt_len = array.shape[0]
    combined_window_size = window_size + forecast_length
    # remove extra so last segment not included at all
    # have the last window end evenly
    max_steps = array.shape[0] - combined_window_size
    if not include_last:
        max_steps = max_steps - forecast_length
    if start_index is None:
        # handle massive stride size relative to data
        start_index = 0
        if stride_size * 6 < array.shape[0]:
            start_index = max_steps % stride_size
    if num_indices > (max_steps / stride_size):
        raise ValueError("num_validations/num_indices too high for this dataset")
    window_idxs = window_id_maker(
        window_size=combined_window_size,
        start_index=start_index,
        max_steps=max_steps,
        stride_size=stride_size,
        skip_size=1,
    )
    # calculate distance between all points and last window of history
    if distance_metric == "nan_euclidean":
        from sklearn.metrics.pairwise import nan_euclidean_distances

        res = np.array(
            [
                nan_euclidean_distances(
                    array[:, a][window_idxs[:, :window_size]],
                    array[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                )
                for a in range(array.shape[1])
            ]
        )
        if include_differenced:
            array_diff = np.diff(array, n=1, axis=0)
            array_diff = np.concatenate([array_diff[0:1], array_diff])
            res_diff = np.array(
                [
                    nan_euclidean_distances(
                        array_diff[:, a][window_idxs[:, :window_size]],
                        array_diff[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                    )
                    for a in range(array_diff.shape[1])
                ]
            )
            res = np.mean([res, res_diff], axis=0)
    else:
        from scipy.spatial.distance import cdist

        res = np.array(
            [
                cdist(
                    array[:, a][window_idxs[:, :window_size]],
                    array[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                    metric=distance_metric,
                )
                for a in range(array.shape[1])
            ]
        )
        if include_differenced:
            array_diff = np.diff(array, n=1, axis=0)
            array_diff = np.concatenate([array_diff[0:1], array_diff])
            res_diff = np.array(
                [
                    cdist(
                        array_diff[:, a][window_idxs[:, :window_size]],
                        array_diff[(tlt_len - window_size) : tlt_len, a].reshape(1, -1),
                        metric=distance_metric,
                    )
                    for a in range(array_diff.shape[1])
                ]
            )
            res = np.mean([res, res_diff], axis=0)
    # find the lowest distance historical windows
    res_sum = np.nansum(res, axis=0)
    num_top = num_indices
    # partial then full sort
    res_idx = np.argpartition(res_sum, num_top, axis=0)[0:num_top]
    res_idx = res_idx[np.argsort(res_sum[res_idx].flatten())]
    if verbose > 1:
        print(
            f"similarity validation distance metrics: {res_sum[res_idx].flatten()} with last window: {res_sum[-1].item()}"
        )
    select_index = index.to_numpy()[window_idxs[res_idx]]
    if select_index.ndim == 3:
        res_shape = select_index.shape
        select_index = select_index.reshape((res_shape[0], res_shape[2]))
    return select_index
