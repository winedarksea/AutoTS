import numpy as np
import pandas as pd


def chunk_reshape(
    arr,
    window_size=10,
    chunk_size=100,
    sample_fraction=None,
    random_seed=7734,
    dtype=np.float32,
):
    """Shifts from (n_records, n_series) to (windows, window_size).
    Multivariate.
    More memory efficient, if not quite as fast as x.reshape(-1, x.shape[-1]) for 3D numpy array.
    """
    if arr.ndim == 1:
        # arr = arr[:,np.newaxis]
        arr = np.reshape(arr, (arr.shape[0], -1))
    num_series = arr.shape[1]
    full_size = int((arr.shape[0] - window_size + 1) * num_series)
    if sample_fraction is None or sample_fraction == 1:
        sample_size = full_size
    elif 0 < sample_fraction < 1:
        sample_size = int(full_size * sample_fraction)
    else:
        sample_size = int(sample_fraction) if sample_fraction < full_size else full_size
    if sample_fraction is not None:
        rng = np.random.default_rng(seed=random_seed)

    # Initialize an empty array if you need to store the entire reshaped data
    reshaped_x = np.empty((sample_size, window_size), dtype=dtype)

    # Process each chunk
    num_chunks = int(num_series / chunk_size)
    if num_chunks < 1:
        num_chunks = 1
    excess_series = num_series % num_chunks
    # samples_per_chunk = int(sample_size / num_chunks)
    samples_per_series = int(sample_size / num_series)
    # excess = int(sample_size % samples_per_chunk)
    excess = int(sample_size % num_series)
    start_spot = 0
    start_slice = 0
    end = chunk_size + excess_series
    end = end if arr.shape[1] > end else arr.shape[1]
    end_spot = samples_per_series * end + excess
    # so we go through each chunk_size of series
    # the first chunk gets the extra series (modulo remainder) + excess samples if uneven
    for start in range(0, num_chunks):
        # print(start_slice, end)
        # print(start_spot, end_spot)
        xA = sliding_window_view(arr[:, start_slice:end], window_size, axis=0)
        if sample_fraction is not None:
            reshaped_x[start_spot:end_spot, :] = rng.choice(
                xA.reshape(-1, xA.shape[-1]), size=end_spot - start_spot, axis=0
            )
        else:
            reshaped_x[start_spot:end_spot, :] = xA.reshape(-1, xA.shape[-1])
        start_spot = int(end_spot)
        end_spot += samples_per_series * chunk_size
        start_slice = end
        end += chunk_size

    return reshaped_x


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
        # I don't think the np.float32 will usually have much accuracy loss
        if False:
            # old way, kept for reference
            x = sliding_window_view(df.to_numpy(dtype=np.float32), phrase_n, axis=0)
            x = x.reshape(-1, x.shape[-1])

            if x.base is None:
                x.resize(-1, x.shape[-1])
            else:
                x = np.resize(x, (-1, x.shape[-1]))
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
        else:
            x = chunk_reshape(
                df.to_numpy(dtype=np.float32),
                phrase_n,
                sample_fraction=max_windows,
                random_seed=random_seed,
            )
            Y = x[:, window_size:]
            if Y.ndim > 1:
                if Y.shape[1] == 1:
                    Y = Y.ravel()
            X = x[:, :window_size]
        if normalize_window:
            div_sum = np.nansum(X, axis=1).reshape(-1, 1)
            X = X / np.where(div_sum == 0, 1, div_sum)
        # regressors
        if str(regression_type).lower() == "user":
            if False:
                # old way, kept for reference
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
            else:
                n = df.shape[0]
                n_cols = df.shape[1]
                # create an i index identical to windows, then select first
                x = chunk_reshape(
                    np.arange(n)[:, None] * np.ones((1, n_cols), dtype=int),
                    phrase_n,
                    sample_fraction=max_windows,
                    random_seed=random_seed,
                    dtype=int,
                )[:, 0:1]
                X = np.concatenate(
                    [X, future_regressor.reindex(df.index).iloc[x.ravel()]], axis=1
                )

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
                rand_slice = df.iloc[z : (z + phrase_n),]
                rand_slice = (
                    rand_slice.reset_index(drop=True)
                    .transpose()
                    .set_index(np.repeat(z, (df.shape[1],)), append=True)
                )
                cX = rand_slice.iloc[:, 0:(window_size)]
                cY = rand_slice.iloc[:, window_size:]
            else:
                cX = df.iloc[z : (z + window_size),]
                cX = pd.DataFrame(cX.stack().reset_index(drop=True)).transpose()
                cY = df.iloc[(z + window_size) : (z + phrase_n),]
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
        cX = df.iloc[z : (z + window_size),]
        cX = (
            cX.reset_index(drop=True)
            .transpose()
            .set_index(np.repeat(z, (shape_1,)), append=True)
        )
    else:
        cX = df.iloc[z : (z + window_size),]
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
    x = sliding_window_view(array, window_size, axis=0, **kwargs)
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


def sliding_window_view(array, window_shape=(0,), axis=None, writeable=False, **kwargs):
    """Toggles between numpy and internal version depending on np.__version__."""
    # check numpy version
    try:
        np_vers = np.__version__.split(".")
        if int(np_vers[0]) <= 1 and int(np_vers[1]) < 20:
            old_verse_flag = True
        else:
            old_verse_flag = False
    except Exception:
        old_verse_flag = True

    if old_verse_flag:
        return rolling_window_view(
            array, window_shape=window_shape, axis=axis, writeable=writeable
        )
    else:
        # only works on numpy >= 1.20
        return np.lib.stride_tricks.sliding_window_view(
            array, window_shape=window_shape, axis=axis, writeable=writeable, **kwargs
        )


def rolling_window_view(array, window_shape=(0,), axis=None, writeable=False):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.

    Based on: https://gist.github.com/seberg/3866040 but designed to match the newer np.sliding_window_view

    Args:
        array (np.array): Array to which the rolling window is applied.
        window_shape (int): Either a single integer to create a window of only the last axis or a
            tuple to create it for the last len(window) axis. 0 can be used as a to ignore a dimension in the window.
        axis (int): If given, must have the same size as window. In this case window is
            interpreted as the size in the dimension given by axis. IE. a window
            of (2, 1) is equivalent to window=2 and axis=-2.

    Returns:
        A view on `array` which is smaller to fit the windows and has windows added
        dimensions (0s not counting), ie. every point of `array` is an array of size
        window.
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window_shape).astype(int)  # maybe crude to cast to int...

    if axis is not None:
        axis = np.atleast_1d(axis)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axis, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    asteps = _asteps

    _wsteps = np.ones_like(window)
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError(
            "`window` * `wsteps` larger then `array` in at least one dimension."
        )

    new_shape = orig_shape  # just renaming...

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    new_shape = np.concatenate((shape, window))
    new_strides = np.concatenate((strides, new_strides))

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(
        array, shape=new_shape, strides=new_strides, writeable=writeable
    )


def window_lin_reg(x, y, w):
    '''From https://stackoverflow.com/questions/70296498/efficient-computation-of-moving-linear-regression-with-numpy-numba/70304475#70304475'''
    sx = sliding_window_view(x, w, axis=0).sum(axis=-1)
    sy = sliding_window_view(y, w, axis=0).sum(axis=-1)
    sx2 = sliding_window_view(x**2, w, axis=0).sum(axis=-1)
    sxy = sliding_window_view(x * y, w, axis=0).sum(axis=-1)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept


def window_sum_nan_mean(x, w, axis=0):
    return np.nanmean(sliding_window_view(x, w, axis=axis), axis=-1)


def window_lin_reg_mean(x, y, w):
    '''From https://stackoverflow.com/questions/70296498/efficient-computation-of-moving-linear-regression-with-numpy-numba/70304475#70304475'''
    sx = window_sum_nan_mean(x, w)
    sy = window_sum_nan_mean(y, w)
    sx2 = window_sum_nan_mean(x**2, w)
    sxy = window_sum_nan_mean(x * y, w)
    slope = (sxy - sx * sy) / (sx2 - sx**2)
    intercept = sy - slope * sx
    return slope, intercept


def window_sum_mean_nan_tail(x, w, axis=0):
    # uses much less memory than the nanmean version
    end_window = (w - 1) - int((w - 1) / 2)
    end_div = np.arange(end_window, w)[::-1]
    summed = np.sum(
        np.nan_to_num(sliding_window_view(x, w, axis=axis), nan=0.0), axis=-1
    )
    return (
        summed
        / np.concatenate([np.ones(x.shape[0] - w - end_window + 1) * w, end_div])[
            : summed.shape[0], None
        ]
    )


def window_sum_mean(x, w, axis=0):
    return np.mean(sliding_window_view(x, w, axis=axis), axis=-1)


def np_2d_arange(start=0, stop=3, step=1, num_columns=4):
    # Create a 1D array using np.arange
    arr = np.arange(start, stop, step)
    result = arr[:, np.newaxis]
    # Repeat the single column 'num_columns' times to create the final array
    return np.broadcast_to(result, (len(arr), num_columns))


def window_lin_reg_mean_no_nan(x, y, w):
    '''From https://stackoverflow.com/questions/70296498/efficient-computation-of-moving-linear-regression-with-numpy-numba/70304475#70304475'''
    sx = window_sum_mean_nan_tail(x, w)
    sy = window_sum_mean_nan_tail(y, w)
    sx2 = window_sum_mean_nan_tail(x**2, w)
    sxy = window_sum_mean_nan_tail(x * y, w)
    slope = (sxy - sx * sy) / (sx2 - sx**2)
    intercept = sy - slope * sx
    return slope, intercept
