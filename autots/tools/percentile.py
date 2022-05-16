# -*- coding: utf-8 -*-
"""Faster percentile and quantile for numpy

Entirely from: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
"""
import numpy as np


def _zvalue_from_index(arr, ind):
    """private helper function to work around the limitation of np.choose() by employing np.take()
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _, nC, nR = arr.shape

    # get linear indices and extract elements with np.take()
    idx = nC * nR * ind + np.arange(nC * nR).reshape((nC, nR))
    return np.take(arr, idx)


def nan_percentile(in_arr, q, method="linear", axis=0, errors="raise"):
    """Given a 3D array, return the given percentiles as input by q.
    Beware this is only tested for the limited case required here, and will not match np fully.
    Args more limited. If errors="rollover" passes to np.nanpercentile where args are not supported.
    """
    flag_2d = False
    if in_arr.ndim == 2:
        arr = np.expand_dims(in_arr, 1)
        flag_2d = True
    else:
        arr = in_arr.copy()
    if (
        axis != 0
        or method not in ["linear", "nearest", "lowest", "highest"]
        or arr.ndim != 3
    ):
        if errors == "rollover":
            return np.nanpercentile(arr, q=q, method=method, axis=axis)
        else:
            raise ValueError("input not supported by internal percentile function")
    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if type(q) is list:
        qs = []
        qs.extend(q)
    elif type(q) is range:
        qs = list(q)
    else:
        qs = [q]
    if len(qs) < 2:
        quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
    else:
        quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

    result = []
    # note this is vectorized for a single quantile but each quantile step is separate
    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        if method == "linear":
            # linear interpolation (like numpy percentile) takes the fractional part of desired position
            floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
            ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

            quant_arr = floor_val + ceil_val
            quant_arr[fc_equal_k_mask] = _zvalue_from_index(
                arr=arr, ind=k_arr.astype(np.int32)
            )[
                fc_equal_k_mask
            ]  # if floor == ceiling take floor value
        elif method == 'nearest':
            f_arr = np.around(k_arr).astype(np.int32)
            quant_arr = _zvalue_from_index(arr=arr, ind=f_arr)
        elif method == 'lowest':
            f_arr = np.floor(k_arr).astype(np.int32)
            quant_arr = _zvalue_from_index(arr=arr, ind=f_arr)
        elif method == 'highest':
            f_arr = np.ceiling(k_arr).astype(np.int32)
            quant_arr = _zvalue_from_index(arr=arr, ind=f_arr)
        else:
            raise ValueError("interpolation method not supported")

        if flag_2d:
            result.append(quant_arr[0])
        else:
            result.append(quant_arr)
    if len(result) == 1:
        return result[0]
    else:
        return np.asarray(result)


def nan_quantile(arr, q, method="linear", axis=0, errors="raise"):
    """Same as nan_percentile but accepts q in range [0, 1].
    Args more limited. If errors="rollover" passes to np.nanpercentile where not supported.
    """
    return nan_percentile(arr, q * 100, method=method, axis=axis, errors=errors)
