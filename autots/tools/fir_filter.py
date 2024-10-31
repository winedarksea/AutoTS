#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 23:59:35 2024

@author: colincatlin
"""
import random
import numpy as np

try:
    from scipy.signal import fftconvolve, firwin, convolve, lfilter
except Exception:
    pass


def apply_fir_filter_to_timeseries(
    data, sampling_frequency, numtaps=512, cutoff_hz=20, window='hamming'
):
    """
    Apply FIR filter to an array of time series data with shape (observations, series).

    Parameters:
    - data: numpy array of shape (observations, series), where each column represents a time series
    - sampling_frequency: The sampling frequency of the time series data (e.g., 365 for daily data)
    - numtaps: Number of taps (filter length)
    - cutoff_hz: The cutoff frequency in Hz (for filtering purposes)
    - window: The windowing function to use for FIR filter design ('hamming', 'hann', etc.)

    Returns:
    - filtered_data: The filtered version of the input data
    """

    # Ensure the data has the correct shape: (observations, series)
    # if data.shape[0] < data.shape[1]:
    #     data = data.T  # Transpose if necessary to match (observations, series)

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist_frequency = 0.5 * sampling_frequency
    cutoff_norm = cutoff_hz / nyquist_frequency

    # Design the FIR filter using the given parameters
    fir_coefficients = firwin(numtaps=numtaps, cutoff=cutoff_norm, window=window)

    # Apply the FIR filter to each time series (each column in the data)
    # Convolve each column with the FIR filter
    filtered_data = np.apply_along_axis(
        lambda x: convolve(x, fir_coefficients, mode='same'), axis=0, arr=data
    )

    return filtered_data


def apply_fir_filter_time_domain(
    data, sampling_frequency, numtaps=512, cutoff_hz=20, window='hamming'
):
    """
    Apply FIR filter using time-domain convolution (lfilter) for smaller memory usage.
    This function has padding issues currently.
    """
    # Ensure the data has the correct shape: (observations, series)
    # if data.shape[0] < data.shape[1]:
    #     data = data.T  # Transpose if necessary to match (observations, series)

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist_frequency = 0.5 * sampling_frequency
    cutoff_norm = cutoff_hz / nyquist_frequency

    # Design the FIR filter
    fir_coefficients = firwin(numtaps=numtaps, cutoff=cutoff_norm, window=window)

    # Apply time-domain filtering (lfilter)
    filtered_data = lfilter(fir_coefficients, 1.0, data, axis=0)

    return filtered_data


def fft_fir_filter_to_timeseries(
    data,
    sampling_frequency,
    numtaps=512,
    cutoff_hz=20,
    window='hamming',
    chunk_size=1000,
):
    """
    Apply FIR filter to an array of time series data with shape (observations, series).

    Parameters:
    - data: numpy array of shape (observations, series), where each column represents a time series
    - sampling_frequency: The sampling frequency of the time series data (e.g., 365 for daily data)
    - numtaps: Number of taps (filter length)
    - cutoff_hz: The cutoff frequency in Hz (for filtering purposes)
    - window: The windowing function to use for FIR filter design ('hamming', 'hann', etc.)

    Returns:
    - filtered_data: The filtered version of the input data
    """
    # Ensure the data has the correct shape: (observations, series)
    # if data.shape[0] < data.shape[1]:
    #     data = data.T  # Transpose if necessary to match (observations, series)

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyquist_frequency = 0.5 * sampling_frequency
    cutoff_norm = cutoff_hz / nyquist_frequency

    if window == 'kaiser':
        beta = 14
        window = ('kaiser', beta)

    # Design the FIR filter using the given parameters
    fir_coefficients = firwin(numtaps=numtaps, cutoff=cutoff_norm, window=window)

    # Pad the beginning of the data to shift edge artifacts to the start
    # pad_width = numtaps - 1
    # padded_data = np.pad(data, ((pad_width, 0), (0, 0)), mode='reflect')
    pad_width_start = numtaps - 1
    pad_width_end = numtaps - 1
    padded_data = np.pad(
        data, ((pad_width_start, pad_width_end), (0, 0)), mode='reflect'
    )

    num_series = data.shape[1]
    if chunk_size is not None and chunk_size < num_series:
        # Filter the data in chunks to reduce memory load
        filtered_data = np.zeros_like(data)

        for start in range(0, num_series, chunk_size):
            end = min(start + chunk_size, num_series)
            chunk = padded_data[:, start:end]
            filtered_chunk = fftconvolve(
                chunk, fir_coefficients[:, np.newaxis], mode='same', axes=0
            )
            filtered_data[:, start:end] = filtered_chunk[
                pad_width_start:-pad_width_end, :
            ]  # [pad_width:, :]
    else:
        # Apply FFT convolution across all time series at once
        filtered_padded_data = fftconvolve(
            padded_data, fir_coefficients[:, np.newaxis], mode='same', axes=0
        )

        # Remove the padding from the start (discard the first `pad_width` samples)
        filtered_data = filtered_padded_data[
            pad_width_start:-pad_width_end, :
        ]  # [pad_width:, :]

    return filtered_data


def generate_random_fir_params(method='random', data_type="time_series"):
    params = {}

    # Random number of taps (filter length)
    params["numtaps"] = random.choices(
        [32, 64, 128, 256, 512, 1024], [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
    )[0]

    if data_type == "audio":
        # Higher cutoff frequencies for audio
        cutoff_choices = [20, 100, 500, 1000, 5000, 10000, 15000]
        cutoff_weights = [0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05]
    else:
        # Lower cutoff frequencies for time series data
        cutoff_choices = [0.01, 0.1, 0.5, 5, 10, 20, 50, 100, 500]
        cutoff_weights = [0.3, 0.3, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01]

    params["cutoff_hz"] = random.choices(cutoff_choices, cutoff_weights)[0]

    # Random window type
    params["window"] = random.choices(
        ["hamming", "hann", "blackman", "kaiser", "tukey", "boxcar", "taylor"],
        [0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05],
    )[0]

    return params


"""
# Example Usage with Time Series Data
sampling_frequency = 365  # Example for daily data with 365 observations per year
num_series = 5  # Number of time series
time_vector = np.arange(365)  # Example time vector (one year of daily data)

# Generate an array of time series signals: Each series has random noise and some seasonality
time_series_data = np.array([
    np.sin(2 * np.pi * time_vector / 365) + 0.5 * np.random.randn(len(time_vector)) 
    for _ in range(num_series)
]).T  # Transposed to (observations, series)

# Apply the FIR filter to the time series data
filtered_time_series = apply_fir_filter_to_timeseries(time_series_data, sampling_frequency, numtaps=256, cutoff_hz=30)

# Output filtered time series
import pandas as pd
pd.DataFrame(filtered_time_series).plot()

# Apply the FIR filter to the time series data
filtered_time_series = apply_fir_filter_time_domain(time_series_data, sampling_frequency, numtaps=256, cutoff_hz=30)

# Output filtered time series
import pandas as pd
pd.DataFrame(filtered_time_series).plot()

# Apply the FIR filter to the time series data
filtered_time_series = fft_fir_filter_to_timeseries(time_series_data, sampling_frequency, numtaps=256, cutoff_hz=30)

# Output filtered time series
import pandas as pd
pd.DataFrame(filtered_time_series).plot()

# Apply the FIR filter to the time series data
filtered_time_series = fft_fir_filter_to_timeseries(time_series_data, sampling_frequency, numtaps=256, cutoff_hz=30, chunk_size=2)

# Output filtered time series
import pandas as pd
pd.DataFrame(filtered_time_series).plot()

"""
