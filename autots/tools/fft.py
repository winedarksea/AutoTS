#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 22:07:37 2023

@author: colincatlin
"""
import numpy as np


def fourier_extrapolation(
    x, forecast_length=10, n_harm=10, detrend='linear', freq_range=None
):
    m, n = x.shape
    t = np.arange(0, m)

    # Detrend
    if detrend == 'linear':
        p = np.polyfit(t, x, 1).T
        x_notrend = x - np.outer(t, p[:, 0])
    elif detrend == 'quadratic':
        p = np.polyfit(t, x, 2).T
        x_notrend = x - np.outer(t**2, p[:, 0]) - np.outer(t, p[:, 1])
    elif detrend is None:
        x_notrend = x
    else:
        raise ValueError(f"Unsupported detrend option: {detrend}")

    # FFT
    x_freqdom = np.fft.fft(x_notrend, axis=0)

    # Frequencies and sorted indices
    f = np.fft.fftfreq(m)
    indexes = np.argsort(np.abs(f))

    # Frequency range filtering
    if freq_range:
        low, high = freq_range
        indexes = [i for i in indexes if low <= np.abs(f[i]) <= high]

    if n_harm is None:
        use_idx = indexes
    elif isinstance(n_harm, (int, float)):
        # handle float as percentage
        if 0 < n_harm < 1:
            use_idx = indexes[: int(len(indexes) * n_harm)]
        # handle negative percentage ie last N percentage
        elif -1 < n_harm < 0:
            use_idx = indexes[int(len(indexes) * n_harm) :]
        elif n_harm <= -1:
            use_idx = indexes[n_harm * 2 :]
        # handle exact number
        else:
            use_idx = indexes[: 1 + n_harm * 2]
    elif isinstance(n_harm, str):
        if "mid" in n_harm:
            midp = int(''.join(filter(str.isdigit, n_harm)))
            use_idx = indexes[midp : midp + 40]
    else:
        raise ValueError(f"n_harm value {n_harm} not recognized")

    t_extended = np.arange(0, m + forecast_length)
    restored_sig = np.zeros((t_extended.size, n))

    # Use harmonics to reconstruct signal
    for i in use_idx:
        ampli = np.abs(x_freqdom[i]) / m
        phase = np.angle(x_freqdom[i])
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t_extended[:, None] + phase)
    """
    # Use harmonics to reconstruct signal
    for i in indexes[10:10 + n_harm * 2]:
    # for i in indexes[-2000:]:
        ampli = np.abs(x_freqdom[i]) / m
        phase = np.angle(x_freqdom[i])
        restored_sig += (ampli * np.cos(2 * np.pi * f[i] * t_extended[:, None] + phase))

    nw = pd.DataFrame((restored_sig + np.outer(t_extended, p[:, 0])), columns=df.columns)
    nw.index = df.index.union(pd.date_range(start=df.index[-1], periods=forecast_length+1, freq='D'))
    col = 'FOODS_2_025_TX_1_evaluation'  # 'wiki_all'
    nw['actual'] = df[col]
    nw[['actual', col]].plot()
    """

    # Add trend back
    if detrend == 'linear':
        return restored_sig + np.outer(t_extended, p[:, 0])
    elif detrend == 'quadratic':
        return (
            restored_sig
            + np.outer(t_extended**2, p[:, 0])
            + np.outer(t_extended, p[:, 1])
        )
    else:
        return restored_sig


class FFT(object):
    def __init__(self, n_harm=10, detrend='linear', freq_range=None):
        self.n_harm = n_harm
        self.detrend = detrend
        self.freq_range = freq_range

    def fit(self, x):
        self.m, self.n = x.shape
        t = np.arange(0, self.m)

        # Detrend
        if self.detrend == 'linear':
            self.p = np.polyfit(t, x, 1).T
            x_notrend = x - np.outer(t, self.p[:, 0])
        elif self.detrend == 'quadratic':
            self.p = np.polyfit(t, x, 2).T
            x_notrend = x - np.outer(t**2, self.p[:, 0]) - np.outer(t, self.p[:, 1])
        elif self.detrend == 'cubic':
            self.p = np.polyfit(t, x, 3).T
            x_notrend = (
                x
                - np.outer(t**3, self.p[:, 0])
                - np.outer(t**2, self.p[:, 1])
                - np.outer(t, self.p[:, 2])
            )
        elif self.detrend == 'quartic':
            self.p = np.polyfit(t, x, 4).T
            x_notrend = (
                x
                - np.outer(t**4, self.p[:, 0])
                - np.outer(t**3, self.p[:, 1])
                - np.outer(t**2, self.p[:, 2])
                - np.outer(t, self.p[:, 3])
            )
        elif self.detrend is None:
            x_notrend = x
        else:
            raise ValueError(f"Unsupported detrend option: {self.detrend}")

        # FFT
        self.x_freqdom = np.fft.fft(x_notrend, axis=0)

        # Frequencies and sorted indices
        self.f = np.fft.fftfreq(self.m)
        indexes = np.argsort(np.abs(self.f))

        # Frequency range filtering
        if self.freq_range:
            low, high = self.freq_range
            indexes = [i for i in indexes if low <= np.abs(self.f[i]) <= high]

        if self.n_harm is None:
            use_idx = indexes
        elif isinstance(self.n_harm, (int, float)):
            # handle float as percentage
            if 0 < self.n_harm < 1:
                use_idx = indexes[: int(len(indexes) * self.n_harm)]
            # handle negative percentage ie last N percentage
            elif -1 < self.n_harm < 0:
                use_idx = indexes[int(len(indexes) * self.n_harm) :]
            elif self.n_harm <= -1:
                use_idx = indexes[self.n_harm * 2 :]
            # handle exact number
            else:
                use_idx = indexes[: 1 + self.n_harm * 2]
        elif isinstance(self.n_harm, str):
            if "mid" in self.n_harm:
                midp = int(''.join(filter(str.isdigit, self.n_harm)))
                use_idx = indexes[midp : midp + 41]
        else:
            raise ValueError(f"n_harm value {self.n_harm} not recognized")
        self.use_idx = use_idx

        return self

    def generate_harmonics_dataframe(self, forecast_length=0):
        extended_m = self.m + forecast_length
        harmonics_data = np.zeros((extended_m, len(self.use_idx) * 2))

        for i, idx in enumerate(self.use_idx):
            freq_component = np.fft.ifft(self.x_freqdom[idx], n=self.m, axis=0)
            extended_freq_component = np.tile(
                freq_component, (extended_m // self.m) + 1
            )[:extended_m]
            harmonics_data[:, 2 * i] = np.real(extended_freq_component).flatten()
            harmonics_data[:, 2 * i + 1] = np.imag(extended_freq_component).flatten()

        return harmonics_data

    def predict(self, forecast_length=0):
        # this rather assumes you care only about historical + fcst of length n after
        t_extended = np.arange(0, self.m + forecast_length)
        restored_sig = np.zeros((t_extended.size, self.n))

        # Use harmonics to reconstruct signal
        for i in self.use_idx:
            ampli = np.abs(self.x_freqdom[i]) / self.m
            phase = np.angle(self.x_freqdom[i])
            restored_sig += ampli * np.cos(
                2 * np.pi * self.f[i] * t_extended[:, None] + phase
            )

        # Add trend back
        if self.detrend == 'linear':
            return restored_sig + np.outer(t_extended, self.p[:, 0])
        elif self.detrend == 'quadratic':
            return (
                restored_sig
                + np.outer(t_extended**2, self.p[:, 0])
                + np.outer(t_extended, self.p[:, 1])
            )
        elif self.detrend == 'cubic':
            return (
                restored_sig
                + np.outer(t_extended**3, self.p[:, 0])
                + np.outer(t_extended**2, self.p[:, 1])
                + np.outer(t_extended, self.p[:, 2])
            )
        elif self.detrend == 'quartic':
            return (
                restored_sig
                + np.outer(t_extended**4, self.p[:, 0])
                + np.outer(t_extended**3, self.p[:, 1])
                + np.outer(t_extended**2, self.p[:, 2])
                + np.outer(t_extended, self.p[:, 3])
            )
        else:
            return restored_sig
