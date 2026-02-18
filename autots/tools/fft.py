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

    def detect_dominant_periods(self, min_period=3, max_periods=5, power_threshold=0.1):
        """Detect dominant seasonal periods from the fitted FFT spectrum.

        Returns
        -------
        list
            List of `(period, relative_strength)` tuples sorted by descending
            relative strength.
        """
        power = np.mean(np.abs(self.x_freqdom) ** 2, axis=1)
        freqs = self.f

        max_period = self.m / 2
        periods = np.full(freqs.shape, np.inf, dtype=float)
        positive_mask = freqs > 0
        np.divide(1.0, freqs, out=periods, where=positive_mask)
        mask = positive_mask & (periods >= min_period) & (periods <= max_period)
        candidate_freqs = freqs[mask]
        candidate_power = power[mask]

        if len(candidate_power) == 0:
            return []

        peak_indices = []
        for i in range(1, len(candidate_power) - 1):
            if (
                candidate_power[i] > candidate_power[i - 1]
                and candidate_power[i] > candidate_power[i + 1]
            ):
                peak_indices.append(i)
        if len(candidate_power) >= 2:
            if candidate_power[0] > candidate_power[1]:
                peak_indices.insert(0, 0)
            if candidate_power[-1] > candidate_power[-2]:
                peak_indices.append(len(candidate_power) - 1)

        if not peak_indices:
            sorted_idx = np.argsort(candidate_power)[::-1][:max_periods]
            max_power = candidate_power[sorted_idx[0]] if len(sorted_idx) > 0 else 1.0
            results = []
            for idx in sorted_idx:
                rel = candidate_power[idx] / (max_power + 1e-12)
                if rel >= power_threshold:
                    results.append((1.0 / candidate_freqs[idx], rel))
            return results

        peak_powers = candidate_power[peak_indices]
        max_peak_power = np.max(peak_powers)
        valid_peaks = [
            (candidate_freqs[i], candidate_power[i])
            for i in peak_indices
            if candidate_power[i] / (max_peak_power + 1e-12) >= power_threshold
        ]

        valid_peaks.sort(key=lambda x: x[1], reverse=True)
        results = []
        for freq, pwr in valid_peaks[:max_periods]:
            period = 1.0 / freq
            rel_strength = pwr / (max_peak_power + 1e-12)
            results.append((period, rel_strength))
        return results

    def generate_harmonics_dataframe(self, forecast_length=0):
        extended_m = self.m + forecast_length
        harmonics_data = np.zeros((extended_m, len(self.use_idx) * 2))
        t_extended = np.arange(0, extended_m)

        for i, idx in enumerate(self.use_idx):
            # Use the frequency domain information to generate harmonics
            # for the extended time period, similar to predict method
            ampli = np.abs(self.x_freqdom[idx]) / self.m
            phase = np.angle(self.x_freqdom[idx])
            # Generate the harmonic component for extended time, summed across series
            # This creates a multivariate summary feature
            harmonic_extended = ampli * np.exp(
                2j * np.pi * self.f[idx] * t_extended[:, None] + 1j * phase
            )
            # Sum across all series to create a single feature per harmonic
            harmonics_data[:, 2 * i] = np.sum(np.real(harmonic_extended), axis=1)
            harmonics_data[:, 2 * i + 1] = np.sum(np.imag(harmonic_extended), axis=1)

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
