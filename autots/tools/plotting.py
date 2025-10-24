# -*- coding: utf-8 -*-
"""Shared plotting utilities for feature-rich time series."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no-cover - optional dependency
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no-cover
    HAS_MATPLOTLIB = False


def _to_timestamp(value):
    """Safely convert assorted datetime formats to pandas.Timestamp."""
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    return pd.Timestamp(value)


def _extract_event(record, keys: Sequence[str]):
    """Return tuple of values for event record that may be tuple or mapping."""
    if isinstance(record, Mapping):
        return tuple(record.get(key) for key in keys)
    if isinstance(record, Sequence):
        # Assume ordering matches keys length
        return tuple(record[i] if i < len(record) else None for i in range(len(keys)))
    return tuple(None for _ in keys)


def _component_array(components: Mapping[str, Iterable[float]], name: str, length: int):
    values = components.get(name)
    if values is None:
        return np.zeros(length)
    return np.asarray(values, dtype=float)


def plot_feature_panels(
    series_name: str,
    date_index: pd.DatetimeIndex,
    series_data: pd.Series | np.ndarray,
    components: Mapping[str, Iterable[float]],
    labels: Mapping[str, Iterable],
    series_type_description: str | None = None,
    scale: float | None = None,
    noise_to_signal: float | None = None,
    figsize=(16, 12),
    title_prefix: str = "Feature Analysis",
    save_path: str | None = None,
    show: bool = True,
):
    """Create a four-panel diagnostic plot shared by generator and detector."""
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")

    if not isinstance(series_data, pd.Series):
        series_data = pd.Series(series_data, index=date_index, name=series_name)

    n = len(date_index)
    trend = _component_array(components, 'trend', n)
    level_shift = _component_array(components, 'level_shift', n)
    seasonality = _component_array(components, 'seasonality', n)
    holidays = _component_array(components, 'holidays', n)
    noise = _component_array(components, 'noise', n)
    anomalies_component = _component_array(components, 'anomalies', n)

    combined_trend = trend + level_shift

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Overall Title
    type_suffix = f" (type: {series_type_description})" if series_type_description else ""
    fig.suptitle(f"{title_prefix}: {series_name}{type_suffix}", fontsize=16, fontweight='bold')

    # Panel 1: Raw series with labeled events
    ax = axes[0]
    ax.plot(date_index, series_data.values, color='tab:blue', alpha=0.75, linewidth=1.2, label='Series')

    anomalies = labels.get('anomalies', [])
    if anomalies:
        y_top = ax.get_ylim()[1]
        for record in anomalies:
            date, magnitude, pattern, duration, shared = _extract_event(
                record,
                ['date', 'magnitude', 'pattern', 'duration', 'shared']
            )
            if date is None:
                continue
            ts = _to_timestamp(date)
            ax.axvline(ts, color='red', alpha=0.35, linestyle='--', linewidth=1.5)
            ax.plot(ts, min(y_top, magnitude if magnitude is not None else y_top * 0.95),
                    marker='v', color='red', markersize=6, alpha=0.8)

    # Trend changepoints
    for record in labels.get('trend_changepoints', []):
        date, *_ = _extract_event(record, ['date', 'prior_slope', 'new_slope'])
        if date is None:
            continue
        ts = _to_timestamp(date)
        ax.axvline(ts, color='green', alpha=0.5, linestyle='-', linewidth=1.2)

    # Level shifts
    for record in labels.get('level_shifts', []):
        date, magnitude, shift_type, shared = _extract_event(
            record,
            ['date', 'magnitude', 'shift_type', 'shared']
        )
        if date is None:
            continue
        ts = _to_timestamp(date)
        ax.axvline(ts, color='purple', alpha=0.5, linestyle=':', linewidth=1.5)

    # Holidays
    holiday_dates = labels.get('holiday_dates', [])
    for date in holiday_dates:
        ts = _to_timestamp(date)
        ax.axvline(ts, color='goldenrod', alpha=0.4, linestyle='-.', linewidth=1.0)

    # Seasonality changepoints
    for record in labels.get('seasonality_changepoints', []):
        date, *_ = _extract_event(record, ['date', 'description'])
        if date is None:
            continue
        ts = _to_timestamp(date)
        ax.axvline(ts, color='darkcyan', alpha=0.35, linestyle='-.', linewidth=1.0)

    legend_elements = [
        Line2D([0], [0], color='tab:blue', linewidth=2, label='Series'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='Anomalies'),
        Line2D([0], [0], color='green', linestyle='-', linewidth=1.2, label='Trend CPs'),
        Line2D([0], [0], color='purple', linestyle=':', linewidth=1.5, label='Level Shifts'),
        Line2D([0], [0], color='goldenrod', linestyle='-.', linewidth=1.0, label='Holidays'),
        Line2D([0], [0], color='darkcyan', linestyle='-.', linewidth=1.0, label='Seasonality CPs'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Series with Key Events', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 2: Trend vs Level Shifts
    ax = axes[1]
    ax.plot(date_index, trend, color='tab:green', linewidth=1.4, alpha=0.8, label='Trend')
    ax.plot(date_index, combined_trend, color='black', linewidth=1.4, alpha=0.85, label='Trend + Level Shifts')

    for record in labels.get('trend_changepoints', []):
        date, *_ = _extract_event(record, ['date', 'prior_slope', 'new_slope'])
        if date is None:
            continue
        ax.axvline(_to_timestamp(date), color='green', alpha=0.35, linestyle='--', linewidth=1)

    for record in labels.get('level_shifts', []):
        date, magnitude, shift_type, shared = _extract_event(
            record,
            ['date', 'magnitude', 'shift_type', 'shared']
        )
        if date is None:
            continue
        ts = _to_timestamp(date)
        ax.axvline(ts, color='purple', alpha=0.45, linestyle=':', linewidth=1.4)
        try:
            idx = date_index.get_loc(ts)
            ax.annotate(
                f"{magnitude:+.1f}" if magnitude is not None else '',
                xy=(ts, combined_trend[idx]),
                xytext=(8, 10), textcoords='offset points', fontsize=8, color='purple',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
            )
        except KeyError:
            pass

    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Trend & Level Shifts', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Seasonality + Holiday impacts
    ax = axes[2]
    ax.plot(date_index, seasonality, color='tab:cyan', linewidth=1.0, alpha=0.7, label='Seasonality')
    ax.plot(date_index, holidays, color='orange', linewidth=1.2, alpha=0.8, label='Holidays')
    ax.plot(date_index, seasonality + holidays, color='tab:blue', linewidth=1.2, alpha=0.75, label='Combined')

    holiday_impacts = labels.get('holiday_impacts', {})
    for date in holiday_impacts.keys():
        ts = _to_timestamp(date)
        ax.axvline(ts, color='orange', alpha=0.25, linestyle='--', linewidth=0.9)
        try:
            idx = date_index.get_loc(ts)
            ax.plot(ts, holidays[idx], marker='o', color='orange', markersize=4, alpha=0.65)
        except KeyError:
            pass

    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Seasonality & Holiday Effects', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Noise and anomaly contributions
    ax = axes[3]
    ax.plot(date_index, noise, color='gray', linewidth=0.7, alpha=0.8, label='Noise')
    ax.plot(date_index, anomalies_component, color='red', linewidth=1.2, alpha=0.75, label='Anomaly Component')

    for record in labels.get('anomalies', []):
        date, magnitude, pattern, duration, shared = _extract_event(
            record,
            ['date', 'magnitude', 'pattern', 'duration', 'shared']
        )
        if date is None:
            continue
        ts = _to_timestamp(date)
        ax.axvline(ts, color='red', alpha=0.2, linestyle='--', linewidth=0.9)
        try:
            idx = date_index.get_loc(ts)
            text = pattern if pattern is not None else 'anomaly'
            ax.annotate(
                f"{text}\n{magnitude:+.1f}" if magnitude is not None else text,
                xy=(ts, anomalies_component[idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.4)
            )
        except KeyError:
            pass

    for record in labels.get('noise_changepoints', []):
        date, *_ = _extract_event(record, ['date', 'from_params', 'to_params'])
        if date is None:
            continue
        ax.axvline(_to_timestamp(date), color='gray', alpha=0.25, linestyle=':', linewidth=1.0)

    ax.set_ylabel('Value', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_title('Noise & Anomaly Components', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Format x-axis for all subplots
    for axis in axes:
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        interval = max(1, max(len(date_index) // 365, 1))
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
        plt.setp(axis.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Summary stats footer
    stats_parts = []
    if scale is not None:
        stats_parts.append(f"Scale: {scale:.1f}x")
    if noise_to_signal is not None:
        stats_parts.append(f"Noise/Signal: {noise_to_signal:.3f}")
    stats_parts.append(f"Trend CPs: {len(labels.get('trend_changepoints', []))}")
    stats_parts.append(f"Level Shifts: {len(labels.get('level_shifts', []))}")
    stats_parts.append(f"Anomalies: {len(labels.get('anomalies', []))}")
    stats_parts.append(f"Holidays: {len(labels.get('holiday_impacts', {}))}")

    fig.text(
        0.5,
        0.02,
        " | ".join(stats_parts),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


__all__ = ['plot_feature_panels', 'HAS_MATPLOTLIB']
