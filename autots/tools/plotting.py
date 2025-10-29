# -*- coding: utf-8 -*-
"""Shared plotting utilities for feature-rich time series."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no-cover - optional dependency
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no-cover
    HAS_MATPLOTLIB = False


colors_list = [
    '#FF00FF',
    '#7FFFD4',
    '#00FFFF',
    '#F5DEB3',
    '#FF6347',
    '#8B008B',
    '#696969',
    '#FFC0CB',
    '#C71585',
    '#008080',
    '#663399',
    '#32CD32',
    '#66CDAA',
    '#A9A9A9',
    '#2F4F4F',
    '#FFDEAD',
    '#800000',
    '#FFDAB9',
    '#D3D3D3',
    '#98FB98',
    '#87CEEB',
    '#A52A2A',
    '#FFA07A',
    '#7FFF00',
    '#E9967A',
    '#1E90FF',
    '#FF69B4',
    '#ADD8E6',
    '#20B2AA',
    '#708090',
    '#B0C4DE',
    '#D8BFD8',
    '#556B2F',
    '#B8860B',
    '#DAA520',
    '#BC8F8F',
    '#CD5C5C',
    '#6A5ACD',
    '#FA8072',
    '#FFD700',
    '#DA70D6',
    '#DC143C',
    '#B22222',
    '#00CED1',
    '#40E0D0',
    '#FF1493',
    '#483D8B',
    '#2E8B57',
    '#D2691E',
    '#8FBC8F',
    '#FF8C00',
    '#FFB6C1',
    '#8A2BE2',
]

# colors you might see in a mosaic or fresco
ancient_roman = [
    '#66023C',  # Tyrian Purple
    '#D4AF37',  # Gold
    '#B55A30',  # Terracotta
    '#5E503F',  # Taupe
    '#DC143C',  # Crimson
    '#D8C3A5',  # Pale Sand
    '#BAA378',  # Olive Tan
    '#3A5F3F',  # Dark Green Serpentine
    '#2E4057',  # Deep Slate Blue
    '#6A7B76',  # Muted Teal
    '#965D62',  # Burnt Rose
    '#7F9B9B',  # Grayish Blue
    '#7C0A02',  # Cinnabar Red
    '#8A3324',  # Burnt Umber
    '#4682B4',  # Steel Blue
    '#CD5C5C',  # Indian Red
    '#B8860B',  # Dark Goldenrod
    '#6B8E23',  # Olive Drab
    '#2E8B57',  # Sea Green
    '#9932CC',  # Dark Orchid
    '#9400D3',  # Dark Violet
    '#4B0082',  # Indigo
    '#6A5ACD',  # Slate Blue
    '#483D8B',  # Dark Slate Blue
    '#DA70D6',  # Orchid
    '#1C1C1C',  # Obsidian Black
    '#D8BFD8',  # Thistle
    '#FF2400',  # Cinnabar Red
    '#1F75FE',  # Egyptian Blue
    '#FFD700',  # Bright Gold
    '#32CD32',  # Bright Verdigris Green
    '#FFA07A',  # Light Coral
    '#FF4500',  # Bright Orange
    '#ADD8E6',  # Light Blue
    '#FFFFE0',  # Light Yellow
    '#00FA9A',  # Medium Spring Green
    '#F4A460',  # Sandy Brown
    '#FFFACD',  # Lemon Chiffon
    '#E9967A',  # Dark Salmon
    '#8B0000',  # Dark Red
    '#2B2B2B',  # Basalt Gray
    '#FF6347',  # Tomato
    '#FF8C00',  # Dark Orange
    '#40E0D0',  # Turquoise
    '#FA8072',  # Salmon
    '#D5C3AA',  # Travertine Beige
    '#EAE6DA',  # Carrara White Marble
]

# grayscale palette useful for overlaying multiple simulation paths
grays = [
    "#838996",
    "#c0c0c0",
    "#dcdcdc",
    "#a9a9a9",
    "#808080",
    "#989898",
    "#757575",
    "#696969",
    "#c9c0bb",
    "#c8c8c8",
    "#323232",
    "#e5e4e2",
    "#778899",
    "#4f666a",
    "#848482",
    "#414a4c",
    "#8a7f80",
    "#c4c3d0",
    "#bebebe",
    "#dbd7d2",
]


def create_seaborn_palette_from_cmap(cmap_name: str = "gist_rainbow", n: int = 10):
    """Return seaborn palette sampling the given matplotlib cmap."""
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")
    try:
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("seaborn is required for create_seaborn_palette_from_cmap") from exc

    cm = plt.get_cmap(cmap_name)
    colors = cm(np.linspace(0, 1, n))
    return sns.color_palette(colors)


def calculate_peak_density(
    model: str,
    data: pd.DataFrame,
    group_col: str = 'Model',
    y_col: str = 'TotalRuntimeSeconds',
):
    """Maximum KDE value for the given model's distribution."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scipy is required for calculate_peak_density") from exc

    model_data = data[data[group_col] == model][y_col]
    kde = gaussian_kde(model_data)
    return np.max(kde(model_data))


def plot_distributions(
    runtimes_data: pd.DataFrame,
    group_col: str = 'Model',
    y_col: str = 'TotalRuntimeSeconds',
    xlim: float | None = None,
    xlim_right: float | None = None,
    title_suffix: str = "",
):
    """Plot runtime density per group with custom palette."""
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")
    try:
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("seaborn is required for plot_distributions") from exc

    single_obs_models = runtimes_data.groupby(group_col).filter(lambda x: len(x) == 1)
    multi_obs_models = runtimes_data.groupby(group_col).filter(lambda x: len(x) > 1)

    if not multi_obs_models.empty:
        average_peak_density = np.mean(
            [
                calculate_peak_density(model, multi_obs_models, group_col, y_col)
                for model in multi_obs_models[group_col].unique()
            ]
        )
    else:
        average_peak_density = 0.0

    unique_models = runtimes_data[group_col].nunique()
    palette = create_seaborn_palette_from_cmap("gist_rainbow", n=unique_models)
    sorted_models = runtimes_data[group_col].value_counts().index.tolist()
    zip_palette = dict(zip(sorted_models, palette))

    fig = plt.figure(figsize=(12, 8))

    sns.kdeplot(
        data=multi_obs_models,
        x=y_col,
        hue=group_col,
        fill=True,
        common_norm=False,
        palette=zip_palette,
        alpha=0.5,
    )

    if not single_obs_models.empty and average_peak_density:
        sns.scatterplot(
            data=single_obs_models,
            x=y_col,
            y=[average_peak_density] * len(single_obs_models),
            hue=group_col,
            palette=zip_palette,
            legend=False,
            marker='o',
        )

    handles, labels = [], []
    for model, color in zip_palette.items():
        handles.append(Line2D([0], [0], linestyle="none", c=color, marker='o'))
        labels.append(model)

    plt.legend(handles, labels, title=group_col)
    plt.title(f'Distribution of {y_col} by {group_col}{title_suffix}', fontsize=16)
    plt.xlabel(f'{y_col}', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    if xlim is not None:
        plt.xlim(left=xlim)
    if xlim_right is not None:
        plt.xlim(right=runtimes_data[y_col].quantile(xlim_right))

    return fig


def plot_forecast_with_intervals(
    plot_df: pd.DataFrame,
    actual_col: str | None = 'actuals',
    forecast_col: str = 'forecast',
    lower_col: str = 'low_forecast',
    upper_col: str = 'up_forecast',
    title: str | None = None,
    colors: Mapping[str, str] | None = None,
    include_bounds: bool = True,
    alpha: float = 0.3,
    band_color: str | None = None,
    interval_label: str | None = "Prediction Interval",
    band_kwargs: Mapping[str, Any] | None = None,
    plot_lines: bool = True,
    ax=None,
    **plot_kwargs,
):
    """Plot forecast (and optionally actuals) with confidence bounds."""
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")

    if ax is None:
        _, ax = plt.subplots()

    if plot_lines:
        columns_to_plot: list[str] = []
        if actual_col and actual_col in plot_df.columns:
            columns_to_plot.append(actual_col)
        if forecast_col and forecast_col in plot_df.columns:
            columns_to_plot.append(forecast_col)
        if not columns_to_plot:
            raise ValueError("plot_df must contain at least one of actual_col or forecast_col.")

        color_mapping = None
        if colors is not None:
            color_mapping = {col: color for col, color in colors.items() if col in columns_to_plot}
            if color_mapping:
                # Create a copy and update with color mapping
                plot_kwargs_copy = dict(plot_kwargs)
                plot_kwargs_copy['color'] = color_mapping
                plot_df[columns_to_plot].plot(ax=ax, title=title, **plot_kwargs_copy)
            else:
                plot_df[columns_to_plot].plot(ax=ax, title=title, **plot_kwargs)
        else:
            plot_df[columns_to_plot].plot(ax=ax, title=title, **plot_kwargs)
    elif title is not None:
        ax.set_title(title)

    if include_bounds and lower_col in plot_df.columns and upper_col in plot_df.columns:
        fill_kwargs = dict(alpha=alpha)
        if band_color is not None:
            fill_kwargs['color'] = band_color
        elif colors is not None:
            fill_color = colors.get(lower_col) or colors.get(upper_col)
            if fill_color is not None:
                fill_kwargs['color'] = fill_color
        if interval_label is not None:
            fill_kwargs['label'] = interval_label
        if band_kwargs:
            fill_kwargs.update(band_kwargs)
        ax.fill_between(
            plot_df.index,
            plot_df[upper_col],
            plot_df[lower_col],
            **fill_kwargs,
        )

    return ax


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

    # Add AutoTS watermark in bottom right
    fig.text(
        0.98,
        0.01,
        "AutoTS",
        ha='right',
        va='bottom',
        fontsize=8,
        alpha=0.3,
        style='italic',
        color='gray',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_risk_score_bar(
    risk_data: pd.Series | np.ndarray,
    index=None,
    bar_color: str = "#6495ED",
    bar_ylim: tuple | list | None = None,
    title: str = "Risk Score",
    ylabel: str = "Risk",
    xlabel: str = "Forecast Horizon",
    ax=None,
    **bar_kwargs,
):
    """Plot risk scores as a bar chart.
    
    Utility function for plotting event risk or similar probability scores.
    
    Args:
        risk_data: Series or array of risk scores to plot
        index: x-axis values; if None, uses range or Series index
        bar_color: color for bars
        bar_ylim: y-axis limits as (min, max) or [min, max]
        title: chart title
        ylabel: y-axis label
        xlabel: x-axis label
        ax: matplotlib axis to plot on; if None, creates new subplot
        **bar_kwargs: additional arguments passed to ax.bar()
        
    Returns:
        matplotlib axis
    """
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        _, ax = plt.subplots()
    
    if isinstance(risk_data, pd.Series):
        if index is None:
            index = risk_data.index
        values = risk_data.values
    else:
        values = np.asarray(risk_data)
        if index is None:
            index = np.arange(len(values))
    
    bar_kwargs_copy = dict(bar_kwargs)
    if 'color' not in bar_kwargs_copy:
        bar_kwargs_copy['color'] = bar_color
    if 'width' not in bar_kwargs_copy:
        bar_kwargs_copy['width'] = 0.6
    
    ax.bar(index, values, **bar_kwargs_copy)
    
    if bar_ylim is not None:
        ax.set_ylim(bar_ylim)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_facecolor("#f9f9f9")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    
    # Rotate x-axis labels if they're datetime-like
    if isinstance(index, pd.Index) and not index.is_numeric():
        ax.tick_params(axis="x", rotation=45)
    
    return ax


def plot_simulation_paths(
    simulations: np.ndarray,
    index=None,
    colors: list | None = None,
    alpha: float = 0.9,
    linewidth: float = 1.2,
    ax=None,
    **plot_kwargs,
):
    """Plot multiple simulation/forecast paths.
    
    Utility for plotting Monte Carlo simulations, motif neighbors, or ensemble members.
    
    Args:
        simulations: 2D array of shape (n_simulations, n_timesteps)
        index: x-axis values; if None, uses range
        colors: list of color strings for each path; if None, uses random grays
        alpha: transparency for lines
        linewidth: width of lines
        ax: matplotlib axis to plot on; if None, creates new subplot
        **plot_kwargs: additional arguments passed to ax.plot()
        
    Returns:
        matplotlib axis
    """
    if not HAS_MATPLOTLIB:  # pragma: no cover - runtime guard
        raise ImportError("matplotlib is required for plotting")
    
    if ax is None:
        _, ax = plt.subplots()
    
    simulations = np.asarray(simulations)
    if simulations.ndim != 2:
        raise ValueError(f"simulations must be 2D, got shape {simulations.shape}")
    
    n_sims, n_steps = simulations.shape
    
    if index is None:
        index = np.arange(n_steps)
    
    if colors is None:
        # Use random gray shades for simulation paths
        import random
        colors = random.choices(grays, k=n_sims)
    
    plot_kwargs_copy = dict(plot_kwargs)
    if 'alpha' not in plot_kwargs_copy:
        plot_kwargs_copy['alpha'] = alpha
    if 'linewidth' not in plot_kwargs_copy:
        plot_kwargs_copy['linewidth'] = linewidth
    
    for idx, series in enumerate(simulations):
        color = colors[idx] if idx < len(colors) else colors[0]
        ax.plot(index, series, color=color, **plot_kwargs_copy)
    
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    
    return ax


__all__ = [
    'plot_feature_panels',
    'plot_distributions',
    'plot_forecast_with_intervals',
    'plot_risk_score_bar',
    'plot_simulation_paths',
    'create_seaborn_palette_from_cmap',
    'calculate_peak_density',
    'colors_list',
    'ancient_roman',
    'grays',
    'HAS_MATPLOTLIB',
]
