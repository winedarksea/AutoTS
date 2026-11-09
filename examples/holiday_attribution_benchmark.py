"""Compare holiday defaults on bundled Wikipedia data and fixed synthetic seeds.

Run from the repository root:
    python -m examples.holiday_attribution_benchmark --output /tmp/holiday-results.json

Both configurations use the corrected calendar utilities. Error denominators
are the original training series' standard deviations, fixed across variants.
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from autots import load_daily
from autots.datasets import SyntheticDailyGenerator
from autots.evaluator.feature_detector import (
    TimeSeriesFeatureDetector,
    FeatureDetectionOptimizer,
)
from autots.tools.easter import easter_offsets

CONFIGURATIONS = {
    "legacy": {
        "global_holiday_anomaly_suppression": True,
        "holiday_params": {"use_lunar_holidays": False, "use_easter_holidays": False},
    },
    "proposed": {
        "global_holiday_anomaly_suppression": None,
        "holiday_params": {"use_lunar_holidays": True, "use_easter_holidays": True},
    },
}


def evaluate(data, variant, generator=None):
    random.seed(42)
    np.random.seed(42)
    train, test = data.iloc[:-28], data.iloc[-28:]
    scale = train.std().replace(0, np.nan)
    start = time.perf_counter()
    detector = TimeSeriesFeatureDetector(**CONFIGURATIONS[variant]).fit(train)
    elapsed = time.perf_counter() - start
    forecast = detector.forecast(28).forecast
    result = {
        "fit_seconds": elapsed,
        "forecast_normalized_mae": float(
            ((forecast - test).abs().mean() / scale).mean()
        ),
        "anomaly_count": sum(map(len, detector.anomalies.values())),
        "level_shift_count": sum(map(len, detector.level_shifts.values())),
        "level_shift_dates": {
            column: [
                str(event["date"] if isinstance(event, dict) else event[0])
                for event in events
            ]
            for column, events in detector.level_shifts.items()
        },
    }
    if generator is not None:
        scorer = FeatureDetectionOptimizer(generator)
        for name, detected, truth, tolerance in [
            ("anomaly", detector.anomalies, generator.anomalies, 1),
            ("holiday", detector.holiday_dates, generator.holiday_dates, 1),
            ("level_shift", detector.level_shifts, generator.level_shifts, 7),
        ]:
            metrics = []
            for column in train:
                true_events = [
                    event
                    for event in truth[column]
                    if scorer.loss_calculator._parse_generic_date(event)
                    <= train.index[-1]
                ]
                metrics.append(
                    scorer._date_detection_stats(
                        detected[column], true_events, tolerance
                    )
                )
            for metric in ["precision", "recall"]:
                result[f"{name}_{metric}"] = float(
                    np.mean([item[metric] for item in metrics])
                )
            if name == "level_shift":
                result["false_level_shifts"] = sum(
                    item["detected_count"] - len(item["matched_date_errors"])
                    for item in metrics
                )
        result["holiday_component_normalized_rmse"] = float(
            np.mean(
                [
                    np.sqrt(
                        np.mean(
                            (
                                np.asarray(detector.components[column]["holidays"])
                                - np.asarray(generator.components[column]["holidays"])[
                                    : len(train)
                                ]
                            )
                            ** 2
                        )
                    )
                    / scale[column]
                    for column in train
                ]
            )
        )
    else:
        result["holiday_profiles"] = {}
        for column, mask in [
            ("wiki_Christmas", (train.index.month == 12) & (train.index.day == 25)),
            (
                "wiki_Thanksgiving",
                (train.index.month == 11)
                & (train.index.dayofweek == 3)
                & (train.index.day >= 22)
                & (train.index.day <= 28),
            ),
            ("wiki_Easter", easter_offsets(train.index).eq(0)),
            (
                "wiki_Chinese_New_Year",
                train.index.isin(
                    pd.to_datetime(
                        [
                            "2017-01-28",
                            "2018-02-16",
                            "2019-02-05",
                            "2020-01-25",
                            "2021-02-12",
                            "2022-02-01",
                            "2023-01-22",
                            "2024-02-10",
                        ]
                    )
                ),
            ),
        ]:
            if column not in train:
                continue
            components = detector.components[column]
            residual = train[column] - sum(
                np.asarray(components[key])
                for key in ["seasonality", "holidays", "trend", "level_shift"]
            )
            result["holiday_profiles"][column] = {
                "anomaly_days": int(
                    detector.anomaly_detector.anomalies.loc[mask, column].eq(-1).sum()
                ),
                "holiday_normalized_mean": float(
                    np.asarray(components["holidays"])[mask].mean() / scale[column]
                ),
                "residual_normalized_rmse": float(
                    np.sqrt(np.mean(residual.loc[mask] ** 2)) / scale[column]
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    results = {}
    wiki = load_daily(long=False).filter(regex="^wiki_").interpolate().ffill().bfill()
    cases = [("wikipedia", wiki, None)]
    for seed in args.seeds:
        generator = SyntheticDailyGenerator(n_days=1460, n_series=12, random_seed=seed)
        cases.append((f"synthetic_{seed}", generator.get_data(), generator))
    for name, data, generator in cases:
        results[name] = {}
        for variant in CONFIGURATIONS:
            print(f"{name}: {variant}", flush=True)
            results[name][variant] = evaluate(data, variant, generator)
            args.output.write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in results[name][variant].items()
                        if key != "level_shift_dates"
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
