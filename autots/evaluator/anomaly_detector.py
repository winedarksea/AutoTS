# -*- coding: utf-8 -*-
"""Anomaly Detector
Created on Mon Jul 18 14:19:55 2022

@author: Colin
"""
import copy
import random
import numpy as np
import pandas as pd
from autots.tools.anomaly_utils import (
    anomaly_new_params,
    detect_anomalies,
    limits_to_anomalies,
    anomaly_df_to_holidays,
    holiday_new_params,
    dates_to_holidays,
    fit_anomaly_classifier,
    score_to_anomaly,
)
from autots.tools.transform import RandomTransform, GeneralTransformer
from autots.tools.impute import FillNA
from autots.evaluator.auto_model import random_model
from autots.evaluator.auto_model import back_forecast

_ANOMALY_DEFAULT_TRANSFORM_DICT = {
    "transformations": {0: "DatepartRegression"},
    "transformation_params": {
        0: {
            "datepart_method": "simple_3",
            "regression_model": {"model": "ElasticNet", "model_params": {}},
        }
    },
}
_UNSET = object()


class AnomalyDetector(object):
    def __init__(
        self,
        output="multivariate",
        method="zscore",
        transform_dict=_UNSET,
        forecast_params=None,
        method_params=None,
        eval_period=None,
        isolated_only=False,
        n_jobs=1,
    ):
        """Detect anomalies on a historic dataset.
        Note anomaly score patterns vary by method.
        Anomaly flag is standard -1 = anomaly; 1 = regular as per sklearn

        Args:
            output (str): 'multivariate' (each series unique outliers), or 'univariate' (all series together for one outlier flag per timestamp)
            method (str): method choosen, from sklearn, AutoTS, and basic stats. Use `.get_new_params()` to see potential models
            transform_dict (dict): option but helpful, often datepart, differencing, or other standard AutoTS transformer params.
                Pass None to disable transforms.
            forecast_params (dict): used to backcast and identify 'unforecastable' values, required only for predict_interval method
            method_params (dict): parameters specific to the method, use `.get_new_params()` to see potential models
            eval_periods (int): only use this length tail of data, currently only implemented for forecast_params forecasting if used
            isolated_only (bool): if True, only standalone anomalies reported
            n_jobs (int): multiprocessing jobs, used by some methods

        Methods:
            detect()
            plot()
            get_new_params()
            score_to_anomaly()  # estimate

        Attributes:
            anomalies
            scores
        """
        if transform_dict is _UNSET:
            transform_dict = copy.deepcopy(_ANOMALY_DEFAULT_TRANSFORM_DICT)
        if method_params is None:
            method_params = {}
        self.output = output
        self.method = method
        self.transform_dict = transform_dict
        self.forecast_params = forecast_params
        self.method_params = method_params
        self.eval_period = eval_period
        self.isolated_only = isolated_only
        self.n_jobs = n_jobs
        self.anomaly_classifier = None
        self.anomalies = None
        self.scores = None

    def detect(self, df):
        """Shared anomaly detection routine."""
        self.df = df.copy()
        self.df_anomaly = df.copy()
        if self.transform_dict is not None:
            model = GeneralTransformer(
                verbose=2, **self.transform_dict
            )  # DATEPART, LOG, SMOOTHING, DIFF, CLIP OUTLIERS with high z score
            # the post selecting by columns is for CenterSplit and any similar renames or expansions
            transformed_df = model.fit_transform(self.df_anomaly)
            # Only select columns that exist in both original and transformed data (from expanding transformers)
            common_cols = [
                col for col in self.df.columns if col in transformed_df.columns
            ]
            if common_cols:
                self.df_anomaly = transformed_df[common_cols]
            else:
                self.df_anomaly = transformed_df

        if self.forecast_params is not None:
            backcast = back_forecast(
                self.df_anomaly,
                n_splits=self.method_params.get("n_splits", "auto"),
                forecast_length=self.method_params.get("forecast_length", 4),
                frequency="infer",
                eval_period=self.eval_period,
                prediction_interval=self.method_params.get("prediction_interval", 0.9),
                **self.forecast_params,
            )
            # don't difference for prediction_interval
            if self.method not in ["prediction_interval"]:
                if self.eval_period is not None:
                    self.df_anomaly = (
                        self.df_anomaly.tail(self.eval_period) - backcast.forecast
                    )
                else:
                    self.df_anomaly = self.df_anomaly - backcast.forecast

        if len(self.df_anomaly.columns) != len(df.columns):
            raise ValueError(
                f"anomaly returned a column mismatch from params {self.method_params} and {self.transform_dict}"
            )
        if not all(self.df_anomaly.columns == df.columns):
            self.df_anomaly.columns = df.columns

        if self.method in ["prediction_interval"]:
            df_for_limits = self.df_anomaly
            if self.eval_period is not None:
                df_for_limits = self.df_anomaly.tail(self.eval_period)
            self.anomalies, self.scores = limits_to_anomalies(
                df_for_limits,
                output=self.output,
                method_params=self.method_params,
                upper_limit=backcast.upper_forecast,
                lower_limit=backcast.lower_forecast,
            )
        else:
            self.anomalies, self.scores = detect_anomalies(
                self.df_anomaly,
                output=self.output,
                method=self.method,
                transform_dict=self.transform_dict,
                method_params=self.method_params,
                eval_period=self.eval_period,
                n_jobs=self.n_jobs,
            )
        if self.isolated_only:
            # replace all anomalies (-1) except those which are isolated (1 before and after)
            mask_minus_one = self.anomalies == -1
            mask_prev_one = self.anomalies.shift(1) == 1
            mask_next_one = self.anomalies.shift(-1) == 1
            mask_replace = mask_minus_one & ~(mask_prev_one & mask_next_one)
            self.anomalies[mask_replace] = 1
        return self.anomalies, self.scores

    def remove_anomalies(self, df=None, fillna=None):
        """Detect and return a copy of the data with anomalies removed (set NaN or filled).

        Args:
            df (pd.DataFrame, optional): data to run detection on. If None, uses previous `detect` input.
            fillna (str, optional): fill method passed to `autots.tools.impute.FillNA`.
        """
        if df is not None:
            _, _ = self.detect(df)
        elif not hasattr(self, "df"):
            raise ValueError(
                "Call `detect(df)` or provide `df` before removing anomalies."
            )
        df_clean = self.df.copy()
        df_clean = df_clean[self.anomalies != -1]
        if fillna is not None:
            df_clean = FillNA(df_clean, method=fillna, window=10)
        return df_clean

    def plot(
        self,
        series_name=None,
        title=None,
        marker_size=None,
        plot_kwargs=None,
        start_date=None,
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        import matplotlib.pyplot as plt

        if series_name is None:
            series_name = random.choice(self.df.columns)
        if title is None:
            title = series_name[0:50] + f" with {self.method} outliers"

        # Filter data by start_date if provided
        df_plot = self.df
        anomalies_plot = self.anomalies
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df_plot = self.df.loc[self.df.index >= start_date]
            anomalies_plot = self.anomalies.loc[self.anomalies.index >= start_date]

        fig, ax = plt.subplots()
        df_plot[series_name].plot(ax=ax, title=title, **plot_kwargs)
        if self.output == "univariate":
            i_anom = anomalies_plot.index[anomalies_plot.iloc[:, 0] == -1]
        else:
            series_anom = anomalies_plot[series_name]
            i_anom = series_anom[series_anom == -1].index
        if len(i_anom) > 0:
            if marker_size is None:
                marker_size = max(20, fig.dpi * 0.45)
            ax.scatter(
                i_anom.tolist(),
                df_plot.loc[i_anom, :][series_name],
                c="red",
                s=marker_size,
            )

    def fit(self, df):
        return self.detect(df)

    def fit_anomaly_classifier(self):
        """Fit a model to predict if a score is an anomaly."""
        self.anomaly_classifier, self.score_categories = fit_anomaly_classifier(
            self.anomalies, self.scores
        )

    def score_to_anomaly(self, scores):
        """A DecisionTree model, used as models are nonstandard (and nonparametric)."""
        if self.anomaly_classifier is None:
            self.fit_anomaly_classifier()
        return score_to_anomaly(scores, self.anomaly_classifier, self.score_categories)

    @staticmethod
    def get_new_params(method="random"):
        """Generate random new parameter combinations.

        Args:
            method (str): 'fast', 'deep', 'default', or any of the anomaly method names (ie 'IQR') to specify only that method
        """
        forecast_params = None
        method_choice, method_params, transform_dict = anomaly_new_params(method=method)
        if transform_dict == "random":
            transform_dict = RandomTransform(
                transformer_list='scalable', transformer_max_depth=2
            )
        if method == "fast":
            preforecast = False
        else:
            preforecast = random.choices([True, False], [0.05, 0.95])[0]

        if preforecast or method_choice == "prediction_interval":
            forecast_params = random_model(
                model_list=['LastValueNaive', 'GLS', 'RRVAR', "SeasonalityMotif"],
                model_prob=[0.8, 0.1, 0.05, 0.05],
                transformer_max_depth=5,
                transformer_list="superfast",
                keyword_format=True,
            )
        return {
            "method": method_choice,
            "transform_dict": transform_dict,
            "forecast_params": forecast_params,
            "method_params": method_params,
        }


class HolidayDetector(object):
    def __init__(
        self,
        anomaly_detector_params=None,
        threshold=0.8,
        min_occurrences=2,
        splash_threshold=0.65,
        use_dayofmonth_holidays=True,
        use_wkdom_holidays=True,
        use_wkdeom_holidays=True,
        use_lunar_holidays=True,
        use_lunar_weekday=False,
        use_islamic_holidays=False,
        use_hebrew_holidays=False,
        use_hindu_holidays=False,
        output: str = "multivariate",
        n_jobs: int = 1,
        auto_relax: bool = False,
        relax_threshold_floor: float = 0.55,
        relax_splash_threshold: float = 0.55,
        relax_rounds: int = 2,
        min_holidays_per_series: int = 1,
        max_holidays_per_series: int = None,
        holiday_selection_strategy: str = "score",
    ):
        """Detect anomalies, then mark as holidays (events, festivals, etc) any that reoccur to a calendar.

        Be aware of timezone, especially combining series from multiple time zones. Dates then may not accurately align.
        Can pick up a holiday on the wrong calendar especially for extended holidays (Christmas) and with short (several years is short here) history.
        Holidays on unusual days or weekdays of month (5th Monday of April) may occur
        No multiyear patterns (election year) are detected - would need lots of history

        Args:
            anomaly_detector_params (dict): anomaly detection params passed to detector class
            threshold (float): percent of date occurrences that must be anomalous (0 - 1)
            splash_threshold (float): None, or % required, avg of nearest 2 neighbors to point
            use* (bool): whether to use these calendars for holiday detection
            output (str): "multivariate" or "univariate", for univariate not all dates_to_holidays styles will work
            auto_relax (bool): if True, progressively relax threshold rules when very few holidays are detected.
            relax_threshold_floor (float): minimum threshold used during relaxed fallback rounds.
            relax_splash_threshold (float): splash threshold used during relaxed fallback rounds.
            relax_rounds (int): number of progressive relaxation rounds.
            min_holidays_per_series (int): minimum desired holiday count per series when auto_relax is enabled.
            max_holidays_per_series (int): optional cap on total holiday rules retained per series.
            holiday_selection_strategy (str): "score" for static scoring cap, or "coverage"
                for anomaly-coverage aware greedy rule selection when max_holidays_per_series is set.

        Methods:
            detect()
            dates_to_holidays()
            plot()
            get_new_params()
        """
        self.anomaly_detector_params = (
            anomaly_detector_params if anomaly_detector_params is not None else {}
        )
        self.threshold = threshold
        self.min_occurrences = min_occurrences
        self.splash_threshold = splash_threshold
        self.use_dayofmonth_holidays = use_dayofmonth_holidays
        self.use_wkdom_holidays = use_wkdom_holidays
        self.use_wkdeom_holidays = use_wkdeom_holidays
        self.use_lunar_holidays = use_lunar_holidays
        self.use_lunar_weekday = use_lunar_weekday
        self.use_islamic_holidays = use_islamic_holidays
        self.use_hebrew_holidays = use_hebrew_holidays
        self.use_hindu_holidays = use_hindu_holidays
        self.n_jobs = n_jobs
        self.output = output
        self.auto_relax = bool(auto_relax)
        self.relax_threshold_floor = float(np.clip(relax_threshold_floor, 0.0, 1.0))
        self.relax_splash_threshold = float(np.clip(relax_splash_threshold, 0.0, 1.0))
        self.relax_rounds = max(0, int(relax_rounds))
        self.min_holidays_per_series = max(0, int(min_holidays_per_series))
        if max_holidays_per_series is None:
            self.max_holidays_per_series = None
        else:
            self.max_holidays_per_series = max(1, int(max_holidays_per_series))
        self.holiday_selection_strategy = str(holiday_selection_strategy).lower()
        if self.holiday_selection_strategy not in {"score", "coverage"}:
            self.holiday_selection_strategy = "score"
        self.anomaly_model = AnomalyDetector(
            output=output, **self.anomaly_detector_params, n_jobs=n_jobs
        )
        self._detection_stats = {}

    @staticmethod
    def _table_names():
        return [
            "day_holidays",
            "wkdom_holidays",
            "wkdeom_holidays",
            "lunar_holidays",
            "lunar_weekday",
            "islamic_holidays",
            "hebrew_holidays",
            "hindu_holidays",
        ]

    def _extract_holidays(
        self,
        anomalies,
        actuals,
        anomaly_scores,
        threshold,
        splash_threshold,
    ):
        return anomaly_df_to_holidays(
            anomalies,
            splash_threshold=splash_threshold,
            threshold=threshold,
            min_occurrences=self.min_occurrences,
            actuals=actuals,
            anomaly_scores=anomaly_scores,
            use_dayofmonth_holidays=self.use_dayofmonth_holidays,
            use_wkdom_holidays=self.use_wkdom_holidays,
            use_wkdeom_holidays=self.use_wkdeom_holidays,
            use_lunar_holidays=self.use_lunar_holidays,
            use_lunar_weekday=self.use_lunar_weekday,
            use_islamic_holidays=self.use_islamic_holidays,
            use_hebrew_holidays=self.use_hebrew_holidays,
            use_hindu_holidays=self.use_hindu_holidays,
        )

    @staticmethod
    def _build_priority_scores(combined):
        score = pd.Series(0.0, index=combined.index, dtype=float)
        if "count" in combined.columns:
            score = score + combined["count"].fillna(0.0).astype(float) * 1.5
        if "occurrence_rate" in combined.columns:
            score = score + combined["occurrence_rate"].fillna(0.0).astype(float) * 2.0
        if "avg_anomaly_score" in combined.columns:
            score = score + combined["avg_anomaly_score"].fillna(0.0).abs().astype(
                float
            )
        return score

    def _rank_cap_rules(self, combined):
        sort_cols = ["series", "_priority_score"]
        asc = [True, False]
        if "count" in combined.columns:
            sort_cols.append("count")
            asc.append(False)
        if "occurrence_rate" in combined.columns:
            sort_cols.append("occurrence_rate")
            asc.append(False)
        combined = combined.sort_values(sort_cols, ascending=asc)
        combined = combined.groupby("series", group_keys=False).head(
            self.max_holidays_per_series
        )
        return combined

    @staticmethod
    def _calendar_rule_mask(calendar_name, row, dates_df):
        try:
            if calendar_name == "day_holidays":
                if pd.isna(row.get("month")) or pd.isna(row.get("day")):
                    return None
                month = int(row["month"])
                day = int(row["day"])
                return (
                    (dates_df["month"] == month) & (dates_df["day"] == day)
                ).to_numpy(dtype=bool)
            if calendar_name == "wkdom_holidays":
                if (
                    pd.isna(row.get("month"))
                    or pd.isna(row.get("weekofmonth"))
                    or pd.isna(row.get("dayofweek"))
                ):
                    return None
                month = int(row["month"])
                weekofmonth = int(row["weekofmonth"])
                dayofweek = int(row["dayofweek"])
                return (
                    (dates_df["month"] == month)
                    & (dates_df["weekofmonth"] == weekofmonth)
                    & (dates_df["dayofweek"] == dayofweek)
                ).to_numpy(dtype=bool)
            if calendar_name == "wkdeom_holidays":
                if (
                    pd.isna(row.get("month"))
                    or pd.isna(row.get("weekfromend"))
                    or pd.isna(row.get("dayofweek"))
                ):
                    return None
                month = int(row["month"])
                weekfromend = int(row["weekfromend"])
                dayofweek = int(row["dayofweek"])
                return (
                    (dates_df["month"] == month)
                    & (dates_df["weekfromend"] == weekfromend)
                    & (dates_df["dayofweek"] == dayofweek)
                ).to_numpy(dtype=bool)
            return None
        except Exception:
            return None

    def _series_anomaly_mask(self, anomalies, series_name):
        if anomalies is None or anomalies.empty:
            return None
        if self.output == "univariate":
            if anomalies.shape[1] < 1:
                return None
            base = anomalies.iloc[:, 0]
        else:
            if series_name not in anomalies.columns:
                return None
            base = anomalies[series_name]
        return base.eq(-1).to_numpy(dtype=bool)

    def _select_rules_by_coverage(self, combined, anomalies, date_index, df_cols):
        if combined.empty:
            return set()

        dates = pd.DatetimeIndex(date_index)
        dates_df = pd.DataFrame(
            {
                "month": dates.month.astype(int),
                "day": dates.day.astype(int),
                "dayofweek": dates.dayofweek.astype(int),
            },
            index=dates,
        )
        dates_df["weekofmonth"] = ((dates_df["day"] - 1) // 7 + 1).astype(int)
        dates_df["weekfromend"] = (
            (dates_df["day"] - dates.daysinmonth.astype(int)) // -7
        ).astype(int)

        rule_masks = {}
        for idx, row in combined.iterrows():
            rule_mask = self._calendar_rule_mask(row.get("_calendar"), row, dates_df)
            if rule_mask is not None and np.any(rule_mask):
                rule_masks[idx] = rule_mask

        selected_indices = set()
        for series_name in df_cols:
            series_rules = combined[combined["series"] == series_name]
            if series_rules.empty:
                continue
            if series_rules.shape[0] <= self.max_holidays_per_series:
                selected_indices.update(series_rules.index.tolist())
                continue

            anomaly_mask = self._series_anomaly_mask(anomalies, series_name)
            if anomaly_mask is None or not np.any(anomaly_mask):
                ranked = self._rank_cap_rules(series_rules)
                selected_indices.update(ranked.index.tolist())
                continue

            covered = np.zeros(len(dates), dtype=bool)
            remaining = series_rules.index.tolist()
            chosen = []

            while remaining and len(chosen) < self.max_holidays_per_series:
                best_idx = None
                best_score = -np.inf
                for rid in remaining:
                    row = series_rules.loc[rid]
                    base_score = float(row.get("_priority_score", 0.0))
                    rule_mask = rule_masks.get(rid)
                    if rule_mask is None:
                        marginal = base_score - 0.1
                    else:
                        predicted = int(rule_mask.sum())
                        total_hits = int((rule_mask & anomaly_mask).sum())
                        new_hits = int((rule_mask & anomaly_mask & ~covered).sum())
                        overlap = int((rule_mask & covered).sum())
                        precision = total_hits / predicted if predicted > 0 else 0.0
                        marginal = (
                            base_score
                            + (3.0 * new_hits)
                            + (1.5 * precision)
                            - (0.15 * overlap)
                            - (0.005 * predicted)
                        )
                    if marginal > best_score:
                        best_score = marginal
                        best_idx = rid
                if best_idx is None:
                    break
                if best_score <= 0 and chosen:
                    break
                chosen.append(best_idx)
                remaining.remove(best_idx)
                picked_mask = rule_masks.get(best_idx)
                if picked_mask is not None:
                    covered = covered | picked_mask

            if not chosen:
                fallback = self._rank_cap_rules(series_rules)
                chosen = fallback.index.tolist()
            selected_indices.update(chosen[: self.max_holidays_per_series])

        return selected_indices

    def _apply_holiday_cap(
        self, holiday_tables, anomalies=None, date_index=None, df_cols=None
    ):
        if self.max_holidays_per_series is None:
            return holiday_tables

        names = self._table_names()
        combined_frames = []
        original_by_name = dict(zip(names, holiday_tables))

        for name in names:
            table = original_by_name.get(name)
            if table is None or table.empty or "series" not in table.columns:
                continue
            tmp = table.copy()
            tmp["_calendar"] = name
            combined_frames.append(tmp)

        if not combined_frames:
            return holiday_tables

        combined = pd.concat(combined_frames, axis=0, ignore_index=True, sort=False)
        combined["_priority_score"] = self._build_priority_scores(combined)

        if (
            self.holiday_selection_strategy == "coverage"
            and anomalies is not None
            and date_index is not None
            and df_cols is not None
        ):
            selected_indices = self._select_rules_by_coverage(
                combined, anomalies, date_index, df_cols
            )
            if selected_indices:
                combined = combined.loc[sorted(selected_indices)]
            else:
                combined = self._rank_cap_rules(combined)
        else:
            combined = self._rank_cap_rules(combined)

        capped_tables = []
        for name in names:
            original = original_by_name.get(name)
            if original is None:
                capped_tables.append(None)
                continue
            subset = combined[combined["_calendar"] == name].drop(
                columns=["_calendar", "_priority_score"], errors="ignore"
            )
            if subset.empty:
                # Preserve schema when possible for downstream consistency.
                capped_tables.append(original.iloc[0:0].copy())
            else:
                capped_tables.append(subset.reset_index(drop=True))
        return tuple(capped_tables)

    @staticmethod
    def _series_holiday_counts(date_index, df_cols, holiday_tables):
        names = HolidayDetector._table_names()
        holidays = dict(zip(names, holiday_tables))
        flags = dates_to_holidays(
            dates=date_index,
            df_cols=df_cols,
            style="series_flag",
            day_holidays=holidays["day_holidays"],
            wkdom_holidays=holidays["wkdom_holidays"],
            wkdeom_holidays=holidays["wkdeom_holidays"],
            lunar_holidays=holidays["lunar_holidays"],
            lunar_weekday=holidays["lunar_weekday"],
            islamic_holidays=holidays["islamic_holidays"],
            hebrew_holidays=holidays["hebrew_holidays"],
            hindu_holidays=holidays["hindu_holidays"],
        )
        if flags is None or flags.empty:
            return pd.Series(0, index=df_cols, dtype=int)
        flags = flags.reindex(columns=df_cols, fill_value=0)
        return flags.sum(axis=0).astype(int)

    @staticmethod
    def _choose_better_candidate(
        current_tables,
        current_counts,
        candidate_tables,
        candidate_counts,
        min_holidays_per_series,
    ):
        current_cov = int((current_counts >= min_holidays_per_series).sum())
        candidate_cov = int((candidate_counts >= min_holidays_per_series).sum())
        if candidate_cov > current_cov:
            return candidate_tables, candidate_counts
        if candidate_cov < current_cov:
            return current_tables, current_counts

        current_total = int(current_counts.sum())
        candidate_total = int(candidate_counts.sum())
        if candidate_cov >= len(current_counts):
            # If both satisfy coverage, prefer the simpler/less noisy candidate.
            if candidate_total < current_total:
                return candidate_tables, candidate_counts
        else:
            # Otherwise prefer coverage-building candidates with more signal.
            if candidate_total > current_total:
                return candidate_tables, candidate_counts
        return current_tables, current_counts

    def _relaxed_anomaly_detector_params(self):
        params = copy.deepcopy(self.anomaly_detector_params)
        method_params = params.get("method_params")
        if method_params is None:
            method_params = {}
            params["method_params"] = method_params
        elif not isinstance(method_params, dict):
            method_params = {}
            params["method_params"] = method_params

        method = str(params.get("method", "")).lower()
        changed = False

        if "alpha" in method_params:
            alpha = float(method_params.get("alpha", 0.05) or 0.05)
            method_params["alpha"] = min(0.2, max(alpha * 2.0, alpha + 0.02))
            changed = True
        if "contamination" in method_params:
            contamination = float(method_params.get("contamination", 0.05) or 0.05)
            method_params["contamination"] = min(
                0.25, max(contamination * 1.5, contamination + 0.03)
            )
            changed = True
        if "iqr_threshold" in method_params:
            iqr = float(method_params.get("iqr_threshold", 2.0) or 2.0)
            method_params["iqr_threshold"] = max(1.2, iqr - 0.5)
            changed = True
        if "responsibility_threshold" in method_params:
            rt = float(method_params.get("responsibility_threshold", 0.05) or 0.05)
            method_params["responsibility_threshold"] = max(0.01, rt * 0.6)
            changed = True

        if not changed and method in {"zscore", "rolling_zscore", "mad"}:
            method_params["alpha"] = 0.1
            changed = True
        if not changed and method == "iqr":
            method_params["iqr_threshold"] = 1.5
            changed = True
        if not changed and method == "ee":
            method_params["contamination"] = 0.12
            changed = True

        return params if changed else None

    def _set_holiday_tables(self, tables):
        (
            self.day_holidays,
            self.wkdom_holidays,
            self.wkdeom_holidays,
            self.lunar_holidays,
            self.lunar_weekday,
            self.islamic_holidays,
            self.hebrew_holidays,
            self.hindu_holidays,
        ) = tables

    def detect(self, df):
        """Run holiday detection. Input wide-style pandas time series."""
        self.anomaly_model.detect(df)
        selected_anomaly_model = self.anomaly_model
        self.df = df
        self.df_cols = df.columns
        if np.min(self.anomaly_model.anomalies.values) != -1:
            print("No anomalies detected.")
        actuals = df if self.output != "univariate" else None
        anomaly_scores = (
            self.anomaly_model.scores if self.output != "univariate" else None
        )

        selected_tables = self._extract_holidays(
            anomalies=self.anomaly_model.anomalies,
            actuals=actuals,
            anomaly_scores=anomaly_scores,
            threshold=self.threshold,
            splash_threshold=self.splash_threshold,
        )
        selected_tables = self._apply_holiday_cap(
            selected_tables,
            anomalies=self.anomaly_model.anomalies,
            date_index=df.index,
            df_cols=self.df_cols,
        )
        selected_counts = self._series_holiday_counts(
            df.index, self.df_cols, selected_tables
        )

        if self.auto_relax and self.relax_rounds > 0:
            # First attempt: relax calendar thresholds while preserving anomaly labels.
            threshold_path = np.linspace(
                float(self.threshold),
                min(float(self.threshold), self.relax_threshold_floor),
                self.relax_rounds + 1,
            )[1:]
            for threshold in threshold_path:
                splash_threshold = self.splash_threshold
                if splash_threshold is None:
                    splash_threshold = self.relax_splash_threshold
                else:
                    splash_threshold = min(
                        float(splash_threshold), self.relax_splash_threshold
                    )
                candidate_tables = self._extract_holidays(
                    anomalies=self.anomaly_model.anomalies,
                    actuals=actuals,
                    anomaly_scores=anomaly_scores,
                    threshold=float(threshold),
                    splash_threshold=splash_threshold,
                )
                candidate_tables = self._apply_holiday_cap(
                    candidate_tables,
                    anomalies=self.anomaly_model.anomalies,
                    date_index=df.index,
                    df_cols=self.df_cols,
                )
                candidate_counts = self._series_holiday_counts(
                    df.index, self.df_cols, candidate_tables
                )
                selected_tables, selected_counts = self._choose_better_candidate(
                    selected_tables,
                    selected_counts,
                    candidate_tables,
                    candidate_counts,
                    self.min_holidays_per_series,
                )

            # Second attempt: if still sparse, rerun anomaly detector with more sensitivity.
            if (selected_counts >= self.min_holidays_per_series).sum() < len(
                selected_counts
            ):
                relaxed_anomaly_params = self._relaxed_anomaly_detector_params()
                if relaxed_anomaly_params is not None:
                    relaxed_anomaly_model = AnomalyDetector(
                        output=self.output,
                        n_jobs=self.n_jobs,
                        **relaxed_anomaly_params,
                    )
                    relaxed_anomaly_model.detect(df)
                    relaxed_scores = (
                        relaxed_anomaly_model.scores
                        if self.output != "univariate"
                        else None
                    )
                    relaxed_tables = self._extract_holidays(
                        anomalies=relaxed_anomaly_model.anomalies,
                        actuals=actuals,
                        anomaly_scores=relaxed_scores,
                        threshold=min(
                            float(self.threshold), self.relax_threshold_floor
                        ),
                        splash_threshold=(
                            self.relax_splash_threshold
                            if self.splash_threshold is None
                            else min(
                                float(self.splash_threshold),
                                self.relax_splash_threshold,
                            )
                        ),
                    )
                    relaxed_tables = self._apply_holiday_cap(
                        relaxed_tables,
                        anomalies=relaxed_anomaly_model.anomalies,
                        date_index=df.index,
                        df_cols=self.df_cols,
                    )
                    relaxed_counts = self._series_holiday_counts(
                        df.index, self.df_cols, relaxed_tables
                    )
                    selected_tables, selected_counts = self._choose_better_candidate(
                        selected_tables,
                        selected_counts,
                        relaxed_tables,
                        relaxed_counts,
                        self.min_holidays_per_series,
                    )
                    if selected_tables is relaxed_tables:
                        selected_anomaly_model = relaxed_anomaly_model

        self.anomaly_model = selected_anomaly_model
        self._set_holiday_tables(selected_tables)
        self._detection_stats = {
            "series_holiday_counts": selected_counts.to_dict(),
            "coverage_series": int(
                (selected_counts >= self.min_holidays_per_series).sum()
            ),
            "total_holiday_days": int(selected_counts.sum()),
        }
        return self

    def plot_anomaly(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Extract start_date if provided in kwargs to pass to the anomaly detector plot method
        self.anomaly_model.plot(**kwargs)

    def plot(
        self,
        series_name=None,
        include_anomalies=True,
        title=None,
        marker_size=None,
        plot_kwargs=None,
        series=None,
        start_date=None,
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        import matplotlib.pyplot as plt

        if series_name is None:
            if series is not None:
                series_name = series
            else:
                series_name = random.choice(self.df.columns)
        if title is None:
            title = (
                series_name[0:50]
                + f" with {self.anomaly_detector_params['method']} holidays"
            )

        # Filter data by start_date if provided
        df_plot = self.df
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            df_plot = self.df.loc[self.df.index >= start_date]

        fig, ax = plt.subplots()
        df_plot[series_name].plot(ax=ax, title=title, **plot_kwargs)
        if marker_size is None:
            marker_size = max(20, fig.dpi * 0.45)
        if include_anomalies:
            # directly copied from above
            if self.anomaly_model.output == "univariate":
                i_anom = self.anomaly_model.anomalies.index[
                    self.anomaly_model.anomalies.iloc[:, 0] == -1
                ]
            else:
                series_anom = self.anomaly_model.anomalies[series_name]
                i_anom = series_anom[series_anom == -1].index

            # Filter anomalies by start_date if provided
            if start_date is not None:
                i_anom = i_anom[i_anom >= start_date]
            # Ensure anomaly indices exist in filtered dataframe
            i_anom = i_anom[i_anom.isin(df_plot.index)]

            if len(i_anom) > 0:
                ax.scatter(
                    i_anom.tolist(),
                    df_plot.loc[i_anom, :][series_name],
                    c="red",
                    s=marker_size,
                )
        # now the actual holidays
        holiday_dates = self.dates_to_holidays(self.df.index, style="series_flag")[
            series_name
        ]
        i_anom = holiday_dates.index[holiday_dates == 1]

        # Filter holidays by start_date if provided
        if start_date is not None:
            i_anom = i_anom[i_anom >= start_date]
        # Ensure holiday indices exist in filtered dataframe
        i_anom = i_anom[i_anom.isin(df_plot.index)]

        if len(i_anom) > 0:
            ax.scatter(
                i_anom.tolist(),
                df_plot.loc[i_anom, :][series_name],
                c="green",
                s=marker_size,
            )

    def dates_to_holidays(
        self, dates, style="flag", holiday_impacts=False, max_features: int = None
    ):
        """Populate date information for a given pd.DatetimeIndex.

        Args:
            dates (pd.DatetimeIndex): list of dates
            day_holidays (pd.DataFrame): list of month/day holidays. Pass None if not available
            style (str): option for how to return information
                "long" - return date, name, series for all holidays in a long style dataframe
                "impact" - returns dates, series with values of sum of impacts (if given) or joined string of holiday names
                'flag' - return dates, holidays flag, (is not 0-1 but rather sum of input series impacted for that holiday and day)
                'prophet' - return format required for prophet. Will need to be filtered on `series` for multivariate case
                'series_flag' - dates, series 0/1 for if holiday occurred in any calendar
            holiday_impacts (dict): a dict passed to .replace contaning values for holiday_names, or str 'value' or 'anomaly_score'
        """
        return dates_to_holidays(
            dates,
            self.df_cols,
            style=style,
            holiday_impacts=holiday_impacts,
            day_holidays=self.day_holidays,
            wkdom_holidays=self.wkdom_holidays,
            wkdeom_holidays=self.wkdeom_holidays,
            lunar_holidays=self.lunar_holidays,
            lunar_weekday=self.lunar_weekday,
            islamic_holidays=self.islamic_holidays,
            hebrew_holidays=self.hebrew_holidays,
            hindu_holidays=self.hindu_holidays,
            max_features=max_features,
        )

    def fit(self, df):
        return self.detect(df)

    def get_detection_stats(self):
        """Return summary stats from the most recent holiday detection run."""
        return copy.deepcopy(self._detection_stats)

    @staticmethod
    def get_new_params(method="random"):
        holiday_params = holiday_new_params(method=method)
        holiday_params['anomaly_detector_params'] = AnomalyDetector.get_new_params(
            method=method
        )
        holiday_params['auto_relax'] = random.choices([True, False], [0.2, 0.8])[0]
        holiday_params['relax_threshold_floor'] = random.choices(
            [0.65, 0.55, 0.5], [0.4, 0.4, 0.2]
        )[0]
        holiday_params['relax_splash_threshold'] = random.choices(
            [0.65, 0.55, 0.45], [0.3, 0.4, 0.3]
        )[0]
        holiday_params['relax_rounds'] = random.choices([1, 2, 3], [0.3, 0.5, 0.2])[0]
        holiday_params['min_holidays_per_series'] = random.choices([1, 2], [0.9, 0.1])[
            0
        ]
        holiday_params['max_holidays_per_series'] = random.choices(
            [None, None, 24, 36, 52], [0.55, 0.2, 0.1, 0.1, 0.05]
        )[0]
        holiday_params['holiday_selection_strategy'] = random.choices(
            ["score", "coverage"], [0.75, 0.25]
        )[0]
        return holiday_params
