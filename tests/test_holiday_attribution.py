"""Annual holiday peer comparisons and detector attribution safety."""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from autots.evaluator.feature_detector import (
    TimeSeriesFeatureDetector,
    FeatureDetectionOptimizer,
)
from autots.tools.easter import easter_offsets
from autots.tools.transform import AnomalyRemoval


class TestNamedHolidaySuppression(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", "2024-12-31")
        self.holidays = pd.to_datetime([f"{year}-12-25" for year in range(2020, 2025)])
        self.data = pd.DataFrame(0.0, index=self.index, columns=["a", "b"])
        self.flags = pd.DataFrame(1, index=self.index, columns=self.data.columns)
        self.flags.loc[self.holidays] = -1
        self.strengths = pd.DataFrame(0.1, index=self.index, columns=self.data.columns)
        self.strengths.loc[self.holidays] = 5.0
        self.named = {
            column: {"Christmas": self.holidays} for column in self.data.columns
        }

    def _filter(self, groups=None, override=1.5, output="multivariate", strengths=True):
        detector = AnomalyRemoval(
            holiday_dates=self.named if groups is None else groups,
            holiday_override_strength=override,
            output=output,
            transform_dict=None,
        )
        detector.df_anomaly = self.data
        detector.anomalies = (
            self.flags.copy()
            if output == "multivariate"
            else self.flags[["a"]].rename(columns={"a": "anomaly"})
        )
        scores = self.strengths if strengths else None
        with patch.object(detector, "_normalized_strengths", return_value=scores):
            detector._filter_holiday_proximate_anomalies()
        return detector.anomalies

    def test_recurring_spikes_and_exceptional_outage(self):
        extreme = self.holidays[2]
        self.strengths.loc[extreme, "a"] = 15.0
        result = self._filter()
        self.assertEqual(result.loc[extreme, "a"], -1)
        self.assertTrue(result.loc[self.holidays, "b"].eq(1).all())
        self.assertEqual(result.loc[self.holidays, "a"].eq(-1).sum(), 1)

    def test_other_series_holidays_do_not_hide_outage(self):
        result = self._filter({"a": {}, "b": self.named["b"]})
        self.assertTrue(result.loc[self.holidays, "a"].eq(-1).all())
        self.assertTrue(result.loc[self.holidays, "b"].eq(1).all())

    def test_missing_peers_and_missing_scores_preserve_flags(self):
        self.data.loc[self.holidays[:3], "a"] = np.nan
        result = self._filter()
        self.assertTrue(result.loc[self.holidays, "a"].eq(-1).all())
        result = self._filter(strengths=False)
        self.assertTrue(result.loc[self.holidays].eq(-1).all().all())
        result = self._filter(override=None)
        self.assertTrue(result.loc[self.holidays].eq(1).all().all())

    def test_unflagged_peers_are_still_observations(self):
        self.flags.loc[self.holidays[:4], "a"] = 1
        self.assertEqual(self._filter().loc[self.holidays[-1], "a"], 1)

    def test_offsets_and_other_holidays_do_not_dilute_reference(self):
        weak = pd.to_datetime([f"{year}-07-04" for year in range(2020, 2025)])
        for column in self.data:
            self.named[column]["weak"] = weak
        self.flags.loc[weak] = -1
        for offset in [-2, -1, 1, 2]:
            self.flags.loc[self.holidays + pd.Timedelta(days=offset)] = -1
        result = self._filter()
        self.assertTrue(result.loc[self.holidays].eq(1).all().all())

    def test_moving_holiday_and_overlap_order_are_stable(self):
        moving = self.index[easter_offsets(self.index).eq(0)]
        self.flags.loc[moving] = -1
        self.strengths.loc[moving] = 5
        self.named["a"] = {"Easter": moving, "single_date": [moving[0]]}
        self.named["b"] = dict(reversed(list(self.named["a"].items())))
        result = self._filter()
        self.assertTrue(result.loc[moving].eq(1).all().all())

    def test_shared_flags_require_every_series_to_qualify(self):
        result = self._filter(output="univariate")
        self.assertTrue(result.loc[self.holidays].eq(1).all().all())
        self.named["b"] = {}
        result = self._filter(output="univariate")
        self.assertTrue(result.loc[self.holidays].eq(-1).all().all())

    def test_new_year_offsets_exclude_the_anchor_year(self):
        anchors = pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])
        dates = anchors - pd.Timedelta(days=1)
        self.flags.loc[dates] = -1
        self.strengths.loc[dates] = 5
        self.strengths.loc[dates[-1], "a"] = 10
        self.named = {column: {"NewYear": anchors} for column in self.data}
        result = self._filter()
        self.assertTrue(result.loc[dates, "b"].eq(1).all())
        self.assertEqual(result.loc[dates[-1], "a"], -1)

    def test_two_years_of_history_keep_flags(self):
        self.named = {column: {"Christmas": self.holidays[:2]} for column in self.data}
        self.assertTrue(self._filter().loc[self.holidays[:2]].eq(-1).all().all())

    def test_intraday_timezone_and_proximity_boundary(self):
        self.index = pd.date_range(
            "2020-01-01", "2024-12-31", freq="12h", tz="America/Chicago"
        )
        self.data = pd.DataFrame(0.0, index=self.index, columns=["a", "b"])
        self.flags = pd.DataFrame(-1, index=self.index, columns=self.data.columns)
        self.strengths = pd.DataFrame(5.0, index=self.index, columns=self.data.columns)
        result = self._filter()
        self.assertEqual(result.loc["2024-12-25 12:00", "a"], 1)
        self.assertEqual(result.loc["2024-12-28 12:00", "a"], -1)


class TestDetectorHolidayAttribution(unittest.TestCase):
    @staticmethod
    def _data():
        index = pd.date_range("2017-01-01", "2024-12-31")
        rng = np.random.default_rng(41)
        data = pd.DataFrame(
            rng.normal(0, 0.1, (len(index), 2)),
            index=index,
            columns=["christmas", "easter"],
        )
        data += 10 + np.sin(np.arange(len(index)) * 2 * np.pi / 7)[:, None]
        data.loc[(index.month == 12) & (index.day == 25), "christmas"] += 20
        data.loc[easter_offsets(index).eq(0), "easter"] += 20
        return data

    @staticmethod
    def _params():
        seasonal = {
            "regression_model": {"model": "FastRidge", "model_params": {}},
            "datepart_method": "dayofweek",
            "transform_dict": None,
            "polynomial_degree": None,
            "holiday_countries_used": False,
        }
        return {
            "global_holiday_anomaly_suppression": None,
            "rough_seasonality_params": seasonal,
            "seasonality_params": seasonal,
            "holiday_params": {"use_easter_holidays": True, "use_lunar_holidays": True},
        }

    def test_attribution_outages_and_forecast_continuity(self):
        data = self._data()
        outage_date = pd.Timestamp("2020-12-25")
        exceptional = pd.Timestamp("2021-12-25")
        data.loc[outage_date, "easter"] -= 60
        data.loc[exceptional, "christmas"] -= 80
        detector = TimeSeriesFeatureDetector(**self._params()).fit(data)
        christmas = data.index[(data.index.month == 12) & (data.index.day == 25)]
        easter = data.index[easter_offsets(data.index).eq(0)]
        for column, dates in [("christmas", christmas), ("easter", easter)]:
            flags = detector.anomaly_detector.anomalies[column]
            ordinary = dates[dates != exceptional] if column == "christmas" else dates
            self.assertTrue(flags.loc[ordinary].eq(1).all())
            component = pd.Series(
                detector.components[column]["holidays"], index=data.index
            )
            self.assertGreaterEqual(component.loc[ordinary].mean(), 16.0)
        self.assertEqual(
            detector.anomaly_detector.anomalies.loc[outage_date, "easter"], -1
        )
        self.assertEqual(
            detector.anomaly_detector.anomalies.loc[exceptional, "christmas"], -1
        )
        future = pd.date_range("2025-04-18", "2025-04-21")
        regressors = detector._build_holiday_regressors_for_index(future, data.columns)
        self.assertEqual(regressors.loc["2025-04-20", "easter_0"], 1)
        self.assertEqual(list(regressors.columns), detector._holiday_regressor_columns)
        prediction = detector.forecast(110)
        holiday_forecast = prediction.components.xs("holidays", axis=1, level=1)
        self.assertGreaterEqual(holiday_forecast.loc["2025-04-20", "easter"], 16.0)

    def test_shared_detection_broadcasts_holiday_dates(self):
        params = self._params()
        params["detection_mode"] = "univariate"
        detector = TimeSeriesFeatureDetector(**params).fit(self._data())
        self.assertIn(pd.Timestamp("2024-12-25"), detector.holiday_dates["christmas"])
        self.assertEqual(
            detector.holiday_dates["christmas"], detector.holiday_dates["easter"]
        )
        self.assertEqual(
            int(detector.anomaly_detector.anomalies.loc["2024-12-25"].iloc[0]), 1
        )

    def test_country_calendar_names_remain_per_series(self):
        data = self._data().rename(columns={"christmas": "us", "easter": "ca"})

        def calendar_flags(index, country, **kwargs):
            month_day = (7, 4) if country == "US" else (7, 1)
            mask = (index.month == month_day[0]) & (index.day == month_day[1])
            return pd.DataFrame({"National Day": mask.astype(float)}, index=index)

        params = self._params()
        params.update(holiday_country="US", holiday_countries={"ca": "CA"})
        detector = TimeSeriesFeatureDetector(**params)
        with patch(
            "autots.evaluator.feature_detector.components.holidays.holiday_flag",
            side_effect=calendar_flags,
        ):
            detector._detect_holidays(data)
        named = detector._named_holiday_dates_temp
        self.assertIn("calendar_US__National Day", named["us"])
        self.assertNotIn("calendar_CA__National Day", named["us"])
        self.assertIn("calendar_CA__National Day", named["ca"])

    def test_configuration_round_trip_and_call_scoping(self):
        for value in [None, True, False]:
            detector = TimeSeriesFeatureDetector(
                global_holiday_anomaly_suppression=value
            )
            detector._apply_detector_params(
                {"global_holiday_anomaly_suppression": value}
            )
            detector.df_original = self._data()
            detector.date_index = detector.df_original.index
            detector._reset_results()
            metadata = json.loads(json.dumps(detector.template["meta"]["config"]))
            self.assertIs(metadata["global_holiday_anomaly_suppression"], value)
        before = copy.deepcopy(detector.anomaly_params)
        named = {"a": {"Christmas": [pd.Timestamp("2024-12-25")]}}
        built = detector._build_anomaly_params(named)
        self.assertEqual(built["holiday_dates"], named)
        built["holiday_dates"]["a"].clear()
        self.assertEqual(detector.anomaly_params, before)
        self.assertTrue(named["a"])
        optimizer = FeatureDetectionOptimizer(synthetic_generator=None)
        self.assertIn(
            "global_holiday_anomaly_suppression", optimizer._default_detector_params()
        )


if __name__ == "__main__":
    unittest.main()
