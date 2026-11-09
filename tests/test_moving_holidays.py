"""Moving calendar boundaries and the public holiday output contracts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from autots.evaluator.anomaly_detector import HolidayDetector
from autots.tools.anomaly_utils import anomaly_df_to_holidays, dates_to_holidays
from autots.tools.calendar import (
    gregorian_to_chinese,
    gregorian_to_christian_lunar,
    gregorian_to_hindu,
)
from autots.tools.easter import easter_offsets, extract_easter_holidays


class TestMovingCalendarBoundaries(unittest.TestCase):
    def test_calendar_horizon_and_scalar_invariance(self):
        dates = pd.date_range("2023-12-01", "2024-05-01")
        for convert in [
            gregorian_to_chinese,
            gregorian_to_christian_lunar,
            gregorian_to_hindu,
        ]:
            full = convert(dates)
            for end in ["2024-01-01", "2024-02-09", "2024-02-10", "2024-03-25"]:
                with self.subTest(calendar=convert.__name__, end=end):
                    prefix = dates[dates <= end]
                    pd.testing.assert_frame_equal(
                        convert(prefix), full.loc[prefix], check_names=False
                    )
                    pd.testing.assert_series_equal(
                        convert([end]).iloc[0], full.loc[end]
                    )
                    forecast_dates = pd.date_range(end, periods=21)
                    pd.testing.assert_frame_equal(
                        convert(forecast_dates),
                        full.loc[forecast_dates],
                        check_names=False,
                    )

    def test_known_lunar_new_year_dates(self):
        dates = pd.to_datetime(
            [
                "2017-01-28",
                "2018-02-16",
                "2019-02-05",
                "2020-01-25",
                "2021-02-12",
                "2022-02-01",
                "2023-01-22",
                "2024-02-10",
                "2025-01-29",
            ]
        )
        result = gregorian_to_chinese(dates)
        np.testing.assert_array_equal(result["lunar_year"], dates.year)
        np.testing.assert_array_equal(result[["lunar_month", "lunar_day"]], 1)
        self.assertNotEqual(
            gregorian_to_chinese(["2024-02-09"]).iloc[0]["lunar_day"], 1
        )


class TestEasterHolidays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range("2017-01-01", "2024-12-31")
        cls.offsets = easter_offsets(cls.dates)
        cls.flags = pd.DataFrame(1.0, index=cls.dates, columns=["easter", "ordinary"])
        cls.flags.loc[cls.offsets.between(-2, 1), "easter"] = -1
        cls.actuals = pd.DataFrame(0.0, index=cls.dates, columns=cls.flags.columns)
        cls.actuals.loc[cls.offsets.between(-2, 1), "easter"] = 10.0

    def test_rules_require_recurrence_and_remain_per_series(self):
        rules = extract_easter_holidays(self.flags, 0.8, 2, self.actuals)
        self.assertEqual(set(rules["series"]), {"easter"})
        self.assertEqual(
            set(rules["holiday_name"]),
            {"easter_-2", "easter_-1", "easter_0", "easter_+1"},
        )
        self.assertTrue(rules["count"].eq(8).all())
        self.assertTrue(rules["avg_value"].eq(10).all())
        single = self.flags.copy()
        single.loc[single.index.year != 2024] = 1
        self.assertTrue(extract_easter_holidays(single, 0.8, 2).empty)
        self.assertEqual(len(anomaly_df_to_holidays(self.flags.copy())), 8)

    def test_future_dates_and_all_output_styles(self):
        rules = extract_easter_holidays(self.flags, 0.8, 2, self.actuals, self.actuals)
        future = pd.date_range("2025-04-17", "2025-04-22")
        self.assertEqual(easter_offsets(pd.DatetimeIndex(["2025-04-20"])).iloc[0], 0)
        for style in ["long", "flag", "prophet", "series_flag", "impact"]:
            with self.subTest(style=style):
                result = dates_to_holidays(
                    future, self.flags.columns, style=style, easter_holidays=rules
                )
                self.assertFalse(result.empty)
                if style == "series_flag":
                    self.assertEqual(result["easter"].sum(), 4)
                    self.assertEqual(result["ordinary"].sum(), 0)
                elif style == "impact":
                    self.assertEqual(result["easter"].sum(), 40)
        limited = dates_to_holidays(
            future,
            self.flags.columns,
            style="flag",
            easter_holidays=rules,
            max_features=2,
        )
        self.assertLessEqual(limited.shape[1], 3)  # two retained features plus Other

    def _detector(self, **kwargs):
        return HolidayDetector(
            use_easter_holidays=True,
            use_dayofmonth_holidays=False,
            use_wkdom_holidays=False,
            use_wkdeom_holidays=False,
            use_lunar_holidays=False,
            **kwargs,
        )

    def test_caps_relaxation_and_older_fitted_detector(self):
        for strategy in ["score", "coverage"]:
            detector = self._detector(
                max_holidays_per_series=1, holiday_selection_strategy=strategy
            )
            with patch.object(detector.anomaly_model, "detect"):
                detector.anomaly_model.anomalies = self.flags
                detector.anomaly_model.scores = self.actuals
                detector.detect(self.actuals)
            self.assertEqual(len(detector.easter_holidays), 1)
            self.assertEqual(detector.get_detection_stats()["coverage_series"], 1)
            self.assertFalse(detector.dates_to_holidays(self.dates).empty)
        sparse = self.flags.copy()
        sparse.loc[sparse.index.year.isin([2017, 2018]), "easter"] = 1
        detector = self._detector(
            threshold=0.9, auto_relax=True, relax_threshold_floor=0.7
        )
        with patch.object(detector.anomaly_model, "detect"):
            detector.anomaly_model.anomalies = sparse
            detector.anomaly_model.scores = self.actuals
            detector.detect(self.actuals)
        self.assertEqual(len(detector.easter_holidays), 4)
        del detector.easter_holidays
        self.assertTrue(detector.dates_to_holidays(self.dates).empty)


if __name__ == "__main__":
    unittest.main()
