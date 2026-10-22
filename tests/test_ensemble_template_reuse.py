"""Forecasting series subsets must not overwrite reusable ensemble templates."""

import copy
import datetime
import json
import unittest

import numpy as np
import pandas as pd

from autots.evaluator.auto_model import create_model_id, model_forecast
from autots.models.basics import AverageValueNaive, LastValueNaive
from autots.models.ensemble import HorizontalEnsemble, MosaicEnsemble


class TestEnsembleTemplateReuse(unittest.TestCase):
    def setUp(self):
        self.horizon = 3
        steps = np.arange(400)
        self.df = pd.DataFrame(
            {'A': 100 + steps / 10, 'B': 50 + np.sin(steps / 7)},
            index=pd.date_range('2023-01-01', periods=400),
        )
        self.transforms = {
            'fillna': 'ffill',
            'transformations': {},
            'transformation_params': {},
        }
        self.models = {}
        self.model_ids = []
        for model in (LastValueNaive(), AverageValueNaive(method='Mean')):
            parameters = model.get_params()
            model_id = create_model_id(model.name, parameters, self.transforms)
            self.models[model_id] = {
                'Model': model.name,
                'ModelParameters': json.dumps(parameters),
                'TransformationParameters': json.dumps(self.transforms),
            }
            self.model_ids.append(model_id)

    def template(self, kind, profiled=False):
        if profiled:
            profiles = (
                'smooth', 'intermittent', 'binary', 'smooth_drift',
                'erratic', 'flat', 'new_product', 'highly_seasonal', 'overall',
            )
            mapping = {
                profile: self.model_ids[n % 2] for n, profile in enumerate(profiles)
            }
        else:
            mapping = dict(zip(self.df.columns, self.model_ids))
        if kind == 'Mosaic':
            mapping = {
                series: {str(period): model for period in range(self.horizon)}
                for series, model in mapping.items()
            }
        return {
            'model_name': kind,
            'model_count': len(self.models),
            'model_metric': kind.lower() + ('-profile' if profiled else ''),
            'models': copy.deepcopy(self.models),
            'series': mapping,
        }

    def forecast(self, template, df):
        return model_forecast(
            'Ensemble', template, self.transforms, df,
            forecast_length=self.horizon, frequency='D', n_jobs=1, verbose=-1,
        )

    def assert_same_forecasts(self, actual, expected):
        for attribute in ('forecast', 'lower_forecast', 'upper_forecast'):
            pd.testing.assert_frame_equal(
                getattr(actual, attribute), getattr(expected, attribute)
            )
        self.assertEqual(actual.model_parameters, expected.model_parameters)

    def test_reusing_templates_for_disjoint_subsets_matches_fresh_calls(self):
        for kind in ('Horizontal', 'Mosaic'):
            for profiled in (False, True):
                for serialized in (False, True):
                    with self.subTest(
                        kind=kind, profiled=profiled, serialized=serialized
                    ):
                        template = self.template(kind, profiled)
                        original = copy.deepcopy(template)
                        transforms = copy.deepcopy(self.transforms)
                        supplied = json.dumps(template) if serialized else template
                        for subset in (['A'], ['B'], ['B', 'A']):
                            df = self.df[subset]
                            actual = self.forecast(supplied, df)
                            expected = self.forecast(copy.deepcopy(original), df)
                            self.assert_same_forecasts(actual, expected)
                            self.assertEqual(template, original)
                            self.assertEqual(self.transforms, transforms)
                            self.assertEqual(actual.forecast.columns.tolist(), subset)
                            self.assertTrue(
                                np.isfinite(actual.forecast.to_numpy()).all()
                            )
                            if kind == 'Mosaic' and profiled:
                                # Profile mosaics intentionally keep profile keys
                                # in result parameters, as in the existing API.
                                self.assertEqual(
                                    actual.model_parameters['series'], original['series']
                                )
                            else:
                                self.assertEqual(
                                    set(actual.model_parameters['series']), set(subset)
                                )

    def test_direct_handlers_preserve_template_and_return_resolved_mapping(self):
        forecast_index = pd.date_range(
            self.df.index[-1] + pd.Timedelta(days=1), periods=self.horizon
        )
        forecasts = {
            model_id: pd.DataFrame({'A': float(n + 1)}, index=forecast_index)
            for n, model_id in enumerate(self.model_ids)
        }
        for kind, handler in (
            ('Horizontal', HorizontalEnsemble), ('Mosaic', MosaicEnsemble)
        ):
            template = self.template(kind)
            original = copy.deepcopy(template)
            result = handler(
                ensemble_params=template,
                forecasts_list=self.model_ids,
                forecasts=forecasts,
                lower_forecasts={key: df - 1 for key, df in forecasts.items()},
                upper_forecasts={key: df + 1 for key, df in forecasts.items()},
                forecasts_runtime={
                    key: datetime.timedelta(0) for key in self.model_ids
                },
                prediction_interval=0.9,
                df_train=self.df[['A']],
            )
            with self.subTest(kind=kind):
                self.assertEqual(template, original)
                self.assertIsNot(result.model_parameters, template)
                self.assertEqual(set(result.model_parameters['series']), {'A'})
                np.testing.assert_array_equal(result.forecast['A'], 1)
