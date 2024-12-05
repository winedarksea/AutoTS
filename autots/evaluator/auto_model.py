"""Mid-level helper functions for AutoTS."""

import sys
import gc
import traceback as tb
import random
from math import ceil
import copy
import numpy as np
import pandas as pd
import datetime
import json
from hashlib import md5
from autots.tools.cpu_count import set_n_jobs
from autots.tools.transform import RandomTransform, GeneralTransformer, shared_trans
from autots.models.base import PredictionObject, ModelObject
from autots.evaluator.metrics import default_scaler, array_last_val
from autots.models.ensemble import (
    EnsembleForecast,
    generalize_horizontal,
    horizontal_aliases,
    parse_horizontal,
    is_mosaic,
)
from autots.tools.shaping import infer_frequency
from autots.models.model_list import (
    no_params,
    recombination_approved,
    no_shared,
    superfast,
    # model_lists,
    model_list_to_dict,
)
from itertools import zip_longest
from autots.models.basics import (
    MotifSimulation,
    LastValueNaive,
    AverageValueNaive,
    SeasonalNaive,
    ConstantNaive,
    Motif,
    SectionalMotif,
    NVAR,
    KalmanStateSpace,
    MetricMotif,
    SeasonalityMotif,
    FFT,
    BallTreeMultivariateMotif,
    BasicLinearModel,
    TVVAR,
    BallTreeRegressionMotif,
)
from autots.models.statsmodels import (
    GLS,
    GLM,
    ETS,
    ARIMA,
    UnobservedComponents,
    DynamicFactor,
    VAR,
    VECM,
    VARMAX,
    Theta,
    ARDL,
    DynamicFactorMQ,
)
from autots.models.arch import ARCH
from autots.models.matrix_var import RRVAR, MAR, TMF, LATC, DMD
from autots.models.sklearn import (
    RollingRegression,
    WindowRegression,
    MultivariateRegression,
    DatepartRegression,
    UnivariateRegression,
    ComponentAnalysis,
    PreprocessingRegression,
)


def create_model_id(
    model_str: str, parameter_dict: dict = {}, transformation_dict: dict = {}
):
    """Create a hash ID which should be unique to the model parameters."""
    if isinstance(parameter_dict, dict):
        str_params = json.dumps(parameter_dict)
    else:
        str_params = str(parameter_dict)
    if isinstance(transformation_dict, dict):
        str_trans = json.dumps(transformation_dict)
    else:
        str_trans = str(transformation_dict)
    str_repr = str(model_str) + str_params + str_trans
    str_repr = ''.join(str_repr.split())
    hashed = md5(str_repr.encode('utf-8')).hexdigest()
    return hashed


def horizontal_template_to_model_list(template):
    """helper function to take template dataframe of ensembles to a single list of models."""
    if "ModelParameters" not in template.columns:
        raise ValueError("Template input does not match expected for models")
    model_list = []
    for idx, row in template.iterrows():
        model_list.extend(list(json.loads(row["ModelParameters"])['models'].keys()))
    return model_list


def ModelMonster(
    model: str,
    parameters: dict = {},
    frequency: str = 'infer',
    prediction_interval: float = 0.9,
    holiday_country: str = 'US',
    startTimeStamps=None,
    forecast_length: int = 14,
    random_seed: int = 2020,
    verbose: int = 0,
    n_jobs: int = None,
    **kwargs,
):
    """Directs strings and parameters to appropriate model objects.

    Args:
        model (str): Name of Model Function
        parameters (dict): Dictionary of parameters to pass through to model
    """
    model = str(model)
    model_lower = model.lower()

    if model in ['ZeroesNaive', 'ConstantNaive']:
        return ConstantNaive(
            frequency=frequency, prediction_interval=prediction_interval, **parameters
        )

    elif model == 'LastValueNaive':
        return LastValueNaive(
            frequency=frequency, prediction_interval=prediction_interval
        )

    elif model == 'AverageValueNaive':
        return AverageValueNaive(
            frequency=frequency, prediction_interval=prediction_interval, **parameters
        )

    elif model == 'SeasonalNaive':
        return SeasonalNaive(
            frequency=frequency, prediction_interval=prediction_interval, **parameters
        )

    elif model == 'GLS':
        return GLS(
            frequency=frequency, prediction_interval=prediction_interval, **parameters
        )

    elif model == 'GLM':
        model = GLM(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model

    elif model == 'ETS':
        model = ETS(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model

    elif model == 'ARIMA':
        model = ARIMA(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model

    elif model in ['FBProphet', "Prophet"]:
        from autots.models.prophet import FBProphet

        model = FBProphet(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model

    elif model == 'RollingRegression':
        model = RollingRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model
    elif model == 'UnivariateRegression':
        model = UnivariateRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            forecast_length=forecast_length,
            **parameters,
        )
        return model

    elif model == 'MultivariateRegression':
        model = MultivariateRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            forecast_length=forecast_length,
            **parameters,
        )
        return model

    elif model == 'UnobservedComponents':
        model = UnobservedComponents(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model

    elif model == 'DynamicFactor':
        model = DynamicFactor(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
        return model

    elif model == 'VAR':
        model = VAR(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
        return model

    elif model == 'VECM':
        model = VECM(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
        return model

    elif model == 'VARMAX':
        model = VARMAX(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
        return model

    elif model == 'GluonTS':
        from autots.models.gluonts import GluonTS

        model = GluonTS(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            **parameters,
        )

        return model

    elif model == 'MotifSimulation':
        model = MotifSimulation(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
        return model
    elif model == 'WindowRegression':
        model = WindowRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
        return model
    elif model == 'TensorflowSTS':
        from autots.models.tfp import TensorflowSTS

        if parameters == {}:
            model = TensorflowSTS(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
            )
        else:
            model = TensorflowSTS(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                seasonal_periods=parameters['seasonal_periods'],
                ar_order=parameters['ar_order'],
                trend=parameters['trend'],
                fit_method=parameters['fit_method'],
                num_steps=parameters['num_steps'],
            )
        return model
    elif model == 'TFPRegression':
        from autots.models.tfp import TFPRegression

        if parameters == {}:
            model = TFPRegression(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
            )
        else:
            model = TFPRegression(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                kernel_initializer=parameters['kernel_initializer'],
                epochs=parameters['epochs'],
                batch_size=parameters['batch_size'],
                optimizer=parameters['optimizer'],
                loss=parameters['loss'],
                dist=parameters['dist'],
                regression_type=parameters['regression_type'],
            )
        return model
    elif model == 'ComponentAnalysis':
        if parameters == {}:
            model = ComponentAnalysis(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                forecast_length=forecast_length,
            )
        else:
            model = ComponentAnalysis(
                frequency=frequency,
                prediction_interval=prediction_interval,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                model=parameters['model'],
                model_parameters=parameters['model_parameters'],
                decomposition=parameters['decomposition'],
                n_components=parameters['n_components'],
                forecast_length=forecast_length,
            )
        return model
    elif model == 'DatepartRegression':
        model = DatepartRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )

        return model
    elif model == 'Greykite':
        from autots.models.greykite import Greykite

        model = Greykite(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )

        return model
    elif model == 'MultivariateMotif':
        return Motif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            multivariate=True,
            **parameters,
        )
    elif model == 'UnivariateMotif':
        return Motif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            multivariate=False,
            **parameters,
        )
    elif model == 'SectionalMotif':
        return SectionalMotif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
    elif model == 'NVAR':
        return NVAR(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
    elif model == 'Theta':
        return Theta(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'ARDL':
        return ARDL(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'NeuralProphet':
        from autots.models.prophet import NeuralProphet

        return NeuralProphet(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=1,
            **parameters,
        )
    elif model == 'DynamicFactorMQ':
        return DynamicFactorMQ(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            **parameters,
        )
    elif model == 'PytorchForecasting':
        from autots.models.pytorch import PytorchForecasting

        return PytorchForecasting(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            **parameters,
        )
    elif model_lower == 'arch':
        return ARCH(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'RRVAR':
        return RRVAR(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model_lower == 'mar':
        return MAR(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'TMF':
        return TMF(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'LATC':
        return LATC(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "KalmanStateSpace":
        return KalmanStateSpace(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            forecast_length=forecast_length,
            **parameters,
        )
    elif model == "MetricMotif":
        return MetricMotif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "SeasonalityMotif":
        return SeasonalityMotif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "Cassandra":
        from autots.models.cassandra import Cassandra  # circular import

        return Cassandra(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            forecast_length=forecast_length,
            **parameters,
        )
    elif model == "PreprocessingRegression":
        return PreprocessingRegression(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            forecast_length=forecast_length,
            **parameters,
        )
    elif model == "MLEnsemble":
        from autots.models.mlensemble import MLEnsemble  # circular import

        return MLEnsemble(
            frequency=frequency,
            forecast_length=forecast_length,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "Motif":
        return Motif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            multivariate=parameters.get("multivariate", False),
            **parameters,
        )
    elif model == "FFT":
        return FFT(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "BallTreeMultivariateMotif":
        return BallTreeMultivariateMotif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model in ["TiDE", "TIDE"]:
        from autots.models.tide import TiDE

        return TiDE(
            frequency=frequency,
            forecast_length=forecast_length,
            prediction_interval=prediction_interval,
            # holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model in ["NeuralForecast", "neuralforecast"]:
        from autots.models.neural_forecast import NeuralForecast

        return NeuralForecast(
            frequency=frequency,
            forecast_length=forecast_length,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'DMD':
        return DMD(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'BasicLinearModel':
        return BasicLinearModel(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'TVVAR':
        return TVVAR(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == 'BallTreeRegressionMotif':
        return BallTreeRegressionMotif(
            frequency=frequency,
            prediction_interval=prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            forecast_length=forecast_length,
            n_jobs=n_jobs,
            **parameters,
        )
    elif model == "":
        raise AttributeError(
            ("Model name is empty. Likely this means AutoTS has not been fit.")
        )
    else:
        raise AttributeError((f"Model String '{model}' not a recognized model type"))


class ModelPrediction(ModelObject):
    """Feed parameters into modeling pipeline. A class object, does NOT work with ensembles.

    Args:
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        fail_on_forecast_nan (bool): if False, return forecasts even if NaN present, if True, raises error if any nan in forecast
        return_model (bool): if True, forecast will have .model and .tranformer attributes set to model object.
        n_jobs (int): number of processes
        current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended

    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """

    def __init__(
        self,
        forecast_length: int,
        transformation_dict: dict,
        model_str: str,
        parameter_dict: dict,
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        no_negatives: bool = False,
        constraint: float = None,
        holiday_country: str = 'US',
        startTimeStamps=None,
        grouping_ids=None,
        fail_on_forecast_nan: bool = True,
        return_model: bool = False,
        random_seed: int = 2020,
        verbose: int = 0,
        n_jobs: int = None,
        current_model_file: str = None,
        model_count: int = 0,
        force_gc: bool = False,
    ):
        self.forecast_length = forecast_length
        self.model_str = model_str
        self.parameter_dict = parameter_dict
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.no_negatives = no_negatives
        self.constraint = constraint
        self.holiday_country = holiday_country
        self.fail_on_forecast_nan = fail_on_forecast_nan
        self.return_model = return_model
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.current_model_file = current_model_file
        self.model_count = model_count
        self.force_gc = force_gc
        # handle still in JSON form
        if isinstance(transformation_dict, str):
            if transformation_dict == "":
                self.transformation_dict = {}
            else:
                self.transformation_dict = json.loads(transformation_dict)
        else:
            self.transformation_dict = transformation_dict
        if isinstance(parameter_dict, str):
            if parameter_dict == "":
                self.parameter_dict = {}
            else:
                self.parameter_dict = json.loads(parameter_dict)
        else:
            self.parameter_dict = parameter_dict
        if model_str == "PreprocessingRegression":
            self.parameter_dict['transformation_dict'] = self.transformation_dict
            self.transformation_dict = {
                'fillna': None,
                'transformations': {},
                'transformation_params': {},
            }
        if self.transformation_dict is None:
            self.transformation_dict = {}
        self.transformer_object = GeneralTransformer(
            **self.transformation_dict,
            n_jobs=self.n_jobs,
            holiday_country=self.holiday_country,
            verbose=self.verbose,
            random_seed=self.random_seed,
            forecast_length=self.forecast_length,
        )
        self.name = "ModelPrediction"
        self._fit_complete = False

    def fit(self, df, future_regressor=None):
        self.df = df
        if self.frequency == "infer":
            self.inferred_frequency = infer_frequency(df)
        else:
            self.inferred_frequency = self.frequency
        self.model = ModelMonster(
            self.model_str,
            parameters=self.parameter_dict,
            frequency=self.inferred_frequency,
            prediction_interval=self.prediction_interval,
            holiday_country=self.holiday_country,
            random_seed=self.random_seed,
            verbose=self.verbose,
            forecast_length=self.forecast_length,
            n_jobs=self.n_jobs,
        )
        transformationStartTime = datetime.datetime.now()
        if self.current_model_file is not None:
            try:
                with open(f'{self.current_model_file}.json', 'w') as f:
                    json.dump(
                        {
                            "model_number": self.model_count,
                            "model_name": self.model_str,
                            "model_param_dict": self.parameter_dict,
                            "model_transform_dict": self.transformation_dict,
                        },
                        f,
                    )
            except Exception as e:
                error_msg = (
                    f"failed to write {self.current_model_file} with error {repr(e)}"
                )
                try:
                    with open(f'{self.current_model_file}_failure.json', 'w') as f:
                        f.write(error_msg)
                except Exception:
                    pass
                print(error_msg)

        df_train_transformed = self.transformer_object._fit(df)

        # make sure regressor has same length. This could be a problem if wrong size regressor is passed.
        if future_regressor is not None:
            future_regressor = future_regressor.reindex(df.index)

        self.transformation_runtime = datetime.datetime.now() - transformationStartTime
        # from autots.evaluator.auto_model import ModelMonster
        self.model = self.model.fit(
            df_train_transformed, future_regressor=future_regressor
        )
        self._fit_complete = True
        return self

    def predict(self, forecast_length=None, future_regressor=None):
        if forecast_length is None:
            forecast_length = self.forecast_length
        if not self._fit_complete:
            raise ValueError("Model not yet fit.")
        df_forecast = self.model.predict(
            forecast_length=forecast_length, future_regressor=future_regressor
        )

        # THIS CHECKS POINT FORECAST FOR NULLS BUT NOT UPPER/LOWER FORECASTS
        # can maybe remove this eventually and just keep the later one
        if self.fail_on_forecast_nan:
            if not np.isfinite(np.max(df_forecast.forecast.to_numpy())):
                raise ValueError(
                    "Model {} returned NaN for one or more series. fail_on_forecast_nan=True".format(
                        self.model_str
                    )
                )

        transformationStartTime = datetime.datetime.now()
        # Inverse the transformations, NULL FILLED IN UPPER/LOWER ONLY
        # forecast inverse MUST come before upper and lower bounds inverse
        df_forecast.forecast = self.transformer_object.inverse_transform(
            df_forecast.forecast
        )
        df_forecast.lower_forecast = self.transformer_object.inverse_transform(
            df_forecast.lower_forecast, fillzero=True, bounds=True
        )
        df_forecast.upper_forecast = self.transformer_object.inverse_transform(
            df_forecast.upper_forecast, fillzero=True, bounds=True
        )

        # CHECK Forecasts are proper length!
        if df_forecast.forecast.shape[0] != self.forecast_length:
            raise ValueError(
                f"Model {self.model_str} returned improper forecast_length. Returned: {df_forecast.forecast.shape[0]} and requested: {self.forecast_length}"
            )

        if df_forecast.forecast.shape[1] != self.df.shape[1]:
            raise ValueError(
                f"{self.model.name} with {self.transformer_object.transformations} failed to return correct number of series. Returned {df_forecast.forecast.shape[1]} and requested: {self.df.shape[1]}"
            )

        df_forecast.transformation_parameters = self.transformation_dict
        # Remove negatives if desired
        # There's df.where(df_forecast.forecast > 0, 0) or  df.clip(lower = 0), not sure which faster
        if self.no_negatives:
            df_forecast.lower_forecast = df_forecast.lower_forecast.clip(lower=0)
            df_forecast.forecast = df_forecast.forecast.clip(lower=0)
            df_forecast.upper_forecast = df_forecast.upper_forecast.clip(lower=0)

        if self.constraint is not None:
            if isinstance(self.constraint, list):
                constraints = self.constraint
                df_forecast = df_forecast.apply_constraints(
                    constraints=constraints,
                    df_train=self.df,
                )
            else:
                constraints = None
                if isinstance(self.constraint, dict):
                    if "constraints" in self.constraint.keys():
                        constraints = self.constraint.get("constraints")
                        constraint_method = None
                        constraint_regularization = None
                        lower_constraint = None
                        upper_constraint = None
                        bounds = True
                    else:
                        constraint_method = self.constraint.get(
                            "constraint_method", "quantile"
                        )
                        constraint_regularization = self.constraint.get(
                            "constraint_regularization", 1
                        )
                        lower_constraint = self.constraint.get("lower_constraint", 0)
                        upper_constraint = self.constraint.get("upper_constraint", 1)
                        bounds = self.constraint.get("bounds", False)
                else:
                    constraint_method = "stdev_min"
                    lower_constraint = float(self.constraint)
                    upper_constraint = float(self.constraint)
                    constraint_regularization = 1
                    bounds = False
                if self.verbose > 3:
                    print(
                        f"Using constraint with method: {constraint_method}, {constraint_regularization}, {lower_constraint}, {upper_constraint}, {bounds}"
                    )

                print(constraints)
                df_forecast = df_forecast.apply_constraints(
                    constraints,
                    self.df,
                    constraint_method,
                    constraint_regularization,
                    upper_constraint,
                    lower_constraint,
                    bounds,
                )

        self.transformation_runtime = self.transformation_runtime + (
            datetime.datetime.now() - transformationStartTime
        )
        df_forecast.transformation_runtime = self.transformation_runtime

        if self.return_model:
            df_forecast.model = self.model
            df_forecast.transformer = self.transformer_object

        # THIS CHECKS POINT FORECAST FOR NULLS BUT NOT UPPER/LOWER FORECASTS
        if self.fail_on_forecast_nan:
            if not np.isfinite(np.max(df_forecast.forecast.to_numpy())):
                raise ValueError(
                    "Model returned NaN due to a preprocessing transformer {}. fail_on_forecast_nan=True".format(
                        str(self.transformation_dict)
                    )
                )
        sys.stdout.flush()
        if self.force_gc:
            gc.collect()

        return df_forecast

    def fit_data(self, df, future_regressor=None):
        self.df = df
        self.model.fit_data(df, future_regressor)

    def fit_predict(
        self,
        df,
        forecast_length,
        future_regressor_train=None,
        future_regressor_forecast=None,
    ):
        self.fit(df, future_regressor=future_regressor_train)
        return self.predict(
            forecast_length=forecast_length, future_regressor=future_regressor_forecast
        )


class TemplateEvalObject(object):
    """Object to contain all the failures!.

    Attributes:
        full_mae_ids (list): list of model_ids corresponding to full_mae_errors
        full_mae_errors (list): list of numpy arrays of shape (rows, columns) appended in order of validation
            only provided for 'mosaic' ensembling
    """

    def __init__(
        self,
        model_results=pd.DataFrame(),
        per_timestamp_smape=pd.DataFrame(),
        per_series_metrics=pd.DataFrame(),
        per_series_mae=None,
        per_series_rmse=None,
        per_series_made=None,
        per_series_contour=None,
        per_series_spl=None,
        per_series_mle=None,
        per_series_imle=None,
        per_series_maxe=None,
        per_series_oda=None,
        per_series_mqae=None,
        per_series_dwae=None,
        per_series_ewmae=None,
        per_series_uwmse=None,
        per_series_smoothness=None,
        per_series_mate=None,
        per_series_matse=None,
        per_series_wasserstein=None,
        per_series_dwd=None,
        model_count: int = 0,
    ):
        self.model_results = model_results
        self.model_count = model_count
        self.per_series_metrics = per_series_metrics
        self.per_series_mae = per_series_mae
        self.per_series_contour = per_series_contour
        self.per_series_rmse = per_series_rmse
        self.per_series_made = per_series_made
        self.per_series_spl = per_series_spl
        self.per_timestamp_smape = per_timestamp_smape
        self.per_series_mle = per_series_mle
        self.per_series_imle = per_series_imle
        self.per_series_maxe = per_series_maxe
        self.per_series_oda = per_series_oda
        self.per_series_mqae = per_series_mqae
        self.per_series_dwae = per_series_dwae
        self.per_series_ewmae = per_series_ewmae
        self.per_series_uwmse = per_series_uwmse
        self.per_series_smoothness = per_series_smoothness
        self.per_series_mate = per_series_mate
        self.per_series_matse = per_series_matse
        self.per_series_wasserstein = per_series_wasserstein
        self.per_series_dwd = per_series_dwd
        self.full_mae_ids = []
        self.full_mae_vals = []
        self.full_mae_errors = []
        self.full_pl_errors = []
        self.squared_errors = []

    def __repr__(self):
        """Print."""
        return 'Results objects, result table at self.model_results (pd.df)'

    def concat(self, another_eval):
        """Merge another TemplateEvalObject onto this one."""
        self.model_results = pd.concat(
            [self.model_results, another_eval.model_results],
            axis=0,
            ignore_index=True,
            sort=False,
        ).reset_index(drop=True)
        self.per_series_metrics = pd.concat(
            [self.per_series_metrics, another_eval.per_series_metrics],
            axis=0,
            sort=False,
        )
        self.per_series_mae = pd.concat(
            [self.per_series_mae, another_eval.per_series_mae], axis=0, sort=False
        )
        self.per_series_made = pd.concat(
            [self.per_series_made, another_eval.per_series_made], axis=0, sort=False
        )
        self.per_series_contour = pd.concat(
            [self.per_series_contour, another_eval.per_series_contour],
            axis=0,
            sort=False,
        )
        self.per_series_rmse = pd.concat(
            [self.per_series_rmse, another_eval.per_series_rmse], axis=0, sort=False
        )
        self.per_series_spl = pd.concat(
            [self.per_series_spl, another_eval.per_series_spl], axis=0, sort=False
        )
        self.per_timestamp_smape = pd.concat(
            [self.per_timestamp_smape, another_eval.per_timestamp_smape],
            axis=0,
            sort=False,
        )
        self.per_series_mle = pd.concat(
            [self.per_series_mle, another_eval.per_series_mle], axis=0, sort=False
        )
        self.per_series_imle = pd.concat(
            [self.per_series_imle, another_eval.per_series_imle], axis=0, sort=False
        )
        self.per_series_maxe = pd.concat(
            [self.per_series_maxe, another_eval.per_series_maxe], axis=0, sort=False
        )
        self.per_series_oda = pd.concat(
            [self.per_series_oda, another_eval.per_series_oda], axis=0, sort=False
        )
        self.per_series_mqae = pd.concat(
            [self.per_series_mqae, another_eval.per_series_mqae], axis=0, sort=False
        )
        self.per_series_dwae = pd.concat(
            [self.per_series_dwae, another_eval.per_series_dwae], axis=0, sort=False
        )
        self.per_series_ewmae = pd.concat(
            [self.per_series_ewmae, another_eval.per_series_ewmae], axis=0, sort=False
        )
        self.per_series_uwmse = pd.concat(
            [self.per_series_uwmse, another_eval.per_series_uwmse], axis=0, sort=False
        )
        self.per_series_smoothness = pd.concat(
            [self.per_series_smoothness, another_eval.per_series_smoothness],
            axis=0,
            sort=False,
        )
        self.per_series_mate = pd.concat(
            [self.per_series_mate, another_eval.per_series_mate], axis=0, sort=False
        )
        self.per_series_matse = pd.concat(
            [self.per_series_matse, another_eval.per_series_matse], axis=0, sort=False
        )
        self.per_series_wasserstein = pd.concat(
            [self.per_series_wasserstein, another_eval.per_series_wasserstein],
            axis=0,
            sort=False,
        )
        self.per_series_dwd = pd.concat(
            [self.per_series_dwd, another_eval.per_series_dwd],
            axis=0,
            sort=False,
        )
        self.full_mae_errors.extend(another_eval.full_mae_errors)
        self.full_pl_errors.extend(another_eval.full_pl_errors)
        self.squared_errors.extend(another_eval.squared_errors)
        self.full_mae_ids.extend(another_eval.full_mae_ids)
        self.full_mae_vals.extend(another_eval.full_mae_vals)
        self.model_count = self.model_count + another_eval.model_count
        return self

    def save(self, filename='initial_results.pickle'):
        """Save results to a file.

        Args:
            filename (str): *.pickle or *.csv. .pickle saves full results
        """
        if filename.endswith('.csv'):
            self.model_results.to_csv(filename, index=False)
        elif filename.endswith('.pickle'):
            import pickle

            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"filename `{filename}` not .csv or .pickle")

    def load(self, filename):
        # might want to add csv handling from auto_ts
        if isinstance(filename, TemplateEvalObject):
            new_obj = filename
        elif filename.endswith('.pickle'):
            import pickle

            new_obj = pickle.load(open(filename, "rb"))
        else:
            raise ValueError("import type not recognized.")
        self = self.concat(new_obj)
        return self


def unpack_ensemble_models(
    template,
    template_cols: list = [
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
    keep_ensemble: bool = True,
    recursive: bool = False,
):
    """Take ensemble models from template and add as new rows.
    Some confusion may exist as Ensembles require both 'Ensemble' column > 0 and model name 'Ensemble'

    Args:
        template (pd.DataFrame): AutoTS template containing template_cols
        keep_ensemble (bool): if False, drop row containing original ensemble
        recursive (bool): if True, unnest ensembles of ensembles...
    """
    if 'Ensemble' not in template.columns:
        template['Ensemble'] = 0
    # handle the fact that recursively, nested Ensembles ensemble flag is set to 0 below
    template['Ensemble'] = np.where(
        ((template['Model'] == 'Ensemble') & (template['Ensemble'] < 1)),
        1,
        template['Ensemble'],
    )
    # alternatively the below could read from 'Model' == 'Ensemble'
    models_to_iterate = template[template['Ensemble'] != 0]['ModelParameters'].copy()
    if not models_to_iterate.empty:
        for index, value in models_to_iterate.items():
            try:
                model_dict = json.loads(value)['models']
            except Exception as e:
                raise ValueError(f"`{value}` is bad model template") from e
            # empty model parameters can exist, but shouldn't...
            if model_dict:
                model_df = pd.DataFrame.from_dict(model_dict, orient='index')
                # it might be wise to just drop the ID column, but keeping for now
                model_df = model_df.rename_axis('ID').reset_index(drop=False)
                # this next line is necessary, albeit confusing
                if 'Ensemble' not in model_df.columns:
                    model_df['Ensemble'] = 0
                # unpack nested ensembles, if recursive specified
                if recursive and 'Ensemble' in model_df['Model'].tolist():
                    model_df = pd.concat(
                        [
                            unpack_ensemble_models(
                                model_df,
                                recursive=True,
                                keep_ensemble=keep_ensemble,
                                template_cols=template_cols,
                            ),
                            model_df,
                        ],
                        axis=0,
                        ignore_index=True,
                        sort=False,
                    ).reset_index(drop=True)
                template = pd.concat(
                    [template, model_df], axis=0, ignore_index=True, sort=False
                ).reset_index(drop=True)
        if not keep_ensemble:
            template = template[template['Ensemble'] == 0]
    template = template.drop_duplicates(subset=template_cols)
    return template


def model_forecast(
    model_name,
    model_param_dict,
    model_transform_dict,
    df_train,
    forecast_length: int,
    frequency: str = 'infer',
    prediction_interval: float = 0.9,
    no_negatives: bool = False,
    constraint: float = None,
    future_regressor_train=None,
    future_regressor_forecast=None,
    holiday_country: str = 'US',
    startTimeStamps=None,
    grouping_ids=None,
    fail_on_forecast_nan: bool = True,
    random_seed: int = 2020,
    verbose: int = 0,
    n_jobs: int = "auto",
    template_cols: list = [
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
    horizontal_subset: list = None,
    return_model: bool = False,
    current_model_file: str = None,
    model_count: int = 0,
    force_gc: bool = False,
    internal_validation: bool = False,
    **kwargs,
):
    """Takes numeric data, returns numeric forecasts.

    Only one model (albeit potentially an ensemble)!
    Horizontal ensembles can not be nested, other ensemble types can be.

    Well, she turned me into a newt.
    A newt?
    I got better. -Python

    Args:
        model_name (str): a string to be direct to the appropriate model, used in ModelMonster
        model_param_dict (dict): dictionary of parameters to be passed into the model.
        model_transform_dict (dict): a dictionary of fillNA and transformation methods to be used
            pass an empty dictionary if no transformations are desired.
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        forecast_length (int): number of periods to forecast
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.
        n_jobs (int): number of CPUs to use when available.
        template_cols (list): column names of columns used as model template
        horizontal_subset (list): columns of df_train to use for forecast, meant for internal use for horizontal ensembling
        fail_on_forecast_nan (bool): if False, return forecasts even if NaN present, if True, raises error if any nan in forecast. True is recommended.
        return_model (bool): if True, forecast will have .model and .tranformer attributes set to model object. Only works for non-ensembles.
        current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended
        force_gc (bool): if True, run gc.collect() after each model
        internal_validation: niche flag to tell that it is running inside a template model search
    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """
    full_model_created = False  # make at least one full model, horziontal only
    # handle JSON inputs of the dicts
    if isinstance(model_param_dict, str):
        if model_param_dict == "":
            model_param_dict = {}
        else:
            model_param_dict = json.loads(model_param_dict)
    if isinstance(model_transform_dict, str):
        if model_transform_dict == "":
            model_transform_dict = {}
        else:
            model_transform_dict = json.loads(model_transform_dict)
    if frequency == "infer":
        frequency = infer_frequency(df_train)
    # handle "auto" n_jobs to an integer of local count
    if n_jobs == 'auto' or not isinstance(n_jobs, int):
        n_jobs = set_n_jobs(n_jobs=n_jobs, verbose=verbose)

    # if an ensemble
    if model_name == 'Ensemble':
        forecasts_runtime = {}
        forecasts = {}
        upper_forecasts = {}
        lower_forecasts = {}
        horizontal_flag = (
            2 if model_param_dict['model_name'].lower() in horizontal_aliases else 1
        )
        template = pd.DataFrame(
            {
                'Model': model_name,
                'ModelParameters': json.dumps(model_param_dict),
                'TransformationParameters': json.dumps(model_transform_dict),
                'Ensemble': horizontal_flag,
            },
            index=[0],
        )
        ens_template = unpack_ensemble_models(
            template, template_cols, keep_ensemble=False, recursive=False
        )
        # horizontal generalization
        if horizontal_flag == 2:
            profiled = "profile" in model_param_dict.get("model_metric")
            if profiled:
                all_series = None
            else:
                available_models = list(model_param_dict['models'].keys())
                known_matches = model_param_dict['series']
                all_series = generalize_horizontal(
                    df_train, known_matches, available_models
                )
        else:
            all_series = None
        total_ens = ens_template.shape[0]
        for index, row in ens_template.iterrows():
            # recursive recursion!
            try:
                if all_series is not None:
                    test_mod = row['ID']
                    horizontal_subset = parse_horizontal(all_series, model_id=test_mod)

                if verbose >= 2:
                    p = f"Ensemble {model_param_dict['model_name']} component {index} of {total_ens} {row['Model']} started"
                    print(p)
                df_forecast = model_forecast(
                    model_name=row['Model'],
                    model_param_dict=row['ModelParameters'],
                    model_transform_dict=row['TransformationParameters'],
                    df_train=df_train,
                    forecast_length=forecast_length,
                    frequency=frequency,
                    prediction_interval=prediction_interval,
                    no_negatives=no_negatives,
                    constraint=constraint,
                    future_regressor_train=future_regressor_train,
                    future_regressor_forecast=future_regressor_forecast,
                    holiday_country=holiday_country,
                    startTimeStamps=startTimeStamps,
                    grouping_ids=grouping_ids,
                    fail_on_forecast_nan=fail_on_forecast_nan,
                    random_seed=random_seed,
                    verbose=verbose,
                    n_jobs=n_jobs,
                    template_cols=template_cols,
                    horizontal_subset=horizontal_subset,
                    current_model_file=current_model_file,
                    model_count=model_count,
                    force_gc=force_gc,
                )
                model_id = create_model_id(
                    df_forecast.model_name,
                    df_forecast.model_parameters,
                    df_forecast.transformation_parameters,
                )
                total_runtime = (
                    df_forecast.fit_runtime
                    + df_forecast.predict_runtime
                    + df_forecast.transformation_runtime
                )
                forecasts_runtime[model_id] = total_runtime
                forecasts[model_id] = df_forecast.forecast
                upper_forecasts[model_id] = df_forecast.upper_forecast
                lower_forecasts[model_id] = df_forecast.lower_forecast
                # print(f"{model_param_dict['model_name']} with shape {df_forecast.forecast.shape}")
            except Exception as e:
                # currently this leaves no key/value for models that fail
                if verbose >= 1:  # 1
                    print(tb.format_exc())
                    p = f"FAILED: Ensemble {model_param_dict['model_name']} component {index + 1} of {total_ens} {row['Model']} with error: {repr(e)}"
                    print(p)
        ens_forecast = EnsembleForecast(
            model_name,
            model_param_dict,
            forecasts_list=list(forecasts.keys()),
            forecasts=forecasts,
            lower_forecasts=lower_forecasts,
            upper_forecasts=upper_forecasts,
            forecasts_runtime=forecasts_runtime,
            prediction_interval=prediction_interval,
            df_train=df_train,
            prematched_series=all_series,
        )
        ens_forecast.runtime_dict = forecasts_runtime
        # POST PROCESSING ONLY
        if model_transform_dict and not internal_validation:
            transformer_object = GeneralTransformer(
                **model_transform_dict,
                n_jobs=n_jobs,
                holiday_country=holiday_country,
                verbose=verbose,
                random_seed=random_seed,
                forecast_length=forecast_length,
            )
            transformer_object.fit(df_train)
            ens_forecast.forecast = transformer_object.inverse_transform(
                ens_forecast.forecast
            )
            ens_forecast.lower_forecast = transformer_object.inverse_transform(
                ens_forecast.lower_forecast, fillzero=True, bounds=True
            )
            ens_forecast.upper_forecast = transformer_object.inverse_transform(
                ens_forecast.upper_forecast, fillzero=True, bounds=True
            )
        # so ensembles bypass the usual checks, so needed again here
        if fail_on_forecast_nan:
            if not np.isfinite(np.max(ens_forecast.forecast.to_numpy())):
                raise ValueError(
                    "Model {} returned NaN for one or more series. fail_on_forecast_nan=True".format(
                        model_name
                    )
                )
        return ens_forecast
    # if not an ensemble
    else:
        # model_str = row_upper['Model']
        # parameter_dict = json.loads(row_upper['ModelParameters'])
        # transformation_dict = json.loads(row_upper['TransformationParameters'])

        # this is needed for horizontal generalization if any models failed, at least one full model on all series
        if model_name in superfast and not full_model_created:
            make_full_flag = True
        else:
            make_full_flag = False
        if (
            horizontal_subset is not None
            and model_name in no_shared
            and all(
                trs not in shared_trans
                for trs in list(model_transform_dict['transformations'].values())
            )
            and not make_full_flag
        ):
            df_train_low = df_train.reindex(copy=True, columns=horizontal_subset)
            # print(f"Reducing to subset for {model_name} with {df_train_low.columns}")
        else:
            df_train_low = df_train.copy()
            full_model_created = True

        model = ModelPrediction(
            forecast_length=forecast_length,
            transformation_dict=model_transform_dict,
            model_str=model_name,
            parameter_dict=model_param_dict,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            constraint=constraint,
            grouping_ids=grouping_ids,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            fail_on_forecast_nan=fail_on_forecast_nan,
            startTimeStamps=startTimeStamps,
            n_jobs=n_jobs,
            return_model=return_model,
            current_model_file=current_model_file,
            model_count=model_count,
            force_gc=force_gc,
        )
        model = model.fit(df_train_low, future_regressor_train)
        return model.predict(
            forecast_length, future_regressor=future_regressor_forecast
        )


def _ps_metric(per_series_metrics, metric, model_id):
    cur_mae = per_series_metrics.loc[metric]
    cur_mae = pd.DataFrame(cur_mae).transpose()
    cur_mae.index = [model_id]
    return cur_mae


def _eval_prediction_for_template(
    df_forecast,
    template_result,
    verbose,
    actuals,
    weights,
    df_trn_arr,
    ensemble,
    scaler,
    cumsum_A,
    diff_A,
    last_of_array,
    validation_round,
    model_str,
    best_smape,
    ensemble_input,
    parameter_dict,
    transformation_dict,
    row,
    post_memory_percent,
    mosaic_used,
    template_start_time,
    current_generation,
    df_train,
    custom_metric=None,
):
    per_ts = True if 'distance' in ensemble else False
    model_error = df_forecast.evaluate(
        actuals,
        series_weights=weights,
        df_train=df_trn_arr,
        per_timestamp_errors=per_ts,
        scaler=scaler,
        cumsum_A=cumsum_A,
        diff_A=diff_A,
        last_of_array=last_of_array,
        column_names=df_train.columns,
        custom_metric=custom_metric,
    )
    if validation_round >= 1 and verbose > 0:
        round_smape = round(
            model_error.avg_metrics['smape'], 2
        )  # should work on both DF and single value
        validation_accuracy_print = "{} - {} with avg smape {}: ".format(
            str(template_result.model_count),
            model_str,
            round_smape,
        )
        if round_smape < best_smape:
            best_smape = round_smape
            try:
                print("\U0001F4C8 " + validation_accuracy_print)
            except Exception:
                print(validation_accuracy_print)
        else:
            print(validation_accuracy_print)
    # for horizontal ensemble, use requested ID and params
    if ensemble_input == 2:
        model_id = create_model_id(model_str, parameter_dict, transformation_dict)
        # it's already json
        deposit_params = row['ModelParameters']
    else:
        # for non horizontal, recreate based on what model actually used (some change)
        model_id = create_model_id(
            df_forecast.model_name,
            df_forecast.model_parameters,
            df_forecast.transformation_parameters,
        )
        deposit_params = json.dumps(df_forecast.model_parameters)
    result = pd.DataFrame(
        {
            'ID': model_id,
            'Model': df_forecast.model_name,
            'ModelParameters': deposit_params,
            'TransformationParameters': json.dumps(
                df_forecast.transformation_parameters
            ),
            'TransformationRuntime': df_forecast.transformation_runtime,
            'FitRuntime': df_forecast.fit_runtime,
            'PredictRuntime': df_forecast.predict_runtime,
            'TotalRuntime': datetime.datetime.now() - template_start_time,
            'Ensemble': ensemble_input,
            'Exceptions': np.nan,
            'Runs': 1,
            'Generation': current_generation,
            'ValidationRound': validation_round,
            'ValidationStartDate': df_forecast.forecast.index[0],
        },
        index=[0],
    )
    if verbose > 1:
        result['PostMemoryPercent'] = post_memory_percent
    a = pd.DataFrame(
        model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')
    ).transpose()
    result = pd.concat(
        [result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis=1
    )
    template_result.model_results = pd.concat(
        [template_result.model_results, result],
        axis=0,
        ignore_index=True,
        sort=False,
    ).reset_index(drop=True)

    ps_metric = model_error.per_series_metrics
    ps_metric["ValidationRound"] = validation_round
    ps_metric.index.name = "autots_eval_metric"
    ps_metric = ps_metric.reset_index(drop=False)
    ps_metric.index = [model_id] * ps_metric.shape[0]
    ps_metric.index.name = "ID"
    template_result.per_series_metrics.append(ps_metric)
    if 'distance' in ensemble:
        cur_smape = model_error.per_timestamp.loc['weighted_smape']
        cur_smape = pd.DataFrame(cur_smape).transpose()
        cur_smape.index = [model_id]
        template_result.per_timestamp_smape = pd.concat(
            [template_result.per_timestamp_smape, cur_smape], axis=0
        )
    if mosaic_used:
        template_result.full_mae_errors.extend([model_error.full_mae_errors])
        template_result.squared_errors.extend([model_error.squared_errors])
        template_result.full_pl_errors.extend(
            [model_error.upper_pl + model_error.lower_pl]
        )
        template_result.full_mae_ids.extend([model_id])
        template_result.full_mae_vals.extend([validation_round])
    return template_result, best_smape


horizontal_post_processors = [
    {
        "fillna": "fake_date",
        "transformations": {"0": "AlignLastValue", "1": "AlignLastValue"},
        "transformation_params": {
            "0": {
                "rows": 1,
                "lag": 1,
                "method": "multiplicative",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": None,
                "threshold_method": "mean",
            },
            "1": {
                "rows": 1,
                "lag": 1,
                "method": "multiplicative",
                "strength": 1.0,
                "first_value_only": True,
                "threshold": 10,
                "threshold_method": "max",
            },
        },
    },  # best competition on vn1
    {
        "fillna": "fake_date",
        "transformations": {"0": "AlignLastValue", "1": "AlignLastValue"},
        "transformation_params": {
            "0": {
                "rows": 4,
                "lag": 28,
                "method": "additive",
                "strength": 0.2,
                "first_value_only": False,
                "threshold": 1,
                "threshold_method": "max",
            },
            "1": {
                "rows": 1,
                "lag": 1,
                "method": "additive",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": 1,
                "threshold_method": "mean",
            },
        },
    },  # best wasserstein on daily
    # {"fillna": "linear", "transformations": {"0": "bkfilter", "1": "DifferencedTransformer", "2": "BKBandpassFilter"}, "transformation_params": {"0": {}, "1": {"lag": 1, "fill": "zero"}, "2": {"low": 12, "high": 32, "K": 6, "lanczos_factor": False, "return_diff": False, "on_transform": False, "on_inverse": True}}},
    {
        "fillna": "rolling_mean_24",
        "transformations": {"0": "bkfilter", "1": "FIRFilter", "2": "AlignLastDiff"},
        "transformation_params": {
            "0": {},
            "1": {
                "numtaps": 128,
                "cutoff_hz": 0.01,
                "window": "blackman",
                "sampling_frequency": 60,
                "on_transform": False,
                "on_inverse": True,
            },
            "2": {
                "rows": 90,
                "displacement_rows": 1,
                "quantile": 1.0,
                "decay_span": 90,
            },
        },
    },  # best smape on daily, observed most effective on eval loop too
    {
        "fillna": "ffill",
        "transformations": {
            "0": "AlignLastValue",
            "1": "SeasonalDifference",
            "2": "AlignLastValue",
        },
        "transformation_params": {
            "0": {
                "rows": 1,
                "lag": 1,
                "method": "additive",
                "strength": 0.7,
                "first_value_only": False,
                "threshold": 1,
                "threshold_method": "max",
            },
            "1": {"lag_1": 7, "method": 20},
            "2": {
                "rows": 1,
                "lag": 28,
                "method": "multiplicative",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": None,
                "threshold_method": "mean",
            },
        },
    },  # best mae on daily, a bit weird otherwise, 1x best mage daily
    {
        "fillna": "median",
        "transformations": {
            "0": "DiffSmoother",
            "1": "AlignLastValue",
            "2": "HistoricValues",
        },
        "transformation_params": {
            "0": {
                "method": "med_diff",
                "method_params": {"distribution": "norm", "alpha": 0.05},
                "transform_dict": None,
                "reverse_alignment": False,
                "isolated_only": False,
                "fillna": "linear",
            },
            "1": {
                "rows": 1,
                "lag": 1,
                "method": "additive",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": None,
                "threshold_method": "mean",
            },
            "2": {"window": 10},
        },
    },  # best mae on daily, 2x observed again as best, 3x
    {
        "fillna": "fake_date",
        "transformations": {
            "0": "AlignLastValue",
            "1": "PositiveShift",
            "2": "HistoricValues",
        },
        "transformation_params": {
            "0": {
                "rows": 1,
                "lag": 1,
                "method": "additive",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": 10,
                "threshold_method": "mean",
            },
            "1": {},
            "2": {"window": 28},
        },
    },  # best competition on VN1
    {
        "fillna": "ffill",
        "transformations": {"0": "FFTFilter", "1": "HistoricValues"},
        "transformation_params": {
            "0": {
                "cutoff": 0.01,
                "reverse": False,
                "on_transform": False,
                "on_inverse": True,
            },
            "1": {"window": None},
        },
    },  # best smape on daily, best on VN1
    {
        "fillna": "linear",
        "transformations": {"0": "Constraint"},
        "transformation_params": {
            "0": {
                "constraint_method": "quantile",
                "constraint_direction": "lower",
                "constraint_regularization": 0.7,
                "constraint_value": 0.5,
                "bounds_only": False,
                "fillna": "linear",
            }
        },
    },  # best on daily, weight of smape and wasserstein, 1x very stable!
    {
        "fillna": "ffill",
        "transformations": {"0": "RegressionFilter", "1": "HistoricValues"},
        "transformation_params": {
            "0": {
                "sigma": 1,
                "rolling_window": 90,
                "run_order": "season_first",
                "regression_params": {
                    "regression_model": {
                        "model": "ElasticNet",
                        "model_params": {
                            "l1_ratio": 0.5,
                            "fit_intercept": False,
                            "selection": "cyclic",
                            "max_iter": 2000,
                        },
                    },
                    "datepart_method": "simple",
                    "polynomial_degree": None,
                    "transform_dict": {
                        "fillna": None,
                        "transformations": {"0": "ScipyFilter"},
                        "transformation_params": {
                            "0": {
                                "method": "savgol_filter",
                                "method_args": {
                                    "window_length": 31,
                                    "polyorder": 3,
                                    "deriv": 0,
                                    "mode": "interp",
                                },
                            }
                        },
                    },
                    "holiday_countries_used": False,
                    "lags": None,
                    "forward_lags": None,
                },
                "holiday_params": None,
                "trend_method": "rolling_mean",
            },
            "1": {"window": 28},
        },
    },  # best on daily, competition, mae
    {
        "fillna": "zero",
        "transformations": {
            "0": "AnomalyRemoval",
            "1": "EWMAFilter",
            "2": "AlignLastValue",
        },
        "transformation_params": {
            "0": {
                "method": "med_diff",
                "method_params": {"distribution": "norm", "alpha": 0.05},
                "fillna": "rolling_mean_24",
                "transform_dict": {
                    "fillna": None,
                    "transformations": {"0": "EWMAFilter"},
                    "transformation_params": {"0": {"span": 7}},
                },
                "isolated_only": False,
                "on_inverse": False,
            },
            "1": {"span": 7},
            "2": {
                "rows": 1,
                "lag": 1,
                "method": "multiplicative",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": 1,
                "threshold_method": "max",
            },
        },
    },  # best on simple ensemble on daily
    {  # best on VN1 competition and MAE
        "fillna": "cubic",
        "transformations": {"0": "ScipyFilter", "1": "DatepartRegression"},
        "transformation_params": {
            "0": {
                "method": "butter",
                "method_args": {
                    "N": 1,
                    "btype": "highpass",
                    "analog": False,
                    "output": "sos",
                    "Wn": 0.024390243902439025,
                },
            },
            "1": {
                "regression_model": {
                    "model": "ElasticNet",
                    "model_params": {
                        "l1_ratio": 0.5,
                        "fit_intercept": True,
                        "selection": "cyclic",
                        "max_iter": 1000,
                    },
                },
                "datepart_method": ["weekdayofmonth", "common_fourier"],
                "polynomial_degree": None,
                "transform_dict": None,
                "holiday_countries_used": False,
                "lags": None,
                "forward_lags": None,
            },
        },
    },
    {  # balanced on wiki daily
        "fillna": "cubic",
        "transformations": {"0": "AlignLastValue", "1": "DatepartRegression"},
        "transformation_params": {
            "0": {
                "rows": 1,
                "lag": 7,
                "method": "multiplicative",
                "strength": 0.9,
                "first_value_only": False,
                "threshold": 3,
                "threshold_method": "max",
            },
            "1": {
                "regression_model": {
                    "model": "ElasticNet",
                    "model_params": {
                        "l1_ratio": 0.5,
                        "fit_intercept": True,
                        "selection": "cyclic",
                        "max_iter": 1000,
                    },
                },
                "datepart_method": "common_fourier",
                "polynomial_degree": None,
                "transform_dict": {
                    "fillna": None,
                    "transformations": {"0": "ClipOutliers"},
                    "transformation_params": {
                        "0": {"method": "clip", "std_threshold": 4}
                    },
                },
                "holiday_countries_used": False,
                "lags": None,
                "forward_lags": None,
            },
        },
    },
    {  # best on VPV, 19.7 smape
        "fillna": "quadratic",
        "transformations": {"0": "AlignLastValue", "1": "ChangepointDetrend"},
        "transformation_params": {
            "0": {
                "rows": 1,
                "lag": 1,
                "method": "multiplicative",
                "strength": 1.0,
                "first_value_only": False,
                "threshold": None,
                "threshold_method": "mean",
            },
            "1": {
                "model": "Linear",
                "changepoint_spacing": 180,
                "changepoint_distance_end": 360,
                "datepart_method": None,
            },
        },
    },
]


def TemplateWizard(
    template,
    df_train,
    df_test,
    weights,
    model_count: int = 0,
    ensemble: list = ["mosaic", "distance"],
    forecast_length: int = 14,
    frequency: str = 'infer',
    prediction_interval: float = 0.9,
    no_negatives: bool = False,
    constraint: float = None,
    future_regressor_train=None,
    future_regressor_forecast=None,
    holiday_country: str = 'US',
    startTimeStamps=None,
    random_seed: int = 2020,
    verbose: int = 0,
    n_jobs: int = None,
    validation_round: int = 0,
    current_generation: int = 0,
    max_generations: str = "0",
    model_interrupt: bool = False,
    grouping_ids=None,
    template_cols: list = [
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
    traceback: bool = False,
    current_model_file: str = None,
    mosaic_used=None,
    force_gc: bool = False,
    additional_msg: str = "",
    custom_metric=None,
):
    """
    Take Template, returns Results.

    There are some who call me... Tim. - Python

    Args:
        template (pandas.DataFrame): containing model str, and json of transformations and hyperparamters
        df_train (pandas.DataFrame): numeric training dataset of DatetimeIndex and series as cols
        df_test (pandas.DataFrame): dataframe of actual values of (forecast length * n series)
        weights (dict): key = column/series_id, value = weight
        ensemble (list): list of ensemble types to prepare metric collection
        forecast_length (int): number of periods to forecast
        transformation_dict (dict): a dictionary of outlier, fillNA, and transformation methods to be used
        model_str (str): a string to be direct to the appropriate model, used in ModelMonster
        frequency (str): str representing frequency alias of time series
        prediction_interval (float): width of errors (note: rarely do the intervals accurately match the % asked for...)
        no_negatives (bool): whether to force all forecasts to be > 0
        constraint (float): when not None, use this value * data st dev above max or below min for constraining forecast values.
        future_regressor_train (pd.Series): with datetime index, of known in advance data, section matching train data
        future_regressor_forecast (pd.Series): with datetime index, of known in advance data, section matching test data
        holiday_country (str): passed through to holiday package, used by a few models as 0/1 regressor.
        startTimeStamps (pd.Series): index (series_ids), columns (Datetime of First start of series)
        validation_round (int): int passed to record current validation.
        current_generation (int): info to pass to print statements
        max_generations (str): info to pass to print statements
        model_interrupt (bool): if True, keyboard interrupts are caught and only break current model eval.
        template_cols (list): column names of columns used as model template
        traceback (bool): include tracebook over just error representation
        current_model_file (str): file path to write to disk of current model params (for debugging if computer crashes). .json is appended
        force_gc (bool): if True, run gc.collect after every model run

    Returns:
        TemplateEvalObject
    """
    best_smape = float("inf")
    template_result = TemplateEvalObject(
        per_series_metrics=[],
    )
    if mosaic_used is None:
        mosaic_used = is_mosaic(ensemble)
    template_result.model_count = model_count
    if isinstance(template, pd.Series):
        template = template.to_frame()
    if verbose > 1:
        try:
            from psutil import virtual_memory
        except Exception:

            class MemObjecty(object):
                def __init__(self):
                    self.percent = np.nan

            def virtual_memory():
                return MemObjecty()

    # template = unpack_ensemble_models(template, template_cols, keep_ensemble = False)

    # minor speedup with one less copy per eval by assuring arrays at this level
    actuals = np.asarray(df_test)
    df_trn_arr = np.asarray(df_train)
    # precompute scaler to save a few miliseconds (saves very little time)
    scaler = default_scaler(df_trn_arr)
    cumsum_A = np.nancumsum(actuals, axis=0)
    last_of_array = array_last_val(df_trn_arr)
    diff_A = np.diff(np.concatenate([last_of_array, actuals]), axis=0)

    template_dict = template.to_dict('records')
    for row in template_dict:
        template_start_time = datetime.datetime.now()
        try:
            model_str = row['Model']
            parameter_dict = json.loads(row['ModelParameters'])
            transformation_dict = json.loads(row['TransformationParameters'])
            ensemble_input = row['Ensemble']
            if ensemble_input == 2 and transformation_dict:
                # SKIP BECAUSE TRANSFORMERS (PRE DEFINED) ARE DONE BELOW TO REDUCE FORECASTS RERUNS
                # ON INTERNAL VALIDATION ONLY ON TEMPLATES
                if verbose >= 1:
                    print(
                        "skipping horizontal with transformation due to that being done on internal validation"
                    )
                continue
            template_result.model_count += 1
            if verbose > 0:
                if validation_round >= 1:
                    base_print = "Model Number: {} of {} with model {} for Validation {}{}".format(
                        str(template_result.model_count),
                        template.shape[0],
                        model_str,
                        str(validation_round),
                        str(additional_msg),
                    )
                else:
                    base_print = (
                        "Model Number: {} with model {} in generation {} of {}".format(
                            str(template_result.model_count),
                            model_str,
                            str(current_generation),
                            str(max_generations),
                        )
                    )
                if verbose > 1:
                    print(
                        base_print
                        + " with params {} and transformations {}{}".format(
                            json.dumps(parameter_dict),
                            json.dumps(transformation_dict),
                            str(additional_msg),
                        )
                    )
                else:
                    print(base_print)
            df_forecast = model_forecast(
                model_name=row['Model'],
                model_param_dict=row['ModelParameters'],
                model_transform_dict=row['TransformationParameters'],
                df_train=df_train,
                forecast_length=forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=constraint,
                future_regressor_train=future_regressor_train,
                future_regressor_forecast=future_regressor_forecast,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
                template_cols=template_cols,
                current_model_file=current_model_file,
                model_count=template_result.model_count,
                force_gc=force_gc,
                internal_validation=True,  # THIS MIGHT BE REDUNTANT, THE CONTINUE ABOVE MAYBE BE ENOUGH
            )
            if verbose > 1:
                post_memory_percent = virtual_memory().percent
            else:
                post_memory_percent = 0.0
            # wrapped up to enable postprocessing
            template_result, best_smape = _eval_prediction_for_template(
                df_forecast,
                template_result,
                verbose,
                actuals,
                weights,
                df_trn_arr,
                ensemble,
                scaler,
                cumsum_A,
                diff_A,
                last_of_array,
                validation_round,
                model_str,
                best_smape,
                ensemble_input,
                parameter_dict,
                transformation_dict,
                row,
                post_memory_percent,
                mosaic_used,
                template_start_time,
                current_generation,
                df_train,
                custom_metric,
            )
            if ensemble_input in [1, 2]:
                # INTERNAL VALIDATION ONLY, POST PROCESSING ONLY
                # more efficent than rerunning the forecasts just to change transformers
                for x in horizontal_post_processors:
                    df_forecast2 = copy.copy(df_forecast)
                    transformer_object = GeneralTransformer(
                        **x,
                        n_jobs=n_jobs,
                        holiday_country=holiday_country,
                        verbose=verbose,
                        random_seed=random_seed,
                        forecast_length=forecast_length,
                    )
                    transformer_object.fit(df_train)
                    df_forecast2.forecast = transformer_object.inverse_transform(
                        df_forecast2.forecast
                    )
                    df_forecast2.lower_forecast = transformer_object.inverse_transform(
                        df_forecast2.lower_forecast, fillzero=True, bounds=True
                    )
                    df_forecast2.upper_forecast = transformer_object.inverse_transform(
                        df_forecast2.upper_forecast, fillzero=True, bounds=True
                    )
                    df_forecast2.transformation_parameters = x
                    template_result.model_count += 1
                    template_result, best_smape = _eval_prediction_for_template(
                        df_forecast2,
                        template_result,
                        verbose,
                        actuals,
                        weights,
                        df_trn_arr,
                        ensemble,
                        scaler,
                        cumsum_A,
                        diff_A,
                        last_of_array,
                        validation_round,
                        model_str,
                        best_smape,
                        ensemble_input,
                        parameter_dict,
                        x,
                        row,
                        post_memory_percent,
                        mosaic_used,
                        template_start_time,
                        current_generation,
                        df_train,
                        custom_metric,
                    )

        except KeyboardInterrupt:
            if model_interrupt:
                fit_runtime = datetime.datetime.now() - template_start_time
                result = pd.DataFrame(
                    {
                        'ID': create_model_id(
                            model_str, parameter_dict, transformation_dict
                        ),
                        'Model': model_str,
                        'ModelParameters': json.dumps(parameter_dict),
                        'TransformationParameters': json.dumps(transformation_dict),
                        'Ensemble': ensemble_input,
                        'TransformationRuntime': datetime.timedelta(0),
                        'FitRuntime': fit_runtime,
                        'PredictRuntime': datetime.timedelta(0),
                        'TotalRuntime': fit_runtime,
                        'Exceptions': "KeyboardInterrupt by user",
                        'Runs': 1,
                        'Generation': current_generation,
                        'ValidationRound': validation_round,
                    },
                    index=[0],
                )
                template_result.model_results = pd.concat(
                    [template_result.model_results, result],
                    axis=0,
                    ignore_index=True,
                    sort=False,
                ).reset_index(drop=True)
                if model_interrupt == "end_generation" and current_generation > 0:
                    break
            else:
                sys.stdout.flush()
                raise KeyboardInterrupt
        except Exception as e:
            if verbose >= 0:
                if traceback:
                    print(
                        'Template Eval Error: {} in model {} in generation {}: {}'.format(
                            ''.join(tb.format_exception(None, e, e.__traceback__)),
                            template_result.model_count,
                            str(current_generation),
                            model_str,
                        )
                    )
                else:
                    print(
                        'Template Eval Error: {} in model {} in generation {}: {}'.format(
                            (repr(e)),
                            template_result.model_count,
                            str(current_generation),
                            model_str,
                        )
                    )
            fit_runtime = datetime.datetime.now() - template_start_time
            result = pd.DataFrame(
                {
                    'ID': create_model_id(
                        model_str, parameter_dict, transformation_dict
                    ),
                    'Model': model_str,
                    'ModelParameters': json.dumps(parameter_dict),
                    'TransformationParameters': json.dumps(transformation_dict),
                    'Ensemble': ensemble_input,
                    'TransformationRuntime': datetime.timedelta(0),
                    'FitRuntime': fit_runtime,
                    'PredictRuntime': datetime.timedelta(0),
                    'TotalRuntime': fit_runtime,
                    'Exceptions': repr(e),
                    'Runs': 1,
                    'Generation': current_generation,
                    'ValidationRound': validation_round,
                },
                index=[0],
            )
            template_result.model_results = pd.concat(
                [template_result.model_results, result],
                axis=0,
                ignore_index=True,
                sort=False,
            ).reset_index(drop=True)
    if template_result.per_series_metrics:
        template_result.per_series_metrics = pd.concat(
            template_result.per_series_metrics, axis=0
        )
        ps = template_result.per_series_metrics
        template_result.per_series_mae = ps[ps['autots_eval_metric'] == 'mae'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_made = ps[ps['autots_eval_metric'] == 'made'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_contour = ps[
            ps['autots_eval_metric'] == 'contour'
        ].drop(columns=['autots_eval_metric', "ValidationRound"])
        template_result.per_series_rmse = ps[ps['autots_eval_metric'] == 'rmse'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_spl = ps[ps['autots_eval_metric'] == 'spl'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_mle = ps[ps['autots_eval_metric'] == 'mle'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_imle = ps[ps['autots_eval_metric'] == 'imle'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_maxe = ps[ps['autots_eval_metric'] == 'maxe'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_oda = ps[ps['autots_eval_metric'] == 'oda'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_mqae = ps[ps['autots_eval_metric'] == 'mqae'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_dwae = ps[ps['autots_eval_metric'] == 'dwae'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_ewmae = ps[ps['autots_eval_metric'] == 'ewmae'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_uwmse = ps[ps['autots_eval_metric'] == 'uwmse'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_smoothness = ps[
            ps['autots_eval_metric'] == 'smoothness'
        ].drop(columns=['autots_eval_metric', "ValidationRound"])
        template_result.per_series_mate = ps[ps['autots_eval_metric'] == 'mate'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_matse = ps[ps['autots_eval_metric'] == 'matse'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
        template_result.per_series_wasserstein = ps[
            ps['autots_eval_metric'] == 'wasserstein'
        ].drop(columns=['autots_eval_metric', "ValidationRound"])
        template_result.per_series_dwd = ps[ps['autots_eval_metric'] == 'dwd'].drop(
            columns=['autots_eval_metric', "ValidationRound"]
        )
    else:
        template_result.per_series_metrics = pd.DataFrame()
        template_result.per_series_mae = pd.DataFrame()
        template_result.per_series_made = pd.DataFrame()
        template_result.per_series_contour = pd.DataFrame()
        template_result.per_series_rmse = pd.DataFrame()
        template_result.per_series_spl = pd.DataFrame()
        template_result.per_series_mle = pd.DataFrame()
        template_result.per_series_imle = pd.DataFrame()
        template_result.per_series_maxe = pd.DataFrame()
        template_result.per_series_oda = pd.DataFrame()
        template_result.per_series_mqae = pd.DataFrame()
        template_result.per_series_dwae = pd.DataFrame()
        template_result.per_series_ewmae = pd.DataFrame()
        template_result.per_series_uwmse = pd.DataFrame()
        template_result.per_series_smoothness = pd.DataFrame()
        template_result.per_series_mate = pd.DataFrame()
        template_result.per_series_matse = pd.DataFrame()
        template_result.per_series_wasserstein = pd.DataFrame()
        template_result.per_series_dwd = pd.DataFrame()
        if verbose > 0 and not template.empty:
            print(f"Generation {current_generation} had all new models fail")
    return template_result


def random_model(
    model_list,
    model_prob,
    transformer_list='fast',
    transformer_max_depth=2,
    models_mode='random',
    counter=15,
    n_models=5,
    keyword_format=False,
):
    """Generate a random model from a given list of models and probabilities."""
    if counter < n_models:
        model_str = model_list[counter]
    else:
        model_str = random.choices(model_list, model_prob, k=1)[0]
    param_dict = ModelMonster(model_str).get_new_params(method=models_mode)
    if counter % 4 == 0:
        trans_dict = RandomTransform(
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            traditional_order=True,
        )
    else:
        trans_dict = RandomTransform(
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
        )
    if keyword_format:
        return {
            'model_name': model_str,
            'model_param_dict': param_dict,
            'model_transform_dict': trans_dict,
        }
    else:
        return {
            'Model': model_str,
            'ModelParameters': json.dumps(param_dict),
            'TransformationParameters': json.dumps(trans_dict),
            'Ensemble': 0,
        }


def RandomTemplate(
    n: int = 10,
    model_list: list = [
        'ZeroesNaive',
        'LastValueNaive',
        'AverageValueNaive',
        'GLS',
        'GLM',
        'ETS',
    ],
    transformer_list: dict = "fast",
    transformer_max_depth: int = 8,
    models_mode: str = "default",
):
    """
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters.

    Args:
        n (int): number of random models to return
    """
    n = abs(int(n))
    template = pd.DataFrame()
    counter = 0
    n_models = len(model_list)
    model_list, model_prob = model_list_to_dict(model_list)
    template_list = []
    while len(template_list) < n:
        # this assures all models get choosen at least once
        random_mod = random_model(
            model_list=model_list,
            model_prob=model_prob,
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            models_mode=models_mode,
            counter=counter,
            n_models=n_models,
        )

        template_list.append(
            pd.DataFrame(
                random_mod,
                index=[0],
            )
        )
        counter += 1
        if counter > (n * 3):
            break
    template = pd.concat(template_list, axis=0, ignore_index=True)
    template = template.drop_duplicates()
    return template


def UniqueTemplates(
    existing_templates,
    new_possibilities,
    selection_cols: list = [
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
):
    """
    Returns unique dataframe rows from new_possiblities not in existing_templates.

    Args:
        selection_cols (list): list of column namess to use to judge uniqueness/match on
    """
    keys = list(new_possibilities[selection_cols].columns.values)
    idx1 = existing_templates.copy().set_index(keys).index
    idx2 = new_possibilities.set_index(keys).index
    new_template = new_possibilities[~idx2.isin(idx1)]
    return new_template


def dict_recombination(a: dict, b: dict):
    """Recombine two dictionaries with identical keys. Return new dict."""
    b_keys = [*b]
    key_size = int(len(b_keys) / 2) if len(b_keys) > 1 else 1
    bs_keys = random.choices(b_keys, k=key_size)
    b_prime = {k: b[k] for k in bs_keys}
    c = {**a, **b_prime}  # overwrites with B
    return c


def trans_dict_recomb(dict_array):
    """Recombine two transformation param dictionaries from array of dicts."""
    empty_trans = (None, {})
    a, b = random.sample(dict_array, 2)
    na_choice = random.sample([a, b], 1)[0]['fillna']

    a_result = [
        (a['transformations'][key], a['transformation_params'][key])
        for key in sorted(a['transformations'].keys())
    ]
    b_result = [
        (b['transformations'][key], b['transformation_params'][key])
        for key in sorted(b['transformations'].keys())
    ]
    combi = zip_longest(a_result, b_result, fillvalue=empty_trans)
    selected = [random.choice(x) for x in combi]
    selected = [x for x in selected if x != empty_trans]
    if not selected:
        selected = [empty_trans]
    selected_vals = list(zip(*selected))
    keys = range(len(selected))
    return {
        "fillna": na_choice,
        "transformations": dict(zip(keys, selected_vals[0])),
        "transformation_params": dict(zip(keys, selected_vals[1])),
    }


def _trans_dicts(
    current_ops,
    best=None,
    n: int = 5,
    transformer_list: dict = {},
    transformer_max_depth: int = 8,
    first_transformer: dict = None,
):
    if first_transformer is None:
        first_transformer = json.loads(
            current_ops.iloc[0, :]['TransformationParameters']
        )
    cur_len = current_ops.shape[0]
    if cur_len > 1:
        # select randomly from best of data
        top_r = cur_len - 1 if cur_len < 6 else int(cur_len / 3)
        r_id = random.randint(0, top_r)
        sec = json.loads(current_ops.iloc[r_id, :]['TransformationParameters'])
    else:
        sec = RandomTransform(
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
            traditional_order=True,
        )
    r = RandomTransform(
        transformer_list=transformer_list,
        transformer_max_depth=transformer_max_depth,
    )
    r2 = RandomTransform(
        transformer_list=transformer_list,
        transformer_max_depth=transformer_max_depth,
    )
    if best is None:
        best = RandomTransform(
            transformer_list=transformer_list,
            transformer_max_depth=transformer_max_depth,
        )
    arr = [first_transformer, sec, best, r, r2]
    trans_dicts = [json.dumps(trans_dict_recomb(arr)) for _ in range(n)]
    return trans_dicts


def NewGeneticTemplate(
    model_results,
    submitted_parameters,
    sort_column: str = "Score",
    sort_ascending: bool = True,
    max_results: int = 50,
    max_per_model_class: int = 5,
    top_n: int = 50,
    template_cols: list = [
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
    transformer_list: dict = {},
    transformer_max_depth: int = 8,
    models_mode: str = "default",
    score_per_series=None,
    recursive_count=0,
    model_list=None,
    # UPDATE RECURSIVE section if adding or removing params
):
    """
    Return new template given old template with model accuracies.

    "No mating!" - Pattern, Sanderson

    Args:
        model_results (pandas.DataFrame): models that have actually been run
        submitted_paramters (pandas.DataFrame): models tried (may have returned different parameters to results)

    """
    new_template_list = []
    if model_list is None:
        model_list = model_results['Model'].unique().tolist()

    # filter existing templates
    sorted_results = model_results[
        (model_results['Ensemble'] == 0)
        & (model_results['Exceptions'].isna())
        & (model_results['Model'].isin(model_list))
    ].copy()
    # remove duplicates by exact same performance
    sorted_results = sorted_results.sort_values(
        by="TotalRuntimeSeconds", ascending=True
    )
    sorted_results.drop_duplicates(
        subset=['ValidationRound', 'smape', 'mae', 'spl'], keep="first", inplace=True
    )
    sorted_results = sorted_results.drop_duplicates(subset=template_cols, keep='first')
    # perform selection
    sorted_results = sorted_results.sort_values(
        by=sort_column, ascending=sort_ascending, na_position='last'
    )
    # find some of the best per series models to mix in
    if score_per_series is not None:
        per_s = score_per_series / (score_per_series.min(axis=0) + 1)
        ad_mods = per_s.quantile(0.3, axis=1).nsmallest(3).index.tolist()
        ad_mods1 = sorted_results[sorted_results["ID"].isin(ad_mods)]
        ad_mods = per_s.quantile(0.1, axis=1).nsmallest(3).index.tolist()
        ad_mods2 = sorted_results[sorted_results["ID"].isin(ad_mods)]
    # take only max of n models per model type to reduce duplication of similar
    if str(max_per_model_class).isdigit():
        sorted_results = (
            sorted_results.sort_values(sort_column, ascending=sort_ascending)
            .groupby('Model')
            .head(max_per_model_class)
        )
        """
        sorted_results = (
            sorted_results
            .groupby('Model')
            .sample(max_per_model_class, weights=((1 + (1 / sorted_results['Score'])) ** 10) - 1)
        )
        """
    # take only n results to proceed
    sorted_results = sorted_results.head(top_n)
    # tack on the per_series best afterwards
    if score_per_series is not None:
        sorted_results = pd.concat(
            [sorted_results, ad_mods1, ad_mods2], axis=0
        ).drop_duplicates(subset=template_cols, keep='first')

    best = json.loads(sorted_results.iloc[0, :]['TransformationParameters'])

    # best models make more kids
    n_list = sorted([1, 2, 3] * int((sorted_results.shape[0] / 3) + 1), reverse=True)
    counter = 0
    # begin the breeding
    sidx = {name: i for i, name in enumerate(list(sorted_results), start=1)}
    for row in sorted_results.itertuples(name=None):
        n = n_list[counter]
        model_type = row[sidx["Model"]]
        # skip models not in the model_list
        if model_type not in model_list:
            continue
        counter += 1
        model_params = row[sidx["ModelParameters"]]
        try:
            trans_params = json.loads(row[sidx["TransformationParameters"]])
        except Exception as e:
            raise ValueError(f"JSON didn't like {row}") from e
        current_ops = sorted_results[sorted_results['Model'] == model_type]
        # only transformers for some models
        if model_type in no_params:
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
                first_transformer=trans_params,
            )
            new_template_list.append(
                pd.DataFrame(
                    {
                        'Model': model_type,
                        'ModelParameters': model_params,
                        'TransformationParameters': trans_dicts,
                        'Ensemble': 0,
                    },
                    index=list(range(n)),
                )
            )
        # recombination for models where mixing params is allowable
        elif model_type in recombination_approved:
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
                first_transformer=trans_params,
            )
            # select the best model of this type
            # fir = json.loads(current_ops.iloc[0, :]['ModelParameters'])
            fir = json.loads(model_params)
            cur_len = current_ops.shape[0]
            # select randomly from best of data
            if cur_len > 1:
                top_r = cur_len - 1 if cur_len < 9 else int(cur_len / 3)
                r_id = random.randint(0, top_r)
                sec = json.loads(current_ops.iloc[r_id, :]['ModelParameters'])
            else:
                sec = ModelMonster(model_type).get_new_params(method=models_mode)
            # select weighted from all
            if cur_len > 4:
                r = json.loads(
                    current_ops['ModelParameters']
                    .sample(1, weights=np.log(np.arange(cur_len) + 1)[::-1])
                    .iloc[0]
                )
            else:
                r = ModelMonster(model_type).get_new_params(method=models_mode)
            # generate new random parameters ('mutations')
            r2 = ModelMonster(model_type).get_new_params(method=models_mode)
            arr = np.array([fir, sec, r2, r])
            model_dicts = list()
            # recombine best and random to create new generation
            for _ in range(n):
                r_sel = np.random.choice(arr, size=2, replace=False)
                a = r_sel[0]
                b = r_sel[1]
                c = dict_recombination(a, b)
                model_dicts.append(json.dumps(c))
            new_template_list.append(
                pd.DataFrame(
                    {
                        'Model': model_type,
                        'ModelParameters': model_dicts,
                        'TransformationParameters': trans_dicts,
                        'Ensemble': 0,
                    },
                    index=list(range(n)),
                )
            )
        # models with params that cannot be mixed, just random mutations
        else:
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
                first_transformer=trans_params,
            )
            model_dicts = list()
            c0 = json.loads(model_params)
            for _ in range(n):
                c = random.choice(
                    [c0, ModelMonster(model_type).get_new_params(method=models_mode)]
                )
                model_dicts.append(json.dumps(c))
            new_template_list.append(
                pd.DataFrame(
                    {
                        'Model': model_type,
                        'ModelParameters': model_dicts,
                        'TransformationParameters': trans_dicts,
                        'Ensemble': 0,
                    },
                    index=list(range(n)),
                )
            )
    new_template = pd.concat(new_template_list, axis=0, ignore_index=True, sort=False)
    """
    # recombination of transforms across models by shifting transforms
    recombination = sorted_results.tail(len(sorted_results.index) - 1).copy()
    recombination['TransformationParameters'] = sorted_results['TransformationParameters'].shift(1).tail(len(sorted_results.index) - 1)
    new_template = pd.concat([new_template,
                              recombination.head(top_n)[template_cols]],
                             axis=0, ignore_index=True, sort=False)
    """
    # remove generated models which have already been tried
    sorted_results = pd.concat(
        [submitted_parameters, sorted_results], axis=0, ignore_index=True, sort=False
    ).reset_index(drop=True)
    new_template = UniqueTemplates(
        sorted_results, new_template, selection_cols=template_cols
    )
    # use recursion to avoid empty returns
    if new_template.empty:
        recursive_count += 1
        if recursive_count > 20:
            print("NewGeneticTemplate max recursion reached")
            return new_template
        else:
            return NewGeneticTemplate(
                model_results=model_results,
                submitted_parameters=submitted_parameters,
                sort_column=sort_column,
                sort_ascending=sort_ascending,
                max_results=max_results,
                max_per_model_class=max_per_model_class,
                top_n=top_n,
                template_cols=template_cols,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
                models_mode=models_mode,
                score_per_series=score_per_series,
                recursive_count=recursive_count,
                model_list=model_list,
            )
    # enjoy the privilege
    elif new_template.shape[0] < max_results:
        return new_template
    else:
        if max_results < 1:
            return new_template.sample(
                frac=max_results,
                weights=np.log(np.arange(new_template.shape[0]) + 2)[::-1] + 1,
            )
        else:
            return new_template.sample(
                max_results,
                weights=np.log(np.arange(new_template.shape[0]) + 2)[::-1] + 1,
            )


def validation_aggregation(
    validation_results,
    df_train=None,
    groupby_cols=[
        'ID',
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ],
):
    """Aggregate a TemplateEvalObject."""
    col_aggs = {
        'Runs': 'sum',
        'smape': 'mean',
        'mae': 'mean',
        'rmse': 'mean',
        'medae': 'mean',
        'made': 'mean',
        'mage': 'mean',
        'custom': 'mean',
        'underestimate': 'sum',
        'mle': 'mean',
        'overestimate': 'sum',
        'imle': 'mean',
        'spl': 'mean',
        'containment': 'mean',
        'contour': 'mean',
        'maxe': 'max',
        'oda': 'mean',
        'dwae': 'mean',
        'mqae': 'mean',
        'ewmae': 'mean',
        'uwmse': 'mean',
        'smoothness': 'mean',
        'mate': 'mean',
        'wasserstein': 'mean',
        'dwd': 'mean',
        'matse': 'mean',
        'smape_weighted': 'mean',
        'mae_weighted': 'mean',
        'rmse_weighted': 'mean',
        'medae_weighted': 'mean',
        'made_weighted': 'mean',
        'mage_weighted': 'mean',
        'custom_weighted': 'mean',
        'mle_weighted': 'mean',
        'imle_weighted': 'mean',
        'spl_weighted': 'mean',
        'maxe_weighted': 'max',
        'oda_weighted': 'mean',
        'dwae_weighted': 'mean',
        'mqae_weighted': 'mean',
        'ewmae_weighted': 'mean',
        'uwmse_weighted': 'mean',
        'smoothness_weighted': 'mean',
        'mate_weighted': 'mean',
        'wasserstein_weighted': 'mean',
        'dwd_weighted': 'mean',
        'matse_weighted': 'mean',
        'containment_weighted': 'mean',
        'contour_weighted': 'mean',
        'TotalRuntimeSeconds': 'mean',
        'Score': 'mean',
    }
    col_names = validation_results.model_results.columns
    col_aggs = {x: y for x, y in col_aggs.items() if x in col_names}
    cols = [x for x in validation_results.model_results.columns if x in col_aggs.keys()]
    # force numeric dytpes. This really shouldn't be necessary but apparently is sometimes (underlying minor bug somewhere)
    validation_results.model_results[cols] = validation_results.model_results[
        cols
    ].apply(pd.to_numeric, errors='coerce')
    validation_results.model_results['TotalRuntimeSeconds'] = (
        validation_results.model_results['TotalRuntime'].dt.total_seconds().round(4)
    )
    # MODEL SELECTION CODE EXPECTS THAT VAL RESULTS WILL NOT CONTAIN ANY NaN/FAILED MODELS
    validation_results.model_results = validation_results.model_results[
        pd.isnull(validation_results.model_results['Exceptions'])
    ]
    validation_results.model_results = validation_results.model_results.replace(
        [np.inf, -np.inf], np.nan
    )
    grouped = validation_results.model_results.groupby(groupby_cols)
    validation_results.model_results = grouped.agg(col_aggs)
    validation_results.model_results = validation_results.model_results.reset_index(
        drop=False
    )
    if df_train is not None:
        scaler = df_train.mean(axis=0)
        scaler[scaler == 0] == np.nan
        scaler = scaler.fillna(df_train.max(axis=0))
        scaler[scaler == 0] == 1
        per_series = (
            (validation_results.per_series_mae.groupby(level=0).max()) / scaler * 100
        )
        per_series_agg = pd.concat(
            [
                per_series.min(axis=1).rename("lowest_series_mape"),
                per_series.idxmin(axis=1).rename("lowest_series_mape_name"),
                per_series.max(axis=1).rename("highest_series_mape"),
                per_series.idxmax(axis=1).rename("highest_series_mape_name"),
            ],
            axis=1,
        )
        validation_results.model_results = validation_results.model_results.merge(
            per_series_agg, left_on='ID', right_index=True, how='left'
        )
    return validation_results


def generate_score(
    model_results,
    metric_weighting: dict = {},
    prediction_interval: float = 0.9,
    return_score_dict: bool = False,
    num_validations=None,
):
    """Generate score based on relative accuracies.

    SMAPE - smaller is better
    MAE - smaller is better
    RMSE -  smaller is better
    MADE - smaller is better
    MLE - smaller is better
    MAGE - smaller is better
    SPL - smaller is better
    ODA - bigger is better
    DWAE - smaller is better
    Contour - bigger is better (is 0 to 1)
    Containment - bigger is better (is 0 to 1)
    Runtime - smaller is better
    """
    smape_weighting = metric_weighting.get('smape_weighting', 0)
    mae_weighting = metric_weighting.get('mae_weighting', 0)
    rmse_weighting = metric_weighting.get('rmse_weighting', 0)
    containment_weighting = metric_weighting.get('containment_weighting', 0)
    runtime_weighting = metric_weighting.get('runtime_weighting', 0)
    spl_weighting = metric_weighting.get('spl_weighting', 0)
    contour_weighting = metric_weighting.get('contour_weighting', 0)
    made_weighting = metric_weighting.get('made_weighting', 0)
    mage_weighting = metric_weighting.get('mage_weighting', 0)
    custom_weighting = metric_weighting.get('custom_weighting', 0)
    mle_weighting = metric_weighting.get('mle_weighting', 0)
    imle_weighting = metric_weighting.get('imle_weighting', 0)
    maxe_weighting = metric_weighting.get('maxe_weighting', 0)
    oda_weighting = metric_weighting.get('oda_weighting', 0)
    mqae_weighting = metric_weighting.get('mqae_weighting', 0)
    dwae_weighting = metric_weighting.get('dwae_weighting', 0)
    ewmae_weighting = metric_weighting.get('ewmae_weighting', 0)
    uwmse_weighting = metric_weighting.get('uwmse_weighting', 0)
    smoothness_weighting = metric_weighting.get('smoothness_weighting', 0)
    mate_weighting = metric_weighting.get('mate_weighting', 0)
    wasserstein_weighting = metric_weighting.get('wasserstein_weighting', 0)
    dwd_weighting = metric_weighting.get('dwd_weighting', 0)
    matse_weighting = metric_weighting.get('matse_weighting', 0)

    score_dict = {"ID": model_results["ID"]}
    # handle various runtime information records
    if 'TotalRuntimeSeconds' in model_results.columns:
        model_results['TotalRuntimeSeconds'] = np.where(
            model_results['TotalRuntimeSeconds'].isna(),
            model_results['TotalRuntimeSeconds'].max(),
            model_results['TotalRuntimeSeconds'],
        )
    else:
        model_results['TotalRuntimeSeconds'] = (
            model_results['TotalRuntime'].dt.total_seconds().round(4)
        )
    model_results = model_results.replace([np.inf, -np.inf], np.nan)
    # not sure why there are negative SMAPE values, but make sure they get dealt with
    if model_results['smape'].min() < 0:
        model_results['smape'] = model_results['smape'].where(
            model_results['smape'] >= 0, model_results['smape'].max()
        )
    # only use results that meet final validation count, if present
    divisor_results = model_results
    if num_validations is not None:
        if "Runs" in model_results.columns:
            divisor_results = model_results[
                model_results["Runs"] >= (num_validations + 1)
            ]
            print(divisor_results["rmse"])
            if divisor_results.empty:
                divisor_results = model_results
    # generate minimizing scores, where smaller = better accuracy
    try:
        # handle NaN in scores...
        # model_results = model_results.fillna(value=model_results.max(axis=0))

        # where smaller is better, are always >=0, beware divide by zero
        smape_scaler = divisor_results['smape_weighted'][
            divisor_results['smape_weighted'] != 0
        ].min()
        smape_score = model_results['smape_weighted'] / smape_scaler
        overall_score = smape_score * smape_weighting
        score_dict['smape'] = smape_score * smape_weighting
        if mae_weighting != 0:
            mae_scaler = divisor_results['mae_weighted'][
                divisor_results['mae_weighted'] != 0
            ].min()
            mae_score = model_results['mae_weighted'] / mae_scaler
            score_dict['mae'] = mae_score * mae_weighting
            overall_score = overall_score + (mae_score * mae_weighting)
        if rmse_weighting != 0:
            rmse_scaler = divisor_results['rmse_weighted'][
                divisor_results['rmse_weighted'] != 0
            ].min()
            rmse_score = model_results['rmse_weighted'] / rmse_scaler
            score_dict['rmse'] = rmse_score * rmse_weighting
            overall_score = overall_score + (rmse_score * rmse_weighting)
        if made_weighting != 0:
            made_scaler = divisor_results['made_weighted'][
                divisor_results['made_weighted'] != 0
            ].min()
            made_score = model_results['made_weighted'] / made_scaler
            score_dict['made'] = made_score * made_weighting
            # fillna, but only if all are nan (forecast_length = 1)
            if pd.isnull(made_score.max()):
                made_score = made_score.fillna(100)
            overall_score = overall_score + (made_score * made_weighting)
        if mage_weighting != 0:
            mage_scaler = divisor_results['mage_weighted'][
                divisor_results['mage_weighted'] != 0
            ].min()
            mage_score = model_results['mage_weighted'] / mage_scaler
            score_dict['mage'] = mage_score * mage_weighting
            overall_score = overall_score + (mage_score * mage_weighting)
        if custom_weighting != 0:
            custom_scaler = divisor_results['custom_weighted'][
                divisor_results['custom_weighted'] != 0
            ].min()
            # potential edge case where weighting is > 0 but not custom metric is provided and is all zeroes
            if not pd.isnull(custom_scaler):
                custom_score = model_results['custom_weighted'] / custom_scaler
            else:
                custom_score = model_results['custom_weighted']
            score_dict['custom'] = custom_score * custom_weighting
            overall_score = overall_score + (custom_score * custom_weighting)
        if mle_weighting != 0:
            mle_scaler = divisor_results['mle_weighted'][
                divisor_results['mle_weighted'] != 0
            ].min()
            mle_score = (
                model_results['mle_weighted'] / mle_scaler / 10
            )  # / 10 due to imbalance often with this
            score_dict['mle'] = mle_score * mle_weighting
            overall_score = overall_score + (mle_score * mle_weighting)
        if imle_weighting != 0:
            imle_scaler = divisor_results['imle_weighted'][
                divisor_results['imle_weighted'] != 0
            ].min()
            imle_score = (
                model_results['imle_weighted'] / imle_scaler / 10
            )  # / 10 due to imbalance often with this
            score_dict['imle'] = imle_score * imle_weighting
            overall_score = overall_score + (imle_score * imle_weighting)
        if maxe_weighting != 0:
            maxe_scaler = divisor_results['maxe_weighted'][
                divisor_results['maxe_weighted'] != 0
            ].min()
            maxe_score = model_results['maxe_weighted'] / maxe_scaler
            score_dict['maxe'] = maxe_score * maxe_weighting
            overall_score = overall_score + (maxe_score * maxe_weighting)
        if mqae_weighting != 0:
            mqae_scaler = divisor_results['mqae_weighted'][
                divisor_results['mqae_weighted'] != 0
            ].min()
            mqae_score = model_results['mqae_weighted'] / mqae_scaler
            score_dict['mqae'] = mqae_score * mqae_weighting
            overall_score = overall_score + (mqae_score * mqae_weighting)
        if dwae_weighting != 0:
            dwae_scaler = divisor_results['dwae_weighted'][
                divisor_results['dwae_weighted'] != 0
            ].min()
            dwae_score = model_results['dwae_weighted'] / dwae_scaler
            score_dict['dwae'] = dwae_score * dwae_weighting
            overall_score = overall_score + (dwae_score * dwae_weighting)
        if ewmae_weighting != 0:
            ewmae_scaler = divisor_results['ewmae_weighted'][
                divisor_results['ewmae_weighted'] != 0
            ].min()
            ewmae_score = model_results['ewmae_weighted'] / ewmae_scaler
            score_dict['ewmae'] = ewmae_score * ewmae_weighting
            overall_score = overall_score + (ewmae_score * ewmae_weighting)
        if uwmse_weighting != 0:
            uwmse_scaler = divisor_results['uwmse_weighted'][
                divisor_results['uwmse_weighted'] != 0
            ].min()
            uwmse_score = model_results['uwmse_weighted'] / uwmse_scaler
            score_dict['uwmse'] = uwmse_score * uwmse_weighting
            overall_score = overall_score + (uwmse_score * uwmse_weighting)
        if mate_weighting != 0:
            mate_scaler = divisor_results['mate_weighted'][
                divisor_results['mate_weighted'] != 0
            ].min()
            mate_score = model_results['mate_weighted'] / mate_scaler
            score_dict['mate'] = mate_score * mate_weighting
            overall_score = overall_score + (mate_score * mate_weighting)
        if wasserstein_weighting != 0:
            wasserstein_scaler = divisor_results['wasserstein_weighted'][
                divisor_results['wasserstein_weighted'] != 0
            ].min()
            wasserstein_score = (
                model_results['wasserstein_weighted'] / wasserstein_scaler
            )
            score_dict['wasserstein'] = wasserstein_score * wasserstein_weighting
            overall_score = overall_score + (wasserstein_score * wasserstein_weighting)
        if dwd_weighting != 0:
            dwd_scaler = divisor_results['dwd_weighted'][
                divisor_results['dwd_weighted'] != 0
            ].min()
            dwd_score = model_results['dwd_weighted'] / dwd_scaler
            score_dict['dwd'] = dwd_score * dwd_weighting
            overall_score = overall_score + (dwd_score * dwd_weighting)
        if matse_weighting != 0:
            matse_scaler = divisor_results['matse_weighted'][
                divisor_results['matse_weighted'] != 0
            ].min()
            matse_score = model_results['matse_weighted'] / matse_scaler
            score_dict['matse'] = matse_score * matse_weighting
            overall_score = overall_score + (matse_score * matse_weighting)
        if smoothness_weighting != 0:
            smoothness_scaler = divisor_results['smoothness_weighted'][
                divisor_results['smoothness_weighted'] != 0
            ].mean()
            smoothness_score = model_results['smoothness_weighted'] / smoothness_scaler
            score_dict['smoothness'] = smoothness_score * smoothness_weighting
            overall_score = overall_score + (smoothness_score * smoothness_weighting)
        if spl_weighting != 0:
            spl_scaler = divisor_results['spl_weighted'][
                divisor_results['spl_weighted'] != 0
            ].min()
            spl_score = model_results['spl_weighted'] / spl_scaler
            score_dict['spl'] = spl_score * spl_weighting
            overall_score = overall_score + (spl_score * spl_weighting)
        smape_median = smape_score.median()
        if runtime_weighting != 0:
            runtime = model_results['TotalRuntimeSeconds'] + 100
            runtime_scaler = runtime.min()  # [runtime != 0]
            runtime_score = runtime / runtime_scaler
            # this scales it into a similar range as SMAPE
            runtime_score = runtime_score * (smape_median / runtime_score.median())
            score_dict['runtime'] = runtime_score * runtime_weighting
            overall_score = overall_score + (runtime_score * runtime_weighting)
        # these have values in the range 0 to 1
        if contour_weighting != 0:
            contour_score = (2 - model_results['contour_weighted']) * smape_median
            score_dict['contour'] = contour_score * contour_weighting
            overall_score = overall_score + (contour_score * contour_weighting)
        if oda_weighting != 0:
            oda_score = (2 - model_results['oda_weighted']) * smape_median
            score_dict['oda'] = oda_score * oda_weighting
            overall_score = overall_score + (oda_score * oda_weighting)
        if containment_weighting != 0:
            containment_score = (
                1 + abs(prediction_interval - model_results['containment_weighted'])
            ) * smape_median
            score_dict['contaiment'] = containment_score * containment_weighting
            overall_score = overall_score + (containment_score * containment_weighting)
        if overall_score.sum() == 0:
            print(
                "something is seriously wrong with one of the input metrics, score is NaN or all 0"
            )
            overall_score = smape_score

    except Exception as e:
        raise KeyError(
            f"""Evaluation Metrics are missing and all models have failed, by an error in template or metrics.
            There are many possible causes for this, bad parameters, environment, or an unreported bug.
            Usually this means you are missing required packages for the models like fbprophet or gluonts,
            or that the models in model_list are inappropriate for your data.
            A new starting template may also help. {repr(e)}"""
        )

    if return_score_dict:
        return overall_score.astype(float), score_dict
    else:
        return overall_score.astype(float)  # need to handle complex values (!)


def generate_score_per_series(
    results_object,
    metric_weighting,
    total_validations=1,
    models_to_use=None,
):
    """Score generation on per_series_metrics for ensembles."""
    mae_weighting = metric_weighting.get('mae_weighting', 0)
    rmse_weighting = metric_weighting.get('rmse_weighting', 0)
    made_weighting = metric_weighting.get('made_weighting', 0)
    spl_weighting = metric_weighting.get('spl_weighting', 0)
    contour_weighting = metric_weighting.get('contour_weighting', 0)
    mle_weighting = metric_weighting.get('mle_weighting', 0)
    imle_weighting = metric_weighting.get('imle_weighting', 0)
    maxe_weighting = metric_weighting.get('maxe_weighting', 0)
    oda_weighting = metric_weighting.get('oda_weighting', 0)
    mqae_weighting = metric_weighting.get('mqae_weighting', 0)
    dwae_weighting = metric_weighting.get('dwae_weighting', 0)
    ewmae_weighting = metric_weighting.get('ewmae_weighting', 0)
    uwmse_weighting = metric_weighting.get('uwmse_weighting', 0)
    smoothness_weighting = metric_weighting.get('smoothness_weighting', 0)
    mate_weighting = metric_weighting.get('mate_weighting', 0)
    matse_weighting = metric_weighting.get('matse_weighting', 0)
    wasserstein_weighting = metric_weighting.get('wasserstein_weighting', 0)
    dwd_weighting = metric_weighting.get('dwd_weighting', 0)

    # there are problems when very small ~e-20 type number are in play
    mae_scaler = results_object.per_series_mae[
        results_object.per_series_mae != 0
    ].round(0)
    mae_scaler = mae_scaler[mae_scaler != 0].min().fillna(1)
    mae_score = results_object.per_series_mae / mae_scaler
    overall_score = mae_score * mae_weighting
    if rmse_weighting != 0:
        rmse_scaler = results_object.per_series_rmse[
            results_object.per_series_rmse.round(20) != 0
        ].round(20)
        rmse_scaler = rmse_scaler.min().fillna(1)
        rmse_score = results_object.per_series_rmse / rmse_scaler
        overall_score = overall_score + (rmse_score * rmse_weighting)
    if made_weighting != 0:
        made_scaler = (
            results_object.per_series_made[results_object.per_series_made != 0]
            .min()
            .fillna(1)
        )
        made_score = results_object.per_series_made / made_scaler
        # fillna but only if ALL are NaN
        # if made_score.isnull().to_numpy().all():
        #     made_score.fillna(0, inplace=True)
        overall_score = overall_score + (made_score * made_weighting)
    if mle_weighting != 0:
        mle_scaler = (
            results_object.per_series_mle[results_object.per_series_mle != 0]
            .min()
            .fillna(1)
        )
        mle_score = (
            results_object.per_series_mle / mle_scaler / 10
        )  # / 10 due to imbalance often with this
        overall_score = overall_score + (mle_score * mle_weighting)
    if imle_weighting != 0:
        imle_scaler = (
            results_object.per_series_imle[results_object.per_series_imle != 0]
            .min()
            .fillna(1)
        )
        imle_score = (
            results_object.per_series_imle / imle_scaler / 10
        )  # / 10 due to imbalance often with this
        overall_score = overall_score + (imle_score * imle_weighting)
    if maxe_weighting != 0:
        maxe_scaler = (
            results_object.per_series_maxe[results_object.per_series_maxe != 0]
            .min()
            .fillna(1)
        )
        maxe_score = results_object.per_series_maxe / maxe_scaler
        overall_score = overall_score + (maxe_score * maxe_weighting)
    if mqae_weighting != 0:
        mqae_scaler = (
            results_object.per_series_mqae[results_object.per_series_mqae != 0]
            .min()
            .fillna(1)
        )
        mqae_score = results_object.per_series_mqae / mqae_scaler
        overall_score = overall_score + (mqae_score * mqae_weighting)
    if dwae_weighting != 0:
        dwae_scaler = (
            results_object.per_series_dwae[results_object.per_series_dwae != 0]
            .min()
            .fillna(1)
        )
        dwae_score = results_object.per_series_dwae / dwae_scaler
        overall_score = overall_score + (dwae_score * dwae_weighting)
    if ewmae_weighting != 0:
        ewmae_scaler = (
            results_object.per_series_ewmae[results_object.per_series_ewmae != 0]
            .min()
            .fillna(1)
        )
        ewmae_score = results_object.per_series_ewmae / ewmae_scaler
        overall_score = overall_score + (ewmae_score * ewmae_weighting)
    if uwmse_weighting != 0:
        uwmse_scaler = (
            results_object.per_series_uwmse[results_object.per_series_uwmse != 0]
            .min()
            .fillna(1)
        )
        uwmse_score = results_object.per_series_uwmse / uwmse_scaler
        overall_score = overall_score + (uwmse_score * uwmse_weighting)
    if mate_weighting != 0:
        mate_scaler = (
            results_object.per_series_mate[results_object.per_series_mate != 0]
            .min()
            .fillna(1)
        )
        mate_score = results_object.per_series_mate / mate_scaler
        overall_score = overall_score + (mate_score * mate_weighting)
    if matse_weighting != 0:
        matse_scaler = (
            results_object.per_series_matse[results_object.per_series_matse != 0]
            .min()
            .fillna(1)
        )
        matse_score = results_object.per_series_matse / matse_scaler
        overall_score = overall_score + (matse_score * matse_weighting)
    if wasserstein_weighting != 0:
        wasserstein_scaler = (
            results_object.per_series_wasserstein[
                results_object.per_series_wasserstein != 0
            ]
            .min()
            .fillna(1)
        )
        wasserstein_score = results_object.per_series_wasserstein / wasserstein_scaler
        overall_score = overall_score + (wasserstein_score * wasserstein_weighting)
    if dwd_weighting != 0:
        dwd_scaler = (
            results_object.per_series_dwd[results_object.per_series_dwd != 0]
            .min()
            .fillna(1)
        )
        dwd_score = results_object.per_series_dwd / dwd_scaler
        overall_score = overall_score + (dwd_score * dwd_weighting)
    if smoothness_weighting != 0:
        smoothness_scaler = (
            results_object.per_series_smoothness[
                results_object.per_series_smoothness != 0
            ]
            .mean()
            .fillna(1)
        )
        smoothness_score = results_object.per_series_smoothness / smoothness_scaler
        overall_score = overall_score + (smoothness_score * smoothness_weighting)
    if spl_weighting != 0:
        spl_scaler = (
            results_object.per_series_spl[results_object.per_series_spl != 0]
            .min()
            .fillna(1)
        )
        spl_score = results_object.per_series_spl / spl_scaler
        overall_score = overall_score + (spl_score * spl_weighting)
    if contour_weighting != 0:
        contour_score = (2 - results_object.per_series_contour) * mae_score.median()
        # handle nan
        if contour_score.isna().all().all():
            print("NaN in Contour in generate_score_per_series")
            if overall_score.sum().sum() == 0:
                overall_score = mae_score
        else:
            overall_score = overall_score + (contour_score * contour_weighting)
    if oda_weighting != 0:
        oda_score = (2 - results_object.per_series_oda) * mae_score.median()
        # handle nan
        if oda_score.isna().all().all():
            print("NaN in Contour in generate_score_per_series")
            if overall_score.sum().sum() == 0:
                overall_score = mae_score
        else:
            overall_score = overall_score + (oda_score * oda_weighting)
    # select only models run through all validations
    # run_count = temp.groupby(level=0).count().mean(axis=1)
    # models_to_use = run_count[run_count >= total_validations].index.tolist()
    if models_to_use is None:
        # remove basic duplicates
        local_results = results_object.model_results.copy()
        local_results = local_results[local_results['Exceptions'].isna()]
        local_results = local_results.sort_values(
            by="TotalRuntimeSeconds", ascending=True
        )
        local_results = local_results.drop_duplicates(
            subset=['ValidationRound', 'smape', 'mae', 'spl'],
            keep="first",
        )
        run_count = local_results[['Model', 'ID']].groupby("ID").count()
        models_to_use = run_count[
            run_count['Model'] >= total_validations
        ].index.tolist()
    overall_score = overall_score[overall_score.index.isin(models_to_use)]
    # remove duplicates for each model, as can occasionally happen
    unique_id_name = "IDxxxx6777y9"  # want it unlikely to be a time series name
    overall_score.index.name = unique_id_name
    overall_score = (
        overall_score.reset_index(drop=False)
        .drop_duplicates()
        .set_index(unique_id_name)
    )
    overall_score.index.name = "ID"
    # take the average score across validations
    overall_score = overall_score.groupby(level=0).mean()
    return overall_score.astype(float)  # need to handle complex values (!)


def back_forecast(
    df,
    model_name,
    model_param_dict,
    model_transform_dict,
    future_regressor_train=None,
    n_splits: int = "auto",
    forecast_length=7,
    frequency="infer",
    prediction_interval=0.9,
    no_negatives=False,
    constraint=None,
    holiday_country="US",
    random_seed=123,
    n_jobs="auto",
    verbose=0,
    eval_periods: int = None,
    current_model_file: str = None,
    force_gc: bool = False,
    **kwargs,
):
    """Create forecasts for the historical training data, ie. backcast or back forecast.

    This actually forecasts on historical data, these are not fit model values as are often returned by other packages.
    As such, this will be slower, but more representative of real world model performance.
    There may be jumps in data between chunks.

    Args are same as for model_forecast except...
    n_splits(int): how many pieces to split data into. Pass 2 for fastest, or "auto" for best accuracy

    Returns a standard prediction object (access .forecast, .lower_forecast, .upper_forecast)

    Args:
        eval_period (int): if passed, only returns results for this many time steps of recent history
    """
    df_train_shape = df.index.shape[0]
    if eval_periods is not None:
        assert (
            eval_periods < df_train_shape
        ), "eval_periods must be less than length of history"
        fore_length = eval_periods
        eval_start = df_train_shape - eval_periods
    else:
        fore_length = df_train_shape
    max_chunk = int(ceil(fore_length / forecast_length))
    if not str(n_splits).isdigit():
        n_splits = max_chunk
    elif n_splits > max_chunk or n_splits < 2:
        n_splits = max_chunk
    else:
        n_splits = int(n_splits)

    chunk_size = fore_length / n_splits
    b_forecast, b_forecast_up, b_forecast_low = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for n in range(n_splits):
        int_idx = int(n * chunk_size)
        int_idx_1 = int((n + 1) * chunk_size)
        inner_forecast_length = int_idx_1 - int_idx
        if eval_periods is not None:
            int_idx = int_idx + eval_start
            int_idx_1 = int_idx_1 + eval_start
        # flip to forecast backwards for the first split
        if n == 0 and eval_periods is None:
            df_split = df.iloc[int_idx_1:].copy()
            df_split = df_split.iloc[::-1]
            df_split.index = df_split.index[::-1]
            result_idx = df.iloc[0:int_idx_1].index
        else:
            df_split = df.iloc[0:int_idx].copy()
        # handle appropriate regressors
        if isinstance(future_regressor_train, pd.DataFrame):
            if n == 0 and eval_periods is None:
                split_regr = future_regressor_train.reindex(df_split.index[::-1])
                split_regr_future = future_regressor_train.reindex(result_idx)
            else:
                split_regr = future_regressor_train.reindex(df_split.index)
                split_regr_future = future_regressor_train.reindex(
                    df.index[int_idx:int_idx_1]
                )
        else:
            split_regr = None
            split_regr_future = None
        try:
            df_forecast = model_forecast(
                model_name=model_name,
                model_param_dict=model_param_dict,
                model_transform_dict=model_transform_dict,
                df_train=df_split,
                forecast_length=inner_forecast_length,
                frequency=frequency,
                prediction_interval=prediction_interval,
                no_negatives=no_negatives,
                constraint=constraint,
                future_regressor_train=split_regr,
                future_regressor_forecast=split_regr_future,
                holiday_country=holiday_country,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
                current_model_file=current_model_file,
                force_gc=force_gc,
            )
            b_forecast = pd.concat([b_forecast, df_forecast.forecast])
            b_forecast_up = pd.concat([b_forecast_up, df_forecast.upper_forecast])
            b_forecast_low = pd.concat([b_forecast_low, df_forecast.lower_forecast])
            # handle index being wrong for the flipped forecast which comes first
            if n == 0 and eval_periods is None:
                b_forecast = b_forecast.iloc[::-1]
                b_forecast_up = b_forecast_up.iloc[::-1]
                b_forecast_low = b_forecast_low.iloc[::-1]
                b_forecast.index = result_idx
                b_forecast_up.index = result_idx
                b_forecast_low.index = result_idx
        except Exception as e:
            print(f"back_forecast split {n} failed with {repr(e)}")
            df_forecast = PredictionObject()
            b_df = pd.DataFrame(
                np.nan, index=df.index[int_idx:int_idx_1], columns=df.columns
            )
            b_forecast = pd.concat([b_forecast, b_df])
            b_forecast_up = pd.concat([b_forecast_up, b_df])
            b_forecast_low = pd.concat([b_forecast_low, b_df])

    # interpolation may hide errors in backcast
    df_forecast.forecast = b_forecast.interpolate('linear')
    df_forecast.upper_forecast = b_forecast_up.interpolate('linear')
    df_forecast.lower_forecast = b_forecast_low.interpolate('linear')
    return df_forecast


def remove_leading_zeros(df):
    """Accepts wide dataframe, returns dataframe with zeroes preceeding any non-zero value as NaN."""
    # keep the last row unaltered to keep metrics happier if all zeroes
    temp = df.head(df.shape[0] - 1)
    temp = temp.abs().cumsum(axis=0).replace(0, np.nan)
    temp = df[~temp.isna()]
    temp = temp.head(df.shape[0] - 1)
    return pd.concat([temp, df.tail(1)], axis=0)
