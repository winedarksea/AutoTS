"""Mid-level helper functions for AutoTS."""
import sys
import traceback as tb
import random
from math import ceil
import numpy as np
import pandas as pd
import datetime
import json
from hashlib import md5
from autots.tools.transform import RandomTransform, GeneralTransformer, shared_trans
from autots.models.ensemble import (
    EnsembleForecast,
    generalize_horizontal,
    horizontal_aliases,
    parse_horizontal,
)
from autots.tools.shaping import infer_frequency
from autots.models.model_list import (
    no_params,
    recombination_approved,
    no_shared,
    superfast,
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


def create_model_id(
    model_str: str, parameter_dict: dict = {}, transformation_dict: dict = {}
):
    """Create a hash ID which should be unique to the model parameters."""
    str_repr = (
        str(model_str) + json.dumps(parameter_dict) + json.dumps(transformation_dict)
    )
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
        return GLS(frequency=frequency, prediction_interval=prediction_interval)

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
        from autots.models.sklearn import RollingRegression

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
        from autots.models.sklearn import UnivariateRegression

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
        from autots.models.sklearn import MultivariateRegression

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
        from autots.models.sklearn import WindowRegression

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
        from autots.models.sklearn import ComponentAnalysis

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
        from autots.models.sklearn import DatepartRegression

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
    else:
        raise AttributeError(
            ("Model String '{}' not a recognized model type").format(model)
        )


def ModelPrediction(
    df_train,
    forecast_length: int,
    transformation_dict: dict,
    model_str: str,
    parameter_dict: dict,
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
    return_model: bool = False,
    random_seed: int = 2020,
    verbose: int = 0,
    n_jobs: int = None,
    current_model_file: str = None,
):
    """Feed parameters into modeling pipeline

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
    transformationStartTime = datetime.datetime.now()
    if current_model_file is not None:
        try:
            with open(f'{current_model_file}.json', 'w') as f:
                json.dump(
                    {
                        "model_name": model_str,
                        "model_param_dict": parameter_dict,
                        "model_transform_dict": transformation_dict,
                    },
                    f,
                )
        except Exception as e:
            error_msg = f"failed to write {current_model_file} with error {repr(e)}"
            try:
                with open(f'{current_model_file}_failure.json', 'w') as f:
                    f.write(error_msg)
            except Exception:
                pass
            print(error_msg)

    transformer_object = GeneralTransformer(**transformation_dict)
    df_train_transformed = transformer_object._fit(df_train)

    # make sure regressor has same length. This could be a problem if wrong size regressor is passed.
    if future_regressor_train is not None:
        future_regressor_train = future_regressor_train.reindex(df_train.index)

    transformation_runtime = datetime.datetime.now() - transformationStartTime
    # from autots.evaluator.auto_model import ModelMonster
    model = ModelMonster(
        model_str,
        parameters=parameter_dict,
        frequency=frequency,
        prediction_interval=prediction_interval,
        holiday_country=holiday_country,
        random_seed=random_seed,
        verbose=verbose,
        forecast_length=forecast_length,
        n_jobs=n_jobs,
    )
    model = model.fit(df_train_transformed, future_regressor=future_regressor_train)
    df_forecast = model.predict(
        forecast_length=forecast_length, future_regressor=future_regressor_forecast
    )

    # THIS CHECKS POINT FORECAST FOR NULLS BUT NOT UPPER/LOWER FORECASTS
    # can maybe remove this eventually and just keep the later one
    if fail_on_forecast_nan:
        if df_forecast.forecast.isnull().any().astype(int).sum() > 0:
            raise ValueError(
                "Model {} returned NaN for one or more series. fail_on_forecast_nan=True".format(
                    model_str
                )
            )

    # CHECK Forecasts are proper length!
    if df_forecast.forecast.shape[0] != forecast_length:
        raise ValueError(f"Model {model_str} returned improper forecast_length")

    transformationStartTime = datetime.datetime.now()
    # Inverse the transformations, NULL FILLED IN UPPER/LOWER ONLY
    df_forecast.forecast = pd.DataFrame(
        transformer_object.inverse_transform(df_forecast.forecast)
    )
    df_forecast.lower_forecast = pd.DataFrame(
        transformer_object.inverse_transform(df_forecast.lower_forecast, fillzero=True)
    )
    df_forecast.upper_forecast = pd.DataFrame(
        transformer_object.inverse_transform(df_forecast.upper_forecast, fillzero=True)
    )

    df_forecast.transformation_parameters = transformation_dict
    # Remove negatives if desired
    # There's df.where(df_forecast.forecast > 0, 0) or  df.clip(lower = 0), not sure which faster
    if no_negatives:
        df_forecast.lower_forecast = df_forecast.lower_forecast.clip(lower=0)
        df_forecast.forecast = df_forecast.forecast.clip(lower=0)
        df_forecast.upper_forecast = df_forecast.upper_forecast.clip(lower=0)

    if constraint is not None:
        if isinstance(constraint, dict):
            constraint_method = constraint.get("constraint_method", "quantile")
            constraint_regularization = constraint.get("constraint_regularization", 1)
            lower_constraint = constraint.get("lower_constraint", 0)
            upper_constraint = constraint.get("upper_constraint", 1)
            bounds = constraint.get("bounds", False)
        else:
            constraint_method = "stdev_min"
            lower_constraint = float(constraint)
            upper_constraint = float(constraint)
            constraint_regularization = 1
            bounds = False
        if verbose > 3:
            print(
                f"Using constraint with method: {constraint_method}, {constraint_regularization}, {lower_constraint}, {upper_constraint}, {bounds}"
            )

        df_forecast = df_forecast.apply_constraints(
            constraint_method,
            constraint_regularization,
            upper_constraint,
            lower_constraint,
            bounds,
            df_train,
        )

    transformation_runtime = transformation_runtime + (
        datetime.datetime.now() - transformationStartTime
    )
    df_forecast.transformation_runtime = transformation_runtime

    if return_model:
        df_forecast.model = model
        df_forecast.transformer = transformer_object

    # THIS CHECKS POINT FORECAST FOR NULLS BUT NOT UPPER/LOWER FORECASTS
    if fail_on_forecast_nan:
        if df_forecast.forecast.isnull().any().astype(int).sum() > 0:
            raise ValueError(
                "Model returned NaN due to a preprocessing transformer {}. fail_on_forecast_nan=True".format(
                    str(transformation_dict)
                )
            )

    return df_forecast


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
        per_series_mae=pd.DataFrame(),
        per_series_rmse=pd.DataFrame(),
        per_series_made=pd.DataFrame(),
        per_series_contour=pd.DataFrame(),
        per_series_spl=pd.DataFrame(),
        per_series_mle=pd.DataFrame(),
        per_series_imle=pd.DataFrame(),
        per_series_maxe=pd.DataFrame(),
        per_series_oda=pd.DataFrame(),
        per_series_mqae=pd.DataFrame(),
        model_count: int = 0,
    ):
        self.model_results = model_results
        self.model_count = model_count
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
        self.full_mae_ids = []
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
        self.full_mae_errors.extend(another_eval.full_mae_errors)
        self.full_pl_errors.extend(another_eval.full_pl_errors)
        self.squared_errors.extend(another_eval.squared_errors)
        self.full_mae_ids.extend(another_eval.full_mae_ids)
        self.model_count = self.model_count + another_eval.model_count
        return self

    def save(self, filename):
        """Save results to a file."""
        if '.csv' in filename:
            self.model_results.to_csv(filename, index=False)
        elif '.pickle' in filename:
            import pickle

            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("filename not .csv or .pickle")


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
    for index, value in models_to_iterate.iteritems():
        model_dict = json.loads(value)['models']
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

    Returns:
        PredictionObject (autots.PredictionObject): Prediction from AutoTS model object
    """
    full_model_created = False  # make at least one full model, horziontal only
    # handle JSON inputs of the dicts
    if isinstance(model_param_dict, str):
        model_param_dict = json.loads(model_param_dict)
    if isinstance(model_transform_dict, str):
        model_transform_dict = json.loads(model_transform_dict)
    if frequency == "infer":
        frequency = infer_frequency(df_train)
    # handle "auto" n_jobs to an integer of local count
    if n_jobs == 'auto':
        from autots.tools import cpu_count

        n_jobs = cpu_count(modifier=0.75)
        if verbose > 0:
            print(f"Auto-detected {n_jobs} cpus for n_jobs.")

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
                if verbose >= 2:
                    p = f"Ensemble {model_param_dict['model_name']} component {index + 1} of {total_ens} {row['Model']} succeeded"
                    print(p)
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

        df_forecast = ModelPrediction(
            df_train_low,
            forecast_length,
            model_transform_dict,
            model_name,
            model_param_dict,
            frequency=frequency,
            prediction_interval=prediction_interval,
            no_negatives=no_negatives,
            constraint=constraint,
            future_regressor_train=future_regressor_train,
            future_regressor_forecast=future_regressor_forecast,
            grouping_ids=grouping_ids,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            fail_on_forecast_nan=fail_on_forecast_nan,
            startTimeStamps=startTimeStamps,
            n_jobs=n_jobs,
            return_model=return_model,
            current_model_file=current_model_file,
        )

        sys.stdout.flush()
        return df_forecast


def _ps_metric(per_series_metrics, metric, model_id):
    cur_mae = per_series_metrics.loc[metric]
    cur_mae = pd.DataFrame(cur_mae).transpose()
    cur_mae.index = [model_id]
    return cur_mae


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

    Returns:
        TemplateEvalObject
    """
    best_smape = float("inf")
    template_result = TemplateEvalObject(
        per_series_mae=[],
        per_series_made=[],
        per_series_contour=[],
        per_series_rmse=[],
        per_series_spl=[],
        per_series_mle=[],
        per_series_imle=[],
        per_series_maxe=[],
        per_series_oda=[],
        per_series_mqae=[],
    )
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

    # precompute scaler to save a few miliseconds (saves very little time)
    scaler = np.nanmean(np.abs(np.diff(df_train[-100:], axis=0)), axis=0)
    fill_val = np.nanmax(scaler)
    fill_val = fill_val if fill_val > 0 else 1
    scaler[scaler == 0] = fill_val
    scaler[np.isnan(scaler)] = fill_val

    template_dict = template.to_dict('records')
    for row in template_dict:
        template_start_time = datetime.datetime.now()
        try:
            model_str = row['Model']
            parameter_dict = json.loads(row['ModelParameters'])
            transformation_dict = json.loads(row['TransformationParameters'])
            ensemble_input = row['Ensemble']
            template_result.model_count += 1
            if verbose > 0:
                if validation_round >= 1:
                    base_print = (
                        "Model Number: {} of {} with model {} for Validation {}".format(
                            str(template_result.model_count),
                            template.shape[0],
                            model_str,
                            str(validation_round),
                        )
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
                        + " with params {} and transformations {}".format(
                            json.dumps(parameter_dict),
                            json.dumps(transformation_dict),
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
            )
            if verbose > 1:
                post_memory_percent = virtual_memory().percent

            per_ts = True if 'distance' in ensemble else False
            full_mae = (
                True if "mosaic" in ensemble or "mosaic-window" in ensemble else False
            )
            model_error = df_forecast.evaluate(
                df_test,
                series_weights=weights,
                df_train=df_train,
                per_timestamp_errors=per_ts,
                full_mae_error=full_mae,
                scaler=scaler,
            )
            if validation_round >= 1 and verbose > 0:
                round_smape = model_error.avg_metrics['smape'].round(2)
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
            model_id = create_model_id(
                df_forecast.model_name,
                df_forecast.model_parameters,
                df_forecast.transformation_parameters,
            )
            result = pd.DataFrame(
                {
                    'ID': model_id,
                    'Model': df_forecast.model_name,
                    'ModelParameters': json.dumps(df_forecast.model_parameters),
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

            template_result.per_series_mae.append(
                _ps_metric(ps_metric, 'mae', model_id)
            )
            template_result.per_series_made.append(
                _ps_metric(ps_metric, 'made', model_id)
            )
            template_result.per_series_contour.append(
                _ps_metric(ps_metric, 'contour', model_id)
            )
            template_result.per_series_rmse.append(
                _ps_metric(ps_metric, 'rmse', model_id)
            )
            template_result.per_series_spl.append(
                _ps_metric(ps_metric, 'spl', model_id)
            )
            template_result.per_series_mle.append(
                _ps_metric(ps_metric, 'mle', model_id)
            )
            template_result.per_series_imle.append(
                _ps_metric(ps_metric, 'imle', model_id)
            )
            template_result.per_series_maxe.append(
                _ps_metric(ps_metric, 'maxe', model_id)
            )
            template_result.per_series_oda.append(
                _ps_metric(ps_metric, 'oda', model_id)
            )
            template_result.per_series_mqae.append(
                _ps_metric(ps_metric, 'mqae', model_id)
            )

            if 'distance' in ensemble:
                cur_smape = model_error.per_timestamp.loc['weighted_smape']
                cur_smape = pd.DataFrame(cur_smape).transpose()
                cur_smape.index = [model_id]
                template_result.per_timestamp_smape = pd.concat(
                    [template_result.per_timestamp_smape, cur_smape], axis=0
                )
            if 'mosaic' in ensemble or 'mosaic-window' in ensemble:
                template_result.full_mae_errors.extend([model_error.full_mae_errors])
                template_result.squared_errors.extend([model_error.squared_errors])
                template_result.full_pl_errors.extend(
                    [model_error.upper_pl + model_error.lower_pl]
                )
                template_result.full_mae_ids.extend([model_id])

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
                        'Template Eval Error: {} in model {}: {}'.format(
                            ''.join(tb.format_exception(None, e, e.__traceback__)),
                            template_result.model_count,
                            model_str,
                        )
                    )
                else:
                    print(
                        'Template Eval Error: {} in model {}: {}'.format(
                            (repr(e)), template_result.model_count, model_str
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
    if template_result.per_series_mae:
        template_result.per_series_mae = pd.concat(
            template_result.per_series_mae, axis=0
        )
        template_result.per_series_made = pd.concat(
            template_result.per_series_made, axis=0
        )
        template_result.per_series_contour = pd.concat(
            template_result.per_series_contour, axis=0
        )
        template_result.per_series_rmse = pd.concat(
            template_result.per_series_rmse, axis=0
        )
        template_result.per_series_spl = pd.concat(
            template_result.per_series_spl, axis=0
        )
        template_result.per_series_mle = pd.concat(
            template_result.per_series_mle, axis=0
        )
        template_result.per_series_imle = pd.concat(
            template_result.per_series_imle, axis=0
        )
        template_result.per_series_maxe = pd.concat(
            template_result.per_series_maxe, axis=0
        )
        template_result.per_series_oda = pd.concat(
            template_result.per_series_oda, axis=0
        )
        template_result.per_series_mqae = pd.concat(
            template_result.per_series_mqae, axis=0
        )
    else:
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
        if verbose > 0 and not template.empty:
            print(f"Generation {current_generation} had all new models fail")
    return template_result


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
    while len(template.index) < n:
        # this assures all models get choosen at least once
        if counter < n_models:
            model_str = model_list[counter]
        else:
            model_str = random.choices(model_list)[0]
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
        row = pd.DataFrame(
            {
                'Model': model_str,
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0,
            },
            index=[0],
        )
        template = pd.concat([template, row], axis=0, ignore_index=True)
        template.drop_duplicates(inplace=True)
        counter += 1
        if counter > (n * 3):
            break
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
):
    fir = json.loads(current_ops.iloc[0, :]['TransformationParameters'])
    cur_len = current_ops.shape[0]
    if cur_len > 1:
        # select randomly from best of data, doesn't handle lengths < 2
        top_r = np.floor((cur_len / 5) + 2)
        r_id = np.random.randint(1, top_r)
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
    arr = [fir, sec, best, r, r2]
    trans_dicts = [json.dumps(trans_dict_recomb(arr)) for _ in range(n)]
    return trans_dicts


def NewGeneticTemplate(
    model_results,
    submitted_parameters,
    sort_column: str = "smape_weighted",
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
):
    """
    Return new template given old template with model accuracies.

    Args:
        model_results (pandas.DataFrame): models that have actually been run
        submitted_paramters (pandas.DataFrame): models tried (may have returned different parameters to results)

    """
    new_template = pd.DataFrame()

    # filter existing templates
    sorted_results = model_results[model_results['Ensemble'] == 0].copy()
    sorted_results = sorted_results.sort_values(
        by=sort_column, ascending=sort_ascending, na_position='last'
    )
    sorted_results = sorted_results.drop_duplicates(subset=template_cols, keep='first')
    if str(max_per_model_class).isdigit():
        sorted_results = (
            sorted_results.sort_values(sort_column, ascending=sort_ascending)
            .groupby('Model')
            .head(max_per_model_class)
            .reset_index()
        )
    sorted_results = sorted_results.sort_values(
        by=sort_column, ascending=sort_ascending, na_position='last'
    ).head(top_n)

    # borrow = ['ComponentAnalysis']
    best = json.loads(sorted_results.iloc[0, :]['TransformationParameters'])

    for model_type in sorted_results['Model'].unique():
        if model_type in no_params:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 3
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
            )
            model_param = current_ops.iloc[0, :]['ModelParameters']
            new_row = pd.DataFrame(
                {
                    'Model': model_type,
                    'ModelParameters': model_param,
                    'TransformationParameters': trans_dicts,
                    'Ensemble': 0,
                },
                index=list(range(n)),
            )
        elif model_type in recombination_approved:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 4
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
            )
            # select the best model of this type
            fir = json.loads(current_ops.iloc[0, :]['ModelParameters'])
            cur_len = current_ops.shape[0]
            if cur_len > 1:
                # select randomly from best of data, doesn't handle lengths < 2
                top_r = np.floor((cur_len / 5) + 2)
                r_id = np.random.randint(1, top_r)
                sec = json.loads(current_ops.iloc[r_id, :]['ModelParameters'])
            else:
                sec = ModelMonster(model_type).get_new_params(method=models_mode)
            # generate new random parameters ('mutations')
            r = ModelMonster(model_type).get_new_params(method=models_mode)
            r2 = ModelMonster(model_type).get_new_params(method=models_mode)
            arr = [fir, sec, r2, r]
            model_dicts = list()
            # recombine best and random to create new generation
            for _ in range(n):
                r_sel = np.random.choice(arr, size=2, replace=False)
                a = r_sel[0]
                b = r_sel[1]
                c = dict_recombination(a, b)
                model_dicts.append(json.dumps(c))
            new_row = pd.DataFrame(
                {
                    'Model': model_type,
                    'ModelParameters': model_dicts,
                    'TransformationParameters': trans_dicts,
                    'Ensemble': 0,
                },
                index=list(range(n)),
            )
        else:
            current_ops = sorted_results[sorted_results['Model'] == model_type]
            n = 3
            trans_dicts = _trans_dicts(
                current_ops,
                best=best,
                n=n,
                transformer_list=transformer_list,
                transformer_max_depth=transformer_max_depth,
            )
            model_dicts = list()
            for _ in range(n):
                c = ModelMonster(model_type).get_new_params(method=models_mode)
                model_dicts.append(json.dumps(c))
            new_row = pd.DataFrame(
                {
                    'Model': model_type,
                    'ModelParameters': model_dicts,
                    'TransformationParameters': trans_dicts,
                    'Ensemble': 0,
                },
                index=list(range(n)),
            )
        new_template = pd.concat(
            [new_template, new_row], axis=0, ignore_index=True, sort=False
        )
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
    ).head(max_results)
    return new_template


def validation_aggregation(validation_results):
    """Aggregate a TemplateEvalObject."""
    groupby_cols = [
        'ID',
        'Model',
        'ModelParameters',
        'TransformationParameters',
        'Ensemble',
    ]
    col_aggs = {
        'Runs': 'sum',
        'smape': 'mean',
        'mae': 'mean',
        'rmse': 'mean',
        'medae': 'mean',
        'made': 'mean',
        'mage': 'mean',
        'mle': 'mean',
        'imle': 'mean',
        'spl': 'mean',
        'containment': 'mean',
        'contour': 'mean',
        'maxe': 'max',
        'oda': 'mean',
        'mqae': 'mean',
        'smape_weighted': 'mean',
        'mae_weighted': 'mean',
        'rmse_weighted': 'mean',
        'medae_weighted': 'mean',
        'made_weighted': 'mean',
        'mage_weighted': 'mean',
        'mle_weighted': 'mean',
        'imle_weighted': 'mean',
        'spl_weighted': 'mean',
        'maxe_weighted': 'max',
        'oda_weighted': 'mean',
        'mqae_weighted': 'mean',
        'containment_weighted': 'mean',
        'contour_weighted': 'mean',
        'TotalRuntimeSeconds': 'mean',
        'Score': 'mean',
    }
    col_names = validation_results.model_results.columns
    col_aggs = {x: y for x, y in col_aggs.items() if x in col_names}
    validation_results.model_results['TotalRuntimeSeconds'] = (
        validation_results.model_results['TotalRuntime'].dt.seconds + 1
    )
    validation_results.model_results = validation_results.model_results[
        pd.isnull(validation_results.model_results['Exceptions'])
    ]
    validation_results.model_results = validation_results.model_results.replace(
        [np.inf, -np.inf], np.nan
    )
    validation_results.model_results = validation_results.model_results.groupby(
        groupby_cols
    ).agg(col_aggs)
    validation_results.model_results = validation_results.model_results.reset_index(
        drop=False
    )
    return validation_results


def generate_score(
    model_results, metric_weighting: dict = {}, prediction_interval: float = 0.9
):
    """Generate score based on relative accuracies.

    SMAPE - smaller is better
    MAE - smaller is better
    RMSE -  smaller is better
    MADE - smaller is better
    MLE - smaller is better
    MAGE - smaller is better
    SPL - smaller is better
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
    mle_weighting = metric_weighting.get('mle_weighting', 0)
    imle_weighting = metric_weighting.get('imle_weighting', 0)
    maxe_weighting = metric_weighting.get('maxe_weighting', 0)
    oda_weighting = metric_weighting.get('oda_weighting', 0)
    mqae_weighting = metric_weighting.get('mqae_weighting', 0)
    # handle various runtime information records
    if 'TotalRuntimeSeconds' in model_results.columns:
        if 'TotalRuntime' in model_results.columns:
            try:
                outz = model_results['TotalRuntime'].dt.seconds
            except Exception:
                outz = model_results['TotalRuntime'].astype(float)
            model_results['TotalRuntimeSeconds'] = np.where(
                model_results['TotalRuntimeSeconds'].isna(),
                outz,
                model_results['TotalRuntimeSeconds'],
            )
        else:
            model_results['TotalRuntimeSeconds'] = np.where(
                model_results['TotalRuntimeSeconds'].isna(),
                model_results['TotalRuntimeSeconds'].max(),
                model_results['TotalRuntimeSeconds'],
            )
    else:
        model_results['TotalRuntimeSeconds'] = model_results['TotalRuntime'].dt.seconds
    # generate minimizing scores, where smaller = better accuracy
    try:
        model_results = model_results.replace([np.inf, -np.inf], np.nan)
        # not sure why there are negative SMAPE values, but make sure they get dealt with
        if model_results['smape'].min() < 0:
            model_results['smape'] = model_results['smape'].where(
                model_results['smape'] >= 0, model_results['smape'].max()
            )
        # handle NaN in scores...
        # model_results = model_results.fillna(value=model_results.max(axis=0))

        # where smaller is better, are always >=0, beware divide by zero
        smape_scaler = model_results['smape_weighted'][
            model_results['smape_weighted'] != 0
        ].min()
        smape_score = model_results['smape_weighted'] / smape_scaler
        overall_score = smape_score * smape_weighting
        if mae_weighting != 0:
            mae_scaler = model_results['mae_weighted'][
                model_results['mae_weighted'] != 0
            ].min()
            mae_score = model_results['mae_weighted'] / mae_scaler
            overall_score = overall_score + (mae_score * mae_weighting)
        if rmse_weighting > 0:
            rmse_scaler = model_results['rmse_weighted'][
                model_results['rmse_weighted'] != 0
            ].min()
            rmse_score = model_results['rmse_weighted'] / rmse_scaler
            overall_score = overall_score + (rmse_score * rmse_weighting)
        if made_weighting > 0:
            made_scaler = model_results['made_weighted'][
                model_results['made_weighted'] != 0
            ].min()
            made_score = model_results['made_weighted'] / made_scaler
            # fillna, but only if all are nan (forecast_length = 1)
            # if pd.isnull(made_score.max()):
            #     made_score.fillna(0, inplace=True)
            overall_score = overall_score + (made_score * made_weighting)
        if mage_weighting > 0:
            mage_scaler = model_results['mage_weighted'][
                model_results['mage_weighted'] != 0
            ].min()
            mage_score = model_results['mage_weighted'] / mage_scaler
            overall_score = overall_score + (mage_score * mage_weighting)
        if mle_weighting > 0:
            mle_scaler = model_results['mle_weighted'][
                model_results['mle_weighted'] != 0
            ].min()
            mle_score = model_results['mle_weighted'] / mle_scaler
            overall_score = overall_score + (mle_score * mle_weighting)
        if imle_weighting > 0:
            imle_scaler = model_results['imle_weighted'][
                model_results['imle_weighted'] != 0
            ].min()
            imle_score = model_results['imle_weighted'] / imle_scaler
            overall_score = overall_score + (imle_score * imle_weighting)
        if maxe_weighting > 0:
            maxe_scaler = model_results['maxe_weighted'][
                model_results['maxe_weighted'] != 0
            ].min()
            maxe_score = model_results['maxe_weighted'] / maxe_scaler
            overall_score = overall_score + (maxe_score * maxe_weighting)
        if mqae_weighting > 0:
            mqae_scaler = model_results['mqae_weighted'][
                model_results['mqae_weighted'] != 0
            ].min()
            mqae_score = model_results['mqae_weighted'] / mqae_scaler
            overall_score = overall_score + (mqae_score * mqae_weighting)
        if spl_weighting > 0:
            spl_scaler = model_results['spl_weighted'][
                model_results['spl_weighted'] != 0
            ].min()
            spl_score = model_results['spl_weighted'] / spl_scaler
            overall_score = overall_score + (spl_score * spl_weighting)
        if runtime_weighting > 0:
            runtime = model_results['TotalRuntimeSeconds'] + 120
            runtime_scaler = runtime[runtime != 0].min()
            runtime_score = runtime / runtime_scaler
            # this scales it into a similar range as SMAPE
            runtime_score = runtime_score * (
                smape_score.median() / runtime_score.median()
            )
            overall_score = overall_score + (runtime_score * runtime_weighting)
        # these have values in the range 0 to 1
        if contour_weighting > 0:
            contour_score = (
                2 - model_results['contour_weighted']
            ) * smape_score.median()
            overall_score = overall_score + (contour_score * contour_weighting)
        if oda_weighting > 0:
            oda_score = (2 - model_results['oda_weighted']) * smape_score.median()
            overall_score = overall_score + (oda_score * oda_weighting)
        if containment_weighting > 0:
            containment_score = (
                1 + abs(prediction_interval - model_results['containment_weighted'])
            ) * smape_score.median()
            overall_score = overall_score + (containment_score * containment_weighting)

    except Exception as e:
        raise KeyError(
            f"""Evaluation Metrics are missing and all models have failed, by an error in template or metrics.
            There are many possible causes for this, bad parameters, environment, or an unreported bug.
            Usually this means you are missing required packages for the models like fbprophet or gluonts,
            or that the models in model_list are inappropriate for your data.
            A new starting template may also help. {repr(e)}"""
        )

    return overall_score


def generate_score_per_series(results_object, metric_weighting, total_validations):
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
    if sum([mae_weighting, rmse_weighting, contour_weighting, spl_weighting]) == 0:
        mae_weighting = 1

    mae_scaler = (
        results_object.per_series_mae[results_object.per_series_mae != 0]
        .min()
        .fillna(1)
    )
    mae_score = results_object.per_series_mae / mae_scaler
    overall_score = mae_score * mae_weighting
    if rmse_weighting > 0:
        rmse_scaler = (
            results_object.per_series_rmse[results_object.per_series_rmse != 0]
            .min()
            .fillna(1)
        )
        rmse_score = results_object.per_series_rmse / rmse_scaler
        overall_score = overall_score + (rmse_score * rmse_weighting)
    if made_weighting > 0:
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
    if mle_weighting > 0:
        mle_scaler = (
            results_object.per_series_mle[results_object.per_series_mle != 0]
            .min()
            .fillna(1)
        )
        mle_score = results_object.per_series_mle / mle_scaler
        overall_score = overall_score + (mle_score * mle_weighting)
    if imle_weighting > 0:
        imle_scaler = (
            results_object.per_series_imle[results_object.per_series_imle != 0]
            .min()
            .fillna(1)
        )
        imle_score = results_object.per_series_imle / imle_scaler
        overall_score = overall_score + (imle_score * imle_weighting)
    if maxe_weighting > 0:
        maxe_scaler = (
            results_object.per_series_maxe[results_object.per_series_maxe != 0]
            .min()
            .fillna(1)
        )
        maxe_score = results_object.per_series_maxe / maxe_scaler
        overall_score = overall_score + (maxe_score * maxe_weighting)
    if mqae_weighting > 0:
        mqae_scaler = (
            results_object.per_series_mqae[results_object.per_series_mqae != 0]
            .min()
            .fillna(1)
        )
        mqae_score = results_object.per_series_mqae / mqae_scaler
        overall_score = overall_score + (mqae_score * mqae_weighting)
    if spl_weighting > 0:
        spl_scaler = (
            results_object.per_series_spl[results_object.per_series_spl != 0]
            .min()
            .fillna(1)
        )
        spl_score = results_object.per_series_spl / spl_scaler
        overall_score = overall_score + (spl_score * spl_weighting)
    if contour_weighting > 0:
        contour_score = (2 - results_object.per_series_contour) * mae_score.median()
        # handle nan
        if contour_score.isna().all().all():
            print("NaN in Contour in generate_score_per_series")
            if overall_score.sum().sum() == 0:
                overall_score = mae_score
        else:
            overall_score = overall_score + (contour_score * contour_weighting)
    if oda_weighting > 0:
        oda_score = (2 - results_object.per_series_oda) * mae_score.median()
        # handle nan
        if oda_score.isna().all().all():
            print("NaN in Contour in generate_score_per_series")
            if overall_score.sum().sum() == 0:
                overall_score = mae_score
        else:
            overall_score = overall_score + (oda_score * oda_weighting)
    # remove basic duplicates
    local_results = results_object.model_results.copy()
    local_results = local_results[local_results['Exceptions'].isna()]
    local_results = local_results.sort_values(by="TotalRuntimeSeconds", ascending=True)
    local_results.drop_duplicates(
        subset=['ValidationRound', 'smape', 'mae', 'spl'], keep="first", inplace=True
    )
    # select only models run through all validations
    # run_count = temp.groupby(level=0).count().mean(axis=1)
    # models_to_use = run_count[run_count >= total_validations].index.tolist()
    run_count = local_results[['Model', 'ID']].groupby("ID").count()
    models_to_use = run_count[run_count['Model'] >= total_validations].index.tolist()
    overall_score = overall_score[overall_score.index.isin(models_to_use)]
    # take the average score across validations
    overall_score = overall_score.groupby(level=0).mean()
    return overall_score


def back_forecast(
    df,
    model_name,
    model_param_dict,
    model_transform_dict,
    future_regressor_train=None,
    n_splits: int = "auto",
    forecast_length=14,
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
            b_df = pd.DataFrame(
                np.nan, index=df.index[int_idx:int_idx_1], columns=df.columns
            )
            b_forecast = pd.concat([b_forecast, b_df])
            b_forecast_up = pd.concat([b_forecast_up, b_df])
            b_forecast_low = pd.concat([b_forecast_low, b_df])

    df_forecast.forecast = b_forecast
    df_forecast.upper_forecast = b_forecast_up
    df_forecast.lower_forecast = b_forecast_low
    return df_forecast


def remove_leading_zeros(df):
    """Accepts wide dataframe, returns dataframe with zeroes preceeding any non-zero value as NaN."""
    # keep the last row unaltered to keep metrics happier if all zeroes
    temp = df.head(df.shape[0] - 1)
    temp = temp.abs().cumsum(axis=0).replace(0, np.nan)
    temp = df[~temp.isna()]
    temp = temp.head(df.shape[0] - 1)
    return pd.concat([temp, df.tail(1)], axis=0)
