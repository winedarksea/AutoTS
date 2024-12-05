"""Tools for generating and forecasting with ensembles of models."""

import re
import datetime
import numpy as np
import pandas as pd
import json
from autots.models.base import PredictionObject
from autots.models.model_list import no_shared
from autots.tools.impute import fill_median
from autots.models.sklearn import retrieve_classifier
from autots.tools.profile import profile_time_series, summarize_series


horizontal_aliases = ['horizontal', 'probabilistic', 'horizontal-max', 'horizontal-min']
# try to include all types in here
full_ensemble_test_list = [
    'simple',
    "distance",
    "horizontal",
    "horizontal-max",
    "mosaic",
    'mosaic-window',
    'mosaic-crosshair',
    'horizontal-min-3',
    "subsample",
    "mlensemble",
    "mosaic-weighted-crosshair-0-10",
    "mosaic-weighted-crosshair_lite-0-10",
    "mosaic-weighted-0-horizontal",
    "mosaic-mae-0-40",
    "mosaic-weighted-0-40",
    "mosaic-spl-3-10",  # this one in particular hard-coded for testing
    "mosaic-mae-profile-0-36",
    "mosaic-weighted-median",
    "mosaic-mae-median-0-30",
    "mosaic-mae-filtered-0-30",
    "mosaic-mae-unpredictability_adjusted-0-30",
    "mosaic-weighted-profile-median-filtered-unpredictability_adjusted-crosshair_lite-3-30",  # maxxed out config
]


def mosaic_or_horizontal(all_series: dict):
    """Take a mosaic or horizontal model and return series or models.

    Args:
        all_series (dict): dict of series: model (or list of models)
    """
    first_value = all_series[next(iter(all_series))]
    if isinstance(first_value, dict):
        return "mosaic"
    else:
        return "horizontal"


def parse_horizontal(all_series: dict, model_id: str = None, series_id: str = None):
    """Take a mosaic or horizontal model and return series or models.

    Args:
        all_series (dict): dict of series: model (or list of models)
        model_id (str): name of model to find series for
        series_id (str): name of series to find models for

    Returns:
        list
    """
    if model_id is None and series_id is None:
        raise ValueError(
            "either series_id or model_id must be specified in parse_horizontal."
        )

    if mosaic_or_horizontal(all_series) == 'mosaic':
        if model_id is not None:
            return [ser for ser, mod in all_series.items() if model_id in mod.values()]
        else:
            return list(set(all_series[series_id].values()))
    else:
        if model_id is not None:
            return [ser for ser, mod in all_series.items() if mod == model_id]
        else:
            # list(set([mod for ser, mod in all_series.items() if ser == series_id]))
            return [all_series[series_id]]


def parse_mosaic(ensemble):
    if ensemble == "mosaic":
        return {
            'metric': 'mae',
            'smoothing_window': None,
            'crosshair': False,
            'n_models': None,
            'profiled': False,
            'filtered': False,
            'unpredictability_adjusted': False,
        }
    elif ensemble in ["mosaic_crosshair", 'mosaic-crosshair']:
        return {
            'metric': 'mae',
            'smoothing_window': None,
            'crosshair': True,
            'n_models': None,
            'profiled': False,
            'filtered': False,
            'unpredictability_adjusted': False,
        }
    elif ensemble in ["mosaic_window", "mosaic-window"]:
        return {
            'metric': 'mae',
            'smoothing_window': 7,
            'crosshair': False,
            'n_models': None,
            'profiled': False,
            'filtered': False,
            'unpredictability_adjusted': False,
        }
    else:
        # mosaic-metric-crosshair-window-n_models
        split = ensemble.split("-")
        len_split = len(split)
        if len_split > 1:
            metric = split[1]
            metric = (
                metric if metric in ['mae', 'spl', 'pl', 'se', 'weighted'] else 'mae'
            )
            penultimate = split[-2]
        else:
            metric = 'mae'
            penultimate = None
        # extract the end which is n_models
        end_split = split[-1]
        end_split_ls = re.findall(r'\d+', end_split)
        if len(end_split_ls) >= 1:
            n_models = int(end_split_ls[0])
        elif end_split == "horizontal":
            n_models = "horizontal"
        else:
            n_models = None
        # zero is considered None here
        if n_models == 0:
            n_models = None
        # extra the penultimate number which is smoothing_window
        penultimate = re.findall(r'\d+', penultimate)
        if len(penultimate) >= 1:
            swindow = int(penultimate[0])
        else:
            swindow = None
        # zero is considered None here
        if swindow == 0:
            swindow = None
        if "crosshair_lite" in ensemble:
            crosshair = 'crosshair_lite'
        elif 'crosshair' in ensemble:
            crosshair = True
        else:
            crosshair = False
        return {
            'metric': metric,
            'smoothing_window': swindow,
            'crosshair': crosshair,
            'n_models': n_models,
            "profiled": "profile" in ensemble,
            "filtered": "filtered" in ensemble,
            "unpredictability_adjusted": "unpredictability_adjusted" in ensemble,
        }


# just a list of horizontal types in general
h_ens_list = [
    'horizontal',
    'probabilistic',
    'hdist',
    "mosaic",
    'mosaic-window',
    'mosaic_window',
    'mosaic_crosshair',
    'mosaic-crosshair',
    'horizontal-max',
    'horizontal-min',
]
mosaic_list = [
    'mosaic',
    'mosaic-window',
    "mosaic_window",
    'mosaic_crosshair',
    "mosaic-crosshair",
]


def is_horizontal(ensemble_list):
    # check for exact matches
    check = any(x in ensemble_list for x in h_ens_list)
    # check for -N matches
    check2 = any("horizontal" in x for x in ensemble_list)
    return check or check2


def is_mosaic(ensemble_list):
    check = any(x in ensemble_list for x in mosaic_list)
    check2 = any("mosaic" in x for x in ensemble_list)
    return check or check2


def BestNEnsemble(
    ensemble_params,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime: dict,
    prediction_interval: float = 0.9,
):
    """Generate mean forecast for ensemble of models.

    model_weights and point_methods other than 'mean' are incompatible

    Args:
        ensemble_params (dict): BestN ensemble param dict
            should have "model_weights": {model_id: weight} where 1 is default weight per model
        forecasts (dict): {forecast_id: forecast dataframe} for all models
            same for lower_forecasts, upper_forecasts
        forecast_runtime (dict): dictionary of {forecast_id: timedelta of runtime}
        prediction_interval (float): metadata on interval
    """
    startTime = datetime.datetime.now()
    forecast_keys = list(forecasts.keys())
    model_weights = dict(ensemble_params.get("model_weights", {}))
    point_method = ensemble_params.get("point_method", "mean")
    ensemble_params['model_weights'] = model_weights
    ensemble_params['point_method'] = point_method
    ensemble_params['models'] = {
        k: v
        for k, v in dict(ensemble_params.get('models')).items()
        if k in forecast_keys
    }

    model_count = len(forecast_keys)
    if model_count < 1:
        raise ValueError("BestN failed, no component models available.")
    sample_df = next(iter(forecasts.values()))
    columnz = sample_df.columns
    indices = sample_df.index

    # this is expected to have to handle NaN
    if point_method in ["median", "midhinge"]:
        forecast_array = np.array(
            [x.values.reshape(1, -1) if x.ndim == 1 else x for x in forecasts.values()]
        )
        l_forecast_array = np.array(
            [
                x.values.reshape(1, -1) if x.ndim == 1 else x
                for x in lower_forecasts.values()
            ]
        )
        u_forecast_array = np.array(
            [
                x.values.reshape(1, -1) if x.ndim == 1 else x
                for x in upper_forecasts.values()
            ]
        )
        # checks only upper and middle, assuming lower follows others in NaN
        # because the nanmedian and nanquantile are much slower than non na version
        nan_flag = np.isnan(np.min(forecast_array)) or np.isnan(
            np.min(u_forecast_array)
        )
        if point_method == "midhinge":
            if nan_flag:
                ens_df = (
                    np.nanquantile(forecast_array, q=0.25, axis=0)
                    + np.nanquantile(forecast_array, q=0.75, axis=0)
                ) / 2
                ens_df_lower = (
                    np.nanquantile(l_forecast_array, q=0.25, axis=0)
                    + np.nanquantile(l_forecast_array, q=0.75, axis=0)
                ) / 2
                ens_df_upper = (
                    np.nanquantile(u_forecast_array, q=0.25, axis=0)
                    + np.nanquantile(u_forecast_array, q=0.75, axis=0)
                ) / 2
            else:
                ens_df = (
                    np.quantile(forecast_array, q=0.25, axis=0)
                    + np.quantile(forecast_array, q=0.75, axis=0)
                ) / 2
                ens_df_lower = (
                    np.quantile(l_forecast_array, q=0.25, axis=0)
                    + np.quantile(l_forecast_array, q=0.75, axis=0)
                ) / 2
                ens_df_upper = (
                    np.quantile(u_forecast_array, q=0.25, axis=0)
                    + np.quantile(u_forecast_array, q=0.75, axis=0)
                ) / 2
        else:
            if nan_flag:
                ens_df = np.nanmedian(forecast_array, axis=0)
                ens_df_lower = np.nanmedian(l_forecast_array, axis=0)
                ens_df_upper = np.nanmedian(u_forecast_array, axis=0)
            else:
                ens_df = np.median(forecast_array, axis=0)
                ens_df_lower = np.median(l_forecast_array, axis=0)
                ens_df_upper = np.median(u_forecast_array, axis=0)

        ens_df = pd.DataFrame(ens_df, index=indices, columns=columnz)
        ens_df_lower = pd.DataFrame(ens_df_lower, index=indices, columns=columnz)
        ens_df_upper = pd.DataFrame(ens_df_upper, index=indices, columns=columnz)
    else:
        # these might be faster but the current method works fine
        # np.average(forecast_array, axis=0, weights=model_weights.values())
        # np.average(l_forecast_array, axis=0, weights=model_weights.values())
        # np.average(u_forecast_array, axis=0, weights=model_weights.values())

        model_divisor = 0
        ens_df = pd.DataFrame(0, index=indices, columns=columnz)
        ens_df_lower = pd.DataFrame(0, index=indices, columns=columnz)
        ens_df_upper = pd.DataFrame(0, index=indices, columns=columnz)
        for idx, x in forecasts.items():
            current_weight = float(model_weights.get(idx, 1))
            ens_df = ens_df + (x * current_weight)
            # also .get(idx, 0)
            ens_df_lower = ens_df_lower + (lower_forecasts[idx] * current_weight)
            ens_df_upper = ens_df_upper + (upper_forecasts[idx] * current_weight)
            model_divisor = model_divisor + current_weight

        ens_df = ens_df / model_divisor
        ens_df_lower = ens_df_lower / model_divisor
        ens_df_upper = ens_df_upper / model_divisor

    ens_runtime = datetime.timedelta(0)
    for x in forecasts_runtime.values():
        ens_runtime = ens_runtime + x

    ens_result = PredictionObject(
        model_name="Ensemble",
        forecast_length=len(ens_df.index),
        forecast_index=ens_df.index,
        forecast_columns=ens_df.columns,
        lower_forecast=ens_df_lower,
        forecast=ens_df,
        upper_forecast=ens_df_upper,
        prediction_interval=prediction_interval,
        predict_runtime=datetime.datetime.now() - startTime,
        fit_runtime=ens_runtime,
        model_parameters=ensemble_params,
    )
    return ens_result


def DistEnsemble(
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
):
    """Generate forecast for distance ensemble."""
    # handle that the inputs are now dictionaries
    forecasts = list(forecasts.values())
    lower_forecasts = list(lower_forecasts.values())
    upper_forecasts = list(upper_forecasts.values())
    forecasts_runtime = list(forecasts_runtime.values())

    first_model_index = forecasts_list.index(ensemble_params['FirstModel'])
    second_model_index = forecasts_list.index(ensemble_params['SecondModel'])
    forecast_length = forecasts[0].shape[0]
    dis_frac = ensemble_params['dis_frac']
    first_bit = int(np.ceil(forecast_length * dis_frac))
    second_bit = int(np.floor(forecast_length * (1 - dis_frac)))

    ens_df = pd.concat(
        [
            forecasts[first_model_index].head(first_bit),
            forecasts[second_model_index].tail(second_bit),
        ]
    )
    ens_df_lower = pd.concat(
        [
            lower_forecasts[first_model_index].head(first_bit),
            lower_forecasts[second_model_index].tail(second_bit),
        ]
    )
    ens_df_upper = pd.concat(
        [
            upper_forecasts[first_model_index].head(first_bit),
            upper_forecasts[second_model_index].tail(second_bit),
        ]
    )

    id_list = list(ensemble_params['models'].keys())
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in id_list]

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in model_indexes:
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result_obj = PredictionObject(
        model_name="Ensemble",
        forecast_length=len(ens_df.index),
        forecast_index=ens_df.index,
        forecast_columns=ens_df.columns,
        lower_forecast=ens_df_lower,
        forecast=ens_df,
        upper_forecast=ens_df_upper,
        prediction_interval=prediction_interval,
        predict_runtime=datetime.timedelta(0),
        fit_runtime=ens_runtime,
        model_parameters=ensemble_params,
    )
    return ens_result_obj


def horizontal_xy(df_train, known):
    """Construct X, Y, X_predict features for generalization."""
    columnz = df_train.columns.tolist()
    X = summarize_series(df_train).transpose()
    X = fill_median(X)
    known_l = list(known.keys())
    unknown = list(set(columnz) - set(known_l))
    Xt = X.loc[known_l]
    Xf = X.loc[unknown]
    Y = np.array(list(known.values()))
    return Xt, Y, Xf


def horizontal_classifier(
    df_train, known: dict, method: str = "whatever", classifier_params=None
):
    """
    CLassify unknown series with the appropriate model for horizontal ensembling.

    Args:
        df_train (pandas.DataFrame): historical data about the series. Columns = series_ids.
        known (dict): dict of series_id: classifier outcome including some but not all series in df_train.

    Returns:
        dict.

    """
    # known = {'EXUSEU': 'xx1', 'MCOILWTICO': 'xx2', 'CSUSHPISA': 'xx3'}
    Xt, Y, Xf = horizontal_xy(df_train, known)

    if classifier_params is None:
        # found using FLAML
        n_neighbors = 5 if Xt.shape[0] > 5 else Xt.shape[0]
        classifier_params = {
            "model": 'KNN',
            "model_params": {'n_neighbors': n_neighbors},
        }
        # newer, but don't like as much
        # RandomForest {'n_estimators': 69, 'max_features': 0.5418860350847585, 'max_leaves': 439, 'criterion': 'gini'}

    clf = retrieve_classifier(
        regression_model=classifier_params,
        verbose=0,
        verbose_bool=False,
        random_seed=2023,
        multioutput=False,
        n_jobs=1,
    )
    clf.fit(Xt, Y)
    result = clf.predict(Xf)
    result_d = dict(zip(Xf.index.tolist(), result))
    # since this only has estimates, overwrite with known that includes more
    final = {**result_d, **known}
    # temp = pd.DataFrame({'series': list(final.keys()), 'model': list(final.values())})
    # temp2 = temp.merge(X, left_on='series', right_index=True)
    return final


def mosaic_xy(df_train, known):
    known.index.name = "forecast_period"
    upload = pd.melt(
        known,
        var_name="series_id",
        value_name="model_id",
        ignore_index=False,
    ).reset_index(drop=False)
    upload['forecast_period'] = upload['forecast_period'].astype(int)
    missing_cols = df_train.columns[
        ~df_train.columns.isin(upload['series_id'].unique())
    ]
    if not missing_cols.empty:
        forecast_p = np.arange(upload['forecast_period'].max() + 1)
        p_full = np.tile(forecast_p, len(missing_cols))
        missing_rows = pd.DataFrame(
            {
                'forecast_period': p_full,
                'series_id': np.repeat(missing_cols.values, len(forecast_p)),
                'model_id': np.nan,
            },
            index=None if len(p_full) > 1 else [0],
        )
        upload = pd.concat([upload, missing_rows])
    X = (
        summarize_series(df_train)
        .transpose()
        .merge(upload, left_index=True, right_on="series_id")
    )
    X = X.set_index("series_id").replace(
        [np.inf, -np.inf], 0
    )  # .drop(columns=['series_id'], inplace=True)
    to_predict = fill_median(X[X['model_id'].isna()].drop(columns=['model_id']))
    X = X[~X['model_id'].isna()]
    Y = X['model_id']
    Xf = X.drop(columns=['model_id'])
    Xf = fill_median(Xf)
    return X, Xf, Y, to_predict


def mosaic_classifier(df_train, known, classifier_params=None):
    """CLassify unknown series with the appropriate model for mosaic ensembles."""
    if classifier_params is None:
        # found using FLAML
        classifier_params = {
            "model": 'RandomForest',
            "model_params": {
                'n_estimators': 169,
                'max_features': 0.25736,
                'max_leaf_nodes': 126,
                'criterion': 'gini',
            },
        }
        # slightly newer, on a mosaic-weighted-0-40
        classifier_params = {
            "model": 'ExtraTrees',
            "model_params": {
                'n_estimators': 62,
                'max_features': 0.181116,
                'max_leaf_nodes': 261,
                'criterion': 'entropy',
            },
        }

    X, Xf, Y, to_predict = mosaic_xy(df_train, known)

    clf = retrieve_classifier(
        regression_model=classifier_params,
        verbose=0,
        verbose_bool=False,
        random_seed=2023,
        multioutput=False,
        n_jobs=1,
    )
    clf.fit(Xf, Y)
    predicted = clf.predict(to_predict)
    result = pd.concat(
        [to_predict.reset_index(drop=False), pd.Series(predicted, name="model_id")],
        axis=1,
    )
    cols_needed = ['model_id', 'series_id', 'forecast_period']
    final = pd.concat(
        [X.reset_index(drop=False)[cols_needed], result[cols_needed]], sort=True, axis=0
    )
    final['forecast_period'] = final['forecast_period'].astype(str)
    final = final.pivot(values="model_id", columns="series_id", index="forecast_period")
    try:
        final = final[df_train.columns]
        if final.isna().to_numpy().sum() > 0:
            raise KeyError("NaN in mosaic generalization")
    except KeyError as e:
        raise ValueError(
            f"mosaic_classifier failed to generalize for all columns: {repr(e)}"
        )
    return final


def generalize_horizontal(
    df_train, known_matches: dict, available_models: list, full_models: list = None
):
    """generalize a horizontal model trained on a subset of all series

    Args:
        df_train (pd.DataFrame): time series data
        known_matches (dict): series:model dictionary for some to all series
        available_models (dict): list of models actually available
        full_models (dict): models that are available for every single series
    """
    org_idx = df_train.columns
    org_list = org_idx.tolist()
    # remove any unnecessary series
    known_matches = {ser: mod for ser, mod in known_matches.items() if ser in org_list}
    # here split for mosaic or horizontal
    if mosaic_or_horizontal(known_matches) == "mosaic":
        # make it a dataframe
        mosaicy = pd.DataFrame.from_dict(known_matches)
        # remove unavailable models
        mosaicy = pd.DataFrame(mosaicy[mosaicy.isin(available_models)])
        # so we can fill some missing by just using a forward fill, should be good enough
        mosaicy = mosaicy.ffill(limit=5)
        mosaicy = mosaicy.bfill(limit=5)
        if mosaicy.isna().any().any() or mosaicy.shape[1] != df_train.shape[1]:
            if full_models is not None:
                k2 = pd.DataFrame(mosaicy[mosaicy.isin(full_models)])
            else:
                k2 = mosaicy.copy()
            final = mosaic_classifier(df_train, known=k2)
            return final.to_dict()
        else:
            return mosaicy.to_dict()

    else:
        # remove any unavailable models
        k = {ser: mod for ser, mod in known_matches.items() if mod in available_models}
        # check if any series are missing from model list
        if not k:
            raise ValueError("Horizontal template has no models matching this data!")
        # test if generalization is needed
        if len(set(org_list) - set(list(k.keys()))) > 0:
            # filter down to only models available for all
            # print(f"Models not available: {[ser for ser, mod in known_matches.items() if mod not in available_models]}")
            # print(f"Series not available: {[ser for ser in df_train.columns if ser not in list(known_matches.keys())]}")
            if full_models is not None:
                k2 = {ser: mod for ser, mod in k.items() if mod in full_models}
            else:
                k2 = k.copy()
            all_series_part = horizontal_classifier(df_train, k2)
            # since this only has "full", overwrite with known that includes more
            all_series = {**all_series_part, **k}
        else:
            all_series = known_matches
        return all_series


def HorizontalEnsemble(
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
    df_train=None,
    prematched_series: dict = None,
):
    """Generate forecast for per_series ensembling."""
    startTime = datetime.datetime.now()
    # this is meant to fill in any failures
    available_models = [mod for mod, fcs in forecasts.items() if fcs.shape[0] > 0]
    train_size = df_train.shape
    # print(f"running inner generalization with training size: {train_size}")
    full_models = [
        mod for mod, fcs in forecasts.items() if fcs.shape[1] == train_size[1]
    ]
    if not full_models:
        print("No full models available for horizontal generalization!")
        full_models = available_models  # hope it doesn't need to fill
    # print(f"FULLMODEL {len(full_models)}: {full_models}")
    if prematched_series is None:
        prematched_series = ensemble_params['series']
    all_series = generalize_horizontal(
        df_train, prematched_series, available_models, full_models
    )
    ensemble_params['series'] = all_series
    # print(f"ALLSERIES {len(all_series.keys())}: {all_series}")

    org_idx = df_train.columns

    forelist, forelist_l, forelist_u = [], [], []
    """
    # forelist = [forecasts.get(mod_id)[series] for series, mod_id in all_series.items()]
    for series, mod_id in all_series.items():
        forelist.append(forecasts[mod_id][series])
        forelist_l.append(lower_forecasts[mod_id][series])
        forelist_u.append(upper_forecasts[mod_id][series])
    """
    # this should be faster if models are reused, but columns won't be in order
    final = pd.DataFrame.from_dict({"0": all_series})
    final = final.reset_index(drop=False).groupby("0").agg(list)
    for row in final.itertuples():
        forelist.append(forecasts[row[0]][row[1]])
        forelist_l.append(lower_forecasts[row[0]][row[1]])
        forelist_u.append(upper_forecasts[row[0]][row[1]])
    # make sure columns align to original
    # reindexing took 170 microseconds for 500 columns so shouldn't be a concern
    forecast_df = pd.concat(forelist, axis=1).reindex(columns=org_idx)
    l_forecast_df = pd.concat(forelist_l, axis=1).reindex(columns=org_idx)
    u_forecast_df = pd.concat(forelist_u, axis=1).reindex(columns=org_idx)

    # combine runtimes
    try:
        ens_runtime = sum(list(forecasts_runtime.values()), datetime.timedelta())
    except Exception:
        ens_runtime = datetime.timedelta(0)

    ens_result = PredictionObject(
        model_name="Ensemble",
        forecast_length=len(forecast_df.index),
        forecast_index=forecast_df.index,
        forecast_columns=forecast_df.columns,
        lower_forecast=l_forecast_df,
        forecast=forecast_df,
        upper_forecast=u_forecast_df,
        prediction_interval=prediction_interval,
        predict_runtime=datetime.datetime.now() - startTime,
        fit_runtime=ens_runtime,
        model_parameters=ensemble_params,
    )
    return ens_result


def HDistEnsemble(
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
):
    """Generate forecast for per_series per distance ensembling."""
    # handle that the inputs are now dictionaries
    forecasts = list(forecasts.values())
    lower_forecasts = list(lower_forecasts.values())
    upper_forecasts = list(upper_forecasts.values())
    forecasts_runtime = list(forecasts_runtime.values())

    id_list = list(ensemble_params['models'].keys())
    mod_dic = {x: idx for idx, x in enumerate(forecasts_list) if x in id_list}
    forecast_length = forecasts[0].shape[0]
    dist_n = int(np.ceil(ensemble_params['dis_frac'] * forecast_length))
    dist_last = forecast_length - dist_n

    forecast_df, u_forecast_df, l_forecast_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for series, mod_id in ensemble_params['series1'].items():
        l_idx = mod_dic[mod_id]
        try:
            c_fore = forecasts[l_idx][series]
            forecast_df = pd.concat([forecast_df, c_fore], axis=1)
        except Exception as e:
            repr(e)
            print(forecasts[l_idx].columns)
            print(forecasts[l_idx].head())
        # upper
        c_fore = upper_forecasts[l_idx][series]
        u_forecast_df = pd.concat([u_forecast_df, c_fore], axis=1)
        # lower
        c_fore = lower_forecasts[l_idx][series]
        l_forecast_df = pd.concat([l_forecast_df, c_fore], axis=1)

    forecast_df2, u_forecast_df2, l_forecast_df2 = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for series, mod_id in ensemble_params['series2'].items():
        l_idx = mod_dic[mod_id]
        try:
            c_fore = forecasts[l_idx][series]
            forecast_df2 = pd.concat([forecast_df2, c_fore], axis=1)
        except Exception as e:
            repr(e)
            print(forecasts[l_idx].columns)
            print(forecasts[l_idx].head())
        # upper
        c_fore = upper_forecasts[l_idx][series]
        u_forecast_df2 = pd.concat([u_forecast_df2, c_fore], axis=1)
        # lower
        c_fore = lower_forecasts[l_idx][series]
        l_forecast_df2 = pd.concat([l_forecast_df2, c_fore], axis=1)

    forecast_df = pd.concat(
        [forecast_df.head(dist_n), forecast_df2.tail(dist_last)], axis=0
    )
    u_forecast_df = pd.concat(
        [u_forecast_df.head(dist_n), u_forecast_df2.tail(dist_last)], axis=0
    )
    l_forecast_df = pd.concat(
        [l_forecast_df.head(dist_n), l_forecast_df2.tail(dist_last)], axis=0
    )

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in list(mod_dic.values()):
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result = PredictionObject(
        model_name="Ensemble",
        forecast_length=len(forecast_df.index),
        forecast_index=forecast_df.index,
        forecast_columns=forecast_df.columns,
        lower_forecast=l_forecast_df,
        forecast=forecast_df,
        upper_forecast=u_forecast_df,
        prediction_interval=prediction_interval,
        predict_runtime=datetime.timedelta(0),
        fit_runtime=ens_runtime,
        model_parameters=ensemble_params,
    )
    return ens_result


def EnsembleForecast(
    ensemble_str,
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
    df_train=None,
    prematched_series: dict = None,
):
    """Return PredictionObject for given ensemble method."""
    ens_model_name = ensemble_params['model_name'].lower().strip()

    s3list = ['best3', 'best3horizontal', 'bestn']
    if ens_model_name in s3list:
        ens_forecast = BestNEnsemble(
            ensemble_params,
            forecasts,
            lower_forecasts,
            upper_forecasts,
            forecasts_runtime,
            prediction_interval,
        )
        return ens_forecast

    elif ens_model_name == 'dist':
        ens_forecast = DistEnsemble(
            ensemble_params,
            forecasts_list,
            forecasts,
            lower_forecasts,
            upper_forecasts,
            forecasts_runtime,
            prediction_interval,
        )
        return ens_forecast

    elif ens_model_name in horizontal_aliases:
        ens_forecast = HorizontalEnsemble(
            ensemble_params,
            forecasts_list,
            forecasts,
            lower_forecasts,
            upper_forecasts,
            forecasts_runtime,
            prediction_interval,
            df_train=df_train,
            prematched_series=prematched_series,
        )
        return ens_forecast

    elif ens_model_name == "mosaic":
        ens_forecast = MosaicEnsemble(
            ensemble_params,
            forecasts_list,
            forecasts=forecasts,
            lower_forecasts=lower_forecasts,
            upper_forecasts=upper_forecasts,
            forecasts_runtime=forecasts_runtime,
            prediction_interval=prediction_interval,
            df_train=df_train,
            prematched_series=prematched_series,
        )
        return ens_forecast

    elif ens_model_name == 'hdist':
        ens_forecast = HDistEnsemble(
            ensemble_params,
            forecasts_list,
            forecasts,
            lower_forecasts,
            upper_forecasts,
            forecasts_runtime,
            prediction_interval,
        )
        return ens_forecast

    else:
        raise ValueError("Ensemble model type not recognized.")


def _generate_distance_ensemble(dis_frac, forecast_length, initial_results):
    """Constructs a distance ensemble dictionary."""
    dis_frac = 0.5
    first_bit = int(np.ceil(forecast_length * dis_frac))
    last_bit = int(np.floor(forecast_length * (1 - dis_frac)))
    not_ens_list = initial_results.model_results[
        initial_results.model_results['Ensemble'] == 0
    ]['ID'].tolist()
    ens_per_ts = initial_results.per_timestamp_smape[
        initial_results.per_timestamp_smape.index.isin(not_ens_list)
    ]
    first_model = ens_per_ts.iloc[:, 0:first_bit].mean(axis=1).idxmin()
    last_model = (
        ens_per_ts.iloc[:, first_bit : (last_bit + first_bit)].mean(axis=1).idxmin()
    )
    ensemble_models = {}
    best3 = (
        initial_results.model_results[
            initial_results.model_results['ID'].isin([first_model, last_model])
        ]
        .drop_duplicates(
            subset=['Model', 'ModelParameters', 'TransformationParameters']
        )
        .set_index("ID")[['Model', 'ModelParameters', 'TransformationParameters']]
    )
    ensemble_models = best3.to_dict(orient='index')
    return {
        'Model': 'Ensemble',
        'ModelParameters': json.dumps(
            {
                'model_name': 'Dist',
                'model_count': 2,
                'model_metric': 'smape',
                'models': ensemble_models,
                'dis_frac': dis_frac,
                'FirstModel': first_model,
                'SecondModel': last_model,
            }
        ),
        'TransformationParameters': '{}',
        'Ensemble': 1,
    }


def _generate_bestn_dict(
    best,
    model_name: str = 'BestN',
    model_metric: str = "best_score",
    model_weights: dict = None,
    point_method: str = None,
):
    ensemble_models = best.to_dict(orient='index')
    if not ensemble_models:
        print(f"BestN returned empty with {model_metric}")
    model_parms = {
        'model_name': model_name,
        'model_count': best.shape[0],
        'model_metric': model_metric,
        'models': ensemble_models,
    }
    if model_weights is not None:
        model_parms['model_weights'] = model_weights
    if point_method is not None:
        model_parms['point_method'] = point_method
    return {
        'Model': 'Ensemble',
        'ModelParameters': json.dumps(model_parms),
        'TransformationParameters': '{}',
        'Ensemble': 1,
    }


def mlens_helper(models, models_source="bestn"):
    from autots.models.mlensemble import MLEnsemble

    mlens = MLEnsemble().get_new_params()
    mlens['models'] = models.to_dict(orient='records')
    mlens['models_source'] = models_source
    return pd.DataFrame(
        {
            'Model': 'MLEnsemble',
            'ModelParameters': json.dumps(mlens),
            'TransformationParameters': '{}',
            'Ensemble': 0,
        },
        index=[0],
    )


def EnsembleTemplateGenerator(
    initial_results,
    forecast_length: int = 14,
    ensemble: str = "simple",
    score_per_series=None,
    use_validation=False,
):
    """Generate class 1 (non-horizontal) ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    ens_temp = initial_results.model_results.drop_duplicates(subset='ID')
    # filter out horizontal ensembles
    ens_temp = ens_temp[ens_temp['Ensemble'] <= 1]
    if 'simple' in ensemble or "mlensemble" in ensemble:
        # best 3, all can be of same model type
        best3nonunique = ens_temp.nsmallest(3, columns=['Score']).set_index("ID")[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ]
        n_models = best3nonunique.shape[0]
        if n_models == 3:
            best3nu_params = pd.DataFrame(
                _generate_bestn_dict(
                    best3nonunique, model_name='BestN', model_metric="best_score"
                ),
                index=[0],
            )
            ensemble_templates = pd.concat([ensemble_templates, best3nu_params], axis=0)
        if "mlensemble" in ensemble:
            ensemble_templates = pd.concat(
                [
                    ensemble_templates,
                    mlens_helper(best3nonunique, models_source='best_score'),
                ],
                axis=0,
                ignore_index=True,
            )
        # Best 5 and Median
        best5nonunique = ens_temp.nsmallest(5, columns=['Score']).set_index("ID")[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ]
        best5_params = pd.DataFrame(
            _generate_bestn_dict(
                best5nonunique,
                model_name='BestN',
                model_metric="bestn_horizontal",
                point_method="median",
            ),
            index=[0],
        )
        if 'simple' in ensemble:
            ensemble_templates = pd.concat(
                [ensemble_templates, best5_params], axis=0, ignore_index=True
            )

        # best 3 and 5, by SMAPE, RMSE, SPL, SMADE
        bestsmape = ens_temp.nsmallest(1, columns=['smape_weighted'])
        bestrmse = ens_temp.nsmallest(2, columns=['rmse_weighted'])
        bestmae = ens_temp.nsmallest(2, columns=['spl_weighted'])
        bestmade = ens_temp.nsmallest(5, columns=['made_weighted'])
        best3metric = pd.concat([bestsmape, bestrmse, bestmae, bestmade], axis=0)
        best3metric = best3metric.drop_duplicates().set_index("ID")[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ]
        best5metric = best3metric.head(5)
        best3metric = best3metric.head(3)
        n_models = best3metric.shape[0]
        if n_models == 3:
            best3m_params = pd.DataFrame(
                _generate_bestn_dict(
                    best3metric, model_name='BestN', model_metric="mixed_metric"
                ),
                index=[0],
            )
            ensemble_templates = pd.concat([ensemble_templates, best3m_params], axis=0)
        if "mlensemble" in ensemble:
            ensemble_templates = pd.concat(
                [
                    ensemble_templates,
                    mlens_helper(best3metric, models_source='mixed_metric'),
                ],
                axis=0,
                ignore_index=True,
            )
        best5m_params = pd.DataFrame(
            _generate_bestn_dict(
                best5metric,
                model_name='BestN',
                model_metric="mixed_metric",
                point_method="median",
            ),
            index=[0],
        )
        ensemble_templates = pd.concat([ensemble_templates, best5m_params], axis=0)
        # best 3, all must be of different model types
        ens_temp = (
            ens_temp.sort_values('Score', ascending=True, na_position='last')
            .groupby('Model')
            .head(1)
            .reset_index(drop=True)
        )
        best3unique = ens_temp.nsmallest(3, columns=['Score']).set_index("ID")[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ]
        best5unique = ens_temp.nsmallest(5, columns=['Score']).set_index("ID")[
            ['Model', 'ModelParameters', 'TransformationParameters']
        ]
        if best3unique.shape[0] == 3:
            best3u_params = pd.DataFrame(
                _generate_bestn_dict(
                    best3unique, model_name='BestN', model_metric="best_score_unique"
                ),
                index=[0],
            )
            ensemble_templates = pd.concat(
                [ensemble_templates, best3u_params], axis=0, ignore_index=True
            )
        if best5unique.shape[0] == 5 and 'simple' in ensemble:
            best5u_params = pd.DataFrame(
                _generate_bestn_dict(
                    best5unique,
                    model_name='BestN',
                    model_metric="best_score_unique",
                    point_method="midhinge",
                ),
                index=[0],
            )
            ensemble_templates = pd.concat(
                [ensemble_templates, best5u_params], axis=0, ignore_index=True
            )

    if 'distance' in ensemble:
        dis_frac = 0.2
        distance_params = pd.DataFrame(
            _generate_distance_ensemble(dis_frac, forecast_length, initial_results),
            index=[0],
        )
        ensemble_templates = pd.concat(
            [ensemble_templates, distance_params], axis=0, ignore_index=True
        )
        dis_frac = 0.5
        distance_params2 = pd.DataFrame(
            _generate_distance_ensemble(dis_frac, forecast_length, initial_results),
            index=[0],
        )
        ensemble_templates = pd.concat(
            [ensemble_templates, distance_params2], axis=0, ignore_index=True
        )
    # in previous versions per_series metrics were only captured if 'horizontal' was passed
    if 'simple' in ensemble or "mlensemble" in ensemble:
        if score_per_series is None:
            per_series = initial_results.per_series_mae
        else:
            per_series = score_per_series
        per_series = per_series[per_series.index.isin(ens_temp['ID'].tolist())]
        # make it ranking based! Need bigger=better for weighting
        per_series_ranked = per_series.rank(ascending=False)
        # choose best n based on score per series
        n = 3
        chosen_ones = per_series_ranked.sum(axis=1).nlargest(n)
        bestn = ens_temp[ens_temp['ID'].isin(chosen_ones.index.tolist())].set_index(
            "ID"
        )[['Model', 'ModelParameters', 'TransformationParameters']]
        n_models = bestn.shape[0]
        if n_models == n:
            best3u_params = pd.DataFrame(
                _generate_bestn_dict(
                    bestn,
                    model_name='BestN',
                    model_metric="bestn_horizontal",
                    model_weights=chosen_ones.to_dict(),
                ),
                index=[0],
            )
            ensemble_templates = pd.concat(
                [ensemble_templates, best3u_params], axis=0, ignore_index=True
            )
        if "mlensemble" in ensemble:
            ensemble_templates = pd.concat(
                [
                    ensemble_templates,
                    mlens_helper(bestn, models_source='bestn_horizontal'),
                ],
                axis=0,
                ignore_index=True,
            )
        # cluster and then make best model per cluster
        if per_series.shape[1] > 4:
            try:
                from sklearn.cluster import AgglomerativeClustering

                max_clusters = 8
                n_clusters = round(per_series.shape[1] / 3)
                n_clusters = max_clusters if n_clusters > max_clusters else n_clusters
                X = per_series_ranked.transpose()
                clstr = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
                series_labels = clstr.labels_
                for cluster in np.unique(series_labels).tolist():
                    current_ps = per_series_ranked[
                        per_series_ranked.columns[series_labels == cluster]
                    ]
                    n = 3
                    chosen_ones = current_ps.sum(axis=1).nlargest(n)
                    bestn = ens_temp[
                        ens_temp['ID'].isin(chosen_ones.index.tolist())
                    ].set_index("ID")[
                        ['Model', 'ModelParameters', 'TransformationParameters']
                    ]
                    n_models = bestn.shape[0]
                    if n_models == n:
                        if "mlensemble" in ensemble:
                            ensemble_templates = pd.concat(
                                [
                                    ensemble_templates,
                                    mlens_helper(
                                        bestn, models_source=f"cluster_{cluster}"
                                    ),
                                ],
                                axis=0,
                                ignore_index=True,
                            )
                        if 'simple' in ensemble:
                            best3u_params = pd.DataFrame(
                                _generate_bestn_dict(
                                    bestn,
                                    model_name='BestN',
                                    model_metric=f"cluster_{cluster}",
                                    model_weights=chosen_ones.to_dict(),
                                ),
                                index=[0],
                            )
                            ensemble_templates = pd.concat(
                                [ensemble_templates, best3u_params],
                                axis=0,
                                ignore_index=True,
                            )
            except Exception as e:
                print(f"cluster-based simple ensemble failed with {repr(e)}")

        mods = pd.Series(dtype='object')
        per_series_des = per_series.copy()
        n_models = 3
        # choose best per series, remove those series, then choose next best
        for x in range(n_models):
            n_dep = 5 if x < 2 else 10
            n_dep = (
                n_dep if per_series_des.shape[0] > n_dep else per_series_des.shape[0]
            )
            models_pos = []
            tr_df = pd.DataFrame()
            for _ in range(n_dep):
                cr_df = pd.DataFrame(per_series_des.idxmin()).transpose()
                tr_df = pd.concat([tr_df, cr_df], axis=0)
                models_pos.extend(per_series_des.idxmin().tolist())
                per_series_des[per_series_des == per_series_des.min()] = np.nan
            cur_mods = pd.Series(models_pos).value_counts()
            cur_mods = cur_mods.sort_values(ascending=False).head(1)
            mods = mods.combine(cur_mods, max, fill_value=0)
            rm_cols = tr_df[tr_df.isin(mods.index.tolist())]
            rm_cols = rm_cols.dropna(how='all', axis=1).columns
            per_series_des = per_series.copy().drop(mods.index, axis=0)
            per_series_des = per_series_des.drop(rm_cols, axis=1)
            if per_series_des.shape[1] == 0:
                per_series_des = per_series.copy().drop(mods.index, axis=0)

        best3 = (
            initial_results.model_results[
                initial_results.model_results['ID'].isin(mods.index.tolist())
            ]
            .drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            .set_index("ID")[['Model', 'ModelParameters', 'TransformationParameters']]
        )
        if "mlensemble" in ensemble:
            ensemble_templates = pd.concat(
                [ensemble_templates, mlens_helper(best3, models_source='horizontal')],
                axis=0,
                ignore_index=True,
            )
        best3_params = pd.DataFrame(
            _generate_bestn_dict(best3, model_name='BestN', model_metric="horizontal"),
            index=[0],
        )
        ensemble_templates = pd.concat(
            [ensemble_templates, best3_params], axis=0, ignore_index=True
        )
    if 'subsample' in ensemble:
        try:
            import random

            if score_per_series is None:
                per_series = initial_results.per_series_mae
            else:
                per_series = score_per_series
            per_series = per_series[per_series.index.isin(ens_temp['ID'].tolist())]
            # make it ranking based! Need bigger=better for weighting
            per_series_ranked = per_series.rank(ascending=False)
            # subsample and then make best model per group
            num_series = per_series.shape[1]
            n_samples = num_series * 2
            max_deep_ensembles = 100
            n_samples = (
                n_samples if n_samples < max_deep_ensembles else max_deep_ensembles
            )
            col_min = 1 if num_series < 3 else 2
            col_max = round(num_series / 2)
            col_max = num_series if col_max > num_series else col_max
            for samp in range(n_samples):
                n_cols = random.randint(col_min, col_max)
                current_ps = per_series_ranked.sample(n=n_cols, axis=1)
                n_largest = random.randint(9, 16)
                n_sample = random.randint(2, 5)
                # randomly choose one of best models
                chosen_ones = current_ps.sum(axis=1).nlargest(n_largest)
                n_sample = (
                    n_sample
                    if n_sample < chosen_ones.shape[0]
                    else chosen_ones.shape[0]
                )
                chosen_ones = chosen_ones.sample(n_sample).sort_values(ascending=False)
                bestn = ens_temp[
                    ens_temp['ID'].isin(chosen_ones.index.tolist())
                ].set_index("ID")[
                    ['Model', 'ModelParameters', 'TransformationParameters']
                ]
                point_method = random.choice(["mean", "median", "midhinge"])
                if point_method in ["median", "midhinge"]:
                    model_weights = None
                else:
                    model_weights = chosen_ones.to_dict()
                best3u_params = pd.DataFrame(
                    _generate_bestn_dict(
                        bestn,
                        model_name='BestN',
                        model_metric=f"subsample_{samp}",
                        model_weights=model_weights,
                        point_method=point_method,
                    ),
                    index=[0],
                )
                ensemble_templates = pd.concat(
                    [ensemble_templates, best3u_params], axis=0, ignore_index=True
                )
        except Exception as e:
            print(f"subsample ensembling failed with error: {repr(e)}")

    return ensemble_templates


def find_pattern(strings, x, sep="-"):
    pattern = f"({x}){sep}(\\d+)"
    results = []

    for string in strings:
        matched = re.search(pattern, string)
        if matched:
            # Extracting the components
            fixed_string, number = matched.groups()
            results.append((fixed_string, int(number)))

    return results


def n_limited_horz(per_series, K, safety_model=False):
    # progressively remove best models to try and achieve wider coverage with a few
    mods = pd.Series()
    per_series_des = per_series.copy()
    if K <= 1:
        safety_model_id = per_series.mean(axis=1).idxmin()
        return [safety_model_id]
    if safety_model:
        safety_model_id = per_series.mean(axis=1).idxmin()

    for x in range(K):
        # gets deeper into the top N per series for the later searches
        n_dep = x + 1
        n_dep = n_dep if per_series_des.shape[0] > n_dep else per_series_des.shape[0]
        models_pos = []
        tr_df = pd.DataFrame()
        # find the most common models at this depth
        for _ in range(n_dep):
            cr_df = pd.DataFrame(per_series_des.idxmin()).transpose()
            tr_df = pd.concat([tr_df, cr_df], axis=0)
            models_pos.extend(per_series_des.idxmin().tolist())
            per_series_des[per_series_des == per_series_des.min()] = np.nan
        cur_mods = pd.Series(models_pos).value_counts()
        cur_mods = cur_mods.sort_values(ascending=False).head(1)
        mods = mods.combine(cur_mods, max, fill_value=0)
        # drop series which have been satisfied by the model selection so far
        rm_cols = tr_df[tr_df.isin(mods.index.tolist())]
        rm_cols = rm_cols.dropna(how='all', axis=1).columns
        per_series_des = per_series.copy().drop(mods.index, axis=0)
        per_series_des = per_series_des.drop(rm_cols, axis=1)
        # if size reaches zero, start back with the full columns
        if per_series_des.shape[1] == 0:
            per_series_des = per_series.copy().drop(mods.index, axis=0)

    min_selected = mods.index.tolist()
    if safety_model:
        if safety_model_id not in min_selected:
            min_selected = min_selected[0 : (K - 1)]
            min_selected.append(safety_model_id)
    return min_selected


def HorizontalTemplateGenerator(
    per_series,
    model_results,
    forecast_length: int = 14,
    ensemble: str = "horizontal",
    subset_flag: bool = True,
    per_series2=None,
    only_specified: bool = False,
):
    """Generate horizontal ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    if 'horizontal-max' in ensemble:
        mods_per_series = per_series.idxmin()
        mods = mods_per_series.unique()
        best5 = (
            model_results[model_results['ID'].isin(mods.tolist())]
            .drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            .set_index("ID")[['Model', 'ModelParameters', 'TransformationParameters']]
        )
        nomen = 'Horizontal'
        metric = 'Score-max'
        if len(mods_per_series) < per_series.shape[1]:
            print(
                "ERROR in Horizontal Generation insufficient series created, horizontal-max"
            )
        else:
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods.shape[0],
                        'model_metric': metric,
                        'models': best5.to_dict(orient='index'),
                        'series': mods_per_series.to_dict(),
                    }
                ),
                'TransformationParameters': '{}',
                'Ensemble': 2,
            }
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat(
                [ensemble_templates, best5_params], axis=0, ignore_index=True
            )
    # this is legacy, replaced by mosaic
    if 'hdist' in ensemble and not subset_flag:
        mods_per_series = per_series.idxmin()
        mods_per_series2 = per_series2.idxmin()
        mods = pd.concat([mods_per_series, mods_per_series2]).unique()
        best5 = (
            model_results[model_results['ID'].isin(mods.tolist())]
            .drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            .set_index("ID")[['Model', 'ModelParameters', 'TransformationParameters']]
        )
        nomen = 'hdist'
        best5_params = {
            'Model': 'Ensemble',
            'ModelParameters': json.dumps(
                {
                    'model_name': nomen,
                    'model_count': mods.shape[0],
                    'models': best5.to_dict(orient='index'),
                    'dis_frac': 0.3,
                    'series1': mods_per_series.to_dict(),
                    'series2': mods_per_series2.to_dict(),
                }
            ),
            'TransformationParameters': '{}',
            'Ensemble': 2,
        }
        best5_params = pd.DataFrame(best5_params, index=[0])
        ensemble_templates = pd.concat(
            [ensemble_templates, best5_params], axis=0, ignore_index=True
        )
    # the idea behind running both is for redundancy in the -max case
    # and this one is better in some testing
    if 'horizontal' in ensemble or (
        'horizontal-max' in ensemble and not only_specified
    ):
        # first generate lists of models by ID that are in shared and no_shared
        no_shared_select = model_results['Model'].isin(no_shared)
        shared_mod_lst = model_results[~no_shared_select]['ID'].tolist()
        no_shared_mod_lst = model_results[no_shared_select]['ID'].tolist()
        # another take on "safety" model
        lowest_score_mod = [model_results.iloc[model_results['Score'].idxmin()]['ID']]
        per_series[per_series.index.isin(shared_mod_lst)]
        # remove those where idxmin is in no_shared
        shared_maxes = per_series.idxmin().isin(shared_mod_lst)
        shr_mx_cols = shared_maxes[shared_maxes].index
        per_series_shareds = per_series.filter(shr_mx_cols, axis=1)
        # select best n shared models
        K = 5
        if False:
            # old method
            use_shared_lst = (
                per_series_shareds.median(axis=1).nsmallest(K).index.tolist()
            )
        else:
            use_shared_lst = n_limited_horz(per_series_shareds, K=K, safety_model=False)
        # combine all of the above as allowed mods
        allowed_list = no_shared_mod_lst + lowest_score_mod + use_shared_lst

        # first select a few of the best shared models
        # Option A: Best overall per model type (by different metrics?)
        # Option B: Best per different clusters...
        # Rank position in score for EACH series
        # Lowest median ranking
        # Lowest Quartile 1 of rankings
        # Normalize and then take Min, Median, or IQ1
        # then choose min from series of these + no_shared
        # make sure no models are included that don't match to any series
        # ENSEMBLE and NO_SHARED (it could be or it could not be)
        # need to TEST cases where all columns are either shared or no_shared!
        per_series_filter = per_series[per_series.index.isin(allowed_list)]
        mods_per_series = per_series_filter.idxmin()
        mods = mods_per_series.unique()
        best5 = (
            model_results[model_results['ID'].isin(mods.tolist())]
            .drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            .set_index("ID")[['Model', 'ModelParameters', 'TransformationParameters']]
        )
        nomen = 'Horizontal'
        metric = 'Score'
        if len(mods_per_series) < per_series.shape[1]:
            print(
                "ERROR in Horizontal Generation insufficient series created, horizontal"
            )
        else:
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods.shape[0],
                        'model_metric': metric,
                        'models': best5.to_dict(orient='index'),
                        'series': mods_per_series.to_dict(),
                    }
                ),
                'TransformationParameters': '{}',
                'Ensemble': 2,
            }
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat(
                [ensemble_templates, best5_params], axis=0, ignore_index=True
            )
    # choose N size ensembles given arg, mostly the same as horizontal-min
    n_horz = find_pattern(ensemble, "horizontal")
    if n_horz:
        for patt in n_horz:
            kn = patt[1]
            min_selected = n_limited_horz(per_series, K=kn, safety_model=True)
            mods_per_series = per_series.loc[min_selected].idxmin()
            best5 = (
                model_results[
                    model_results['ID'].isin(mods_per_series.unique().tolist())
                ]
                .drop_duplicates(
                    subset=['Model', 'ModelParameters', 'TransformationParameters']
                )
                .set_index("ID")[
                    ['Model', 'ModelParameters', 'TransformationParameters']
                ]
            )
            nomen = 'Horizontal'
            metric = 'Score-n'
            if len(mods_per_series) < per_series.shape[1]:
                print(
                    "ERROR in Horizontal Generation insufficient series created, horizontal-min"
                )
            else:
                best5_params = {
                    'Model': 'Ensemble',
                    'ModelParameters': json.dumps(
                        {
                            'model_name': nomen,
                            'model_count': mods_per_series.unique().shape[0],
                            'model_metric': metric,
                            'models': best5.to_dict(orient='index'),
                            'series': mods_per_series.to_dict(),
                        }
                    ),
                    'TransformationParameters': '{}',
                    'Ensemble': 2,
                }
                best5_params = pd.DataFrame(best5_params, index=[0])
                ensemble_templates = pd.concat(
                    [ensemble_templates, best5_params], axis=0, ignore_index=True
                )
    # this is a "greedy" approach to choosing a smaller number of models
    n_horz_min = find_pattern(ensemble, "horizontal-min")
    if 'horizontal-min' in ensemble or n_horz_min:
        if 'horizontal-min' in ensemble:
            n_horz_min.append(('horizontal-min', 15))
        for patt in n_horz_min:
            K = patt[1]
            min_selected = n_limited_horz(per_series, K=K, safety_model=False)
            mods_per_series = per_series.loc[min_selected].idxmin()
            best5 = (
                model_results[
                    model_results['ID'].isin(mods_per_series.unique().tolist())
                ]
                .drop_duplicates(
                    subset=['Model', 'ModelParameters', 'TransformationParameters']
                )
                .set_index("ID")[
                    ['Model', 'ModelParameters', 'TransformationParameters']
                ]
            )
            nomen = 'Horizontal'
            metric = 'Score-min'
            if len(mods_per_series) < per_series.shape[1]:
                print(
                    "ERROR in Horizontal Generation insufficient series created, horizontal-min"
                )
            else:
                best5_params = {
                    'Model': 'Ensemble',
                    'ModelParameters': json.dumps(
                        {
                            'model_name': nomen,
                            'model_count': mods_per_series.unique().shape[0],
                            'model_metric': metric,
                            'models': best5.to_dict(orient='index'),
                            'series': mods_per_series.to_dict(),
                        }
                    ),
                    'TransformationParameters': '{}',
                    'Ensemble': 2,
                }
                best5_params = pd.DataFrame(best5_params, index=[0])
                ensemble_templates = pd.concat(
                    [ensemble_templates, best5_params], axis=0, ignore_index=True
                )
    return ensemble_templates


def generate_crosshair_score(error_matrix, method=None):
    # handle nan, assume the worst
    if np.isnan(np.sum(error_matrix)):
        error_matrix[np.isnan(error_matrix)] = np.nanmax(error_matrix)
    # 'lite' only takes the weighted axis down a series not from other series
    if method == 'crosshair_lite':
        return error_matrix + (np.median(error_matrix, axis=0) / 3)
    else:
        arr_size = error_matrix.size
        base_weight = 0.001 / arr_size
        sum_error = np.sum(error_matrix) * base_weight

        cross_base = error_matrix * (base_weight * 50)
        row_sums = cross_base.sum(axis=1)
        col_sums = cross_base.sum(axis=0)
        outer_sum = np.add.outer(row_sums, col_sums)

        return error_matrix + sum_error + outer_sum


def generate_crosshair_score_list(error_list):
    # unfinished 3d version
    arr_size = error_list[-1].size
    base_weight = 0.001 / arr_size

    full_arr = np.array(error_list)

    sum_error = np.sum(full_arr, axis=(-1, 1))

    cross_base = full_arr * (base_weight * 50)
    row_sums = cross_base.sum(axis=2)
    col_sums = cross_base.sum(axis=1)
    # stops working here
    outer_sum = np.add.outer(row_sums, col_sums)  # noqa

    return list(full_arr + sum_error)  # + outer_sum


def _custom_min_max_scaler(df, min_value=0.1, power=0.5):
    # Calculate min and max for each column
    if power is not None:
        # this reduces the skew of the strongest outliers when power = 0.5
        df = np.power(df.astype(float).copy(), power)
    col_min = df.min()
    col_max = df.max()

    # Apply the inverted min-max scaling formula
    scaled_df = min_value + (1 - min_value) * ((col_max - df) / (col_max - col_min))

    return scaled_df.round(3)


def create_unpredictability_score(
    full_mae_errors,
    full_mae_vals,
    total_vals,
    df_wide,
    validation_test_indexes,
    scale=False,
):
    results = []
    threshold = np.nanmedian(full_mae_errors) * 1.1
    for val in range(total_vals):
        errors_array = np.array(
            [
                x
                for y, x in sorted(
                    zip(full_mae_vals, full_mae_errors), key=lambda pair: pair[0]
                )
                if y == val
            ]
        )
        # filters by models performance across ALL series, which may sometimes be a limitation
        performance_summary = np.nanmedian(errors_array, axis=(1, 2))
        filtered_models = errors_array[performance_summary <= threshold].copy()
        if filtered_models.shape[0] <= 1:
            inner_threshold = np.median(performance_summary) * 1.2
            filtered_models = errors_array[
                performance_summary <= inner_threshold
            ].copy()
        # median_error = np.nanmedian(filtered_models, axis=0)
        min_error = np.nanquantile(filtered_models, q=0.01, axis=0)
        # score = (median_error * 0.01 + min_error)  # where min was actual min
        score = min_error
        # score = score / np.min(score)
        score = pd.DataFrame(
            score, index=validation_test_indexes[val], columns=df_wide.columns
        )
        # scale
        score = score / pd.DataFrame(
            index=validation_test_indexes[val], columns=df_wide.columns
        ).fillna(df_wide.mean())
        results.append(score)

    if scale:
        return _custom_min_max_scaler(pd.concat(results).sort_index(), min_value=0.1)
    else:
        return pd.concat(results).sort_index()


def process_mosaic_arrays(
    local_results,
    full_mae_ids,
    full_mae_errors,
    total_vals=None,
    models_to_use=None,
    smoothing_window=None,
    filtered=False,
    unpredictability_adjusted=False,
    validation_test_indexes=None,
    full_mae_vals=None,
    df_wide=None,
):
    # sort by runtime then drop duplicates on metric results to remove functionally equivalent model duplication
    local_results = local_results.sort_values(by="TotalRuntimeSeconds", ascending=True)
    temp = local_results.drop_duplicates(
        subset=['ValidationRound', 'smape', 'mae', 'spl'],
        keep="first",
    )
    # there is still a possible edge case where a model matches different, but equal models on each validation round but is better overall
    # but as models being identical on point and probabilistic and this occurring seems unlikely
    local_results = local_results[local_results["ID"].isin(temp["ID"].unique())]
    # remove slow models... tbd
    # select only models run through all validations
    # previous version was failing to remove models that failed on validation
    run_count = (
        local_results[local_results["Exceptions"].isnull()][['Model', 'ID']]
        .groupby("ID")
        .count()
    )
    if filtered:
        # this one has dedupe currently and can handle greater
        # but I'm not 100% on this dedupe overall being correct, hence not using for all
        fully_validated = run_count[run_count['Model'] >= total_vals].index.tolist()
    else:
        fully_validated = run_count[run_count['Model'] == total_vals].index.tolist()
    if models_to_use is None:
        models_to_use = fully_validated
    else:
        # so the logic makes it that it must be EXACTLY the right number of vals
        # which can create problems, some models get duplicated and will be excluded
        filtered_use = list(set(models_to_use).intersection(set(fully_validated)))
        if len(filtered_use) > 1:
            models_to_use = filtered_use
    # begin figuring out which are the min models for each point
    id_array = np.array([y for y in sorted(full_mae_ids) if y in models_to_use])
    if unpredictability_adjusted:
        scores = create_unpredictability_score(
            full_mae_errors,
            full_mae_vals,
            total_vals,
            df_wide,
            validation_test_indexes,
            scale=False,
        )
        weight_dict = {
            idx: scores.reindex(val).to_numpy()
            for idx, val in enumerate(validation_test_indexes)
        }
        full_mae_errors_use = [
            x * weight_dict[y] for y, x in zip(full_mae_vals, full_mae_errors)
        ]
    else:
        full_mae_errors_use = full_mae_errors

    # remove models that are above median overall
    if filtered:
        threshold = np.nanmedian(full_mae_errors_use) * 1.2  # could change this level
        seen = set()
        errors_array = []
        id_array = []
        rubbish = []
        for idz, errz, valz in sorted(
            zip(full_mae_ids, full_mae_errors_use, full_mae_vals),
            key=lambda pair: pair[0],
        ):
            if idz in models_to_use:
                if np.nanmedian(errz) <= threshold:
                    rubbish.append(idz)
                if (idz, str(valz)) not in seen:
                    # dedupe if dupes exist, which they shouldn't but they do sometimes...
                    seen.add((idz, str(valz)))
                    errors_array.append(errz)
                    id_array.append(idz)
        errors_array2 = []
        id_array2 = []
        for errz, idz in zip(errors_array, id_array):
            if idz not in rubbish:
                errors_array2.append(errz)
                id_array2.append(idz)
        errors_array = np.array(errors_array2)
        id_array = np.array(id_array2)
    else:
        seen = set()
        errors_array = []
        id_array = []
        for idz, errz, valz in sorted(
            zip(full_mae_ids, full_mae_errors_use, full_mae_vals),
            key=lambda pair: pair[0],
        ):
            if idz in models_to_use and (idz, str(valz)) not in seen:
                seen.add((idz, str(valz)))
                errors_array.append(errz)
                id_array.append(idz)
        errors_array = np.array(errors_array)
        id_array = np.array(id_array)

    if smoothing_window is not None:
        from scipy.ndimage import uniform_filter1d

        errors_array = uniform_filter1d(
            np.nan_to_num(errors_array), size=smoothing_window, axis=1
        )
    return id_array, errors_array


def parse_forecast_length(forecast_length):
    # make progressively larger chunks
    if forecast_length <= 2:
        chunks = [list(range(forecast_length))]
    elif forecast_length <= 6:
        chunks = [[0, 1], list(range(2, forecast_length))]
    elif forecast_length <= 12:
        chunks = [[0, 1], [2, 3, 4, 5], list(range(5, forecast_length))]
    else:
        start = int(forecast_length * 0.08)
        mid = int(forecast_length * 0.32)
        chunks = [
            list(range(0, start)),
            list(range(start, mid)),
            list(range(mid, forecast_length)),
        ]
    return chunks


def generate_mosaic_template(
    initial_results,
    full_mae_ids,
    num_validations,
    col_names,
    full_mae_errors,
    smoothing_window=None,
    metric_name="MAE",
    models_to_use=None,
    id_to_group_mapping: dict = None,
    filtered: bool = False,
    unpredictability_adjusted: bool = False,
    validation_test_indexes=None,
    full_mae_vals=None,
    df_wide=None,
    **kwargs,
):
    """Generate an ensemble template from results."""
    total_vals = num_validations + 1
    local_results = initial_results.copy()
    id_array, errors_array = process_mosaic_arrays(
        local_results,
        full_mae_ids,
        full_mae_errors,
        total_vals=total_vals,
        models_to_use=models_to_use,
        smoothing_window=smoothing_window,
        filtered=filtered,
        unpredictability_adjusted=unpredictability_adjusted,
        validation_test_indexes=validation_test_indexes,
        full_mae_vals=full_mae_vals,
        df_wide=df_wide,
    )
    checksum = pd.Series(id_array).value_counts()
    # should be the same because all should have the same num validations
    assert (
        checksum.min() == checksum.max()
    ), f"id array wrong in mosaic generation, {checksum.min()}, {checksum.max()}, {len(errors_array)}"
    # window across multiple time steps to smooth the result
    name = "Mosaic"
    # since it is sorted by id and filtered to only those run through all vals, this is the slice step after each val
    slice_points = np.arange(0, errors_array.shape[0], step=total_vals)
    id_sliced = id_array[slice_points]
    if id_to_group_mapping is None:
        best_points = np.add.reduceat(errors_array, slice_points, axis=0).argmin(axis=0)
        model_id_array = pd.DataFrame(
            np.take(id_sliced, best_points), columns=col_names
        )
    else:
        # group by profile
        res = []
        for group in set(list(id_to_group_mapping.values())):
            idz = [
                col_names.get_loc(key)
                for key, value in id_to_group_mapping.items()
                if value == group and key in col_names
            ]
            if len(idz) < 1:
                pass
            else:
                subsetz = errors_array[:, :, idz].mean(axis=2)
                best_points = np.add.reduceat(subsetz, slice_points, axis=0).argmin(
                    axis=0
                )
                res.append(
                    pd.DataFrame(np.take(id_sliced, best_points), columns=[group])
                )
        # add on the overall for any missing groups
        best_points = np.add.reduceat(
            errors_array.mean(axis=2), slice_points, axis=0
        ).argmin(axis=0)
        res.append(pd.DataFrame(np.take(id_sliced, best_points), columns=["overall"]))
        # combines
        model_id_array = pd.concat(res, axis=1)

    used_models = pd.unique(model_id_array.values.flatten())
    used_models_results = local_results[
        ["ID", "Model", "ModelParameters", "TransformationParameters"]
    ].drop_duplicates(subset='ID')
    used_models_results = used_models_results[
        used_models_results['ID'].isin(used_models)
    ].set_index("ID")

    ensemble_params = {
        'Model': 'Ensemble',
        'ModelParameters': json.dumps(
            {
                'model_name': name,
                'model_count': used_models_results.shape[0],
                'smoothing_window': smoothing_window,
                'model_metric': metric_name,
                'models': used_models_results.to_dict(orient='index'),
                'series': model_id_array.to_dict(orient='dict'),
            }
        ),
        'TransformationParameters': '{}',
        'Ensemble': 2,
    }
    ensemble_template = pd.DataFrame(ensemble_params, index=[0])
    return ensemble_template


def mosaic_to_horizontal(ModelParameters, forecast_period: int = 0):
    """Take a mosaic template and pull a single forecast step as a horizontal model.

    Args:
        ModelParameters (dict): the json.loads() of the ModelParameters of a mosaic ensemble template
        forecast_period (int): when to choose the model, starting with 0
            where 0 would be the first forecast datestamp, 1 would be the second, and so on
            must be less than forecast_length that the model was trained on.
    Returs:
        ModelParameters (dict)
    """
    if str(ModelParameters['model_name']).lower() != "mosaic":
        raise ValueError("Input parameters are not recognized as a mosaic ensemble.")
    all_models = ModelParameters['series']
    result = {k: v[str(forecast_period)] for k, v in all_models.items()}
    model_result = {
        k: v for k, v in ModelParameters['models'].items() if k in result.values()
    }
    return {
        'model_name': "horizontal",
        'model_count': len(model_result),
        "model_metric": "mosaic_conversion",
        'models': model_result,
        'series': result,
    }


def MosaicEnsemble(
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
    df_train=None,
    prematched_series: dict = None,
):
    """Generate forecast for mosaic ensembling.

    Args:
        prematched_series (dict): from outer horizontal generalization, possibly different than params
    """
    # work with forecast_lengths longer or shorter than provided by template

    # handle profiled mosaic
    profiled = "profile" in ensemble_params.get("model_metric")
    medianed = "median" in ensemble_params.get("model_metric")
    if profiled:
        profiled = (
            profile_time_series(df_train).set_index("SERIES").to_dict()["PROFILE"]
        )
        known_matches = ensemble_params['series']
        valid_values = list(known_matches.keys())
        profiled = {
            key: value if value in valid_values else "overall"
            for key, value in profiled.items()
        }
        # json.loads(ensemble_params["ModelParameters"])["series"][profiled[col]]
        prematched_series = {
            col: known_matches[profiled[col]] for col in df_train.columns
        }
    elif prematched_series is None:
        prematched_series = ensemble_params['series']
    # this is meant to fill in any failures
    startTime = datetime.datetime.now()
    sample_idx = next(iter(forecasts.values())).index
    available_models = [mod for mod, fcs in forecasts.items() if fcs.shape[0] > 0]
    train_size = df_train.shape
    full_models = [
        mod for mod, fcs in forecasts.items() if fcs.shape[1] == train_size[1]
    ]
    if not full_models:
        print("No full models available for mosaic generalization.")
        full_models = available_models  # hope it doesn't need to fill
    all_series = generalize_horizontal(
        df_train,
        prematched_series,
        available_models=available_models,
        full_models=full_models,
    )

    org_idx = df_train.columns

    final = pd.DataFrame.from_dict(all_series)
    final.index.name = "forecast_period"
    if medianed:
        # this doesn't assure all three models are unique, but mostly should
        nx1 = final.shift(1).bfill()
        nx2 = final.shift(-1).ffill()
        nx3 = final.shift(2).bfill()
        nx4 = final.shift(-2).ffill()
        nx1 = nx1.where(final != nx1, nx2)
        nx2 = nx2.where((nx1 != nx2) & (final != nx2), nx3)
        nx2 = nx2.where((nx1 != nx2) & (final != nx2), nx4)

    forecast_df, u_forecast_df, l_forecast_df = _buildup_mosaics(
        final,
        sample_idx,
        forecasts,
        upper_forecasts,
        lower_forecasts,
        available_models,
        org_idx,
    )
    if medianed:
        forecast_df2, u_forecast_df2, l_forecast_df2 = _buildup_mosaics(
            nx1,
            sample_idx,
            forecasts,
            upper_forecasts,
            lower_forecasts,
            available_models,
            org_idx,
        )
        forecast_df3, u_forecast_df3, l_forecast_df3 = _buildup_mosaics(
            nx2,
            sample_idx,
            forecasts,
            upper_forecasts,
            lower_forecasts,
            available_models,
            org_idx,
        )
        # Stack the three DataFrames into a 3D NumPy array
        stacked = np.stack(
            [forecast_df.to_numpy(), forecast_df2.to_numpy(), forecast_df3.to_numpy()],
            axis=2,
        )
        median_array = np.median(stacked, axis=2)
        forecast_df = pd.DataFrame(
            median_array, index=forecast_df.index, columns=forecast_df.columns
        )
        # upper
        stacked = np.stack(
            [
                u_forecast_df.to_numpy(),
                u_forecast_df2.to_numpy(),
                u_forecast_df3.to_numpy(),
            ],
            axis=2,
        )
        median_array = np.median(stacked, axis=2)
        u_forecast_df = pd.DataFrame(
            median_array, index=u_forecast_df.index, columns=u_forecast_df.columns
        )
        # lower
        stacked = np.stack(
            [
                l_forecast_df.to_numpy(),
                l_forecast_df2.to_numpy(),
                l_forecast_df3.to_numpy(),
            ],
            axis=2,
        )
        median_array = np.median(stacked, axis=2)
        l_forecast_df = pd.DataFrame(
            median_array, index=l_forecast_df.index, columns=l_forecast_df.columns
        )

    # combine runtimes
    try:
        ens_runtime = sum(list(forecasts_runtime.values()), datetime.timedelta())
    except Exception:
        ens_runtime = datetime.timedelta(0)

    # don't overwrite with mapping if profile type mosaic is used
    if not profiled:
        ensemble_params['series'] = all_series
    ens_result = PredictionObject(
        model_name="Ensemble",
        forecast_length=len(sample_idx),
        forecast_index=sample_idx,
        forecast_columns=org_idx,
        lower_forecast=l_forecast_df,
        forecast=forecast_df,
        upper_forecast=u_forecast_df,
        prediction_interval=prediction_interval,
        predict_runtime=datetime.datetime.now() - startTime,
        fit_runtime=ens_runtime,
        model_parameters=ensemble_params,
    )
    return ens_result


def _buildup_mosaics(
    final,
    sample_idx,
    forecasts,
    upper_forecasts,
    lower_forecasts,
    available_models,
    org_idx,
):
    melted = pd.melt(
        final,
        var_name="series_id",
        value_name="model_id",
        ignore_index=False,
    ).reset_index(drop=False)
    melted["forecast_period"] = pd.to_numeric(
        melted["forecast_period"], downcast="integer"
    )
    max_forecast_period = melted["forecast_period"].max()
    # handle forecast length being longer than template
    len_sample_index = len(sample_idx)
    if len_sample_index > (max_forecast_period + 1):
        print("Mosaic forecast length longer than template provided.")
        base_df = melted[melted['forecast_period'] == max_forecast_period]
        needed_stamps = len_sample_index - (max_forecast_period + 1)
        newdf = pd.DataFrame(np.repeat(base_df.to_numpy(), needed_stamps, axis=0))
        newdf.columns = base_df.columns
        newdf['forecast_period'] = np.tile(
            np.arange(max_forecast_period + 1, needed_stamps + 1 + max_forecast_period),
            base_df.shape[0],
        )
        melted = pd.concat([melted, newdf])
    elif len_sample_index < (max_forecast_period + 1):
        print("Mosaic forecast length less than template provided.")
        melted = melted[melted['forecast_period'] < len_sample_index]

    fore, u_fore, l_fore = [], [], []
    row = (0, "Unknown", "Unknown", "Unknown")
    try:
        # maybe this could be sped up by something like numpy take
        for row in melted.itertuples():
            fore.append(forecasts[row[3]][row[2]].iloc[row[1]])
            u_fore.append(upper_forecasts[row[3]][row[2]].iloc[row[1]])
            l_fore.append(lower_forecasts[row[3]][row[2]].iloc[row[1]])
    except Exception as e:
        m0 = f"{row[3]} in available_models: {row[3] in available_models}, "
        mi = (
            m0
            + f"In forecast: {row[3] in forecasts.keys()}, in upper: {row[3] in upper_forecasts.keys()}, in Lower: {row[3] in lower_forecasts.keys()}"
        )
        raise ValueError(
            f"Mosaic Ensemble failed on model {row[3]} series {row[2]} and period {row[1]} due to missing model: {e} "
            + mi
        ) from e
    melted['forecast'] = (
        fore  # [forecasts[row[3]][row[2]].iloc[row[1]] for row in melted.itertuples()]
    )
    melted['upper_forecast'] = (
        u_fore  # [upper_forecasts[row[3]][row[2]].iloc[row[1]] for row in melted.itertuples()]
    )
    melted['lower_forecast'] = (
        l_fore  # [lower_forecasts[row[3]][row[2]].iloc[row[1]] for row in melted.itertuples()]
    )

    forecast_df = melted.pivot(
        values="forecast", columns="series_id", index="forecast_period"
    )
    forecast_df.index = sample_idx
    u_forecast_df = melted.pivot(
        values="upper_forecast", columns="series_id", index="forecast_period"
    )
    u_forecast_df.index = sample_idx
    l_forecast_df = melted.pivot(
        values="lower_forecast", columns="series_id", index="forecast_period"
    )
    l_forecast_df.index = sample_idx
    # make sure columns align to original
    forecast_df = forecast_df.reindex(columns=org_idx)
    u_forecast_df = u_forecast_df.reindex(columns=org_idx)
    l_forecast_df = l_forecast_df.reindex(columns=org_idx)
    return forecast_df, u_forecast_df, l_forecast_df
