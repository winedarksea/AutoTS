"""Tools for generating and forecasting with ensembles of models."""
import datetime
import numpy as np
import pandas as pd
import json
from autots.models.base import PredictionObject
from autots.models.model_list import no_shared


horizontal_aliases = ['horizontal', 'probabilistic']


def BestNEnsemble(
    ensemble_params,
    forecasts_list,
    forecasts,
    lower_forecasts,
    upper_forecasts,
    forecasts_runtime,
    prediction_interval,
):
    """Generate mean forecast for ensemble of models."""
    # id_list = list(ensemble_params['models'].keys())
    # does it handle missing models well?
    # model_indexes = [x for x in forecasts.keys() if x in id_list]
    model_count = len(forecasts.keys())
    if model_count < 1:
        raise ValueError("BestN failed, no component models available.")
    sample_df = next(iter(forecasts.values()))
    columnz = sample_df.columns
    indices = sample_df.index

    ens_df = pd.DataFrame(0, index=indices, columns=columnz)
    for idx, x in forecasts.items():
        ens_df = ens_df + x
    ens_df = ens_df / model_count

    ens_df_lower = pd.DataFrame(0, index=indices, columns=columnz)
    for idx, x in lower_forecasts.items():
        ens_df_lower = ens_df_lower + x
    ens_df_lower = ens_df_lower / model_count

    ens_df_upper = pd.DataFrame(0, index=indices, columns=columnz)
    for idx, x in upper_forecasts.items():
        ens_df_upper = ens_df_upper + x
    ens_df_upper = ens_df_upper / model_count

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
        predict_runtime=datetime.timedelta(0),
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

    ens_df = (
        forecasts[first_model_index]
        .head(first_bit)
        .append(forecasts[second_model_index].tail(second_bit))
    )
    ens_df_lower = (
        lower_forecasts[first_model_index]
        .head(first_bit)
        .append(lower_forecasts[second_model_index].tail(second_bit))
    )
    ens_df_upper = (
        upper_forecasts[first_model_index]
        .head(first_bit)
        .append(upper_forecasts[second_model_index].tail(second_bit))
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


def summarize_series(df):
    """Summarize time series data. For now just df.describe()."""
    df_sum = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return df_sum


def horizontal_classifier(df_train, known: dict, method: str = "whatever"):
    """
    CLassify unknown series with the appropriate model for horizontal ensembling.

    Args:
        df_train (pandas.DataFrame): historical data about the series. Columns = series_ids.
        known (dict): dict of series_id: classifier outcome including some but not all series in df_train.

    Returns:
        dict.

    """
    # known = {'EXUSEU': 'xx1', 'MCOILWTICO': 'xx2', 'CSUSHPISA': 'xx3'}
    columnz = df_train.columns.tolist()
    X = summarize_series(df_train).transpose()
    known_l = list(known.keys())
    unknown = list(set(columnz) - set(known_l))
    Xt = X.loc[known_l]
    Xf = X.loc[unknown]
    Y = np.array(list(known.values()))
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(Xt, Y)
    result = clf.predict(Xf)
    result_d = dict(zip(Xf.index.tolist(), result))
    # since this only has estimates, overwrite with known that includes more
    final = {**result_d, **known}
    # temp = pd.DataFrame({'series': list(final.keys()), 'model': list(final.values())})
    # temp2 = temp.merge(X, left_on='series', right_index=True)
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
    # remove any unavailable models or unnecessary series
    known_matches = {ser: mod for ser, mod in known_matches.items() if ser in org_list}
    k = {ser: mod for ser, mod in known_matches.items() if mod in available_models}
    # check if any series are missing from model list
    if not k:
        raise ValueError("Horizontal template has no models matching this data!")
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
    # this is meant to fill in any failures
    available_models = list(forecasts.keys())
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
    # print(f"ALLSERIES {len(all_series.keys())}: {all_series}")

    org_idx = df_train.columns

    forecast_df, u_forecast_df, l_forecast_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for series, mod_id in all_series.items():
        try:
            c_fore = forecasts[mod_id][series]
            forecast_df = pd.concat([forecast_df, c_fore], axis=1)
        except Exception as e:
            print(f"Horizontal ensemble unable to add model {repr(e)}")
        # upper
        c_fore = upper_forecasts[mod_id][series]
        u_forecast_df = pd.concat([u_forecast_df, c_fore], axis=1)
        # lower
        c_fore = lower_forecasts[mod_id][series]
        l_forecast_df = pd.concat([l_forecast_df, c_fore], axis=1)
    # make sure columns align to original
    forecast_df.reindex(columns=org_idx)
    u_forecast_df.reindex(columns=org_idx)
    l_forecast_df.reindex(columns=org_idx)
    # combine runtimes
    ens_runtime = datetime.timedelta(0)
    for idx, x in forecasts_runtime.items():
        ens_runtime = ens_runtime + x

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
    s3list = ['best3', 'best3horizontal', 'bestn']
    if ensemble_params['model_name'].lower().strip() in s3list:
        ens_forecast = BestNEnsemble(
            ensemble_params,
            forecasts_list,
            forecasts,
            lower_forecasts,
            upper_forecasts,
            forecasts_runtime,
            prediction_interval,
        )
        return ens_forecast

    if ensemble_params['model_name'].lower().strip() == 'dist':
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

    if ensemble_params['model_name'].lower().strip() in horizontal_aliases:
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

    if ensemble_params['model_name'].lower().strip() == 'hdist':
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


def EnsembleTemplateGenerator(
    initial_results, forecast_length: int = 14, ensemble: str = "simple"
):
    """Generate ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    if 'simple' in ensemble:
        ens_temp = initial_results.model_results.drop_duplicates(subset='ID')
        ens_temp = ens_temp[ens_temp['Ensemble'] == 0]
        # best 3, all can be of same model type
        best3nonunique = ens_temp.nsmallest(3, columns=['Score'])
        n_models = best3nonunique.shape[0]
        if n_models == 3:
            ensemble_models = {}
            for index, row in best3nonunique.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            best3nu_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': 'BestN',
                        'model_count': n_models,
                        'model_metric': 'best_score',
                        'models': ensemble_models,
                    }
                ),
                'TransformationParameters': '{}',
                'Ensemble': 1,
            }
            best3nu_params = pd.DataFrame(best3nu_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates, best3nu_params], axis=0)
        # best 3, by SMAPE, RMSE, SPL
        bestsmape = ens_temp.nsmallest(1, columns=['smape_weighted'])
        bestrmse = ens_temp.nsmallest(2, columns=['rmse_weighted'])
        bestmae = ens_temp.nsmallest(3, columns=['spl_weighted'])
        best3metric = pd.concat([bestsmape, bestrmse, bestmae], axis=0)
        best3metric = best3metric.drop_duplicates().head(3)
        n_models = best3metric.shape[0]
        if n_models == 3:
            ensemble_models = {}
            for index, row in best3metric.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            best3m_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': 'BestN',
                        'model_count': n_models,
                        'model_metric': 'mixed_metric',
                        'models': ensemble_models,
                    }
                ),
                'TransformationParameters': '{}',
                'Ensemble': 1,
            }
            best3m_params = pd.DataFrame(best3m_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates, best3m_params], axis=0)
        # best 3, all must be of different model types
        ens_temp = (
            ens_temp.sort_values('Score', ascending=True, na_position='last')
            .groupby('Model')
            .head(1)
            .reset_index(drop=True)
        )
        best3unique = ens_temp.nsmallest(3, columns=['Score'])
        # only run if there are more than 3 model types available...
        n_models = best3unique.shape[0]
        if n_models == 3:
            ensemble_models = {}
            for index, row in best3unique.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            best3u_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': 'BestN',
                        'model_count': n_models,
                        'model_metric': 'best_score_unique',
                        'models': ensemble_models,
                    }
                ),
                'TransformationParameters': '{}',
                'Ensemble': 1,
            }
            best3u_params = pd.DataFrame(best3u_params, index=[0])
            ensemble_templates = pd.concat(
                [ensemble_templates, best3u_params], axis=0, ignore_index=True
            )

    if 'distance' in ensemble:
        dis_frac = 0.2
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
            ens_per_ts.iloc[:, first_bit: (last_bit + first_bit)].mean(axis=1).idxmin()
        )
        ensemble_models = {}
        best3 = initial_results.model_results[
            initial_results.model_results['ID'].isin([first_model, last_model])
        ].drop_duplicates(
            subset=['Model', 'ModelParameters', 'TransformationParameters']
        )
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters'],
            }
            ensemble_models[row['ID']] = temp_dict
        best3u_params = {
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
        best3u_params = pd.DataFrame(best3u_params, index=[0])
        ensemble_templates = pd.concat(
            [ensemble_templates, best3u_params], axis=0, ignore_index=True
        )

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
            ens_per_ts.iloc[:, first_bit: (last_bit + first_bit)].mean(axis=1).idxmin()
        )
        ensemble_models = {}
        best3 = initial_results.model_results[
            initial_results.model_results['ID'].isin([first_model, last_model])
        ].drop_duplicates(
            subset=['Model', 'ModelParameters', 'TransformationParameters']
        )
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters'],
            }
            ensemble_models[row['ID']] = temp_dict
        best3u_params = {
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
        best3u_params = pd.DataFrame(best3u_params, index=[0])
        ensemble_templates = pd.concat(
            [ensemble_templates, best3u_params], axis=0, ignore_index=True
        )
    if ('horizontal' in ensemble) or ('probabilistic' in ensemble):
        # per_series = model.initial_results.per_series_mae.copy()
        if 'horizontal' in ensemble:
            per_series = initial_results.per_series_mae.copy()
        elif 'probabilistic' in ensemble:
            per_series = initial_results.per_series_spl.copy()
        mods = pd.Series()
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

        ensemble_models = {}
        best3 = initial_results.model_results[
            initial_results.model_results['ID'].isin(mods.index.tolist())
        ].drop_duplicates(
            subset=['Model', 'ModelParameters', 'TransformationParameters']
        )
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters'],
            }
            ensemble_models[row['ID']] = temp_dict
        best3_params = {
            'Model': 'Ensemble',
            'ModelParameters': json.dumps(
                {
                    'model_name': 'BestN',
                    'model_count': n_models,
                    'model_metric': 'horizontal',
                    'models': ensemble_models,
                }
            ),
            'TransformationParameters': '{}',
            'Ensemble': 1,
        }
        best3_params = pd.DataFrame(best3_params, index=[0])
        ensemble_templates = pd.concat(
            [ensemble_templates, best3_params], axis=0, ignore_index=True
        )
    return ensemble_templates


def HorizontalTemplateGenerator(
    per_series,
    model_results,
    forecast_length: int = 14,
    ensemble: str = "horizontal",
    subset_flag: bool = True,
    per_series2=None,
):
    """Generate horizontal ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    ensy = ['horizontal', 'probabilistic', 'hdist']
    if any(x in ensemble for x in ensy):
        if ('horizontal-max' in ensemble) or ('probabilistic-max' in ensemble):
            mods_per_series = per_series.idxmin()
            mods = mods_per_series.unique()
            ensemble_models = {}
            best5 = model_results[
                model_results['ID'].isin(mods.tolist())
            ].drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            metric = 'MAE' if 'horizontal' in ensemble else 'SPL'
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods.shape[0],
                        'model_metric': metric,
                        'models': ensemble_models,
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
        if 'hdist' in ensemble and not subset_flag:
            mods_per_series = per_series.idxmin()
            mods_per_series2 = per_series2.idxmin()
            mods = pd.concat([mods_per_series, mods_per_series2]).unique()
            ensemble_models = {}
            best5 = model_results[
                model_results['ID'].isin(mods.tolist())
            ].drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'hdist'
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods.shape[0],
                        'models': ensemble_models,
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
        if ('horizontal' in ensemble) or ('probabilistic' in ensemble):
            # first generate lists of models by ID that are in shared and no_shared
            no_shared_select = model_results['Model'].isin(no_shared)
            shared_mod_lst = model_results[~no_shared_select]['ID'].tolist()
            no_shared_mod_lst = model_results[no_shared_select]['ID'].tolist()
            lowest_score_mod = [model_results.iloc[model_results['Score'].idxmin()]['ID']]
            per_series[per_series.index.isin(shared_mod_lst)]
            # remove those where idxmin is in no_shared
            shared_maxes = per_series.idxmin().isin(shared_mod_lst)
            shr_mx_cols = shared_maxes[shared_maxes].index
            per_series_shareds = per_series.filter(shr_mx_cols, axis=1)
            # select best n shared models (NEEDS improvement)
            n_md = 5
            use_shared_lst = per_series_shareds.median(axis=1).nsmallest(n_md).index.tolist()
            # combine all of the above as allowed mods
            allowed_list = no_shared_mod_lst + lowest_score_mod + use_shared_lst
            per_series_filter = per_series[per_series.index.isin(allowed_list)]
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
            # concern: choose lots of models, slower to run initial
            mods_per_series = per_series_filter.idxmin()
            mods = mods_per_series.unique()
            ensemble_models = {}
            best5 = model_results[
                model_results['ID'].isin(mods.tolist())
            ].drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            metric = 'TAX' if 'horizontal' in ensemble else 'SPL'
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods.shape[0],
                        'model_metric': metric,
                        'models': ensemble_models,
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
        if ('horizontal-min' in ensemble) or ('probabilistic-min' in ensemble):
            mods = pd.Series()
            per_series_des = per_series.copy()
            n_models = 15
            # choose best per series, remove those series, then choose next best
            for x in range(n_models):
                n_dep = x + 1
                n_dep = (
                    n_dep
                    if per_series_des.shape[0] > n_dep
                    else per_series_des.shape[0]
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

            mods_per_series = per_series.loc[mods.index].idxmin()
            ensemble_models = {}
            best5 = model_results[
                model_results['ID'].isin(mods_per_series.unique().tolist())
            ].drop_duplicates(
                subset=['Model', 'ModelParameters', 'TransformationParameters']
            )
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters'],
                }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            metric = 'MAE' if 'horizontal' in ensemble else 'SPL'
            best5_params = {
                'Model': 'Ensemble',
                'ModelParameters': json.dumps(
                    {
                        'model_name': nomen,
                        'model_count': mods_per_series.unique().shape[0],
                        'model_metric': metric,
                        'models': ensemble_models,
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
