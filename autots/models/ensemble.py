import datetime
import numpy as np
import pandas as pd
import json
from autots.evaluator.auto_model import PredictionObject
from autots.evaluator.auto_model import create_model_id


def Best3Ensemble(ensemble_params, forecasts_list, forecasts,
                  lower_forecasts, upper_forecasts, forecasts_runtime,
                  prediction_interval):
    """Generate forecast for ensemble of 3 models."""
    id_list = list(ensemble_params['models'].keys())
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in id_list]

    ens_df = pd.DataFrame(0, index=forecasts[0].index, columns=forecasts[0].columns)
    for idx, x in enumerate(forecasts):
        if idx in model_indexes:
            ens_df = ens_df + forecasts[idx]
    ens_df = ens_df / len(model_indexes)

    ens_df_lower = pd.DataFrame(0, index=forecasts[0].index, columns=forecasts[0].columns)
    for idx, x in enumerate(lower_forecasts):
        if idx in model_indexes:
            ens_df_lower = ens_df_lower + lower_forecasts[idx]
    ens_df_lower = ens_df_lower / len(model_indexes)

    ens_df_upper = pd.DataFrame(0, index=forecasts[0].index, columns=forecasts[0].columns)
    for idx, x in enumerate(upper_forecasts):
        if idx in model_indexes:
            ens_df_upper = ens_df_upper + upper_forecasts[idx]
    ens_df_upper = ens_df_upper / len(model_indexes)

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in model_indexes:
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result = PredictionObject(model_name="Ensemble",
                                  forecast_length=len(ens_df.index),
                                  forecast_index=ens_df.index,
                                  forecast_columns=ens_df.columns,
                                  lower_forecast=ens_df_lower,
                                  forecast=ens_df, upper_forecast=ens_df_upper,
                                  prediction_interval=prediction_interval,
                                  predict_runtime=datetime.timedelta(0),
                                  fit_runtime=ens_runtime,
                                  model_parameters=ensemble_params
                                  )
    return ens_result


def DistEnsemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):
    """Generate forecast for distance ensemble."""
    first_model_index = forecasts_list.index(ensemble_params['FirstModel'])
    second_model_index = forecasts_list.index(ensemble_params['SecondModel'])
    forecast_length = forecasts[0].shape[0]
    dis_frac = ensemble_params['dis_frac']
    first_bit = int(np.ceil(forecast_length * dis_frac))
    second_bit = int(np.floor(forecast_length * (1 - dis_frac)))

    ens_df = forecasts[first_model_index].head(first_bit).append(forecasts[second_model_index].tail(second_bit))
    ens_df_lower = lower_forecasts[first_model_index].head(first_bit).append(lower_forecasts[second_model_index].tail(second_bit))
    ens_df_upper = upper_forecasts[first_model_index].head(first_bit).append(upper_forecasts[second_model_index].tail(second_bit))

    id_list = list(ensemble_params['models'].keys())
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in id_list]

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in model_indexes:
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result_obj = PredictionObject(model_name="Ensemble",
                                      forecast_length=len(ens_df.index),
                                      forecast_index=ens_df.index,
                                      forecast_columns=ens_df.columns,
                                      lower_forecast=ens_df_lower,
                                      forecast=ens_df,
                                      upper_forecast=ens_df_upper,
                                      prediction_interval=prediction_interval,
                                      predict_runtime=datetime.timedelta(0),
                                      fit_runtime=ens_runtime,
                                      model_parameters=ensemble_params
                                      )
    return ens_result_obj


def HorizontalEnsemble(ensemble_params, forecasts_list, forecasts,
                       lower_forecasts, upper_forecasts, forecasts_runtime,
                       prediction_interval):
    """Generate forecast for per_series ensembling."""
    id_list = list(ensemble_params['models'].keys())
    mod_dic = {x: idx for idx, x in enumerate(forecasts_list) if x in id_list}

    forecast_df, u_forecast_df, l_forecast_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for series, mod_id in ensemble_params['series'].items():
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

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in list(mod_dic.values()):
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result = PredictionObject(model_name="Ensemble",
                                  forecast_length=len(forecast_df.index),
                                  forecast_index=forecast_df.index,
                                  forecast_columns=forecast_df.columns,
                                  lower_forecast=l_forecast_df,
                                  forecast=forecast_df,
                                  upper_forecast=u_forecast_df,
                                  prediction_interval=prediction_interval,
                                  predict_runtime=datetime.timedelta(0),
                                  fit_runtime=ens_runtime,
                                  model_parameters=ensemble_params
                                  )
    return ens_result


def HDistEnsemble(ensemble_params, forecasts_list, forecasts,
                       lower_forecasts, upper_forecasts, forecasts_runtime,
                       prediction_interval):
    """Generate forecast for per_series per distance ensembling."""
    id_list = list(ensemble_params['models'].keys())
    mod_dic = {x: idx for idx, x in enumerate(forecasts_list) if x in id_list}
    forecast_length = forecasts[0].shape[0]
    dist_n = int(np.ceil(ensemble_params['dis_frac'] * forecast_length))
    dist_last = forecast_length - dist_n


    forecast_df, u_forecast_df, l_forecast_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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

    forecast_df2, u_forecast_df2, l_forecast_df2 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
        
    forecast_df = pd.concat([forecast_df.head(dist_n),
                             forecast_df2.tail(dist_last)], axis=0)
    u_forecast_df = pd.concat([u_forecast_df.head(dist_n),
                               u_forecast_df2.tail(dist_last)], axis=0)
    l_forecast_df = pd.concat([l_forecast_df.head(dist_n),
                               l_forecast_df2.tail(dist_last)], axis=0)

    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in list(mod_dic.values()):
            ens_runtime = ens_runtime + forecasts_runtime[idx]

    ens_result = PredictionObject(model_name="Ensemble",
                                  forecast_length=len(forecast_df.index),
                                  forecast_index=forecast_df.index,
                                  forecast_columns=forecast_df.columns,
                                  lower_forecast=l_forecast_df,
                                  forecast=forecast_df,
                                  upper_forecast=u_forecast_df,
                                  prediction_interval=prediction_interval,
                                  predict_runtime=datetime.timedelta(0),
                                  fit_runtime=ens_runtime,
                                  model_parameters=ensemble_params
                                  )
    return ens_result



def EnsembleForecast(ensemble_str, ensemble_params, forecasts_list,
                     forecasts, lower_forecasts, upper_forecasts,
                     forecasts_runtime, prediction_interval):
    """Return PredictionObject for given ensemble method."""
    s3list = ['best3', 'best3horizontal']
    if ensemble_params['model_name'].lower().strip() in s3list:
        ens_forecast = Best3Ensemble(
            ensemble_params, forecasts_list, forecasts, lower_forecasts,
            upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast

    if ensemble_params['model_name'].lower().strip() == 'dist':
        ens_forecast = DistEnsemble(
            ensemble_params, forecasts_list, forecasts, lower_forecasts,
            upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast

    hlist = ['horizontal', 'probabilistic']
    if ensemble_params['model_name'].lower().strip() in hlist:
        ens_forecast = HorizontalEnsemble(
            ensemble_params, forecasts_list, forecasts, lower_forecasts,
            upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast
    
    if ensemble_params['model_name'].lower().strip() == 'hdist':
        ens_forecast = HDistEnsemble(
            ensemble_params, forecasts_list, forecasts, lower_forecasts,
            upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast


def EnsembleTemplateGenerator(initial_results,
                              forecast_length: int = 14,
                              ensemble: str = "simple"
                              ):
    """Generate ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    if 'simple' in ensemble:
        ens_temp = initial_results.model_results.drop_duplicates(subset='ID')
        ens_temp = ens_temp[ens_temp['Ensemble'] == 0]
        # best 3, all can be of same model type
        best3nonunique = ens_temp.nsmallest(3, columns=['Score'])
        if best3nonunique.shape[0] == 3:
            ensemble_models = {}
            for index, row in best3nonunique.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            best3nu_params = {'Model': 'Ensemble',
                              'ModelParameters':
                                  json.dumps({'model_name': 'Best3',
                                              'models': ensemble_models}),
                              'TransformationParameters': '{}',
                              'Ensemble': 1}
            best3nu_params = pd.DataFrame(best3nu_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best3nu_params],
                                           axis=0)
        # best 3, by SMAPE, RMSE, SPL
        bestsmape = ens_temp.nsmallest(1, columns=['smape_weighted'])
        bestrmse = ens_temp.nsmallest(2, columns=['rmse_weighted'])
        bestmae = ens_temp.nsmallest(3, columns=['spl_weighted'])
        best3metric = pd.concat([bestsmape, bestrmse, bestmae], axis=0)
        best3metric = best3metric.drop_duplicates().head(3)
        if best3metric.shape[0] == 3:
            ensemble_models = {}
            for index, row in best3metric.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            best3m_params = {'Model': 'Ensemble',
                             'ModelParameters':
                                 json.dumps({'model_name': 'Best3',
                                             'models': ensemble_models}),
                                 'TransformationParameters': '{}',
                                 'Ensemble': 1}
            best3m_params = pd.DataFrame(best3m_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best3m_params],
                                           axis=0)
        # best 3, all must be of different model types
        ens_temp = ens_temp.sort_values(
            'Score', ascending=True, na_position='last'
            ).groupby('Model').head(1).reset_index(drop=True)
        best3unique = ens_temp.nsmallest(3, columns=['Score'])
        # only run if there are more than 3 model types available...
        if best3unique.shape[0] == 3:
            ensemble_models = {}
            for index, row in best3unique.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            best3u_params = {'Model': 'Ensemble',
                             'ModelParameters':
                                 json.dumps({'model_name': 'Best3',
                                             'models': ensemble_models}),
                             'TransformationParameters': '{}',
                             'Ensemble': 1}
            best3u_params = pd.DataFrame(best3u_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best3u_params],
                                           axis=0, ignore_index=True)

    if 'distance' in ensemble:
        dis_frac = 0.2
        first_bit = int(np.ceil(forecast_length * dis_frac))
        last_bit = int(np.floor(forecast_length * (1 - dis_frac)))
        not_ens_list = initial_results.model_results[initial_results.model_results['Ensemble'] == 0]['ID'].tolist()
        ens_per_ts = initial_results.per_timestamp_smape[initial_results.per_timestamp_smape.index.isin(not_ens_list)]
        first_model = ens_per_ts.iloc[:, 0:first_bit].mean(axis=1).idxmin()
        last_model = ens_per_ts.iloc[:, first_bit:(last_bit + first_bit)].mean(axis=1).idxmin()
        ensemble_models = {}
        best3 = initial_results.model_results[initial_results.model_results['ID'].isin([first_model, last_model])].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters']
                }
            ensemble_models[row['ID']] = temp_dict
        best3u_params = {'Model': 'Ensemble',
                         'ModelParameters':
                             json.dumps({'model_name': 'Dist',
                                         'models': ensemble_models,
                                         'dis_frac': dis_frac,
                                         'FirstModel': first_model,
                                         'SecondModel': last_model
                                         }),
                         'TransformationParameters': '{}',
                         'Ensemble': 1}
        best3u_params = pd.DataFrame(best3u_params, index=[0])
        ensemble_templates = pd.concat([ensemble_templates,
                                        best3u_params],
                                       axis=0, ignore_index=True)

        dis_frac = 0.5
        first_bit = int(np.ceil(forecast_length * dis_frac))
        last_bit = int(np.floor(forecast_length * (1 - dis_frac)))
        not_ens_list = initial_results.model_results[initial_results.model_results['Ensemble'] == 0]['ID'].tolist()
        ens_per_ts = initial_results.per_timestamp_smape[initial_results.per_timestamp_smape.index.isin(not_ens_list)]
        first_model = ens_per_ts.iloc[:, 0:first_bit].mean(axis=1).idxmin()
        last_model = ens_per_ts.iloc[:, first_bit:(last_bit + first_bit)].mean(axis=1).idxmin()
        ensemble_models = {}
        best3 = initial_results.model_results[initial_results.model_results['ID'].isin([first_model, last_model])].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters']
                }
            ensemble_models[row['ID']] = temp_dict
        best3u_params = {'Model': 'Ensemble',
                         'ModelParameters':
                             json.dumps({'model_name': 'Dist',
                                         'models': ensemble_models,
                                         'dis_frac': dis_frac,
                                         'FirstModel': first_model,
                                         'SecondModel': last_model
                                         }),
                         'TransformationParameters': '{}',
                         'Ensemble': 1}
        best3u_params = pd.DataFrame(best3u_params, index=[0])
        ensemble_templates = pd.concat([ensemble_templates,
                                        best3u_params],
                                       axis=0, ignore_index=True)
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
            n_dep = n_dep if per_series_des.shape[0] > n_dep else per_series_des.shape[0]
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
        best3 = initial_results.model_results[initial_results.model_results['ID'].isin(mods.index.tolist())].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
        for index, row in best3.iterrows():
            temp_dict = {
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': row['TransformationParameters']
                }
            ensemble_models[row['ID']] = temp_dict
        best3_params = {'Model': 'Ensemble',
                         'ModelParameters':
                             json.dumps({'model_name': 'best3horizontal',
                                         'models': ensemble_models
                                         }),
                         'TransformationParameters': '{}',
                         'Ensemble': 1}
        best3_params = pd.DataFrame(best3_params, index=[0])
        ensemble_templates = pd.concat([ensemble_templates,
                                        best3_params],
                                       axis=0, ignore_index=True)
    return ensemble_templates


def HorizontalTemplateGenerator(per_series, model_results,
                                forecast_length: int = 14,
                                ensemble: str = "horizontal",
                                subset_flag: bool = True,
                                per_series2 = None
                                ):
    """Generate horizontal ensemble templates given a table of results."""
    ensemble_templates = pd.DataFrame()
    ensy = ['horizontal', 'probabilistic', 'hdist']
    if any(x in ensemble for x in ensy) and not subset_flag:
        if ('horizontal-max' in ensemble) or ('probabilistic-max' in ensemble):
            mods_per_series = per_series.idxmin()
            mods = mods_per_series.unique()
            ensemble_models = {}
            best5 = model_results[model_results['ID'].isin(mods.tolist())].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            best5_params = {'Model': 'Ensemble',
                            'ModelParameters':
                                json.dumps({'model_name': nomen,
                                            'model_count': mods.shape[0],
                                            'models': ensemble_models,
                                            'series': mods_per_series.to_dict()
                                            }),
                            'TransformationParameters': '{}',
                            'Ensemble': 2}
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best5_params],
                                           axis=0, ignore_index=True)
        elif ('hdist' in ensemble):
            mods_per_series = per_series.idxmin()
            mods_per_series2 = per_series2.idxmin()
            mods = pd.concat([mods_per_series, mods_per_series2]).unique()
            ensemble_models = {}
            best5 = model_results[model_results['ID'].isin(mods.tolist())].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'hdist'
            best5_params = {'Model': 'Ensemble',
                            'ModelParameters':
                                json.dumps({'model_name': nomen,
                                            'model_count': mods.shape[0],
                                            'models': ensemble_models,
                                            'dis_frac': 0.3,
                                            'series1': mods_per_series.to_dict(),
                                            'series2': mods_per_series2.to_dict()
                                            }),
                            'TransformationParameters': '{}',
                            'Ensemble': 2}
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best5_params],
                                           axis=0, ignore_index=True)
        else:
            """
            mods = pd.Series()
            per_series_des = per_series.copy()
            n_models = 5
            # choose best per series, remove those series, then choose next best
            for x in range(n_models):
                n_dep = 5 if x < 2 else 10
                n_dep = n_dep if per_series_des.shape[0] > n_dep else per_series_des.shape[0]
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
                rm_cols = rm_cols.dropna(how='all',axis=1).columns
                per_series_des = per_series.copy().drop(mods.index, axis=0)
                per_series_des = per_series_des.drop(rm_cols, axis=1)
                if per_series_des.shape[1] == 0:
                    per_series_des = per_series.copy().drop(mods.index, axis=0)

            mods_per_series = per_series.loc[mods.index].idxmin()
            ensemble_models = {}
            best5 = model_results[model_results['ID'].isin(mods_per_series.unique().tolist())].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            best5_params = {'Model': 'Ensemble',
                            'ModelParameters':
                                json.dumps({'model_name': nomen,
                                            'model_count': mods_per_series.unique().shape[0],
                                            'models': ensemble_models,
                                            'series': mods_per_series.to_dict()
                                            }),
                            'TransformationParameters': '{}',
                            'Ensemble': 2}
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best5_params],
                                           axis=0, ignore_index=True)
            """
            mods = pd.Series()
            per_series_des = per_series.copy()
            n_models = 15
            # choose best per series, remove those series, then choose next best
            for x in range(n_models):
                n_dep = x + 1
                n_dep = n_dep if per_series_des.shape[0] > n_dep else per_series_des.shape[0]
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
                rm_cols = rm_cols.dropna(how='all',axis=1).columns
                per_series_des = per_series.copy().drop(mods.index, axis=0)
                per_series_des = per_series_des.drop(rm_cols, axis=1)
                if per_series_des.shape[1] == 0:
                    per_series_des = per_series.copy().drop(mods.index, axis=0)

            mods_per_series = per_series.loc[mods.index].idxmin()
            ensemble_models = {}
            best5 = model_results[model_results['ID'].isin(mods_per_series.unique().tolist())].drop_duplicates(subset=['Model', 'ModelParameters', 'TransformationParameters'])
            for index, row in best5.iterrows():
                temp_dict = {
                    'Model': row['Model'],
                    'ModelParameters': row['ModelParameters'],
                    'TransformationParameters': row['TransformationParameters']
                    }
                ensemble_models[row['ID']] = temp_dict
            nomen = 'Horizontal' if 'horizontal' in ensemble else 'Probabilistic'
            best5_params = {'Model': 'Ensemble',
                            'ModelParameters':
                                json.dumps({'model_name': nomen,
                                            'model_count': mods_per_series.unique().shape[0],
                                            'models': ensemble_models,
                                            'series': mods_per_series.to_dict()
                                            }),
                            'TransformationParameters': '{}',
                            'Ensemble': 2}
            best5_params = pd.DataFrame(best5_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best5_params],
                                           axis=0, ignore_index=True)            
    return ensemble_templates
