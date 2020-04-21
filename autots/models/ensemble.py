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
                                      forecast=ens_df, upper_forecast=ens_df_upper,
                                      prediction_interval=prediction_interval,
                                      predict_runtime=datetime.timedelta(0),
                                      fit_runtime=ens_runtime,
                                      model_parameters=ensemble_params
                                      )
    return ens_result_obj


def HorizontalEnsemble(ensemble_params, forecasts_list, forecasts,
                  lower_forecasts, upper_forecasts, forecasts_runtime,
                  prediction_interval):
    """Generate forecast for ensemble of 3 models."""
    id_list = list(ensemble_params['models'].keys())
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in id_list]

    ensemble_params['series']

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


def EnsembleForecast(ensemble_str, ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):
    """
    Returns PredictionObject for given ensemble method
    """
    if ensemble_params['model_name'].lower().strip() == 'best3':
        ens_forecast = Best3Ensemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast

    if ensemble_params['model_name'].lower().strip() == 'dist':
        ens_forecast = DistEnsemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast
    
    if ensemble_params['model_name'].lower().strip() == 'horizontal':
        ens_forecast = HorizontalEnsemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast


def EnsembleTemplateGenerator(initial_results,
                              forecast_length: int = 14,
                              ensemble: str = "simple",
                              subset_flag: bool = True
                              ):
    ensemble_templates = pd.DataFrame()
    if 'simple' in ensemble:
        ens_temp = initial_results.model_results
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
        # best 3, by SMAPE, RMSE, MAE
        bestsmape = ens_temp.nsmallest(1, columns=['smape_weighted'])
        bestrmse = ens_temp.nsmallest(2, columns=['rmse_weighted'])
        bestmae = ens_temp.nsmallest(3, columns=['mae_weighted'])
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
        ens_temp = ens_temp.sort_values('Score', ascending=True, na_position='last').groupby('Model').head(1).reset_index(drop=True)
        best3unique = ens_temp.nsmallest(3, columns=['Score'])
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
    if 'horizontal' in ensemble:
        # per_series = model.initial_results.per_series_mae.copy()
        per_series = initial_results.per_series_mae.copy()
        per_series_des = initial_results.per_series_mae.copy()
        max_models = 3
        n_depth = 5 if per_series.shape[0] > 5 else per_series.shape[0]
        models_pos = []
        for _ in range(n_depth):
            models_pos.extend(per_series_des.idxmin().tolist())
            per_series_des[per_series_des == per_series_des.min()] = np.nan
        mods = pd.Series(models_pos).value_counts()
        mods = mods.sort_values(ascending=False).head(max_models)

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
                             json.dumps({'model_name': 'Best3',
                                         'models': ensemble_models
                                         }),
                         'TransformationParameters': '{}',
                         'Ensemble': 1}
        best3_params = pd.DataFrame(best3_params, index=[0])
        ensemble_templates = pd.concat([ensemble_templates,
                                        best3_params],
                                       axis=0, ignore_index=True)
        if not subset_flag:
            per_series_des = initial_results.per_series_mae.copy()
            max_models = 5
            n_depth = 5 if per_series.shape[0] > 5 else per_series.shape[0]
            models_pos = []
            for _ in range(n_depth):
                models_pos.extend(per_series_des.idxmin().tolist())
                per_series_des[per_series_des == per_series_des.min()] = np.nan
            mods = pd.Series(models_pos).value_counts()
            mods = mods.sort_values(ascending=False).head(max_models)
            mods_per_series = per_series.loc[mods.index].idxmin()
    
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
                                 json.dumps({'model_name': 'Horizontal',
                                             'models': ensemble_models,
                                             'series': mods_per_series.to_dict()
                                             }),
                             'TransformationParameters': '{}',
                             'Ensemble': 1}
            best3_params = pd.DataFrame(best3_params, index=[0])
            ensemble_templates = pd.concat([ensemble_templates,
                                            best3_params],
                                           axis=0, ignore_index=True)
    return ensemble_templates
