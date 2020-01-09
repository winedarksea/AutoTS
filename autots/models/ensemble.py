import datetime
import numpy as np
import pandas as pd
import json
from autots.evaluator.auto_model import PredictionObject
from autots.evaluator.auto_model import create_model_id    


def Best3Ensemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):
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
    
    ens_result = PredictionObject(model_name = "Best3Ensemble",
                                          forecast_length=len(ens_df.index),
                                          forecast_index = ens_df.index,
                                          forecast_columns = ens_df.columns,
                                          lower_forecast=ens_df_lower,
                                          forecast=ens_df, upper_forecast=ens_df_upper,
                                          prediction_interval= prediction_interval,
                                          predict_runtime= datetime.timedelta(0),
                                          fit_runtime = ens_runtime,
                                          model_parameters = ensemble_params
                                          )
    return ens_result

def Dist2080Ensemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):   

    first_model_index = forecasts_list.index(ensemble_params['FirstModel'])
    last_model_index = forecasts_list.index(ensemble_params['LastModel'])
    forecast_length = forecasts[0].shape[0]
    first_bit = int(np.ceil(forecast_length * 0.2))
    last_bit = int(np.floor(forecast_length * 0.8))
    
    ens_df = forecasts[first_model_index].head(first_bit).append(forecasts[last_model_index].tail(last_bit))
    ens_df_lower = lower_forecasts[first_model_index].head(first_bit).append(lower_forecasts[last_model_index].tail(last_bit))
    ens_df_upper = upper_forecasts[first_model_index].head(first_bit).append(upper_forecasts[last_model_index].tail(last_bit))
    
    id_list = list(ensemble_params['models'].keys())
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in id_list]
    
    ens_runtime = datetime.timedelta(0)
    for idx, x in enumerate(forecasts_runtime):
        if idx in model_indexes:
            ens_runtime = ens_runtime + forecasts_runtime[idx]
    
    ens_result_obj = PredictionObject(model_name = "Dist2080Ensemble",
                                          forecast_length=len(ens_df.index),
                                          forecast_index = ens_df.index,
                                          forecast_columns = ens_df.columns,
                                          lower_forecast=ens_df_lower,
                                          forecast=ens_df, upper_forecast=ens_df_upper,
                                          prediction_interval= prediction_interval,
                                          predict_runtime= datetime.timedelta(0),
                                          fit_runtime = ens_runtime,
                                          model_parameters = ensemble_params
                                          )
    return ens_result_obj

def EnsembleForecast(ensemble_str, ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):
    """
    Returns PredictionObject for given ensemble method
    """
    if ensemble_str.lower().strip() == 'best3ensemble':
        #from autots.models.ensemble import Best3Ensemble
        ens_forecast = Best3Ensemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast
    
    if ensemble_str.lower().strip() == "dist2080ensemble":
        #from autots.models.ensemble import Dist2080Ensemble
        ens_forecast = Dist2080Ensemble(ensemble_params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
        return ens_forecast
    

from autots.evaluator.auto_model import TemplateEvalObject
from autots.evaluator.metrics import PredictionEval
def EnsembleEvaluate(ensemble_forecasts_list: list, df_test, weights, model_count: int = 0):
    """
    Accepts a list of Prediction Objects and returns a TemplateEvalObject
    """
    ens_eval = TemplateEvalObject()
    ens_eval.model_count = model_count
    for ensemble_forecast in ensemble_forecasts_list:
        try:
            ens_eval.model_count += 1
            print("Model Number: {} with model {}".format(str(ens_eval.model_count), ensemble_forecast.model_name))
            model_error = PredictionEval(ensemble_forecast, df_test, series_weights = weights)
            model_id = create_model_id(ensemble_forecast.model_name, ensemble_forecast.model_parameters, ensemble_forecast.transformation_parameters)
            total_runtime = ensemble_forecast.fit_runtime + ensemble_forecast.predict_runtime + ensemble_forecast.transformation_runtime
            result = pd.DataFrame({
                    'ID': model_id,
                    'Model': ensemble_forecast.model_name,
                    'ModelParameters': json.dumps(ensemble_forecast.model_parameters),
                    'TransformationParameters': json.dumps(ensemble_forecast.transformation_parameters),
                    'TransformationRuntime': ensemble_forecast.transformation_runtime,
                    'FitRuntime': ensemble_forecast.fit_runtime,
                    'PredictRuntime': ensemble_forecast.predict_runtime,
                    'TotalRuntime': total_runtime,
                    'Ensemble': 1,
                    'Exceptions': np.nan,
                    'Runs': 1
                    }, index = [0])
            a = pd.DataFrame(model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')).transpose()
            result = pd.concat([result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis = 1)
            
            ens_eval.model_results = pd.concat([ens_eval.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['smape']).transpose()
            temp.index = result['ID'] 
            ens_eval.model_results_per_timestamp_smape = ens_eval.model_results_per_timestamp_smape.append(temp)
            temp = pd.DataFrame(model_error.per_timestamp_metrics.loc['mae']).transpose()
            temp.index = result['ID']  
            ens_eval.model_results_per_timestamp_mae = ens_eval.model_results_per_timestamp_mae.append(temp)
            temp = pd.DataFrame(model_error.per_series_metrics.loc['smape']).transpose()
            temp.index = result['ID']            
            ens_eval.model_results_per_series_smape = ens_eval.model_results_per_series_smape.append(temp)
            temp = pd.DataFrame(model_error.per_series_metrics.loc['mae']).transpose()
            temp.index = result['ID']
            ens_eval.model_results_per_series_mae = ens_eval.model_results_per_series_mae.append(temp)
            
        except Exception as e:
            model_str = ensemble_forecast.model_name
            result = pd.DataFrame({
                'ID': create_model_id(model_str, {}, {}),
                'Model': model_str,
                'ModelParameters': json.dumps({}),
                'TransformationParameters': json.dumps({}),
                'Ensemble': 1,
                'TransformationRuntime': datetime.timedelta(0),
                'FitRuntime': datetime.timedelta(0),
                'PredictRuntime': datetime.timedelta(0),
                'TotalRuntime': datetime.timedelta(0),
                'Exceptions': str(e),
                'Runs': 1
                }, index = [0])
            ens_eval.model_results = pd.concat([ens_eval.model_results, result], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    return ens_eval