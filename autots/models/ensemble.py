import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import PredictionObject

def Best3Ensemble(model_results, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval):
    best3 = model_results.nsmallest(3, columns = ['smape'])
    ensemble_models = {}
    for index, row in best3.iterrows():
        temp_dict = {'Model': row['Model'],
         'ModelParameters': row['ModelParameters'],
         'TransformationParameters': row['TransformationParameters']
         }
        ensemble_models[row['ID']] = temp_dict
    model_indexes = [idx for idx, x in enumerate(forecasts_list) if x in best3['ID'].tolist()]
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
                                          model_parameters = {'models': ensemble_models}
                                          )
    return ens_result
