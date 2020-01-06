import numpy as np
import pandas as pd
import datetime
import json

forecast_length = 14

weighted = False
frequency = '1D'
prediction_interval = 0.9
no_negatives = False
seed = 425
holiday_country = 'US'
ensemble = True

df_long = pd.read_csv("SampleTimeSeries.csv")
df_long['date'] = pd.to_datetime(df_long['date'], infer_datetime_format = True)


#from autots.datasets.fred import get_fred_data
#df_long = get_fred_data('XXXXXXXXXXXxx')


from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = frequency, na_tolerance = 0.95,
                       drop_data_older_than_periods = 1000, aggfunc = 'first')


preord_regressor = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index )



from autots.tools.shaping import values_to_numeric
categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

from autots.tools.profile import data_profile
profile_df = data_profile(df_wide_numeric)

from autots.tools.shaping import categorical_inverse
df_cat_inverse = categorical_inverse(categorical_transformer, df_wide_numeric)


from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)
preord_regressor_train = preord_regressor[df_train.index]
preord_regressor_test = preord_regressor[df_test.index]

if weighted == False:
    weights = {x:1 for x in df_train.columns}



from autots.evaluator.auto_model import ModelNames
from autots.tools.transform import RandomTransform
from autots.evaluator.auto_model import ModelMonster
def RandomTemplate(n: int = 10):
    """"
    Returns a template dataframe of randomly generated transformations, models, and hyperparameters
    
    Args:
        n (int): number of random models to return
    """
    n = abs(int(n))
    template = pd.DataFrame()
    counter = 0
    while (len(template.index) < n):
        model_str = np.random.choice(ModelNames)
        param_dict = ModelMonster(model_str).get_new_params()
        trans_dict = RandomTransform()
        row = pd.DataFrame({
                'Model': model_str,
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0
                }, index = [0])
        template = pd.concat([template, row], axis = 0, ignore_index = True)
        template.drop_duplicates(inplace = True)
        counter += 1
        if counter > (n * 3):
            break
    return template
    


model_results = pd.DataFrame()
model_results_per_timestamp_smape = pd.DataFrame()
model_results_per_timestamp_mae = pd.DataFrame()
model_results_per_series_mae = pd.DataFrame()
model_results_per_series_smape = pd.DataFrame()

forecasts = []
upper_forecasts = []
lower_forecasts = []
forecasts_list = []
forecasts_runtime = []

from autots.evaluator.auto_model import TemplateWizard    

model_count = 0
template = RandomTemplate(30)
template_result = TemplateWizard(template, df_train, df_test, weights,
                                 model_count = model_count, ensemble = ensemble, 
                                 forecast_length = forecast_length, frequency=frequency, 
                                  prediction_interval=prediction_interval, 
                                  no_negatives=no_negatives,
                                  preord_regressor_train = preord_regressor_train,
                                  preord_regressor_forecast = preord_regressor_test, 
                                  holiday_country = holiday_country,
                                  startTimeStamps = profile_df.loc['FirstDate'])
model_count = template_result.model_count
model_results = pd.concat([model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
model_results_per_timestamp_smape = model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
model_results_per_timestamp_mae = model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
model_results_per_series_smape = model_results_per_series_smape.append(template_result.model_results_per_series_smape)
model_results_per_series_mae = model_results_per_series_mae.append(template_result.model_results_per_series_mae)
if ensemble:
    forecasts_list.extend(template_result.forecasts_list)
    forecasts_runtime.extend(template_result.forecasts_runtime)
    forecasts.extend(template_result.forecasts)
    upper_forecasts.extend(template_result.upper_forecasts)
    lower_forecasts.extend(template_result.lower_forecasts)


if ensemble:
    best3 = model_results[model_results['Ensemble'] == 0].nsmallest(3, columns = ['smape'])
    ensemble_models = {}
    for index, row in best3.iterrows():
        temp_dict = {'Model': row['Model'],
         'ModelParameters': row['ModelParameters'],
         'TransformationParameters': row['TransformationParameters']
         }
        ensemble_models[row['ID']] = temp_dict
    best3params = {'models': ensemble_models}    
    
    from autots.models.ensemble import EnsembleForecast
    ens_forecast = EnsembleForecast("Best3Ensemble", best3params, forecasts_list, forecasts, lower_forecasts, upper_forecasts, forecasts_runtime, prediction_interval)
    
    
    first_bit = int(np.ceil(forecast_length * 0.2))
    last_bit = int(np.floor(forecast_length * 0.8))
    ens_per_ts = model_results_per_timestamp_smape[model_results_per_timestamp_smape.index.isin(model_results[model_results['Ensemble'] == 0]['ID'].tolist())]
    first_model = ens_per_ts.iloc[:,0:first_bit].mean(axis = 1).idxmin()
    last_model = ens_per_ts.iloc[:,first_bit:(last_bit + first_bit)].mean(axis = 1).idxmin()
    ensemble_models = {}
    best3 = model_results[model_results['ID'].isin([first_model,last_model])].drop_duplicates(subset = ['Model','ModelParameters','TransformationParameters'])
    for index, row in best3.iterrows():
        temp_dict = {'Model': row['Model'],
         'ModelParameters': row['ModelParameters'],
         'TransformationParameters': row['TransformationParameters']
         }
        ensemble_models[row['ID']] = temp_dict
    dist2090params = {'models': ensemble_models,
                      'FirstModel':first_model,
                      'LastModel':last_model} 


"""
unpack ensembles if in template!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Verbosity
Ensemble 2080
Consolidate repeat models in model_results, and per_series/per_timestamp

Genetic Algorithm
    Intial template (random, expert, expert+random) (user only, user add-on)
    For X generations:
        Select N best algorithms (by multiple metrics, MAE, SMAPE)
            Generate Y new models
            Generate random new parameters
            recombine existing and random
            Check if combination already tested
    Pass Z models into Multiple Validation
Multiple validation
Predict method

Combine multiple metrics into 'score'
Ranked by score
    nearest neighbor score - is time much slower than similar? is MAE much better than similar for SMAPE?

Things needing testing:
    Confirm per_series weighting
    Passing in Start Dates - (Test)


"""



















from autots.tools.shaping import subset_series

df_subset = subset_series(df_wide_numeric, n = 10, na_tolerance = 0.5, random_state = 425)




# to gluon ds
# to xgboost ds

# train/test split + cross validation
    # to be more sensitive to NaNs in train/test split count
    # option to skip if na_tolerance is not met in both train and test.
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# then fill na
# *args, **kwargs
