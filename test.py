import numpy as np
import pandas as pd
import datetime
import json
import copy

forecast_length = 14

weights = {'categoricalDayofWeek': 1,
           'randomNegative': 1,
         'sp500high': 4,
         'wabashaTemp': 1}
weighted = False
frequency = '1D'
aggfunc = 'first'
prediction_interval = 0.9
no_negatives = False
random_seed = 425
holiday_country = 'US'
ensemble = True
subset = 200
na_tolerance = 0.95
metric_weighting = {'smape_weighting' : 10, 'mae_weighting' : 1,
    'rmse_weighting' : 5, 'containment_weighting' : 1, 'runtime_weighting' : 0}
drop_most_recent = 1
num_validations = 2
models_to_validate = 10
# 'backwards' or 'even'
validation_method = 'even'
max_generations = 10
verbose = 1

random_seed = abs(int(random_seed))
import random
random.seed(random_seed)
np.random.seed(random_seed)

template_cols = ['Model','ModelParameters','TransformationParameters','Ensemble']

from autots.datasets import load_toy_daily
df_long = load_toy_daily()

#from autots.datasets.fred import get_fred_data
#df_long = get_fred_data('XXXXXXXXXXXxx')


from autots.tools.shaping import long_to_wide

df_wide = long_to_wide(df_long, date_col = 'date', value_col = 'value',
                       id_col = 'series_id', frequency = frequency, na_tolerance = na_tolerance,
                       drop_data_older_than_periods = 1000, aggfunc = aggfunc,
                       drop_most_recent = drop_most_recent)

if weighted == False:
    weights = {x:1 for x in df_wide.columns}
if weighted == True:
    # handle not all weights being provided
    weights = {col:(weights[col] if col in weights else 1) for col in df_wide.columns}
    # handle non-numeric inputs
    weights = {key:(abs(float(weights[key])) if str(weights[key]).isdigit() else 1) for key in weights}
    
preord_regressor = pd.Series(np.random.randint(0, 100, size = len(df_wide.index)), index = df_wide.index )



from autots.tools.shaping import values_to_numeric
categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

from autots.tools.profile import data_profile
profile_df = data_profile(df_wide_numeric)

from autots.tools.shaping import categorical_inverse
df_cat_inverse = categorical_inverse(categorical_transformer, df_wide_numeric)



from autots.tools.shaping import subset_series
df_subset = subset_series(df_wide_numeric, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)

if weighted == False:
    current_weights = {x:1 for x in df_subset.columns}
if weighted == True:
    current_weights = {x: weights[x] for x in df_subset.columns}
    
from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
preord_regressor_train = preord_regressor[df_train.index]
preord_regressor_test = preord_regressor[df_test.index]




def generate_score(model_results, metric_weighting: dict = {}, prediction_interval: float = 0.9):
    """
    Generates score based on relative accuracies
    """
    try:
        smape_weighting = metric_weighting['smape_weighting']
    except:
        smape_weighting = 9
    try:
        mae_weighting = metric_weighting['mae_weighting']
    except:
        mae_weighting = 1
    try:
        rmse_weighting = metric_weighting['rmse_weighting']
    except:
        rmse_weighting = 5
    try:
        containment_weighting = metric_weighting['containment_weighting']
    except:
        containment_weighting = 1
    try:
        runtime_weighting = metric_weighting['runtime_weighting'] * 0.1
    except:
        runtime_weighting = 0.5
    smape_score = model_results['smape_weighted']/model_results['smape_weighted'].min(skipna=True) # smaller better
    rmse_score = model_results['rmse_weighted']/model_results['rmse_weighted'].min(skipna=True) # smaller better
    mae_score = model_results['mae_weighted']/model_results['mae_weighted'].min(skipna=True) # smaller better
    containment_score = (abs(prediction_interval - model_results['containment'])) # from 0 to 1, smaller better
    runtime_score = model_results['TotalRuntime']/(model_results['TotalRuntime'].min(skipna=True) + datetime.timedelta(minutes = 1)) # smaller better
    return (smape_score * smape_weighting) + (mae_score * mae_weighting) + (rmse_score * rmse_weighting) + (containment_score * containment_weighting) + (runtime_score * runtime_weighting)



from autots.evaluator.auto_model import TemplateEvalObject
main_results = TemplateEvalObject()

from autots.evaluator.auto_model import NewGeneticTemplate
from autots.evaluator.auto_model import RandomTemplate
from autots.evaluator.auto_model import TemplateWizard    
from autots.evaluator.auto_model import unpack_ensemble_models
model_count = 0
initial_template = RandomTemplate(40)
initial_template = unpack_ensemble_models(initial_template, template_cols, keep_ensemble = False)
submitted_parameters = initial_template.copy()
template_result = TemplateWizard(initial_template, df_train, df_test, current_weights,
                                 model_count = model_count, ensemble = ensemble, 
                                 forecast_length = forecast_length, frequency=frequency, 
                                  prediction_interval=prediction_interval, 
                                  no_negatives=no_negatives,
                                  preord_regressor_train = preord_regressor_train,
                                  preord_regressor_forecast = preord_regressor_test, 
                                  holiday_country = holiday_country,
                                  startTimeStamps = profile_df.loc['FirstDate'],
                                  template_cols = template_cols, random_seed = random_seed, verbose = verbose)
model_count = template_result.model_count
main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting,prediction_interval = prediction_interval)
main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
if ensemble:
    main_results.forecasts_list.extend(template_result.forecasts_list)
    main_results.forecasts_runtime.extend(template_result.forecasts_runtime)
    main_results.forecasts.extend(template_result.forecasts)
    main_results.upper_forecasts.extend(template_result.upper_forecasts)
    main_results.lower_forecasts.extend(template_result.lower_forecasts)


current_generation = 0
# eventually, have this break if accuracy improvement plateaus before max_generations
while current_generation < max_generations:
    current_generation += 1
    if verbose > 0:
        print("New generation: {}".format(current_generation))
    new_template = NewGeneticTemplate(main_results.model_results, submitted_parameters=submitted_parameters, sort_column = "Score", 
                       sort_ascending = True, max_results = 40, top_n = 15, template_cols=template_cols)
    submitted_parameters = pd.concat([submitted_parameters, new_template], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    
    template_result = TemplateWizard(new_template, df_train, df_test, current_weights,
                                 model_count = model_count, ensemble = ensemble, 
                                 forecast_length = forecast_length, frequency=frequency, 
                                  prediction_interval=prediction_interval, 
                                  no_negatives=no_negatives,
                                  preord_regressor_train = preord_regressor_train,
                                  preord_regressor_forecast = preord_regressor_test, 
                                  holiday_country = holiday_country,
                                  startTimeStamps = profile_df.loc['FirstDate'],
                                  template_cols = template_cols,
                                  random_seed = random_seed, verbose = verbose)
    model_count = template_result.model_count
    main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
    
    main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
    main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
    main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
    main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
    if ensemble:
        main_results.forecasts_list.extend(template_result.forecasts_list)
        main_results.forecasts_runtime.extend(template_result.forecasts_runtime)
        main_results.forecasts.extend(template_result.forecasts)
        main_results.upper_forecasts.extend(template_result.upper_forecasts)
        main_results.lower_forecasts.extend(template_result.lower_forecasts)




if ensemble:
    ensemble_forecasts_list = []
    
    best3 = main_results.model_results[main_results.model_results['Ensemble'] == 0].nsmallest(3, columns = ['Score'])
    ensemble_models = {}
    for index, row in best3.iterrows():
        temp_dict = {'Model': row['Model'],
         'ModelParameters': row['ModelParameters'],
         'TransformationParameters': row['TransformationParameters']
         }
        ensemble_models[row['ID']] = temp_dict
    best3params = {'models': ensemble_models}    
    
    from autots.models.ensemble import EnsembleForecast
    best3_ens_forecast = EnsembleForecast("Best3Ensemble", best3params, main_results.forecasts_list, main_results.forecasts, main_results.lower_forecasts, main_results.upper_forecasts, main_results.forecasts_runtime, prediction_interval)
    ensemble_forecasts_list.append(best3_ens_forecast)
    
    first_bit = int(np.ceil(forecast_length * 0.2))
    last_bit = int(np.floor(forecast_length * 0.8))
    ens_per_ts = main_results.model_results_per_timestamp_smape[main_results.model_results_per_timestamp_smape.index.isin(main_results.model_results[main_results.model_results['Ensemble'] == 0]['ID'].tolist())]
    first_model = ens_per_ts.iloc[:,0:first_bit].mean(axis = 1).idxmin()
    last_model = ens_per_ts.iloc[:,first_bit:(last_bit + first_bit)].mean(axis = 1).idxmin()
    ensemble_models = {}
    best3 = main_results.model_results[main_results.model_results['ID'].isin([first_model,last_model])].drop_duplicates(subset = ['Model','ModelParameters','TransformationParameters'])
    for index, row in best3.iterrows():
        temp_dict = {'Model': row['Model'],
         'ModelParameters': row['ModelParameters'],
         'TransformationParameters': row['TransformationParameters']
         }
        ensemble_models[row['ID']] = temp_dict
    dist2080params = {'models': ensemble_models,
                      'FirstModel':first_model,
                      'LastModel':last_model} 
    dist2080_ens_forecast = EnsembleForecast("Dist2080Ensemble", dist2080params, main_results.forecasts_list, main_results.forecasts, main_results.lower_forecasts, main_results.upper_forecasts, main_results.forecasts_runtime, prediction_interval)
    ensemble_forecasts_list.append(dist2080_ens_forecast)

    from autots.models.ensemble import EnsembleEvaluate
    ens_template_result = EnsembleEvaluate(ensemble_forecasts_list, df_test = df_test, weights = current_weights, model_count = model_count)
    
    model_count = ens_template_result.model_count
    main_results.model_results = pd.concat([main_results.model_results, ens_template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    main_results.model_results['Score'] = generate_score(main_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
    main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(ens_template_result.model_results_per_timestamp_smape)
    main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(ens_template_result.model_results_per_timestamp_mae)
    main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(ens_template_result.model_results_per_series_smape)
    main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(ens_template_result.model_results_per_series_mae)


num_validations = abs(int(num_validations))
max_possible = int(np.floor(len(df_wide_numeric.index)/forecast_length))
if max_possible < (num_validations + 1):
    num_validations = max_possible - 1
    if num_validations < 0:
        num_validations = 0
    print("Too many training validations for length of data provided, decreasing num_validations to {}".format(num_validations))

validation_template = main_results.model_results.sort_values(by = "Score", ascending = True, na_position = 'last').drop_duplicates(subset = template_cols).head(models_to_validate)[template_cols]
if not ensemble:
    validation_template[validation_template['Ensemble'] == 0]
    
validation_results = copy.copy(main_results) 

from autots.evaluator.auto_model import validation_aggregation
if num_validations > 0:
    if validation_method == 'backwards':
        for y in range(num_validations):
            # gradually remove the end
            current_slice = df_wide_numeric.head(len(df_wide_numeric.index) - (y+1) * forecast_length)
            # subset series (if used) and take a new train/test split
            df_subset = subset_series(current_slice, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)
            if weighted == False:
                current_weights = {x:1 for x in df_subset.columns}
            if weighted == True:
                current_weights = {x: weights[x] for x in df_subset.columns}                
            df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
            preord_regressor_train = preord_regressor[df_train.index]
            preord_regressor_test = preord_regressor[df_test.index]

            template_result = TemplateWizard(validation_template, df_train, df_test, current_weights,
                                         model_count = model_count, ensemble = ensemble, 
                                         forecast_length = forecast_length, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_test, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = profile_df.loc['FirstDate'],
                                          template_cols = template_cols, random_seed = random_seed, verbose = verbose)
            model_count = template_result.model_count
            validation_results.model_results = pd.concat([validation_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            validation_results.model_results['Score'] = generate_score(validation_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
            validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
            validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
            validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
            validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
        validation_results = validation_aggregation(validation_results)

    if validation_method == 'even':
        for y in range(num_validations):
            # /num_validations biases it towards the last segment (which I prefer), /(num_validations + 1) would remove that
            validation_size = int(np.floor((len(df_wide_numeric.index) - forecast_length)/num_validations))
            current_slice = df_wide_numeric.head(validation_size * (y+1) + forecast_length)
            # subset series (if used) and take a new train/test split
            df_subset = subset_series(current_slice, list((weights.get(i)) for i in df_wide_numeric.columns), n = subset, na_tolerance = na_tolerance, random_state = random_seed)
            if weighted == False:
                current_weights = {x:1 for x in df_subset.columns}
            if weighted == True:
                current_weights = {x: weights[x] for x in df_subset.columns}                
            df_train, df_test = simple_train_test_split(df_subset, forecast_length = forecast_length)
            preord_regressor_train = preord_regressor[df_train.index]
            preord_regressor_test = preord_regressor[df_test.index]

            template_result = TemplateWizard(validation_template, df_train, df_test, current_weights,
                                         model_count = model_count, ensemble = ensemble, 
                                         forecast_length = forecast_length, frequency=frequency, 
                                          prediction_interval=prediction_interval, 
                                          no_negatives=no_negatives,
                                          preord_regressor_train = preord_regressor_train,
                                          preord_regressor_forecast = preord_regressor_test, 
                                          holiday_country = holiday_country,
                                          startTimeStamps = profile_df.loc['FirstDate'],
                                          template_cols = template_cols,
                                          random_seed = random_seed, verbose = verbose)
            model_count = template_result.model_count
            validation_results.model_results = pd.concat([validation_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
            validation_results.model_results['Score'] = generate_score(validation_results.model_results, metric_weighting = metric_weighting, prediction_interval = prediction_interval)
            validation_results.model_results_per_timestamp_smape = validation_results.model_results_per_timestamp_smape.append(template_result.model_results_per_timestamp_smape)
            validation_results.model_results_per_timestamp_mae = validation_results.model_results_per_timestamp_mae.append(template_result.model_results_per_timestamp_mae)
            validation_results.model_results_per_series_smape = validation_results.model_results_per_series_smape.append(template_result.model_results_per_series_smape)
            validation_results.model_results_per_series_mae = validation_results.model_results_per_series_mae.append(template_result.model_results_per_series_mae)
        validation_results = validation_aggregation(validation_results)

class AutoTS(object):
    def __init__(self):
        pass
    def fit(self):
        pass
    def predict(self):
        pass
    def export_template(output_format: str = 'csv', models: str = 'best'):
        """"
        output_format = 'csv' or 'json'
        models = 'best', 'validation', or 'all'
        """
        print("Not yet implemented")
    def get_params(self):
        pass

model_results_per_series_smape = main_results.model_results_per_series_smape
model_results = main_results.model_results
temp = validation_results.model_results

"""
Recombine best two of each model, if two or more present
Duplicates still seem to be occurring in the genetic template runs
Inf appearing in MAE and RMSE (possibly all NaN in test)
Na Tolerance for test in simple_train_test_split
Relative/Absolute Imports and reduce package reloading
User regressor to sklearn model regression_type

Predict method
    PredictWitch + Inverse Categorical
    Regressor Flag - warns if Regressor provided in train but not forecast (regression_type == 'User')
Import/export template
Sklearn models

ARIMA + Detrend fails

Things needing testing:
    Confirm per_series weighting works properly
    Passing in Start Dates - (Test)
    Different frequencies
    Various verbose inputs
    Test holidays on non-daily data


"""

"""
The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout. The frequency of the messages increases with the verbosity level. If it more than 10, all iterations are reported."
In addition, the Glossary (search for "verbose") says this:
"Logging is not handled very consistently in Scikit-learn at present, but when it is provided as an option, the verbose parameter is usually available to choose no logging (set to False). Any True value should enable some logging, but larger integers (e.g. above 10) may be needed for full verbosity. Verbose logs are usually printed to Standard Output. Estimators should not produce any output on Standard Output with the default verbose setting."
https://stackoverflow.com/questions/29995249/verbose-argument-in-scikit-learn
"""
# to gluon ds
# to xgboost ds
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# *args, **kwargs
