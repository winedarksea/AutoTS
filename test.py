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

import random
random.seed(seed)
np.random.seed(seed)

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

if weighted == False:
    weights = {x:1 for x in df_wide_numeric.columns}

from autots.tools.shaping import simple_train_test_split
df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)
preord_regressor_train = preord_regressor[df_train.index]
preord_regressor_test = preord_regressor[df_test.index]


#from autots.tools.shaping import subset_series
#df_subset = subset_series(df_wide_numeric, n = 10, na_tolerance = 0.5, random_state = seed)


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

def generate_score(model_results):
    """
    Generates score based on relative accuracies
    """
    smape_weighting = 9
    rmse_weighting = 6
    containment_weighting = 1
    runtime_weighting = 0.5
    smape_score = model_results['smape_weighted']/model_results['smape_weighted'].min(skipna=True) # smaller better
    rmse_score = model_results['rmse_weighted']/model_results['rmse_weighted'].min(skipna=True) # smaller better
    containment_score = 2/(model_results['containment']+1) # from 1 to 2, smaller better
    runtime_score = model_results['TotalRuntime']/(model_results['TotalRuntime'].min(skipna=True) + datetime.timedelta(minutes = 1))
    return (smape_score * smape_weighting) + (rmse_score * rmse_weighting) + (containment_score * containment_weighting) + (runtime_score * runtime_weighting)

def UniqueTemplates(existing_templates, new_possibilities, selection_cols: list = ['Model','ModelParameters','TransformationParameters','Ensemble']):
    """
    Returns unique dataframe rows from new_possiblities not in existing_templates
    
    Args:
        selection_cols (list): list of column namess to use to judge uniqueness/match on
    """
    keys = list(new_possibilities[selection_cols].columns.values)
    idx1 = existing_templates.copy().set_index(keys).index
    idx2 = new_possibilities.set_index(keys).index
    new_template = new_possibilities[~idx2.isin(idx1)]
    return new_template

def NewGeneticTemplate(model_results, sort_column: str = "smape_weighted", 
                       sort_ascending: bool = True, max_results: int = 40,
                       top_n: int = 15):
    """
    Returns new template given old template with model accuracies
    """
    new_template = pd.DataFrame()
    template_cols = ['Model','ModelParameters','TransformationParameters','Ensemble']
    sorted_results = model_results[model_results['Ensemble'] == 0].copy().sort_values(by = sort_column, ascending = sort_ascending, na_position = 'last')
    # mutation
    for index, row in sorted_results.drop_duplicates(subset = "Model", keep = 'first').head(top_n).iterrows():
        param_dict = ModelMonster(row['Model']).get_new_params()
        trans_dict = RandomTransform()
        new_row = pd.DataFrame({
                'Model': row['Model'],
                'ModelParameters': json.dumps(param_dict),
                'TransformationParameters': row['TransformationParameters'],
                'Ensemble': 0
                }, index = [0])
        new_template = pd.concat([new_template, new_row], axis = 0, ignore_index = True, sort = False)
        new_row = pd.DataFrame({
                'Model': row['Model'],
                'ModelParameters': row['ModelParameters'],
                'TransformationParameters': json.dumps(trans_dict),
                'Ensemble': 0
                }, index = [0])
        new_template = pd.concat([new_template, new_row], axis = 0, ignore_index = True, sort = False)

    # recombination of transforms across models
    recombination = sorted_results.tail(len(sorted_results.index) - 1).copy()
    recombination['TransformationParameters'] = sorted_results['TransformationParameters'].shift(1).tail(len(sorted_results.index) - 1)
    new_template = pd.concat([new_template, recombination.head(top_n)[template_cols]], axis = 0, ignore_index = True, sort = False)
    
    # internal recombination of model parameters, not implemented because some options are mutually exclusive.
    # Recombine best two of each model, if two or more present
    
    # remove generated models which have already been tried
    new_template = UniqueTemplates(sorted_results, new_template, selection_cols = template_cols).head(max_results)
    return new_template   

from autots.evaluator.auto_model import TemplateEvalObject
main_results = TemplateEvalObject()


from autots.evaluator.auto_model import TemplateWizard    

model_count = 0
initial_template = RandomTemplate(40)
template_result = TemplateWizard(initial_template, df_train, df_test, weights,
                                 model_count = model_count, ensemble = ensemble, 
                                 forecast_length = forecast_length, frequency=frequency, 
                                  prediction_interval=prediction_interval, 
                                  no_negatives=no_negatives,
                                  preord_regressor_train = preord_regressor_train,
                                  preord_regressor_forecast = preord_regressor_test, 
                                  holiday_country = holiday_country,
                                  startTimeStamps = profile_df.loc['FirstDate'])
model_count = template_result.model_count
main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
main_results.model_results['Score'] = generate_score(main_results.model_results)
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

generations = 3 # 3 = 24.221 SMAPE, 24.6159 SMAPE for 5 gen, 24.108 10 gen, 23.4256 20 gen
for x in range(generations):
    new_template = NewGeneticTemplate(main_results.model_results, sort_column = "Score", 
                       sort_ascending = True, max_results = 40, top_n = 15)
    # use multiple metrics to create multiple template sets
    
    template_result = TemplateWizard(new_template, df_train, df_test, weights,
                                 model_count = model_count, ensemble = ensemble, 
                                 forecast_length = forecast_length, frequency=frequency, 
                                  prediction_interval=prediction_interval, 
                                  no_negatives=no_negatives,
                                  preord_regressor_train = preord_regressor_train,
                                  preord_regressor_forecast = preord_regressor_test, 
                                  holiday_country = holiday_country,
                                  startTimeStamps = profile_df.loc['FirstDate'])
    model_count = template_result.model_count
    main_results.model_results = pd.concat([main_results.model_results, template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    main_results.model_results['Score'] = generate_score(main_results.model_results)
    
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
    ens_template_result = EnsembleEvaluate(ensemble_forecasts_list, df_test = df_test, weights = weights, model_count = model_count)
    
    model_count = ens_template_result.model_count
    main_results.model_results = pd.concat([main_results.model_results, ens_template_result.model_results], axis = 0, ignore_index = True, sort = False).reset_index(drop = True)
    main_results.model_results['Score'] = generate_score(main_results.model_results)
    main_results.model_results_per_timestamp_smape = main_results.model_results_per_timestamp_smape.append(ens_template_result.model_results_per_timestamp_smape)
    main_results.model_results_per_timestamp_mae = main_results.model_results_per_timestamp_mae.append(ens_template_result.model_results_per_timestamp_mae)
    main_results.model_results_per_series_smape = main_results.model_results_per_series_smape.append(ens_template_result.model_results_per_series_smape)
    main_results.model_results_per_series_mae = main_results.model_results_per_series_mae.append(ens_template_result.model_results_per_series_mae)

num_validations = 0
validation_method = 'even'
num_validations = abs(int(num_validations))
max_possible = int(np.floor(len(df_wide_numeric.index)/forecast_length))
if max_possible < (num_validations + 1):
    num_validations = max_possible - 1
    if num_validations < 0:
        num_validations = 0
    print("Too many training validations for length of data provided, decreasing num_validations to {}".format(num_validations))

# work backwards approach
# slice into num bits (less the bit used above)
# across different series??
if num_validations > 0:
    if validation_method == 'backwards':
        for y in range(num_validations):
            current_slice = df_wide_numeric.head(len(df_wide_numeric.index) - (y+1) * forecast_length)
            df_train, df_test = simple_train_test_split(current_slice, forecast_length = forecast_length)
            preord_regressor_train = preord_regressor[df_train.index]
            preord_regressor_test = preord_regressor[df_test.index]
    if validation_method == 'even':
        df_wide_numeric.index
    from autots.tools.shaping import simple_train_test_split
    df_train, df_test = simple_train_test_split(df_wide_numeric, forecast_length = forecast_length)
    preord_regressor_train = preord_regressor[df_train.index]
    preord_regressor_test = preord_regressor[df_test.index]


    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=num_validations, max_train_size=None)
    tscv.split(df_wide.index)
    for train_index, test_index in tscv.split(df_wide_numeric):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    for y in range(num_validations):
        pass
    

"""
unpack ensembles if in template!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Recombine best two of each model, if two or more present

Verbosity
Consolidate repeat models in model_results, and per_series/per_timestamp
Subset series

Multiple validation
Predict method

ARIMA + Detrend fails

Ranked by score
    nearest neighbor score - is time much slower than similar? is MAE much better than similar for SMAPE?

Things needing testing:
    Confirm per_series weighting works properly
    Passing in Start Dates - (Test)
    Different frequencies


"""











# to gluon ds
# to xgboost ds
# GENERATOR of series for per series methods
# trim series to first actual value
    # gluon start
    # per series, trim before first na
    # from regressions, remove rows based on % columns that are NaN
# *args, **kwargs
