"""Informal testing script."""
# pragma pylint: disable=W293,E251,D407,E501
import numpy as np
import pandas as pd
from autots.datasets import load_toy_daily
from autots.datasets import load_toy_hourly
from autots.datasets import load_toy_monthly
from autots.datasets import load_toy_yearly
from autots.datasets import load_toy_weekly
from autots import AutoTS
from autots.evaluator.auto_ts import fake_regressor


forecast_length = 4
df_long = load_toy_daily()

# df_long = df_long[df_long['series_id'] == 'GS10']

weights_daily = {'categoricalDayofWeek': 5,
                 'randomNegative': 1,
                 'sp500high': 2,
                 'wabashaTemp': 1}

weights_hourly = {'traffic_volume': 10}

model_list = [
              # 'ZeroesNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS',
              'GLM', 'ETS','RollingRegression', 'ARIMA',
               'FBProphet', 'UnobservedComponents'
               ,'VECM', 'DynamicFactor'
              #,'VARMAX', 'GluonTS'
              ]
model_list = 'superfast'
# model_list = ['MofitSimulation', 'GLM','ZeroesNaive', 'LastValueNaive', 'AverageValueNaive', 'GLS', 'SeasonalNaive']
model_list = ['RollingRegression', 'LastValueNaive']

metric_weighting = {'smape_weighting': 10, 'mae_weighting': 1,
                    'rmse_weighting': 5, 'containment_weighting': 1,
                    'runtime_weighting': 0, 'lower_mae_weighting': 0,
                    'upper_mae_weighting': 0, 'contour_weighting': 2
                    }


model = AutoTS(forecast_length=forecast_length, frequency='infer',
               prediction_interval=0.9, ensemble=False, weighted=False,
               max_generations=1, num_validations=2, validation_method='even',
               model_list=model_list, initial_template='General+Random',
               metric_weighting=metric_weighting, models_to_validate=50,
               max_per_model_class=10,
               drop_most_recent=1, verbose=1)


preord_regressor_train, preord_regressor_forecast = fake_regressor(df_long, dimensions= 1, forecast_length = forecast_length, date_col = 'datetime', value_col = 'value', id_col = 'series_id')
preord_regressor_train2d, preord_regressor_forecast2d = fake_regressor(df_long, dimensions= 4, forecast_length = forecast_length, date_col = 'datetime', value_col = 'value', id_col = 'series_id')

# model = model.fit(df_long, date_col = 'datetime', value_col = 'value', id_col = 'series_id')
"""
model = model.fit(df_long, weights = weights_hourly,
                  result_file = 'test_results.csv',
                  date_col = 'datetime', value_col = 'value',
                  id_col = 'series_id')  # and weighted = True
"""
model = model.fit(df_long, preord_regressor=preord_regressor_train2d,
                  date_col='datetime', value_col='value', id_col='series_id')

print(model.best_model['Model'].iloc[0])
print(model.best_model['ModelParameters'].iloc[0])
print(model.best_model['TransformationParameters'].iloc[0])

prediction = model.predict(preord_regressor=preord_regressor_forecast2d)
# point forecasts dataframe
forecasts_df = prediction.forecast
# accuracy of all tried model results (not including cross validation)
initial_results = model.initial_results.model_results
# validation results
validation_results = model.validation_results.model_results

error_results = model.error_templates


"""
Import/Export

example_filename = "example_export.csv" #.csv/.json
model.export_template(example_filename, models = 'best', n = 15, max_per_model_class = 3)
model = model.import_template(example_filename, method = 'add on')
print("Overwrite template is: {}".format(str(model.initial_template)))
"""

"""
Things needing testing:
    With and without regressor
    With and without weighting
    Different frequencies
    Various verbose inputs

Edgey Cases:
        Single Time Series
        Forecast Length of 1
        Very short training data
"""

# %%
from autots.tools.shaping import long_to_wide
df_wide = long_to_wide(df_long, date_col = 'datetime', value_col = 'value',
                       id_col='series_id', frequency='infer', aggfunc='first')

# df = df_wide[df_wide.columns[0:3]].fillna(0).astype(float)

from autots.tools.shaping import values_to_numeric
categorical_transformer = values_to_numeric(df_wide)
df_wide_numeric = categorical_transformer.dataframe

df = df_wide_numeric.tail(50).fillna(0).astype(float)


"""
https://packaging.python.org/tutorials/packaging-projects/

python -m pip install --user --upgrade setuptools wheel
cd /to project directory
python setup.py sdist bdist_wheel
twine upload dist/*

Merge dev to master on GitHub and create release (include .tar.gz)
"""
# *args, **kwargs

#%%
"""
Help correlate errors with parameters
"""
if error_results.shape[0] > 0:
    import json
    from sklearn.preprocessing import OneHotEncoder
    # test = initial_results[ initial_results['TransformationParameters'].str.contains('FastICA')]

    cols = ['Model', 'ModelParameters',
            'TransformationParameters', 'Exceptions']
    all_results = pd.concat([initial_results[cols],
                             error_results[cols]], axis=0)
    all_results = all_results.drop_duplicates()
    all_results['ExceptionFlag'] = (~all_results['Exceptions'].isna()).astype(int)
    all_results = all_results[all_results['ExceptionFlag'] > 0]
    all_results = all_results.reset_index(drop=True)
    
    trans_df = all_results['TransformationParameters'].apply(json.loads)
    trans_df = pd.io.json.json_normalize(trans_df)  # .fillna(value='NaN')
    trans_cols1 = trans_df.columns
    trans_df = trans_df.astype(str).replace('nan', 'NaNZ')
    trans_transformer = OneHotEncoder(sparse=False).fit(trans_df)
    trans_df = pd.DataFrame(trans_transformer.transform(trans_df))
    # trans_cols = [item for sublist in trans_transformer.categories_ for item in sublist]
    trans_cols = np.array([x1 + x2 for x1, x2 in zip(
        trans_cols1, trans_transformer.categories_)])
    trans_cols = [item for sublist in trans_cols for item in sublist]
    trans_df.columns = trans_cols
    
    model_df = all_results['ModelParameters'].apply(json.loads)
    model_df = pd.io.json.json_normalize(model_df)  # .fillna(value='NaN')
    model_cols1 = model_df.columns
    model_df = model_df.astype(str).replace('nan', 'NaNZ')
    model_transformer = OneHotEncoder(sparse=False).fit(model_df)
    model_df = pd.DataFrame(model_transformer.transform(model_df))
    model_cols = np.array([x1 + x2 for x1, x2 in zip(
        model_cols1, model_transformer.categories_)])
    model_cols = [item for sublist in model_cols for item in sublist]
    model_df.columns = model_cols
    
    modelstr_df = all_results['Model']
    modelstr_transformer = OneHotEncoder(sparse=False).fit(
        modelstr_df.values.reshape(-1, 1))
    modelstr_df = pd.DataFrame(modelstr_transformer.transform(
        modelstr_df.values.reshape(-1, 1)))
    modelstr_df.columns = modelstr_transformer.categories_[0]
    
    except_df = all_results['Exceptions'].copy()
    except_df = except_df.where(except_df.duplicated(), 'UniqueError')
    except_transformer = OneHotEncoder(sparse=False).fit(
        except_df.values.reshape(-1, 1))
    except_df = pd.DataFrame(except_transformer.transform(
        except_df.values.reshape(-1, 1)))
    except_df.columns = except_transformer.categories_[0]
    
    test = pd.concat([except_df, all_results[['ExceptionFlag']],
                      modelstr_df, model_df, trans_df], axis=1)
    # test_cols = [column for column in test.columns if 'NaNZ' not in column]
    # test = test[test_cols]
    test_corr = test.corr()[except_df.columns]
    """
    try:
        from mlxtend.frequent_patterns import association_rules
        from mlxtend.frequent_patterns import apriori
        import re
        freq_itemsets = apriori(test.drop('ExceptionFlag', axis=1),
                                min_support=0.3, use_colnames=True)
        rules = association_rules(freq_itemsets)
        err_rules = pd.DataFrame()
        for err in except_df.columns:
            err = re.sub('[^a-zA-Z0-9\s]', '', err)
            edf = rules[
                rules['consequents'].astype(
                    str).str.replace('[^a-zA-Z0-9\s]', '').str.contains(err)]
            err_rules = pd.concat([err_rules, edf],
                                     axis=0, ignore_index=True)
        err_rules = err_rules.drop_duplicates()
    except Exception as e:
        print(repr(e))
    """
#%%