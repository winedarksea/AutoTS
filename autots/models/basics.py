"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject


class Zeroes(ModelObject):
    def __init__(self, name):
        ModelObject.__init__(self, name)
    def fit(self, df):
        self.startTime = datetime.datetime.now()
        self.shape = df.shape
        return self
    def predict(self, forecast_length, regressor):
        
        return df
        
try:
    startTime = datetime.datetime.now()
    Zeroes.forecast = (np.zeros((forecast_length,len(train.columns))))
    Zeroes.runtime = datetime.datetime.now() - startTime
    
    Zeroes.mae = pd.DataFrame(mae(test.values, Zeroes.forecast)).mean(axis=0, skipna = True)
    Zeroes.overall_mae = np.nanmean(Zeroes.mae)
    Zeroes.smape = smape(test.values, Zeroes.forecast)
    Zeroes.overall_smape = np.nanmean(Zeroes.smape)
except Exception as e:
    print(e)
    error_list.extend([traceback.format_exc()])

currentResult = pd.DataFrame({
        'method': Zeroes.name, 
        'runtime': Zeroes.runtime, 
        'overall_smape': Zeroes.overall_smape, 
        'overall_mae': Zeroes.overall_mae,
        'object_name': 'Zeroes'
        }, index = [0])
model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)

LastValue = ModelResult("Last Value Naive")
try:
    startTime = datetime.datetime.now()
    LastValue.forecast = np.tile(train.tail(1).values, (forecast_length,1))
    LastValue.runtime = datetime.datetime.now() - startTime
    
    LastValue.mae = pd.DataFrame(mae(test.values, LastValue.forecast)).mean(axis=0, skipna = True)
    LastValue.overall_mae = np.nanmean(LastValue.mae)
    LastValue.smape = smape(test.values, LastValue.forecast)
    LastValue.overall_smape = np.nanmean(LastValue.smape)
except Exception as e:
    print(e)
    error_list.extend([traceback.format_exc()])

currentResult = pd.DataFrame({
        'method': LastValue.name, 
        'runtime': LastValue.runtime, 
        'overall_smape': LastValue.overall_smape, 
        'overall_mae': LastValue.overall_mae,
        'object_name': 'LastValue'
        }, index = [0])
model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)

MedValue = ModelResult("Median Naive")
try:
    startTime = datetime.datetime.now()
    MedValue.forecast = np.tile(train.median(axis = 0).values, (forecast_length,1))
    MedValue.runtime = datetime.datetime.now() - startTime
    
    MedValue.mae = pd.DataFrame(mae(test.values, MedValue.forecast)).mean(axis=0, skipna = True)
    MedValue.overall_mae = np.nanmean(MedValue.mae)
    MedValue.smape = smape(test.values, MedValue.forecast)
    MedValue.overall_smape = np.nanmean(MedValue.smape)
except Exception as e:
    print(e)
    error_list.extend([traceback.format_exc()])

currentResult = pd.DataFrame({
        'method': MedValue.name, 
        'runtime': MedValue.runtime, 
        'overall_smape': MedValue.overall_smape, 
        'overall_mae': MedValue.overall_mae,
        'object_name': 'MedValue'
        }, index = [0])
model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
