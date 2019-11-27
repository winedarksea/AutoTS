"""
Comparing Time Series Models' Point Forecast Accuracies

We only use Daily data for this example, much of it business day only, 
but this could easily be used for monthly or other frequency data

Incoming data format is a 'long' data format:
    Just three columns expected: date, series_id, value

Missing data here is handled with a fill-forward when necessary. 
For intermittent data, filling with zero may be better

Most ts models don't need much parameter tuning. 
Tuning GluonTS epochs, context_length and adding features is one possibility

pip install fredapi # if using samples
conda install -c conda-forge fbprophet
pip install mxnet==1.4.1
    pip install mxnet-cu90mkl==1.4.1 # if you want GPU and have Intel CPU
pip install gluonts==0.4.0
    pip install git+https://github.com/awslabs/gluon-ts.git #if you want dev version
pip install pmdarima==1.4.0 
pip uninstall numpy # might be necessary, even twice, followed by the following
pip install numpy==1.17.4 # gluonts likes to force numpy back to 1.14, but 1.17 seems to be fine with it
pip install sktime==0.3.1
"""
import traceback
import numpy as np
import pandas as pd
import datetime


use_sample_data = True # whether to use sample FRED time series or use a new dataset
fredkey = 'XXXXXXXXXX' # get from https://research.stlouisfed.org/docs/api/fred/
forecast_length = 90 # how much you will be predicting
frequency = '1D'  # 'Offset aliases' from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
no_negatives = True # True if all forecasts should be positive
series_to_sample = 50 # number of series to evaluate on
na_tolerance = 0.15 # 0 to 1, drop time series if up to this percent of values are NOT NaN
drop_most_recent = False # if to drop the most recent date and value (useful if most recent is incomplete month, say)
figures = False # whether to plot some sample time series
drop_data_older_than_years = 5 # at what point to cut old data
fill_na_zero = False # use for intermittent time series, where NA just means zero (sales, say)
output_name = "eval_table.csv"


""" 
Plot Time Series
"""
if figures:
    try:
        # plot one and save
        # series = 'SP500'
        series = timeseries_seriescols.columns[0]
        ax = timeseries_seriescols[series].fillna(method = 'ffill').plot()
        fig = ax.get_figure()
        # fig.savefig((series + '.png'), dpi=300)
        
        # plot multiple time series all on same scale
        from sklearn.preprocessing import MinMaxScaler
        ax = pd.DataFrame(MinMaxScaler().fit_transform(timeseries_seriescols)).sample(5, axis = 1).plot()
        ax.get_legend().remove()
        fig = ax.get_figure()
        # fig.savefig('MultipleTimeSeries.png', dpi=300)
    except Exception:
        pass


train = timeseries_seriescols.head(len(timeseries_seriescols.index) - forecast_length)
test = timeseries_seriescols.tail(forecast_length)



class ModelResult(object):
    def __init__(self, name=None, forecast=None, mae=None, overall_mae=-1, smape=None, overall_smape=-1, runtime=datetime.timedelta(0)):
        self.name = name
        self.forecast = forecast
        self.mae = mae
        self.overall_mae = overall_mae
        self.smape = smape
        self.overall_smape = overall_smape
        self.runtime = runtime

    def __repr__(self):
        return "Time Series Model Result: " + str(self.name)
    def __str__(self):
        return "Time Series Model Result: " + str(self.name)
    def result_message(self):
        print("TS Method: " + str(self.name) + " of Avg SMAPE: " + str(self.overall_smape))

class EvaluationReturn(object):
    def __init__(self, model_performance = np.nan, per_series_mae = np.nan, per_series_smape = np.nan, errors = np.nan):
        self.model_performance = model_performance
        self.per_series_mae = per_series_mae
        self.per_series_smape = per_series_smape
        self.errors = errors
subset = series_to_sample

def model_evaluator(train, test, subset = 'All'):
    model_results = pd.DataFrame(columns = ['method', 'runtime', 'overall_smape', 'overall_mae', 'object_name'])
    error_list = []
    
    # take a subset of the availabe series to speed up
    if isinstance(subset, (int, float, complex)) and not isinstance(subset, bool):
        if subset > len(train.columns):
            subset = len(train.columns) 
        train = train.sample(subset, axis = 1, random_state = 425, replace = False)
        test = test[train.columns]
    
    trainStartDate = min(train.index)
    trainEndDate = max(train.index)
    testStartDate = min(test.index)
    testEndDate = max(test.index)
    """
    Naives
    """
    Zeroes = ModelResult("All Zeroes")
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
    
    """
    Linear Regression
    """
    GLM = ModelResult("GLM")
    try:
        from statsmodels.regression.linear_model import GLS
        startTime = datetime.datetime.now()
        glm_model = GLS(train.values, (train.index.astype( int ).values), missing = 'drop').fit()
        GLM.forecast = glm_model.predict(test.index.astype( int ).values)
        if no_negatives:
            GLM.forecast = GLM.forecast.clip(min = 0)
        GLM.runtime = datetime.datetime.now() - startTime
        
        GLM.mae = pd.DataFrame(mae(test.values, GLM.forecast)).mean(axis=0, skipna = True)
        GLM.overall_mae = np.nanmean(GLM.mae)
        GLM.smape = smape(test.values, GLM.forecast)
        GLM.overall_smape = np.nanmean(GLM.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GLM.name, 
            'runtime': GLM.runtime, 
            'overall_smape': GLM.overall_smape, 
            'overall_mae': GLM.overall_mae,
            'object_name': 'GLM'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GLM.result_message()
    """
    ETS Forecast
    
    Here I copy the comb benchmark of the M4 competition:
        the simple arithmetic average of Single, Holt and Damped exponential smoothing
        
    http://www.statsmodels.org/stable/tsa.html
    """
    
    sETS = ModelResult("SimpleETS")
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            sesModel = SimpleExpSmoothing(current_series).fit()
            sesPred = sesModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                sesPred = sesPred.where(sesPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, sesPred], axis = 1)
        sETS.forecast = forecast.values
        sETS.runtime = datetime.datetime.now() - startTime
        
        sETS.mae = pd.DataFrame(mae(test.values, sETS.forecast)).mean(axis=0, skipna = True)
        sETS.overall_mae = np.nanmean(sETS.mae)
        sETS.smape = smape(test.values, sETS.forecast)
        sETS.overall_smape = np.nanmean(sETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': sETS.name, 
            'runtime': sETS.runtime, 
            'overall_smape': sETS.overall_smape, 
            'overall_mae': sETS.overall_mae,
            'object_name': 'sETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    sETS.result_message()
    
    
    ETS = ModelResult("ETS")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            esModel = ExponentialSmoothing(current_series, damped = False).fit()
            esPred = esModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                esPred = esPred.where(esPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, esPred], axis = 1)
        ETS.forecast = forecast.values
        ETS.runtime = datetime.datetime.now() - startTime
        
        ETS.mae = pd.DataFrame(mae(test.values, ETS.forecast)).mean(axis=0, skipna = True)
        ETS.overall_mae = np.nanmean(ETS.mae)
        ETS.smape = smape(test.values, ETS.forecast)
        ETS.overall_smape = np.nanmean(ETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': ETS.name, 
            'runtime': ETS.runtime, 
            'overall_smape': ETS.overall_smape, 
            'overall_mae': ETS.overall_mae,
            'object_name': 'ETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ETS.result_message()
    """
    Damped ETS
    """
    dETS = ModelResult("dampedETS")
    try:
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            esModel = ExponentialSmoothing(current_series, damped = True, trend = 'add').fit()
            esPred = esModel.predict(start=testStartDate, end=testEndDate)
            if no_negatives:
                esPred = esPred.where(esPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, esPred], axis = 1)
        dETS.forecast = forecast.values
        dETS.runtime = datetime.datetime.now() - startTime
        
        dETS.mae = pd.DataFrame(mae(test.values, dETS.forecast)).mean(axis=0, skipna = True)
        dETS.overall_mae = np.nanmean(dETS.mae)
        dETS.smape = smape(test.values, dETS.forecast)
        dETS.overall_smape = np.nanmean(dETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': dETS.name, 
            'runtime': dETS.runtime, 
            'overall_smape': dETS.overall_smape, 
            'overall_mae': dETS.overall_mae,
            'object_name': 'dETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    dETS.result_message()
    """
    Markov AutoRegression - new to statsmodels 1.10, make sure you have recent version
    """
    """
    MarkovReg = ModelResult("MarkovRegression")
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = MarkovRegression(current_series, k_regimes=3, trend='nc', switching_variance=True).fit(em_iter=20,search_reps=20)
                maPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                try: # ETS if the above failed
                    sesModel = SimpleExpSmoothing(current_series).fit()
                    maPred = sesModel.predict(start=testStartDate, end=testEndDate)
                except Exception:
                    maPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    maPred = pd.Series(np.where(maPred > 0, maPred, 0))
                except Exception:
                    maPred = maPred.where(maPred > 0, 0) 
            forecast = pd.concat([forecast, maPred], axis = 1)
        MarkovReg.forecast = forecast.values
        MarkovReg.runtime = datetime.datetime.now() - startTime
        
        MarkovReg.mae = pd.DataFrame(mae(test.values, MarkovReg.forecast)).mean(axis=0, skipna = True)
        MarkovReg.overall_mae = np.nanmean(MarkovReg.mae)
        MarkovReg.smape = smape(test.values, MarkovReg.forecast)
        MarkovReg.overall_smape = np.nanmean(MarkovReg.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': MarkovReg.name, 
            'runtime': MarkovReg.runtime, 
            'overall_smape': MarkovReg.overall_smape, 
            'overall_mae': MarkovReg.overall_mae,
            'object_name': 'MarkovReg'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    MarkovAuto = ModelResult("MarkovAutoregression")
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = MarkovAutoregression(current_series, k_regimes=2, order=4, switching_ar=False).fit(em_iter=20,search_reps=20)
                maPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                maPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    maPred = pd.Series(np.where(maPred > 0, maPred, 0))
                except Exception:
                    maPred = maPred.where(maPred > 0, 0) 
            forecast = pd.concat([forecast, maPred], axis = 1)
        MarkovAuto.forecast = forecast.values
        MarkovAuto.runtime = datetime.datetime.now() - startTime
        
        MarkovAuto.mae = pd.DataFrame(mae(test.values, MarkovAuto.forecast)).mean(axis=0, skipna = True)
        MarkovAuto.overall_mae = np.nanmean(MarkovAuto.mae)
        MarkovAuto.smape = smape(test.values, MarkovAuto.forecast)
        MarkovAuto.overall_smape = np.nanmean(MarkovAuto.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': MarkovAuto.name, 
            'runtime': MarkovAuto.runtime, 
            'overall_smape': MarkovAuto.overall_smape, 
            'overall_mae': MarkovAuto.overall_mae,
            'object_name': 'MarkovAuto'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    """
    
    UnComp = ModelResult("UnobservedComponents")
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        unrestricted_model = {
            'level': 'local linear trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
        }
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = UnobservedComponents(current_series, **unrestricted_model).fit(method='powell')
                ucPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                try: # ETS if the above failed
                    sesModel = SimpleExpSmoothing(current_series).fit()
                    ucPred = sesModel.predict(start=testStartDate, end=testEndDate)
                except Exception:
                    ucPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    ucPred = pd.Series(np.where(ucPred > 0, ucPred, 0))
                except Exception:
                    ucPred = ucPred.where(ucPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, ucPred], axis = 1)
        UnComp.forecast = forecast.values
        UnComp.runtime = datetime.datetime.now() - startTime
        
        UnComp.mae = pd.DataFrame(mae(test.values, UnComp.forecast)).mean(axis=0, skipna = True)
        UnComp.overall_mae = np.nanmean(UnComp.mae)
        UnComp.smape = smape(test.values, UnComp.forecast)
        UnComp.overall_smape = np.nanmean(UnComp.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': UnComp.name, 
            'runtime': UnComp.runtime, 
            'overall_smape': UnComp.overall_smape, 
            'overall_mae': UnComp.overall_mae,
            'object_name': 'UnComp'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    UnComp.result_message()
    """
    Simple ARIMA
    """
    SARIMA = ModelResult("ARIMA 101")
    try:
        from statsmodels.tsa.arima_model import ARIMA
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                model = ARIMA(current_series, order=(1,0,1)).fit()
                saPred = model.predict(start=testStartDate, end=testEndDate)
            except Exception:
                saPred = (np.zeros((forecast_length,)))
            if no_negatives:
                try:
                    saPred = pd.Series(np.where(saPred > 0, saPred, 0))
                except Exception:
                    saPred = saPred.where(saPred > 0, 0)   # replace all negatives with zeroes, remove if you want negatives!
            forecast = pd.concat([forecast, saPred], axis = 1)
        SARIMA.forecast = forecast.values
        SARIMA.runtime = datetime.datetime.now() - startTime
        
        SARIMA.mae = pd.DataFrame(mae(test.values, SARIMA.forecast)).mean(axis=0, skipna = True)
        SARIMA.overall_mae = np.nanmean(SARIMA.mae)
        SARIMA.smape = smape(test.values, SARIMA.forecast)
        SARIMA.overall_smape = np.nanmean(SARIMA.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': SARIMA.name, 
            'runtime': SARIMA.runtime, 
            'overall_smape': SARIMA.overall_smape, 
            'overall_mae': SARIMA.overall_mae,
            'object_name': 'SARIMA'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    SARIMA.result_message()
    """
    Prophet
    expects data in a specific format: a 'ds' column for dates and a 'y' column for values
    can handle missing data
    https://facebook.github.io/prophet/
	conda install -c conda-forge fbprophet
    """
    
    ProphetResult = ModelResult("Prophet")
    
    try:
        from fbprophet import Prophet
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            
            m = Prophet().fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            fcst = m.predict(future)
            if no_negatives:
                fcst.loc[~(fcst['yhat'] > 0), 'yhat'] = 0   
            fcst = fcst.tail(forecast_length) # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis = 1)
        ProphetResult.forecast = forecast.values
        ProphetResult.runtime = datetime.datetime.now() - startTime
        
        ProphetResult.mae = pd.DataFrame(mae(test.values, ProphetResult.forecast)).mean(axis=0, skipna = True)
        ProphetResult.overall_mae = np.nanmean(ProphetResult.mae)
        ProphetResult.smape = smape(test.values, ProphetResult.forecast)
        ProphetResult.overall_smape = np.nanmean(ProphetResult.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': ProphetResult.name, 
            'runtime': ProphetResult.runtime, 
            'overall_smape': ProphetResult.overall_smape, 
            'overall_mae': ProphetResult.overall_mae,
            'object_name': 'ProphetResult'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ProphetResult.result_message()
    
    ProphetResultHoliday = ModelResult("Prophet w Holidays")
    try:
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        for series in train.columns:
            current_series = train.copy()
            current_series['y'] = current_series[series]
            current_series['ds'] = current_series.index
            
            m = Prophet()
            m.add_country_holidays(country_name='US')
            m.fit(current_series)
            future = m.make_future_dataframe(periods=forecast_length)
            fcst = m.predict(future)
            if no_negatives:
                fcst.loc[~(fcst['yhat'] > 0), 'yhat'] = 0   
            fcst = fcst.tail(forecast_length) # remove the backcast
            forecast = pd.concat([forecast, fcst['yhat']], axis = 1)
        ProphetResultHoliday.forecast = forecast.values
        ProphetResultHoliday.runtime = datetime.datetime.now() - startTime
        
        ProphetResultHoliday.mae = pd.DataFrame(mae(test.values, ProphetResultHoliday.forecast)).mean(axis=0, skipna = True)
        ProphetResultHoliday.overall_mae = np.nanmean(ProphetResultHoliday.mae)
        ProphetResultHoliday.smape = smape(test.values, ProphetResultHoliday.forecast)
        ProphetResultHoliday.overall_smape = np.nanmean(ProphetResultHoliday.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': ProphetResultHoliday.name, 
            'runtime': ProphetResultHoliday.runtime, 
            'overall_smape': ProphetResultHoliday.overall_smape, 
            'overall_mae': ProphetResultHoliday.overall_mae,
            'object_name': 'ProphetResultHoliday'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    ProphetResultHoliday.result_message()
    
    """
    AutoARIMA
    pip install pmdarima  (==1.4.0 to play with GluonTS==0.4.0)
    Install pmdarima after installing GluonTS to prevent numpy issues
    """
    AutoArima = ModelResult("Auto ARIMA S7")
    try:
        from pmdarima.arima import auto_arima
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        model_orders = []
        for series in train.columns:
            try:
                current_series = train[series].copy()
                current_series = current_series.fillna(method='ffill').fillna(method='bfill')
                current_series = current_series.reset_index(drop = True)
                model = auto_arima(current_series, error_action='ignore', seasonal=True, m=7, suppress_warnings = True)
                saPred = model.predict(n_periods=forecast_length)
                model_orders.extend([model.order])
            except Exception:
                saPred = (np.zeros((forecast_length,)))
                model_orders.extend([(0,0,0)])
            if no_negatives:
                try:
                    saPred = pd.Series(np.where(saPred > 0, saPred, 0))
                except Exception:
                    saPred = np.where(saPred > 0, saPred, 0)
            forecast = pd.concat([forecast, pd.Series(saPred)], axis = 1)
        AutoArima.forecast = forecast.values
        AutoArima.runtime = datetime.datetime.now() - startTime
        
        AutoArima.mae = pd.DataFrame(mae(test.values, AutoArima.forecast)).mean(axis=0, skipna = True)
        AutoArima.overall_mae = np.nanmean(AutoArima.mae)
        AutoArima.smape = smape(test.values, AutoArima.forecast)
        AutoArima.overall_smape = np.nanmean(AutoArima.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    
    currentResult = pd.DataFrame({
            'method': AutoArima.name, 
            'runtime': AutoArima.runtime, 
            'overall_smape': AutoArima.overall_smape, 
            'overall_mae': AutoArima.overall_mae,
            'object_name': 'AutoArima'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    AutoArima.result_message()
    
    """
    GluonTS
    https://gluon-ts.mxnet.io/
    
    Gluon has a nice built in evaluator, but that is not used here
    First install mxnet if you haven't already (in my case, pip install mxnet-cu90mkl==1.4.1)
    pip install gluonts==0.4.0
    pip install git+https://github.com/awslabs/gluon-ts.git
    pip install numpy==1.17.4 after gluon, because gluon seems okay with new version,
    but most things aren't okay with 1.14 numpy
    """
    try:
        try:
            from gluonts.transform import FieldName # old way (0.3.3 and older)
        except Exception:
            from gluonts.dataset.field_names import FieldName # new way
        
        gluon_train = train.fillna(method='ffill').fillna(method='bfill').transpose()
        if frequency == "MS" or frequency == "1MS":
            gluon_freq = "1M"
        else:
            gluon_freq = frequency
        ts_metadata = {'num_series': len(gluon_train.index),
                              'forecast_length': forecast_length,
                              'freq': gluon_freq,
                              'gluon_start': [gluon_train.columns[0] for _ in range(len(gluon_train.index))],
                              'context_length': 2 * forecast_length
                             }
        from gluonts.dataset.common import ListDataset
        
        test_ds = ListDataset([{FieldName.TARGET: target, 
                                 FieldName.START: start
                                 # FieldName.FEAT_DYNAMIC_REAL: custDayOfWeek
                                 } 
                                for (target, start) in zip( # , custDayOfWeek
                                        gluon_train.values, 
                                        ts_metadata['gluon_start']
                                        # custDayOfWeek
                                        )],
                                freq=ts_metadata['freq']
                                )
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
        
    GluonNPTS = ModelResult("Gluon NPTS")
    try:
        startTime = datetime.datetime.now()
        from gluonts.model.npts import NPTSEstimator
        estimator = NPTSEstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'])
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        GluonNPTS.forecast = forecast.values
        GluonNPTS.runtime = datetime.datetime.now() - startTime
        
        GluonNPTS.mae = pd.DataFrame(mae(test.values, GluonNPTS.forecast)).mean(axis=0, skipna = True)
        GluonNPTS.overall_mae = np.nanmean(GluonNPTS.mae)
        GluonNPTS.smape = smape(test.values, GluonNPTS.forecast)
        GluonNPTS.overall_smape = np.nanmean(GluonNPTS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonNPTS.name, 
            'runtime': GluonNPTS.runtime, 
            'overall_smape': GluonNPTS.overall_smape, 
            'overall_mae': GluonNPTS.overall_mae,
            'object_name': 'GluonNPTS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonNPTS.result_message()
    
    
    GluonDeepAR = ModelResult("Gluon DeepARE")
    try:
        startTime = datetime.datetime.now()
        from gluonts.model.deepar import DeepAREstimator
        from gluonts.trainer import Trainer
        estimator = DeepAREstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=20,
                                                     num_batches_per_epoch=200)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonDeepAR.forecast = forecast.values
        GluonDeepAR.runtime = datetime.datetime.now() - startTime
        
        GluonDeepAR.mae = pd.DataFrame(mae(test.values, GluonDeepAR.forecast)).mean(axis=0, skipna = True)
        GluonDeepAR.overall_mae = np.nanmean(GluonDeepAR.mae)
        GluonDeepAR.smape = smape(test.values, GluonDeepAR.forecast)
        GluonDeepAR.overall_smape = np.nanmean(GluonDeepAR.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonDeepAR.name, 
            'runtime': GluonDeepAR.runtime, 
            'overall_smape': GluonDeepAR.overall_smape, 
            'overall_mae': GluonDeepAR.overall_mae,
            'object_name': 'GluonDeepAR'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonDeepAR.result_message()
    
        
    GluonMQCNN = ModelResult("Gluon MQCNN")
    try:
        startTime = datetime.datetime.now()
        # from gluonts.trainer import Trainer
        from gluonts.model.seq2seq import MQCNNEstimator
        estimator = MQCNNEstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=50)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonMQCNN.forecast = forecast.values
        GluonMQCNN.runtime = datetime.datetime.now() - startTime
        
        GluonMQCNN.mae = pd.DataFrame(mae(test.values, GluonMQCNN.forecast)).mean(axis=0, skipna = True)
        GluonMQCNN.overall_mae = np.nanmean(GluonMQCNN.mae)
        GluonMQCNN.smape = smape(test.values, GluonMQCNN.forecast)
        GluonMQCNN.overall_smape = np.nanmean(GluonMQCNN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonMQCNN.name, 
            'runtime': GluonMQCNN.runtime, 
            'overall_smape': GluonMQCNN.overall_smape, 
            'overall_mae': GluonMQCNN.overall_mae,
            'object_name': 'GluonMQCNN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonMQCNN.result_message()
    
    GluonSFF = ModelResult("Gluon SFF")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
        estimator = SimpleFeedForwardEstimator(
            prediction_length=ts_metadata['forecast_length'],
            context_length=ts_metadata['context_length'],
            freq=ts_metadata['freq'],
            trainer=Trainer(epochs=10, 
                            learning_rate=1e-3, 
                            hybridize=False, 
                            num_batches_per_epoch=100
                           ))
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonSFF.forecast = forecast.values
        GluonSFF.runtime = datetime.datetime.now() - startTime
        
        GluonSFF.mae = pd.DataFrame(mae(test.values, GluonSFF.forecast)).mean(axis=0, skipna = True)
        GluonSFF.overall_mae = np.nanmean(GluonSFF.mae)
        GluonSFF.smape = smape(test.values, GluonSFF.forecast)
        GluonSFF.overall_smape = np.nanmean(GluonSFF.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonSFF.name, 
            'runtime': GluonSFF.runtime, 
            'overall_smape': GluonSFF.overall_smape, 
            'overall_mae': GluonSFF.overall_mae,
            'object_name': 'GluonSFF'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonSFF.result_message()
    
    GluonTransformer = ModelResult("Gluon Transformer 20 epoch")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.transformer import TransformerEstimator
        estimator = TransformerEstimator(
            prediction_length=ts_metadata['forecast_length'],
            context_length=ts_metadata['context_length'],
            freq=ts_metadata['freq'],
            trainer=Trainer(epochs=20))
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonTransformer.forecast = forecast.values
        GluonTransformer.runtime = datetime.datetime.now() - startTime
        
        GluonTransformer.mae = pd.DataFrame(mae(test.values, GluonTransformer.forecast)).mean(axis=0, skipna = True)
        GluonTransformer.overall_mae = np.nanmean(GluonTransformer.mae)
        GluonTransformer.smape = smape(test.values, GluonTransformer.forecast)
        GluonTransformer.overall_smape = np.nanmean(GluonTransformer.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    currentResult = pd.DataFrame({
            'method': GluonTransformer.name, 
            'runtime': GluonTransformer.runtime, 
            'overall_smape': GluonTransformer.overall_smape, 
            'overall_mae': GluonTransformer.overall_mae,
            'object_name': 'GluonTransformer'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonTransformer.result_message()
    
    GluonTransformer150 = ModelResult("Gluon Transformer 150 epoch")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.transformer import TransformerEstimator
        estimator = TransformerEstimator(
            prediction_length=ts_metadata['forecast_length'],
            context_length=ts_metadata['context_length'],
            freq=ts_metadata['freq'],
            trainer=Trainer(epochs=150))
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonTransformer150.forecast = forecast.values
        GluonTransformer150.runtime = datetime.datetime.now() - startTime
        
        GluonTransformer150.mae = pd.DataFrame(mae(test.values, GluonTransformer150.forecast)).mean(axis=0, skipna = True)
        GluonTransformer150.overall_mae = np.nanmean(GluonTransformer150.mae)
        GluonTransformer150.smape = smape(test.values, GluonTransformer150.forecast)
        GluonTransformer150.overall_smape = np.nanmean(GluonTransformer150.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonTransformer150.name, 
            'runtime': GluonTransformer150.runtime, 
            'overall_smape': GluonTransformer150.overall_smape, 
            'overall_mae': GluonTransformer150.overall_mae,
            'object_name': 'GluonTransformer150'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonTransformer150.result_message()
    
    GluonDeepState = ModelResult("Gluon DeepState")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.deepstate import DeepStateEstimator
        estimator = DeepStateEstimator(
            prediction_length=ts_metadata['forecast_length'],
            past_length=ts_metadata['context_length'],
            freq=ts_metadata['freq'],
            use_feat_static_cat=False,
            cardinality = [1],
            trainer=Trainer(ctx='cpu', epochs=20))
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonDeepState.forecast = forecast.values
        GluonDeepState.runtime = datetime.datetime.now() - startTime
        
        GluonDeepState.mae = pd.DataFrame(mae(test.values, GluonDeepState.forecast)).mean(axis=0, skipna = True)
        GluonDeepState.overall_mae = np.nanmean(GluonDeepState.mae)
        GluonDeepState.smape = smape(test.values, GluonDeepState.forecast)
        GluonDeepState.overall_smape = np.nanmean(GluonDeepState.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonDeepState.name, 
            'runtime': GluonDeepState.runtime, 
            'overall_smape': GluonDeepState.overall_smape, 
            'overall_mae': GluonDeepState.overall_mae,
            'object_name': 'GluonDeepState'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonDeepState.result_message()
    
    GluonDeepFactor = ModelResult("Gluon DeepFactor")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.deep_factor import DeepFactorEstimator
        estimator = DeepFactorEstimator(freq=ts_metadata['freq'],
                                    context_length=ts_metadata['context_length'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=50)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonDeepFactor.forecast = forecast.values
        GluonDeepFactor.runtime = datetime.datetime.now() - startTime
        
        GluonDeepFactor.mae = pd.DataFrame(mae(test.values, GluonDeepFactor.forecast)).mean(axis=0, skipna = True)
        GluonDeepFactor.overall_mae = np.nanmean(GluonDeepFactor.mae)
        GluonDeepFactor.smape = smape(test.values, GluonDeepFactor.forecast)
        GluonDeepFactor.overall_smape = np.nanmean(GluonDeepFactor.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonDeepFactor.name, 
            'runtime': GluonDeepFactor.runtime, 
            'overall_smape': GluonDeepFactor.overall_smape, 
            'overall_mae': GluonDeepFactor.overall_mae,
            'object_name': 'GluonDeepFactor'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonDeepFactor.result_message()
    
    GluonWavenet = ModelResult("Gluon Wavenet")
    try:
        startTime = datetime.datetime.now()
        from gluonts.trainer import Trainer
        from gluonts.model.wavenet import WaveNetEstimator
        estimator = WaveNetEstimator(freq=ts_metadata['freq'],
                                    prediction_length=ts_metadata['forecast_length'] 
                                    ,trainer=Trainer(epochs=80)
                                    )
        
        forecast = pd.DataFrame()
        GluonPredictor = estimator.train(test_ds)
        gluon_results = GluonPredictor.predict(test_ds)
        i = 0
        for result in gluon_results:
            currentCust = gluon_train.index[i]
            rowForecast = pd.DataFrame({
                    "ForecastDate": pd.date_range(start = result.start_date, periods = ts_metadata['forecast_length'], freq = ts_metadata['freq']),
                    "series_id": currentCust,
                    # "Quantile10thForecast": (result.quantile(0.1)),
                    "MedianForecast": (result.quantile(0.5)),
                    # "Quantile90thForecast": (result.quantile(0.9))
                    })
            if no_negatives:
                rowForecast['MedianForecast'] = rowForecast['MedianForecast'].clip(lower = 0)
            forecast = pd.concat([forecast, rowForecast], ignore_index = True).reset_index(drop = True)
            i += 1
        forecast = forecast.pivot_table(values='MedianForecast', index='ForecastDate', columns='series_id')
        forecast = forecast[test.columns]
        
        GluonWavenet.forecast = forecast.values
        GluonWavenet.runtime = datetime.datetime.now() - startTime
        
        GluonWavenet.mae = pd.DataFrame(mae(test.values, GluonWavenet.forecast)).mean(axis=0, skipna = True)
        GluonWavenet.overall_mae = np.nanmean(GluonWavenet.mae)
        GluonWavenet.smape = smape(test.values, GluonWavenet.forecast)
        GluonWavenet.overall_smape = np.nanmean(GluonWavenet.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GluonWavenet.name, 
            'runtime': GluonWavenet.runtime, 
            'overall_smape': GluonWavenet.overall_smape, 
            'overall_mae': GluonWavenet.overall_mae,
            'object_name': 'GluonWavenet'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    GluonWavenet.result_message()
        
    SktimeKNN = ModelResult("Sktime KNN")
    try:
        from sktime.transformers.compose import Tabulariser
        from sklearn.neighbors import KNeighborsRegressor
        from sktime.pipeline import Pipeline
        from sktime.highlevel.tasks import ForecastingTask
        from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        fh_vals = [x + 1 for x in range(forecast_length)]
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            current_series_df = pd.DataFrame(pd.Series([current_series.reset_index(drop = True)]), columns = ['outcome'])
            steps = [
                ('tabularise', Tabulariser()),
                ('clf', KNeighborsRegressor() ) # RandomForestRegressor(n_estimators=10))
            ]
            estimator = Pipeline(steps)
            
            task = ForecastingTask(target='outcome', fh= fh_vals,
                                   metadata=current_series_df)
            
            s = Forecasting2TSRReductionStrategy(estimator=estimator)
            s.fit(task, current_series_df)
            srfPred = s.predict().values
            if no_negatives:
                try:
                    srfPred = pd.Series(np.where(srfPred > 0, srfPred, 0))
                except Exception:
                    srfPred = srfPred.where(srfPred > 0, 0) 
    
            forecast = pd.concat([forecast, srfPred], axis = 1)
        SktimeKNN.forecast = forecast.values
        SktimeKNN.runtime = datetime.datetime.now() - startTime
        
        SktimeKNN.mae = pd.DataFrame(mae(test.values, SktimeKNN.forecast)).mean(axis=0, skipna = True)
        SktimeKNN.overall_mae = np.nanmean(SktimeKNN.mae)
        SktimeKNN.smape = smape(test.values, SktimeKNN.forecast)
        SktimeKNN.overall_smape = np.nanmean(SktimeKNN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': SktimeKNN.name, 
            'runtime': SktimeKNN.runtime, 
            'overall_smape': SktimeKNN.overall_smape, 
            'overall_mae': SktimeKNN.overall_mae,
            'object_name': 'SktimeKNN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    SktimeKNN.result_message()

    SktimeAda = ModelResult("Sktime Adaboost 10 Estimators")
    try:
        from sktime.transformers.compose import Tabulariser
        from sklearn.ensemble import AdaBoostRegressor
        from sktime.pipeline import Pipeline
        from sktime.highlevel.tasks import ForecastingTask
        from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
        startTime = datetime.datetime.now()
        forecast = pd.DataFrame()
        fh_vals = [x + 1 for x in range(forecast_length)]
        for series in train.columns:
            current_series = train[series].copy()
            current_series = current_series.fillna(method='ffill').fillna(method='bfill')
            current_series_df = pd.DataFrame(pd.Series([current_series.reset_index(drop = True)]), columns = ['outcome'])
            steps = [
                ('tabularise', Tabulariser()),
                ('clf', AdaBoostRegressor(n_estimators=10) ) # RandomForestRegressor(n_estimators=10))
            ]
            estimator = Pipeline(steps)
            
            task = ForecastingTask(target='outcome', fh= fh_vals,
                                   metadata=current_series_df)
            
            s = Forecasting2TSRReductionStrategy(estimator=estimator)
            s.fit(task, current_series_df)
            srfPred = s.predict().values
            if no_negatives:
                try:
                    srfPred = pd.Series(np.where(srfPred > 0, srfPred, 0))
                except Exception:
                    srfPred = srfPred.where(srfPred > 0, 0) 
    
            forecast = pd.concat([forecast, srfPred], axis = 1)
        SktimeAda.forecast = forecast.values
        SktimeAda.runtime = datetime.datetime.now() - startTime
        
        SktimeAda.mae = pd.DataFrame(mae(test.values, SktimeAda.forecast)).mean(axis=0, skipna = True)
        SktimeAda.overall_mae = np.nanmean(SktimeAda.mae)
        SktimeAda.smape = smape(test.values, SktimeAda.forecast)
        SktimeAda.overall_smape = np.nanmean(SktimeAda.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': SktimeAda.name, 
            'runtime': SktimeAda.runtime, 
            'overall_smape': SktimeAda.overall_smape, 
            'overall_mae': SktimeAda.overall_mae,
            'object_name': 'SktimeAda'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    SktimeAda.result_message()
    
    RandForest = ModelResult("Random Forest Lag 1")
    try:
        from sklearn.ensemble import RandomForestRegressor
        startTime = datetime.datetime.now()
        sktraindata = train.dropna(how = 'all', axis = 0).fillna(method='ffill').fillna(method='bfill')
        Y = sktraindata.drop(sktraindata.head(1).index)
        X = sktraindata.drop(sktraindata.tail(1).index)
        
        regr = RandomForestRegressor(random_state=425, n_estimators=100)
        regr.fit(X, Y) 
        
        forecast = regr.predict(sktraindata.tail(1).values)
        for x in range(forecast_length - 1):
            rfPred = regr.predict(forecast[-1, :].reshape(1, -1))
            if no_negatives:
                rfPred[rfPred < 0] = 0
            forecast = np.append(forecast, rfPred, axis = 0)
        RandForest.forecast = forecast
        RandForest.runtime = datetime.datetime.now() - startTime
        
        RandForest.mae = pd.DataFrame(mae(test.values, RandForest.forecast)).mean(axis=0, skipna = True)
        RandForest.overall_mae = np.nanmean(RandForest.mae)
        RandForest.smape = smape(test.values, RandForest.forecast)
        RandForest.overall_smape = np.nanmean(RandForest.smape)
        del(sktraindata)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': RandForest.name, 
            'runtime': RandForest.runtime, 
            'overall_smape': RandForest.overall_smape, 
            'overall_mae': RandForest.overall_mae,
            'object_name': 'RandForest'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    RandForest.result_message()
    
    LagRegr = ModelResult("ElasticNet Lag 1")
    try:
        from sklearn.linear_model import MultiTaskElasticNet
        startTime = datetime.datetime.now()
        sktraindata = train.dropna(how = 'all', axis = 0).fillna(method='ffill').fillna(method='bfill')
        Y = sktraindata.drop(sktraindata.head(1).index)
        X = sktraindata.drop(sktraindata.tail(1).index)
        
        regr = MultiTaskElasticNet(alpha = 1.0)
        regr.fit(X, Y) 
        
        forecast = regr.predict(sktraindata.tail(1).values)
        for x in range(forecast_length - 1):
            rfPred = regr.predict(forecast[-1, :].reshape(1, -1))
            if no_negatives:
                rfPred[rfPred < 0] = 0
            forecast = np.append(forecast, rfPred, axis = 0)
        LagRegr.forecast = forecast
        LagRegr.runtime = datetime.datetime.now() - startTime
        
        LagRegr.mae = pd.DataFrame(mae(test.values, LagRegr.forecast)).mean(axis=0, skipna = True)
        LagRegr.overall_mae = np.nanmean(LagRegr.mae)
        LagRegr.smape = smape(test.values, LagRegr.forecast)
        LagRegr.overall_smape = np.nanmean(LagRegr.smape)
        del(sktraindata)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': LagRegr.name, 
            'runtime': LagRegr.runtime, 
            'overall_smape': LagRegr.overall_smape, 
            'overall_mae': LagRegr.overall_mae,
            'object_name': 'LagRegr'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    LagRegr.result_message()
    
    RandFeaturedForest = ModelResult("Random Forest Lag 1, Roll 7 std, Roll 30 mean")
    try:
        from sklearn.ensemble import RandomForestRegressor
        startTime = datetime.datetime.now()
        sktraindata = train.dropna(how = 'all', axis = 0).fillna(method='ffill').fillna(method='bfill')
        Y = sktraindata.drop(sktraindata.head(2).index) 
        Y.columns = [x for x in range(len(Y.columns))]
       
        def X_maker(df):
            X = pd.concat([df, df.rolling(30,min_periods = 1).mean(), df.rolling(7,min_periods = 1).std()], axis = 1)
            X.columns = [x for x in range(len(X.columns))]
            X.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            return X
        X = X_maker(sktraindata)
        X = X.drop(X.tail(1).index).drop(X.head(1).index)
     
        regr = RandomForestRegressor(random_state=425, n_estimators=100)
        regr.fit(X, Y)
        
        forecast = pd.DataFrame()
        sktraindata.columns = [x for x in range(len(sktraindata.columns))]
        for x in range(forecast_length):
            rfPred =  pd.DataFrame(regr.predict(X_maker(sktraindata).tail(1).values))
            if no_negatives:
                rfPred[rfPred < 0] = 0
            forecast = pd.concat([forecast, rfPred], axis = 0, ignore_index = True)
            sktraindata = pd.concat([sktraindata, rfPred], axis = 0, ignore_index = True)
        RandFeaturedForest.forecast = forecast.values
        RandFeaturedForest.runtime = datetime.datetime.now() - startTime
       
        RandFeaturedForest.mae = pd.DataFrame(mae(test.values, RandFeaturedForest.forecast)).mean(axis=0, skipna = True)
        RandFeaturedForest.overall_mae = np.nanmean(RandFeaturedForest.mae)
        RandFeaturedForest.smape = smape(test.values, RandFeaturedForest.forecast)
        RandFeaturedForest.overall_smape = np.nanmean(RandFeaturedForest.smape)
        del(sktraindata)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
     
    currentResult = pd.DataFrame({
            'method': RandFeaturedForest.name,
            'runtime': RandFeaturedForest.runtime,
            'overall_smape': RandFeaturedForest.overall_smape,
            'overall_mae': RandFeaturedForest.overall_mae,
            'object_name': 'RandFeaturedForest'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    RandFeaturedForest.result_message()
    
    """
    Simple Ensembles
    """
    AllModelEnsemble = ModelResult("All Model Ensemble")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        all_models = [GLM, sETS, ETS, dETS, ProphetResult, ProphetResultHoliday, 
     AutoArima, SARIMA, UnComp, GluonNPTS, GluonDeepAR, GluonMQCNN, GluonSFF,
     GluonWavenet, GluonDeepFactor, GluonDeepState, GluonTransformer, 
     SktimeKNN, SktimeAda, RandFeaturedForest, RandForest, LagRegr]
        for modelmethod in all_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        AllModelEnsemble.forecast = master_array/n
        AllModelEnsemble.runtime = runtime
        
        AllModelEnsemble.mae = pd.DataFrame(mae(test.values, AllModelEnsemble.forecast)).mean(axis=0, skipna = True)
        AllModelEnsemble.overall_mae = np.nanmean(AllModelEnsemble.mae)
        AllModelEnsemble.smape = smape(test.values, AllModelEnsemble.forecast)
        AllModelEnsemble.overall_smape = np.nanmean(AllModelEnsemble.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': AllModelEnsemble.name, 
            'runtime': AllModelEnsemble.runtime, 
            'overall_smape': AllModelEnsemble.overall_smape, 
            'overall_mae': AllModelEnsemble.overall_mae,
            'object_name': 'AllModelEnsemble'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    M4 = ModelResult("ETS M4 Comb")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        ensemble_models = [sETS, ETS, dETS]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        M4.forecast = master_array/n
        M4.runtime = runtime
        
        M4.mae = pd.DataFrame(mae(test.values, M4.forecast)).mean(axis=0, skipna = True)
        M4.overall_mae = np.nanmean(M4.mae)
        M4.smape = smape(test.values, M4.forecast)
        M4.overall_smape = np.nanmean(M4.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': M4.name, 
            'runtime': M4.runtime, 
            'overall_smape': M4.overall_smape, 
            'overall_mae': M4.overall_mae,
            'object_name': 'M4'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    bestN = ModelResult("Best 3 Ensemble")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        
        bestNames = model_results[model_results['overall_smape'] > 0].sort_values('overall_smape', ascending = True).head(3)['object_name'].values
        n = 0
        runtime = datetime.timedelta(0)
        for modelmethod in bestNames:
            modelmethod_obj = eval(modelmethod) # globals()[modelmethod] # getattr(sys.modules[__name__], modelmethod)
            if modelmethod_obj.overall_smape != -1:
                master_array = master_array + modelmethod_obj.forecast
                n += 1
                runtime = runtime + modelmethod_obj.runtime
    
        bestN.forecast = master_array/n
        bestN.runtime = runtime
        
        bestN.mae = pd.DataFrame(mae(test.values, bestN.forecast)).mean(axis=0, skipna = True)
        bestN.overall_mae = np.nanmean(bestN.mae)
        bestN.smape = smape(test.values, bestN.forecast)
        bestN.overall_smape = np.nanmean(bestN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': bestN.name, 
            'runtime': bestN.runtime, 
            'overall_smape': bestN.overall_smape, 
            'overall_mae': bestN.overall_mae,
            'object_name': 'bestN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    EPA = ModelResult("ETS + Prophet + ARIMA")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        ensemble_models = [AutoArima, ETS, ProphetResult]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        EPA.forecast = master_array/n
        EPA.runtime = runtime
        
        EPA.mae = pd.DataFrame(mae(test.values, EPA.forecast)).mean(axis=0, skipna = True)
        EPA.overall_mae = np.nanmean(EPA.mae)
        EPA.smape = smape(test.values, EPA.forecast)
        EPA.overall_smape = np.nanmean(EPA.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': EPA.name, 
            'runtime': EPA.runtime, 
            'overall_smape': EPA.overall_smape, 
            'overall_mae': EPA.overall_mae,
            'object_name': 'EPA'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    
    GE = ModelResult("GluonMQCNN + ETS")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        ensemble_models = [GluonMQCNN, ETS]
        runtime = datetime.timedelta(0)
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        GE.forecast = master_array/n
        GE.runtime = runtime
        
        GE.mae = pd.DataFrame(mae(test.values, GE.forecast)).mean(axis=0, skipna = True)
        GE.overall_mae = np.nanmean(GE.mae)
        GE.smape = smape(test.values, GE.forecast)
        GE.overall_smape = np.nanmean(GE.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    currentResult = pd.DataFrame({
            'method': GE.name, 
            'runtime': GE.runtime, 
            'overall_smape': GE.overall_smape, 
            'overall_mae': GE.overall_mae,
            'object_name': 'GE'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    GEAR = ModelResult("GluonDeepARE + ETS")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        ensemble_models = [GluonMQCNN, ETS]
        for modelmethod in ensemble_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        GEAR.forecast = master_array/n
        GEAR.runtime = runtime
        
        GEAR.mae = pd.DataFrame(mae(test.values, GEAR.forecast)).mean(axis=0, skipna = True)
        GEAR.overall_mae = np.nanmean(GEAR.mae)
        GEAR.smape = smape(test.values, GEAR.forecast)
        GEAR.overall_smape = np.nanmean(GEAR.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': GEAR.name, 
            'runtime': GEAR.runtime, 
            'overall_smape': GEAR.overall_smape, 
            'overall_mae': GEAR.overall_mae,
            'object_name': 'GEAR'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    RAETS = ModelResult("ETS + Random Forest")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        all_models = [ETS, RandForest]
        for modelmethod in all_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        RAETS.forecast = master_array/n
        RAETS.runtime = runtime
        
        RAETS.mae = pd.DataFrame(mae(test.values, RAETS.forecast)).mean(axis=0, skipna = True)
        RAETS.overall_mae = np.nanmean(RAETS.mae)
        RAETS.smape = smape(test.values, RAETS.forecast)
        RAETS.overall_smape = np.nanmean(RAETS.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': RAETS.name, 
            'runtime': RAETS.runtime, 
            'overall_smape': RAETS.overall_smape, 
            'overall_mae': RAETS.overall_mae,
            'object_name': 'RAETS'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    RATKN = ModelResult("SktimeKNN + Random Forest")
    try:
        master_array = np.zeros((test.shape[0], test.shape[1]))
        n = 0
        runtime = datetime.timedelta(0)
        all_models = [SktimeKNN, RandForest]
        for modelmethod in all_models:
            if modelmethod.overall_smape != -1:
                master_array = master_array + modelmethod.forecast
                runtime = runtime + modelmethod.runtime
                n += 1
    
        RATKN.forecast = master_array/n
        RATKN.runtime = runtime
        
        RATKN.mae = pd.DataFrame(mae(test.values, RATKN.forecast)).mean(axis=0, skipna = True)
        RATKN.overall_mae = np.nanmean(RATKN.mae)
        RATKN.smape = smape(test.values, RATKN.forecast)
        RATKN.overall_smape = np.nanmean(RATKN.smape)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    currentResult = pd.DataFrame({
            'method': RATKN.name, 
            'runtime': RATKN.runtime, 
            'overall_smape': RATKN.overall_smape, 
            'overall_mae': RATKN.overall_mae,
            'object_name': 'RATKN'
            }, index = [0])
    model_results = pd.concat([model_results, currentResult], ignore_index = True).reset_index(drop = True)
    
    
    per_series_smape = np.zeros((len(test.columns),))
    try:
        finishedNames = model_results[model_results['overall_smape'] > 0].sort_values('overall_smape', ascending = True)['object_name'].values
        for modelmethod in finishedNames:
            modelmethod_obj = eval(modelmethod) # getattr(sys.modules[__name__], modelmethod) # or eval()
            if modelmethod_obj.overall_smape != -1:
                per_series_smape = np.vstack((per_series_smape, modelmethod_obj.smape))
        per_series_smape = np.delete(per_series_smape, (0), axis=0) # remove the zeros
        per_series_smape = pd.DataFrame(per_series_smape, columns = test.columns)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    per_series_mae = np.zeros((len(test.columns),))
    try:
        finishedNames = model_results[model_results['overall_mae'] > 0].sort_values('overall_mae', ascending = True)['object_name'].values
        for modelmethod in finishedNames:
            modelmethod_obj =  eval(modelmethod) #getattr(sys.modules[__name__], modelmethod)
            if modelmethod_obj.overall_smape != -1:
                per_series_mae = np.vstack((per_series_mae, modelmethod_obj.smape))
        per_series_mae = np.delete(per_series_mae, (0), axis=0) # remove the zeros
        per_series_mae = pd.DataFrame(per_series_mae, columns = test.columns)
    except Exception as e:
        print(e)
        error_list.extend([traceback.format_exc()])
    
    final_result = EvaluationReturn("run_result")
    
    final_result.model_performance = model_results
    final_result.per_series_mae = per_series_mae
    final_result.per_series_smape = per_series_smape
    final_result.errors = error_list
    return final_result

evaluator_result = model_evaluator(train, test, subset = series_to_sample)
evaluator_result.errors
eval_table = evaluator_result.model_performance.sort_values('overall_smape', ascending = True)
eval_table.to_csv(output_name, index = False)
print("Complete at: " + str(datetime.datetime.now()))

"""
Handle missing with better options
    Slice out missing days and pretend it's continuous (make up dates, remove week day effects)
    Fill forward and backwards
    Fill 0
Handle different length time series (time series that don't start until later)
Run multiple test segments
Limit the context length to a standard length?

Profile metadata of series (number NaN imputed, context length, etc)

Capture volatility of results - basically get methods that do great on many even if they absymally fail on others
    Median sMAPE

Forecast distance blending (mix those more accurate in short term and long term) ensemble

For monthly, account for the last, incomplete, month
"""