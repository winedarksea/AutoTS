"""
Naives and Others Requiring No Additional Packages Beyond Numpy and Pandas
"""
import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int
from autots.tools.probabilistic import Point_to_Probability, historic_quantile


class ZeroesNaive(ModelObject):
    """Naive forecasting predicting a dataframe of zeroes (0's)
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "ZeroesNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country=holiday_country,
                             random_seed=random_seed, verbose=verbose)
    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        df = self.basic_profile(df)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, future_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(np.zeros((forecast_length,(self.train_shape[1]))), columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length) )
        if just_point_forecast:
            return df
        else:
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df.index,
                                          forecast_columns=df.columns,
                                          lower_forecast=df,
                                          forecast=df, upper_forecast=df,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            
            return prediction
        
    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}
    
    def get_params(self):
        """Return dict of current parameters
        """
        return {}

class LastValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the last series value
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """
    def __init__(self, name: str = "LastValueNaive", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             holiday_country=holiday_country,
                             random_seed=random_seed)
    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied 

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.last_values = df.tail(1).values
        # self.df_train = df
        self.lower, self.upper = historic_quantile(
            df, prediction_interval=self.prediction_interval)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, future_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(
            np.tile(self.last_values, (forecast_length,1)),
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length))
        if just_point_forecast:
            return df
        else:
            # upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df, prediction_interval = self.prediction_interval, method = 'historic_quantile')
            upper_forecast = df.astype(float) + (self.upper * 0.8)
            lower_forecast = df.astype(float) - (self.lower * 0.8)
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df.index,
                                          forecast_columns=df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            return prediction

    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        return {}

    def get_params(self):
        """Return dict of current parameters
        """
        return {}

class AverageValueNaive(ModelObject):
    """Naive forecasting predicting a dataframe of the series' median values

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

    """

    def __init__(self, name: str = "AverageValueNaive",
                 frequency: str = 'infer',
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 method: str = 'Median'):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             holiday_country=holiday_country,
                             random_seed=random_seed,
                             verbose=verbose)
        self.method = method

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied. 

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        if str(self.method).lower() == 'median':
            self.average_values = df.median(axis = 0).values
        if str(self.method).lower() == 'mean':
            self.average_values = df.mean(axis = 0).values
        if str(self.method).lower() == 'mode':
            self.average_values = df.mode(axis = 0).iloc[0].fillna(df.median(axis=0)).values
        self.fit_runtime = datetime.datetime.now() - self.startTime
        self.lower, self.upper = historic_quantile(
            df, prediction_interval=self.prediction_interval)
        return self

    def predict(self, forecast_length: int, future_regressor=[],
                just_point_forecast=False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
 
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        df = pd.DataFrame(
            np.tile(self.average_values, (forecast_length, 1)),
            columns=self.column_names,
            index=self.create_forecast_index(forecast_length=forecast_length))
        if just_point_forecast:
            return df
        else:
            upper_forecast = df.astype(float) + self.upper
            lower_forecast = df.astype(float) - self.lower
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df.index,
                                          forecast_columns=df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            return prediction

    def get_new_params(self,method: str = 'random'):
        """Returns dict of new parameters for parameter tuning
        """
        method_choice = np.random.choice(a=['Median', 'Mean', 'Mode'],
                                         size=1, p=[0.3, 0.6, 0.1]).item()
        return {
                'method': method_choice
                }

    def get_params(self):
        """Return dict of current parameters."""
        return {
                'method': self.method
                }


class SeasonalNaive(ModelObject):
    """Naive forecasting predicting a dataframe with seasonal (lag) forecasts.

    Concerto No. 2 in G minor, Op. 8, RV 315

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        method (str): Either 'LastValue' (use last value of lag n) or 'Mean' (avg of all lag n)
        lag_1 (int): The lag of the seasonality, should int > 1.
        lag_2 (int): Optional second lag of seasonality which is averaged with first lag to produce forecast.

    """

    def __init__(self, name: str = "SeasonalNaive", frequency: str = 'infer',
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 lag_1: int = 7, lag_2: int = None, method: str = 'LastValue'):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             holiday_country=holiday_country,
                             random_seed=random_seed, verbose=verbose)
        self.lag_1 = abs(int(lag_1))
        self.lag_2 = lag_2
        if str(self.lag_2).isdigit():
            self.lag_2 = abs(int(self.lag_2))
            if str(self.lag_2) == str(self.lag_1):
                self.lag_2 = 1
        self.method = method

    def fit(self, df, future_regressor = []):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)
        self.df_train = df

        df_length = (self.train_shape[0])
        self.tile_values_lag_2 = None
        if self.method in ['Mean', 'Median']:
            tile_index = np.tile(np.arange(self.lag_1),
                                 int(np.ceil(df_length/self.lag_1)))
            tile_index = tile_index[len(tile_index)-(df_length):]
            df.index = tile_index
            if self.method == "Median":
                self.tile_values_lag_1 = df.groupby(level=0, axis=0).median()
            else:
                self.tile_values_lag_1 = df.groupby(level=0, axis=0).mean()
            if str(self.lag_2).isdigit():
                if self.lag_2 == 1:
                    self.tile_values_lag_2 = df.tail(self.lag_2)
                else:
                    tile_index = np.tile(np.arange(self.lag_2),
                                         int(np.ceil(df_length/self.lag_2)))
                    tile_index = tile_index[len(tile_index)-(df_length):]
                    df.index = tile_index
                    if self.method == "Median":
                        self.tile_values_lag_2 = df.groupby(
                            level=0, axis=0).median()
                    else:
                        self.tile_values_lag_2 = df.groupby(
                            level=0, axis=0).mean()
        else:
            self.method == 'LastValue'
            self.tile_values_lag_1 = df.tail(self.lag_1)
            if str(self.lag_2).isdigit():
                self.tile_values_lag_2 = df.tail(self.lag_2)
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int,
                future_regressor = [], just_point_forecast: bool = False):
        """Generate forecast data immediately following dates of .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        tile_len = len(self.tile_values_lag_1.index)
        df = pd.DataFrame(np.tile(self.tile_values_lag_1, (int(np.ceil(forecast_length/tile_len)),1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
        if str(self.lag_2).isdigit():
            y = pd.DataFrame(np.tile(self.tile_values_lag_2, (int(np.ceil(forecast_length/len(self.tile_values_lag_2.index))), 1))[0:forecast_length], columns = self.column_names, index = self.create_forecast_index(forecast_length=forecast_length))
            df = (df + y) / 2
        # df = df.apply(pd.to_numeric, errors='coerce')
        df = df.astype(float)
        if just_point_forecast:
            return df
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train, df, method='inferred_normal',
                prediction_interval=self.prediction_interval)

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df.index,
                                          forecast_columns=df.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df,
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        lag_1_choice = seasonal_int()
        lag_2_choice = np.random.choice(a=['None',
                                           seasonal_int(include_one=True)],
                                        size=1, p=[0.3, 0.7]).item()
        if str(lag_2_choice) == str(lag_1_choice):
            lag_2_choice = 1
        method_choice = np.random.choice(a=['Mean', 'Median', 'LastValue'],
                                         size=1,
                                         p=[0.4, 0.2, 0.4]).item()
        return {
                'method': method_choice,
                'lag_1': lag_1_choice,
                'lag_2': lag_2_choice
                }

    def get_params(self):
        """Return dict of current parameters."""
        return {
                'method': self.method,
                'lag_1': self.lag_1,
                'lag_2': self.lag_2
                }


class MotifSimulation(ModelObject):
    """More dark magic created by the evil mastermind of this project.
    Basically a highly-customized KNN
    
    Warning: if you are forecasting many steps (large forecast_length), and interested in probabilistic upper/lower forecasts, then set recency_weighting <= 0, and have a larger cutoff_minimum
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        
        phrase_len (int): length of motif vectors to compare as samples
        comparison (str): method to process data before comparison, 'magnitude' is original data
        shared (bool): whether to compare motifs across all series together, or separately
        distance_metric (str): passed through to sklearn pairwise_distances
        max_motifs (float): number of motifs to compare per series. If less 1, used as % of length training data
        recency_weighting (float): amount to the value of more recent data.
        cutoff_threshold (float): lowest value of distance metric to allow into forecast
        cutoff_minimum (int): minimum number of motif vectors to include in forecast.
        point_method (str): summarization method to choose forecast on, 'sample', 'mean', 'sign_biased_mean', 'median'
    """

    def __init__(self, name: str = "MotifSimulation", frequency: str = 'infer',
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020,
                 phrase_len: str = '5',
                 comparison: str = 'magnitude_pct_change_sign',
                 shared: bool = False,
                 distance_metric: str = 'l2',
                 max_motifs: float = 50,
                 recency_weighting: float = 0.1,
                 cutoff_threshold: float = 0.9,
                 cutoff_minimum: int = 20,
                 point_method: str = 'median',
                 verbose: int = 1
                 ):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             holiday_country=holiday_country,
                             random_seed=random_seed)
        self.phrase_len = phrase_len
        self.comparison = comparison
        self.shared = shared
        self.distance_metric = distance_metric
        self.max_motifs = max_motifs
        self.recency_weighting = recency_weighting
        self.cutoff_threshold = cutoff_threshold
        self.cutoff_minimum = cutoff_minimum
        self.point_method = point_method
        
    def fit(self, df, future_regressor = []):
        """Train algorithm given data supplied.
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 
        """
        if abs(float(self.max_motifs)) > 1:
            self.max_motifs_n = abs(int(self.max_motifs))
        elif float(self.max_motifs) == 1:
            self.max_motifs_n = 2
        elif float(self.max_motifs) < 1:
            self.max_motifs_n = int(abs(np.floor(self.max_motifs * df.shape[0])))
        else:
            self.max_motifs = 10
            self.max_motifs_n = 10
        
        # convert strings of phrase length into int lengths
        if str(self.phrase_len).isdigit():
            self.phrase_n = int(self.phrase_len)
        elif self.phrase_len == '10thN':
            self.phrase_n = int(np.floor((df.shape[0])/10))
        elif self.phrase_len == '100thN':
            self.phrase_n = int(np.floor((df.shape[0])/100))
        elif self.phrase_len == '1000thN':
            self.phrase_n = int(np.floor((df.shape[0])/1000))
        else:
            self.phrase_len = '20thN'
            self.phrase_n = int(np.floor((df.shape[0])/20))
        if (self.phrase_n > df.shape[0]) or (self.phrase_n <= 1):
            # raise ValueError("phrase_len is inappropriate for the length of training data provided")
            self.phrase_n = 3
            self.phrase_len = 3
        # df = df_wide[df_wide.columns[0:3]].fillna(0).astype(float)
        df = self.basic_profile(df)
        """
        comparison = 'pct_change' # pct_change, pct_change_sign, magnitude_pct_change_sign, magnitude, magnitude_pct_change
        distance_metric = 'hamming'
        max_motifs_n = 100
        phrase_n = 5
        shared = False
        recency_weighting = 0.1
        cutoff_threshold = 0.8
        cutoff_minimum = 20
        prediction_interval = 0.9
        na_threshold = 0.1
        point_method = 'sample'
        """
        phrase_n = abs(int(self.phrase_n))
        max_motifs_n = abs(int(self.max_motifs_n))
        comparison = self.comparison
        distance_metric = self.distance_metric
        shared = self.shared
        recency_weighting = float(self.recency_weighting)
        cutoff_threshold = float(self.cutoff_threshold)
        cutoff_minimum = abs(int(self.cutoff_minimum))
        prediction_interval = float(self.prediction_interval)
        na_threshold = 0.1
        point_method = self.point_method

        # transform the data into different views (contour = percent_change)
        if 'pct_change' in comparison:
            if comparison in ['magnitude_pct_change', 'magnitude_pct_change_sign']:
                original_df = df.copy()
            df = df.replace([0], np.nan)
            df = df.fillna(abs(df[df != 0]).min()).fillna(0.1)
            last_row = df.tail(1)
            df = df.pct_change(periods=1, fill_method='ffill').tail(df.shape[0] - 1).fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
        else:
            self.comparison = 'magnitude'

        if 'pct_change_sign' in comparison:
            last_motif = df.where(df >= 0, -1).where(df <= 0, 1).tail(phrase_n)
        else:
            last_motif = df.tail(phrase_n)

        numbers = np.random.choice((df.shape[0] - phrase_n),
                                   size=max_motifs_n,
                                   replace=False)

        # make this faster
        motif_vecs = pd.DataFrame()
        # takes random slices of the time series and rearranges as phrase_n length vectors
        for z in numbers:
            rand_slice = df.iloc[z:(z + phrase_n), ]
            rand_slice = rand_slice.reset_index(drop=True).transpose().set_index(np.repeat(z, (df.shape[1], )), append=True)
            motif_vecs = pd.concat([motif_vecs, rand_slice], axis=0)

        if 'pct_change_sign' in comparison:
            motif_vecs = motif_vecs.where(motif_vecs >= 0, -1).where(motif_vecs <= 0, 1)

        # compare the motif vectors to the most recent vector of the series
        from sklearn.metrics.pairwise import pairwise_distances
        if shared:
            comparative = pd.DataFrame(
                pairwise_distances(motif_vecs, last_motif.transpose(),
                                   metric=distance_metric))
            comparative.index = motif_vecs.index
            comparative.columns = last_motif.columns
        if not shared:
            # make this faster
            comparative = pd.DataFrame()
            for column in last_motif.columns:
                x = motif_vecs[motif_vecs.index.get_level_values(0) == column]
                y = last_motif[column].values.reshape(1, -1)
                current_comparative = pd.DataFrame(
                    pairwise_distances(x, y, metric=distance_metric))
                current_comparative.index = x.index
                current_comparative.columns = [column]
                comparative = pd.concat([comparative, current_comparative],
                                        axis=0, sort=True)
            comparative = comparative.groupby(level=[0, 1]).sum(min_count=0)

        # comparative is a df of motifs (in index) with their value to each series (per column)
        if recency_weighting != 0:
            rec_weights = np.repeat(((comparative.index.get_level_values(1))/df.shape[0]).values.reshape(-1,1) * recency_weighting, len(comparative.columns), axis=1)
            comparative = comparative.add(rec_weights, fill_value=0)

        # make this faster
        upper_forecasts = pd.DataFrame()
        forecasts = pd.DataFrame()
        lower_forecasts = pd.DataFrame()
        for col in comparative.columns:
            # comparative.idxmax()
            vals = comparative[col].sort_values(ascending=False)
            if not shared:
                vals = vals[vals.index.get_level_values(0) == col]
            vals = vals[vals > cutoff_threshold]
            if vals.shape[0] < cutoff_minimum:
                vals = comparative[col].sort_values(ascending=False)
                if not shared:
                    vals = vals[vals.index.get_level_values(0) == col]
                vals = vals.head(cutoff_minimum)

            pos_forecasts = pd.DataFrame()
            for val_index, val_value in vals.items():
                sec_start = (val_index[1] + phrase_n)
                if comparison in ['magnitude_pct_change', 'magnitude_pct_change_sign']:
                    current_pos = original_df[val_index[0]].iloc[sec_start + 1:]
                else:
                    current_pos = df[val_index[0]].iloc[sec_start:]
                pos_forecasts = pd.concat([pos_forecasts, current_pos.reset_index(drop=True)], axis=1, sort=False)

            thresh = int(np.ceil(pos_forecasts.shape[1] * na_threshold))
            if point_method == 'mean':
                current_forecast = pos_forecasts.mean(axis=1)
            elif point_method == 'sign_biased_mean':
                axis_means = pos_forecasts.mean(axis=0)
                if axis_means.mean() > 0:
                    pos_forecasts = pos_forecasts[pos_forecasts.columns[~(axis_means < 0)]]
                else:
                    pos_forecasts = pos_forecasts[pos_forecasts.columns[~(axis_means > 0)]]
                current_forecast = pos_forecasts.mean(axis=1)
            elif point_method == 'sample':
                current_forecast = pos_forecasts.sample(n=1, axis=1,
                                                        weights=vals.values)
            else:
                point_method = 'median'
                current_forecast = pos_forecasts.median(axis=1)
            # current_forecast.columns = [col]
            forecasts = pd.concat([forecasts, current_forecast],
                                  axis=1, sort=False)

            if point_method == 'sample':
                n_samples = int(np.ceil(pos_forecasts.shape[1]/2))
                current_forecast = pos_forecasts.sample(
                    n=n_samples, axis=1, weights=vals.values
                    ).dropna(thresh=thresh, axis=0).quantile(
                        q=[(1 - (prediction_interval* 1.1)),
                           (prediction_interval * 1.1)], axis=1).transpose()
            else:
                current_forecast = pos_forecasts.dropna(
                    thresh=thresh, axis=0).quantile(
                        q=[(1 - prediction_interval), prediction_interval],
                        axis=1).transpose()
            # current_forecast.columns = [col, col]
            lower_forecasts = pd.concat([lower_forecasts, current_forecast.iloc[:, 0]], axis=1, sort=False)
            upper_forecasts = pd.concat([upper_forecasts, current_forecast.iloc[:, 1]], axis=1, sort=False)
        forecasts.columns = comparative.columns
        lower_forecasts.columns = comparative.columns
        upper_forecasts.columns = comparative.columns

        if comparison in ['pct_change', 'pct_change_sign']:
            forecasts = (forecasts + 1).replace([0], np.nan)
            forecasts = forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            forecasts = pd.concat([last_row.reset_index(drop=True), (forecasts)], axis=0, sort=False).cumprod()
            upper_forecasts = (upper_forecasts + 1).replace([0], np.nan)
            upper_forecasts = upper_forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            upper_forecasts = pd.concat([last_row.reset_index(drop=True), (upper_forecasts)], axis=0, sort=False).cumprod()
            lower_forecasts = (lower_forecasts + 1).replace([0], np.nan)
            lower_forecasts = lower_forecasts.fillna(abs(df[df != 0]).min()).fillna(0.1)
            lower_forecasts = pd.concat([last_row.reset_index(drop=True), (lower_forecasts)], axis=0, sort=False).cumprod()

        self.forecasts = forecasts
        self.lower_forecasts = lower_forecasts
        self.upper_forecasts = upper_forecasts

        """
        In fit phase, only select motifs.
            table: start index, weight, column it applies to, and count of rows that follow motif
            slice into possible motifs
            compare motifs (efficiently)
            choose the motifs to use for each series
                if not shared, can drop column part of index ref
            combine the following values into forecasts
                consider the weights
                magnitude and percentage change
            account for forecasts not running the full length of forecast_length
                if longer than comparative, append na df then ffill

        Profile speed and which code to improve first
            Remove for loops
            Quantile not be calculated until after pos_forecasts narrowed down to only forecast length
            https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
        """
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, future_regressor = [], just_point_forecast = False):
        """Generates forecast data immediately following dates of index supplied to .fit()
        
        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor, not used
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts
            
        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        
        forecasts = self.forecasts.head(forecast_length)
        if forecasts.shape[0] < forecast_length:
            extra_len = forecast_length - forecasts.shape[0]
            empty_frame = pd.DataFrame(index=np.arange(extra_len), columns=forecasts.columns)
            forecasts = pd.concat([forecasts, empty_frame], axis=0, sort=False).fillna(method='ffill')
        forecasts.columns = self.column_names
        forecasts.index = self.create_forecast_index(forecast_length=forecast_length)
        
        if just_point_forecast:
            return forecasts
        else:
            lower_forecasts = self.lower_forecasts.head(forecast_length)
            upper_forecasts = self.upper_forecasts.head(forecast_length)
            if lower_forecasts.shape[0] < forecast_length:
                extra_len = forecast_length - lower_forecasts.shape[0]
                empty_frame = pd.DataFrame(index=np.arange(extra_len), columns=lower_forecasts.columns)
                lower_forecasts = pd.concat([lower_forecasts, empty_frame], axis=0, sort=False).fillna(method='ffill')
            lower_forecasts.columns = self.column_names
            lower_forecasts.index = self.create_forecast_index(forecast_length=forecast_length)
            
            if upper_forecasts.shape[0] < forecast_length:
                extra_len = forecast_length - upper_forecasts.shape[0]
                empty_frame = pd.DataFrame(index=np.arange(extra_len), columns=upper_forecasts.columns)
                upper_forecasts = pd.concat([upper_forecasts, empty_frame], axis=0, sort=False).fillna(method='ffill')
            upper_forecasts.columns = self.column_names
            upper_forecasts.index = self.create_forecast_index(forecast_length=forecast_length)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name = self.name,
                                          forecast_length=forecast_length,
                                          forecast_index = forecasts.index,
                                          forecast_columns = forecasts.columns,
                                          lower_forecast=lower_forecasts,
                                          forecast=forecasts, upper_forecast=upper_forecasts,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime = self.fit_runtime,
                                          model_parameters = self.get_params())
            
            return prediction
        
    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        comparison_choice = np.random.choice(
            a=['pct_change', 'pct_change_sign',
               'magnitude_pct_change_sign',
               'magnitude', 'magnitude_pct_change'], size=1,
            p=[0.2, 0.1, 0.4, 0.2, 0.1]).item()
        phrase_len_choice = np.random.choice(
            a=[5, 10, 20, '10thN', '100thN', '1000thN', '20thN'],
            p=[0.4, 0.1, 0.3, 0.01, 0.1, 0.08, 0.01], size=1).item()
        shared_choice = np.random.choice(
            a=[True, False], size=1, p=[0.05, 0.95]).item()
        distance_metric_choice = np.random.choice(
            a=['other', 'hamming', 'cityblock', 'cosine',
               'euclidean', 'l1', 'l2', 'manhattan'], size=1,
            p=[0.2, 0.05, 0.1, 0.1,
               0.1, 0.2, 0.24, 0.01]).item()
        if distance_metric_choice == 'other':
            distance_metric_choice = np.random.choice(
                a=['braycurtis', 'canberra', 'chebyshev', 'correlation',
                   'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                   'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                   'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
                size=1).item()
        max_motifs_choice = float(np.random.choice(
            a=[20, 50, 0.05, 0.2, 0.5], size=1,
            p=[0.4, 0.1, 0.3, 0.19, 0.01]).item())
        recency_weighting_choice = np.random.choice(
            a=[0, 0.5, 0.1, 0.01, -0.01, 0.001], size=1,
            p=[0.5, 0.02, 0.05, 0.35, 0.05, 0.03]).item()
        cutoff_threshold_choice = np.random.choice(
            a=[0.7, 0.9, 0.99, 1.5], size=1, p=[0.1, 0.1, 0.4, 0.4]).item()
        cutoff_minimum_choice = np.random.choice(
            a=[5, 10, 20, 50, 100, 200], size=1,
            p=[0.05, 0.05, 0.2, 0.2, 0.4, 0.1]).item()
        point_method_choice = np.random.choice(
            a=['median', 'sample', 'mean', 'sign_biased_mean'], size=1,
            p=[0.5, 0.1, 0.3, 0.1]).item()

        return {
                'phrase_len': phrase_len_choice,
                'comparison': comparison_choice,
                'shared': shared_choice,
                'distance_metric': distance_metric_choice,
                'max_motifs': max_motifs_choice,
                'recency_weighting': recency_weighting_choice,
                'cutoff_threshold': cutoff_threshold_choice,
                'cutoff_minimum': cutoff_minimum_choice,
                'point_method': point_method_choice
                }
    
    def get_params(self):
        """Return dict of current parameters."""
        return {
                'phrase_len': self.phrase_len,
                'comparison': self.comparison,
                'shared': self.shared,
                'distance_metric' : self.distance_metric,
                'max_motifs': self.max_motifs,
                'recency_weighting': self.recency_weighting,
                'cutoff_threshold': self.cutoff_threshold,
                'cutoff_minimum': self.cutoff_minimum,
                'point_method': self.point_method
                }


"""
model = MotifSimulation()
model = model.fit(df_wide.fillna(0)[df_wide.columns[0:5]].astype(float))
prediction = model.predict(forecast_length = 14)
prediction.forecast
"""