import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import sts
except Exception:  # except ImportError
    _has_tf = False
else:
    _has_tf = True

class TensorflowSTS(ModelObject):
    """STS from TensorflowProbability.

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast

        regression_type (str): type of regression (None, 'User', or 'Holiday')
    """

    def __init__(self, name: str = "TensorflowSTS", frequency: str = 'infer',
                 prediction_interval: float = 0.9,
                 regression_type: str = None, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 0,
                 trend: str = 'local',
                 seasonal_periods: int = None,
                 ar_order: int = None, fit_method: str = 'hmc',
                 num_steps: int = 200
                 ):
        ModelObject.__init__(self, name, frequency, prediction_interval,
                             regression_type=regression_type,
                             holiday_country=holiday_country,
                             random_seed=random_seed,
                             verbose=verbose)
        self.seasonal_periods = seasonal_periods
        self.ar_order = ar_order
        self.trend = trend
        self.fit_method = fit_method
        self.num_steps = num_steps

    def _build_model(self, observed_time_series, trend = 'local',
                    seasonal_periods = 12, ar_order = 1):
        if trend == 'semilocal':
            trend = sts.SemiLocalLinearTrend(observed_time_series=observed_time_series,
                                         name='trend')
        else:
            trend = sts.LocalLinearTrend(observed_time_series=observed_time_series,
                                         name='trend')
        mods_list = [trend]
        if str(seasonal_periods).isdigit():
            seasonal = tfp.sts.SmoothSeasonal(
                period=seasonal_periods,
                observed_time_series=observed_time_series,
                name='seasonal',
                frequency_multipliers=[1, 2, 3])
            mods_list.append(seasonal)
        if str(ar_order).isdigit():
            autoregressive = sts.Autoregressive(
                order=int(ar_order),
                observed_time_series=observed_time_series,
                name='autoregressive')
            mods_list.append(autoregressive)
        model = sts.Sum(mods_list,
                        observed_time_series=observed_time_series)
        return model
        

    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        df = self.basic_profile(df)

        demand2d = df.transpose().values
        # models expect shape: [..., num_timesteps] or [..., num_timesteps, 1]
        demand_model = self._build_model(demand2d, 
                                         trend=self.trend,
                                         seasonal_periods=self.seasonal_periods,
                                         ar_order=self.ar_order)
        
        if self.fit_method == 'variational':
            # Build the variational surrogate posteriors `qs`.
            variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
                model=demand_model)
            q_samples = variational_posteriors.sample(int(self.num_steps))      
        
        else:
            self.fit_method = 'hmc'
            q_samples, kernel_results = tfp.sts.fit_with_hmc(demand_model, demand2d,
                                                           num_results=self.num_steps,
                                                           seed=self.random_seed
                                                           )
        self.q_samples = q_samples
        self.demand2d = demand2d
        self.demand_model = demand_model

        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int,
                preord_regressor = [], just_point_forecast = False):
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
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        forecast_dist = tfp.sts.forecast(
            model=self.demand_model,
            observed_time_series=self.demand2d,
            parameter_samples=self.q_samples,
            num_steps_forecast=forecast_length,
            include_observation_noise=True)

        forecast = forecast_dist.mean().numpy()[..., 0]

        forecast = pd.DataFrame(forecast,
                                index=self.column_names,
                                columns=test_index).transpose()

        if just_point_forecast:
            return forecast
        else:
            prediction_interval = self.prediction_interval
            # assume follows rules of normal because those are conventional
            from scipy.stats import norm
            adj = norm.sf(abs(prediction_interval))*2
            p_int = 1 - ((1 - prediction_interval) / 2)
            adj = norm.ppf(p_int)
            forecast_scale = forecast_dist.stddev().numpy()[..., 0] 
            upper_forecast = forecast.transpose().values + (forecast_scale * adj)
            lower_forecast = forecast.transpose().values - (forecast_scale * adj)
            lower_forecast = pd.DataFrame(lower_forecast,
                                          index=self.column_names,
                                          columns=test_index).transpose()
            upper_forecast = pd.DataFrame(upper_forecast,
                                          index=self.column_names,
                                          columns=test_index).transpose()
            # alternatively this followed by quantile
            # forecast_samples = self.forecast_dist.sample(10)[..., 0]
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=test_index,
                                          forecast_columns=forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=forecast,
                                          upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())

            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        fit_method_choice = np.random.choice(['hmc', 'variational'],
                                             size=1).item()
        num_steps_choice = np.random.choice([50, 200], size=1).item()
        ar_order_choice = np.random.choice([None, 1, 2, 7], size=1,
                                           p=[0.5, 0.3, 0.1, 0.1]).item()
        seasonal_periods_choice = np.random.choice([None, seasonal_int()],
                                                   size=1).item()
        trend_choice = np.random.choice(['local', 'semilocal'], size=1).item()

        parameter_dict = {
            'fit_method': fit_method_choice,
            'num_steps': num_steps_choice,
            'ar_order': ar_order_choice,
            'seasonal_periods': seasonal_periods_choice,
            'trend': trend_choice
                        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'fit_method': self.fit_method,
            'num_steps': self.num_steps,
            'ar_order': self.ar_order,
            'seasonal_periods': self.seasonal_periods,
            'trend': self.trend
                        }
        return parameter_dict


class TFPRegression(ModelObject):
    """Tensorflow Probability regression.
    
    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        regression_type (str): type of regression (None, 'User')
    """
    def __init__(self, name: str = "TFPRegression", frequency: str = 'infer', 
                 prediction_interval: float = 0.9, holiday_country: str = 'US',
                 random_seed: int = 2020, verbose: int = 1,
                 regression_type: str = None):
        ModelObject.__init__(self, name, frequency, prediction_interval, 
                             regression_type=regression_type,
                             holiday_country=holiday_country, random_seed=random_seed,
                             verbose=verbose)
        self.family = family
        self.constant = constant
        
    def fit(self, df, preord_regressor = []):
        """Train algorithm given data supplied 
        
        Args:
            df (pandas.DataFrame): Datetime Indexed 

        num_inducing_points = 40
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[1], dtype=x.dtype),
            tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
            tfp.layers.VariationalGaussianProcess(
                num_inducing_points=num_inducing_points,
                kernel_provider=RBFKernelFn(dtype=x.dtype),
                event_shape=[1],
                inducing_index_points_initializer=tf.constant_initializer(
                    np.linspace(*x_range, num=num_inducing_points,
                                dtype=x.dtype)[..., np.newaxis]),
                unconstrained_observation_noise_variance_initializer=(
                    tf.constant_initializer(
                        np.log(np.expm1(1.)).astype(x.dtype))),
            ),
        ])
        funky_loss = lambda y, rv_y: rv_y.variational_loss(
            y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
        yhats = [model(x_tst) for _ in range(100)]
        """
        # Build model.
        model = tfk.Sequential([
          tf.keras.layers.Dense(1 + 1),
          tfp.layers.DistributionLambda(
              lambda t: tfd.Normal(loc=t[..., :1],
                                   scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
        ])
        model = tfk.Sequential([
            tfkl.Dense(tfpl.IndependentPoisson.params_size(1)),
            tfpl.IndependentPoisson(1)
        ])
        negloglik=lambda y, model: -model.log_prob(y)
        # Do inference.
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05),
                      loss=negloglik)
        model.fit(x, y, epochs=100, verbose=False)
        
        # Make predictions.
        yhat = model(x_tst)
        mean = yhat.mean()
        stddev = yhat.stddev()
        mean_plus_2_stddev = mean - 2. * stddev
        mean_minus_2_stddev = mean + 2. * stddev

        df = self.basic_profile(df)
        self.df_train = df
        if self.verbose > 1:
            self.verbose = True
        else:
            self.verbose = False
        if self.regression_type == 'User':
            if ((np.array(preord_regressor).shape[0]) != (df.shape[0])):
                self.regression_type = None
            else:
                self.preord_regressor_train = preord_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(self, forecast_length: int, preord_regressor = [], just_point_forecast = False):
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
        test_index = self.create_forecast_index(forecast_length=forecast_length)
        from statsmodels.api import GLM
        X = (pd.to_numeric(self.df_train.index, errors = 'coerce',downcast='integer').values)
        if self.constant in [True, 'True', 'true']:
            from statsmodels.tools import add_constant
            X = add_constant(X, has_constant='add')
        if self.regression_type == 'User':
            if self.preord_regressor_train.ndim == 1:
                self.preord_regressor_train = np.array(self.preord_regressor_train).reshape(-1, 1)
            X = np.concatenate((X.reshape(-1, 1), self.preord_regressor_train), axis = 1)
        forecast = pd.DataFrame()
        self.df_train = self.df_train.replace(0, np.nan)
        fill_vals = self.df_train.abs().min(axis = 0, skipna = True)
        self.df_train = self.df_train.fillna(fill_vals).fillna(0.1)
        for y in self.df_train.columns:
            current_series = self.df_train[y]
            if str(self.family).lower() == 'poisson':
                from statsmodels.genmod.families.family import Poisson
                model = GLM(current_series.values, X, family= Poisson(), missing = 'drop').fit(disp = self.verbose)
            else:
                self.family = 'Gaussian'
                model = GLM(current_series.values, X, missing = 'drop').fit()
            Xf = pd.to_numeric(test_index, errors = 'coerce',downcast='integer').values
            if self.constant or self.constant == 'True':
                Xf = add_constant(Xf, has_constant='add')
            if self.regression_type == 'User':
                if preord_regressor.ndim == 1:
                    preord_regressor = np.array(preord_regressor).reshape(-1, 1)
                Xf = np.concatenate((Xf.reshape(-1, 1), preord_regressor), axis = 1)   
            current_forecast = model.predict((Xf))
            forecast = pd.concat([forecast, pd.Series(current_forecast)], axis = 1)
        df_forecast = pd.DataFrame(forecast)
        df_forecast.columns = self.column_names
        df_forecast.index = test_index
        if just_point_forecast:
            return df_forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(self.df_train, df_forecast, prediction_interval = self.prediction_interval)
            
            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(model_name=self.name,
                                          forecast_length=forecast_length,
                                          forecast_index=df_forecast.index,
                                          forecast_columns=df_forecast.columns,
                                          lower_forecast=lower_forecast,
                                          forecast=df_forecast, upper_forecast=upper_forecast,
                                          prediction_interval=self.prediction_interval,
                                          predict_runtime=predict_runtime,
                                          fit_runtime=self.fit_runtime,
                                          model_parameters=self.get_params())
            
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        family_choice = np.random.choice(
            a=['Gaussian', 'Poisson', 'Binomial',
               'NegativeBinomial', 'Tweedie', 'Gamma'], size=1,
            p=[0.1, 0.3, 0.1, 0.3, 0.1, 0.1]).item()
        constant_choice = np.random.choice(a=[False, True], size=1,
                                           p=[0.95, 0.05]).item()
        regression_type_choice = np.random.choice(a=[None, 'User'], size=1,
                                                  p=[0.8, 0.2]).item()
        return {'family': family_choice,
                'constant': constant_choice,
                'regression_type': regression_type_choice
                }

    def get_params(self):
        """Return dict of current parameters."""
        return {
                'family': self.family,
                'constant': self.constant,
                'regression_type': self.regression_type
                }

"""
test = TensorflowSTS().fit(df).predict(14)
test.forecast
"""
