import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject, PredictionObject, seasonal_int
from autots.tools.probabilistic import Point_to_Probability

class DatepartRegression(ModelObject):
    """Regression not on series but datetime

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
    """

    def __init__(
        self,
        name: str = "DatepartRegression",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        holiday_country: str = 'US',
        random_seed: int = 2020,
        verbose: int = 0,
        forecast_length: int = 1,
        n_jobs: int = None,
        regression_model: dict = {
            "model": 'DecisionTree',
            "model_params": {"max_depth": 5, "min_samples_split": 2},
        },
        datepart_method: str = 'expanded',
        regression_type: str = None,
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            regression_type=regression_type,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.name = 'DatepartRegression'
        self.regression_model = regression_model
        self.datepart_method = datepart_method

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        # if external regressor, do some check up
        if self.regression_type is not None:
            if (np.array(future_regressor).shape[0]) != (df.shape[0]):
                self.regression_type = None

        y = df.values

        X = date_part(df.index, method=self.datepart_method)
        if self.regression_type == 'User':
            X = pd.concat(
                [X, future_regressor], axis=0
            )

        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=0,
            verbose_bool=False,
            random_seed=2020,
        )
        self.df_train = df
        self.model = self.model.fit(X, y)
        self.shape = df.shape
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
        """Generate forecast data immediately following dates of index supplied to .fit().

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        predictStartTime = datetime.datetime.now()
        index = self.create_forecast_index(forecast_length=forecast_length)
        X = date_part(index, method=self.datepart_method)
        if self.regression_type == 'User':
            X = pd.concat(
                [X, future_regressor], axis=0
            )

        forecast = pd.DataFrame(self.model.predict(X.values))

        forecast.columns = self.column_names
        forecast.index = index

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train,
                forecast,
                method='inferred_normal',
                prediction_interval=self.prediction_interval,
            )

            predict_runtime = datetime.datetime.now() - predictStartTime
            prediction = PredictionObject(
                model_name=self.name,
                forecast_length=forecast_length,
                forecast_index=forecast.index,
                forecast_columns=forecast.columns,
                lower_forecast=lower_forecast,
                forecast=forecast,
                upper_forecast=upper_forecast,
                prediction_interval=self.prediction_interval,
                predict_runtime=predict_runtime,
                fit_runtime=self.fit_runtime,
                model_parameters=self.get_params(),
            )
            return prediction

    def get_new_params(self, method: str = 'random'):
        """Return dict of new parameters for parameter tuning."""
        model_choice = generate_regressor_params()
        datepart_choice = np.random.choice(
            a=["recurring", "simple", "expanded"], size=1, p=[0.2, 0.2, 0.2, 0.2, 0.2]
        ).item()
        regression_choice = np.random.choice(
            a=[None, 'User'], size=1, p=[0.7, 0.3]
        ).item()
        parameter_dict = {
            'regression_model': model_choice,
            'datepart_method': datepart_choice,
            'regression_type': regression_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters."""
        parameter_dict = {
            'regression_model': self.regression_model,
            'datepart_method': self.datepart_method,
            'regression_type': self.regression_type,
        }
        return parameter_dict
