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
        **kwargs,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.name = 'DatepartRegressionTransformer'
        self.regression_model = regression_model
        self.datepart_method = datepart_method

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied.

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")

        y = df.values
        from autots.models.sklearn import date_part

        X = date_part(df.index, method=self.datepart_method)
        from autots.models.sklearn import retrieve_regressor

        self.model = retrieve_regressor(
            regression_model=self.regression_model,
            verbose=0,
            verbose_bool=False,
            random_seed=2020,
        )
        self.model = self.model.fit(X, y)
        self.shape = df.shape
        return self

    def fit_transform(self, df):
        """Fit and Return Detrended DataFrame.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        self.fit(df)
        return self.transform(df)

    def transform(self, df):
        """Return detrended data.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        from autots.models.sklearn import date_part

        X = date_part(df.index, method=self.datepart_method)
        y = pd.DataFrame(self.model.predict(X))
        y.columns = df.columns
        y.index = df.index
        df = df - y
        return df

    def inverse_transform(self, df):
        """Return data to original form.

        Args:
            df (pandas.DataFrame): input dataframe
        """
        try:
            df = df.astype(float)
        except Exception:
            raise ValueError("Data Cannot Be Converted to Numeric Float")
        from autots.models.sklearn import date_part

        X = date_part(df.index, method=self.datepart_method)
        y = pd.DataFrame(self.model.predict(X))
        y.columns = df.columns
        y.index = df.index
        df = df + y
        return df