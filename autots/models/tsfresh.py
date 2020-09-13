"""
tsfresh - automated feature extraction

n_jobs>1 causes Windows issues, sometimes maybe
"""

import datetime
import numpy as np
import pandas as pd
from autots.evaluator.auto_model import ModelObject
from autots.evaluator.auto_model import PredictionObject
from autots.tools.probabilistic import Point_to_Probability

try:
    from tsfresh.feature_extraction import (
        extract_features,
        EfficientFCParameters,
        MinimalFCParameters,
    )
except Exception:  # except ImportError
    _has_tsfresh = False
else:
    _has_tsfresh = True


class TSFreshRegressor(ModelObject):
    """Sklearn + TSFresh feature generation

    Args:
        name (str): String to identify class
        frequency (str): String alias of datetime index frequency or else 'infer'
        prediction_interval (float): Confidence interval for probabilistic forecast
        holiday (bool): If true, include holiday flags
        regression_type (str): type of regression (None, 'User')

    """

    def __init__(
        self,
        name: str = "TSFreshRegressor",
        frequency: str = 'infer',
        prediction_interval: float = 0.9,
        regression_type: str = None,
        holiday_country: str = 'US',
        verbose: int = 0,
        random_seed: int = 2020,
        regression_model: str = 'Adaboost',
        max_timeshift: int = 10,
        feature_selection: str = None,
    ):
        ModelObject.__init__(
            self,
            name,
            frequency,
            prediction_interval,
            regression_type=regression_type,
            holiday_country=holiday_country,
            random_seed=random_seed,
            verbose=verbose,
        )
        self.regression_model = regression_model
        self.max_timeshift = max_timeshift
        self.feature_selection = feature_selection

    def fit(self, df, future_regressor=[]):
        """Train algorithm given data supplied

        Args:
            df (pandas.DataFrame): Datetime Indexed
        """
        if not _has_tsfresh:
            raise ImportError("Package tsfresh is required")

        df = self.basic_profile(df)

        self.df_train = df
        self.regressor_train = future_regressor
        self.fit_runtime = datetime.datetime.now() - self.startTime
        return self

    def predict(
        self,
        forecast_length: int,
        future_regressor=[],
        just_point_forecast: bool = False,
    ):
        """Generates forecast data immediately following dates of index supplied to .fit()

        Args:
            forecast_length (int): Number of periods of data to forecast ahead
            regressor (numpy.Array): additional regressor
            just_point_forecast (bool): If True, return a pandas.DataFrame of just point forecasts

        Returns:
            Either a PredictionObject of forecasts and metadata, or
            if just_point_forecast == True, a dataframe of point forecasts
        """
        if not _has_tsfresh:
            raise ImportError("Package tsfresh is required")
        # num_subsamples = 10
        predictStartTime = datetime.datetime.now()

        # from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import make_forecasting_frame

        # from sklearn.ensemble import AdaBoostRegressor
        from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute

        # from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

        max_timeshift = 10
        regression_model = 'Adaboost'
        feature_selection = None

        max_timeshift = self.max_timeshift
        regression_model = self.regression_model
        feature_selection = self.feature_selection

        sktraindata = self.df_train.copy()

        X = pd.DataFrame()
        y = pd.DataFrame()
        counter = 0
        for column in sktraindata.columns:
            df_shift, current_y = make_forecasting_frame(
                sktraindata[column],
                kind="time_series",
                max_timeshift=max_timeshift,
                rolling_direction=1,
            )
            # disable_progressbar = True MinimalFCParameters EfficientFCParameters
            current_X = extract_features(
                df_shift,
                column_id="id",
                column_sort="time",
                column_value="value",
                impute_function=tsfresh_impute,
                show_warnings=False,
                default_fc_parameters=EfficientFCParameters(),
                n_jobs=1,
            )  #
            current_X["feature_last_value"] = current_y.shift(1)
            current_X.rename(columns=lambda x: str(counter) + '_' + x, inplace=True)

            X = pd.concat([X, current_X], axis=1)
            y = pd.concat([y, current_y], axis=1)
            counter += 1

        # drop constant features
        X = X.loc[:, X.apply(pd.Series.nunique) != 1]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        y = y.fillna(method='ffill').fillna(method='bfill')

        if feature_selection == 'Variance':
            from sklearn.feature_selection import VarianceThreshold

            sel = VarianceThreshold(threshold=(0.15))
            X = pd.DataFrame(sel.fit_transform(X))
        if feature_selection == 'Percentile':
            from sklearn.feature_selection import SelectPercentile, chi2

            X = pd.DataFrame(
                SelectPercentile(chi2, percentile=20).fit_transform(X, y[y.columns[0]])
            )
        if feature_selection == 'DecisionTree':
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.feature_selection import SelectFromModel

            clf = DecisionTreeRegressor()
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)

            X = model.transform(X)
        if feature_selection == 'Lasso':
            from sklearn.linear_model import MultiTaskLasso
            from sklearn.feature_selection import SelectFromModel

            clf = MultiTaskLasso(max_iter=2000)
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)

            X = model.transform(X)

        """
         decisionTreeList = X.columns[model.get_support()]
         LassoList = X.columns[model.get_support()]
         
         feature_list = decisionTreeList.to_list()
         set([x for x in feature_list if feature_list.count(x) > 1])
         from collections import Counter
         repeat_features = Counter(feature_list)
         repeat_features = repeat_features.most_common(20)
        """

        # Drop first line
        X = X.iloc[
            1:,
        ]
        y = y.iloc[1:]

        y = y.fillna(method='ffill').fillna(method='bfill')

        index = self.create_forecast_index(forecast_length=forecast_length)

        if regression_model == 'ElasticNet':
            from sklearn.linear_model import MultiTaskElasticNet

            regr = MultiTaskElasticNet(alpha=1.0, random_state=self.random_seed)
        elif regression_model == 'DecisionTree':
            from sklearn.tree import DecisionTreeRegressor

            regr = DecisionTreeRegressor(random_state=self.random_seed)
        elif regression_model == 'MLP':
            from sklearn.neural_network import MLPRegressor

            # relu/tanh lbfgs/adam layer_sizes (100) (10)
            regr = MLPRegressor(
                hidden_layer_sizes=(10, 25, 10),
                verbose=self.verbose_bool,
                max_iter=200,
                activation='tanh',
                solver='lbfgs',
                random_state=self.random_seed,
            )
        elif regression_model == 'KNN':
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.neighbors import KNeighborsRegressor

            regr = MultiOutputRegressor(
                KNeighborsRegressor(random_state=self.random_seed)
            )
        elif regression_model == 'Adaboost':
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.ensemble import AdaBoostRegressor

            regr = MultiOutputRegressor(
                AdaBoostRegressor(n_estimators=200)
            )  # , random_state=self.random_seed))
        else:
            regression_model = 'RandomForest'
            from sklearn.ensemble import RandomForestRegressor

            regr = RandomForestRegressor(
                random_state=self.random_seed, n_estimators=1000, verbose=self.verbose
            )

        regr.fit(X, y)

        combined_index = self.df_train.index.append(index)
        forecast = pd.DataFrame()
        sktraindata.columns = [x for x in range(len(sktraindata.columns))]

        for x in range(forecast_length):
            x_dat = pd.DataFrame()
            y_dat = pd.DataFrame()
            counter = 0
            for column in sktraindata.columns:
                df_shift, current_y = make_forecasting_frame(
                    sktraindata.tail(max_timeshift)[column],
                    kind="time_series",
                    max_timeshift=max_timeshift,
                    rolling_direction=1,
                )
                # disable_progressbar = True MinimalFCParameters EfficientFCParameters
                current_X = extract_features(
                    df_shift,
                    column_id="id",
                    column_sort="time",
                    column_value="value",
                    impute_function=tsfresh_impute,
                    show_warnings=False,
                    n_jobs=1,
                    default_fc_parameters=EfficientFCParameters(),
                )  # default_fc_parameters=MinimalFCParameters(),
                current_X["feature_last_value"] = current_y.shift(1)

                current_X.rename(columns=lambda x: str(counter) + '_' + x, inplace=True)

                x_dat = pd.concat([x_dat, current_X], axis=1)
                y_dat = pd.concat([y_dat, current_y], axis=1)
                counter += 1

            x_dat = x_dat[X.columns]
            rfPred = pd.DataFrame(regr.predict(x_dat.tail(1).values))

            forecast = pd.concat([forecast, rfPred], axis=0, ignore_index=True)
            sktraindata = pd.concat([sktraindata, rfPred], axis=0, ignore_index=True)
            sktraindata.index = combined_index[: len(sktraindata.index)]

        forecast.columns = self.column_names
        forecast.index = index

        if just_point_forecast:
            return forecast
        else:
            upper_forecast, lower_forecast = Point_to_Probability(
                self.df_train, forecast, prediction_interval=self.prediction_interval
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
        """Returns dict of new parameters for parameter tuning"""
        max_timeshift_choice = np.random.choice(
            a=[5, 10, 20], size=1, p=[0.05, 0.9, 0.05]
        ).item()
        regression_model_choice = np.random.choice(
            a=['RandomForest', 'ElasticNet', 'MLP', 'DecisionTree', 'KNN', 'Adaboost'],
            size=1,
            p=[0.02, 0.01, 0.01, 0.05, 0.01, 0.9],
        ).item()
        feature_selection_choice = None
        parameter_dict = {
            'max_timeshift': max_timeshift_choice,
            'regression_model': regression_model_choice,
            'feature_selection': feature_selection_choice,
        }
        return parameter_dict

    def get_params(self):
        """Return dict of current parameters"""
        parameter_dict = {
            'max_timeshift': self.max_timeshift,
            'regression_model': self.regression_model,
            'feature_selection': self.feature_selection,
        }
        return parameter_dict


"""
model = TSFreshRegressor()
model = model.fit(df_wide[df_wide.columns[0:2]].fillna(method='ffill').fillna(method='bfill').tail(50))
prediction = model.predict(forecast_length = 3)
prediction.forecast
"""
