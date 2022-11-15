# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:45:01 2021

@author: Colin
"""
import timeit
import platform
from autots import model_forecast, __version__, load_linear


class Benchmark(object):
    def __init__(self):
        """Benchmark.
        Lower is better.
        Results may not be comparable across versions.
        Total runtime includes only standard package model runtimes.
        """
        self.version = __version__
        self.total_runtime = 0
        self.platform = platform.platform()
        self.node = platform.node()
        self.python_version = platform.python_version()
        self.results = {"not": "run (yet)"}
        self.avg_naive_runtime = 0
        self.sect_motif_runtime = 0
        self.nvar_runtime = 0
        self.datepart_trees_runtime = 0
        self.datepart_svm_runtime = 0
        self.theta_runtime = 0
        self.arima_runtime = 0
        self.tensorflow_rnn_runtime = 0
        self.tensorflow_cnn_runtime = 0
        self.gluonts_runtime = 0
        self.multivariate_knn_runtime = 0
        self.prophet_runtime = 0
        self.sklearn_mlp_runtime = 0

    def __repr__(self):
        """Print."""
        return f"Benchmark runtime: {self.total_runtime} see .results for details"

    def run(
        self,
        n_jobs: int = "auto",
        times: int = 3,
        random_seed: int = 123,
        base_models_only=False,
        verbose: int = 0,
    ):
        """Run benchmark.

        Args:
            n_jobs (int): passed to model_forecast for n cpus
            times (int): number of times to run benchmark models (returns avg of n times)
            random_seed (int): random seed, increases consistency
            base_models_only (bool): if True, doesn't attempt Tensorflow, GluonTS, or Prophet models
        """
        small_df = load_linear(
            long=False, shape=(200, 20), introduce_random=2, random_seed=random_seed
        )

        for _ in range(times):
            print("Beginning AverageValueNaive")
            start_time = timeit.default_timer()
            df = load_linear(
                long=False,
                shape=(200, 1000),
                introduce_random=2,
                random_seed=random_seed,
            )
            df_forecast = model_forecast(
                model_name="AverageValueNaive",
                model_param_dict={"method": "Mean"},
                model_transform_dict={
                    "fillna": "mean",
                    "transformations": {"0": "DifferencedTransformer"},
                    "transformation_params": {"0": {}},
                },
                df_train=df,
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.avg_naive_runtime = (
                self.avg_naive_runtime + timeit.default_timer() - start_time
            )

            print("Beginning SectionalMotif")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="SectionalMotif",
                model_param_dict={
                    "window": 15,
                    "point_method": "mean",
                    "distance_metric": "euclidean",
                    "include_differenced": True,
                    "k": 3,
                    "stride_size": 1,
                    "regression_type": None,
                },
                model_transform_dict={
                    "fillna": "pad",
                    "transformations": {"0": "PowerTransformer", "1": "Round"},
                    "transformation_params": {
                        "0": {},
                        "1": {"decimals": 0, "on_transform": False, "on_inverse": True},
                    },
                },
                df_train=df,
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.sect_motif_runtime = (
                self.sect_motif_runtime + timeit.default_timer() - start_time
            )

            print("Beginning NVAR")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="NVAR",
                model_param_dict={
                    "k": 2,
                    "ridge_param": 0.002,
                    "warmup_pts": 1,
                    "seed_pts": 1,
                    "seed_weighted": None,
                    "batch_size": 10,
                    "batch_method": "std_sorted",
                },
                model_transform_dict={
                    "fillna": "quadratic",
                    "transformations": {
                        "0": "Log",
                        "1": "DifferencedTransformer",
                        "2": "RollingMeanTransformer",
                        "3": "Round",
                    },
                    "transformation_params": {
                        "0": {},
                        "1": {},
                        "2": {"fixed": False, "window": 28},
                        "3": {"decimals": 0, "on_transform": False, "on_inverse": True},
                    },
                },
                df_train=df[df.columns[0:600]],
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.nvar_runtime = self.nvar_runtime + timeit.default_timer() - start_time

            print("Beginning Datepart RandomForest")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="DatepartRegression",
                model_param_dict={
                    "regression_model": {
                        "model": "RandomForest",
                        "model_params": {
                            "n_estimators": 1000,
                            "min_samples_leaf": 2,
                            "bootstrap": True,
                        },
                    },
                    "datepart_method": "simple_binarized",
                    "regression_type": None,
                },
                model_transform_dict={
                    "fillna": "ffill",
                    "transformations": {"0": "MaxAbsScaler"},
                    "transformation_params": {"0": {}},
                },
                df_train=df,
                forecast_length=15,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.datepart_trees_runtime = (
                self.datepart_trees_runtime + timeit.default_timer() - start_time
            )

            print("Beginning Datepart SVM")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="DatepartRegression",
                model_param_dict={
                    "regression_model": {"model": "SVM", "model_params": {}},
                    "datepart_method": "simple",
                    "regression_type": None,
                },
                model_transform_dict={
                    "fillna": "median",
                    "transformations": {"0": "PowerTransformer"},
                    "transformation_params": {"0": {}},
                },
                df_train=df,
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.datepart_svm_runtime = (
                self.datepart_svm_runtime + timeit.default_timer() - start_time
            )

            print("Beginning Theta")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="Theta",
                model_param_dict={
                    "deseasonalize": True,
                    "difference": True,
                    "use_test": True,
                    "method": "auto",
                    "period": None,
                    "theta": 1.4,
                    "use_mle": False,
                },
                model_transform_dict={
                    "fillna": "quadratic",
                    "transformations": {
                        "0": "Log",
                        "1": "Slice",
                        "2": "RollingMeanTransformer",
                    },
                    "transformation_params": {
                        "0": {},
                        "1": {"method": 0.9},
                        "2": {"fixed": False, "window": 28},
                    },
                },
                df_train=df[df.columns[0:150]],
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.theta_runtime = (
                self.theta_runtime + timeit.default_timer() - start_time
            )

            print("Beginning ARIMA")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="ARIMA",
                model_param_dict={'p': 7, 'd': 1, 'q': 1, 'regression_type': None},
                model_transform_dict={
                    "fillna": "median",
                    "transformations": {
                        "0": "Detrend",
                    },
                    "transformation_params": {
                        "0": {'model': 'Linear'},
                    },
                },
                df_train=df[df.columns[0:60]],
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.arima_runtime = (
                self.arima_runtime + timeit.default_timer() - start_time
            )

            print("Beginning Multivariate KNN")
            start_time = timeit.default_timer()
            df_forecast = model_forecast(
                model_name="MultivariateRegression",
                model_param_dict={
                    "regression_model": {
                        "model": "KNN",
                        "model_params": {"n_neighbors": 5, "weights": "uniform"},
                    },
                    "mean_rolling_periods": 30,
                    "macd_periods": None,
                    "std_rolling_periods": 7,
                    "max_rolling_periods": 60,
                    "min_rolling_periods": 60,
                    "quantile90_rolling_periods": 5,
                    "quantile10_rolling_periods": None,
                    "ewm_alpha": None,
                    "ewm_var_alpha": None,
                    "additional_lag_periods": None,
                    "abs_energy": False,
                    "rolling_autocorr_periods": None,
                    "datepart_method": "simple",
                    "polynomial_degree": None,
                    "regression_type": None,
                    "window": 7,
                    "holiday": False,
                    "probabilistic": False,
                },
                model_transform_dict={
                    "fillna": "fake_date",
                    "transformations": {"0": "MaxAbsScaler"},
                    "transformation_params": {"0": {}},
                },
                df_train=df[df.columns[0:100]],
                forecast_length=12,
                frequency="D",
                prediction_interval=0.9,
                random_seed=random_seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            self.multivariate_knn_runtime = (
                self.multivariate_knn_runtime + timeit.default_timer() - start_time
            )
            try:
                print("Beginning MLP")
                start_time = timeit.default_timer()
                df_forecast = model_forecast(
                    model_name="WindowRegression",
                    model_param_dict={
                        "regression_model": {
                            "model": "MLP",
                            "model_params": {
                                'hidden_layer_sizes': (25, 15, 25),
                                'max_iter': 200,
                                'activation': 'tanh',
                                'solver': 'adam',
                                'early_stopping': True,
                                'learning_rate_init': 0.01,
                            },
                        },
                        "window_size": 10,
                        "input_dim": "univariate",
                        "output_dim": "forecast_length",
                        "normalize_window": False,
                        "max_windows": 5000,
                        "regression_type": None,
                    },
                    model_transform_dict={
                        "fillna": "ffill_mean_biased",
                        "transformations": {
                            "0": "MaxAbsScaler",
                        },
                        "transformation_params": {"0": {}},
                    },
                    df_train=df,
                    forecast_length=12,
                    frequency="D",
                    prediction_interval=0.9,
                    random_seed=random_seed,
                    verbose=verbose,
                    n_jobs=n_jobs,
                )
                self.sklearn_mlp_runtime = (
                    self.sklearn_mlp_runtime + timeit.default_timer() - start_time
                )
            except Exception as e:
                print(f"sklearn mlp failed with: {repr(e)}")

            if not base_models_only:
                try:
                    print("Beginning KerasRNN")
                    start_time = timeit.default_timer()
                    df_forecast = model_forecast(
                        model_name="WindowRegression",
                        model_param_dict={
                            "regression_model": {
                                "model": "KerasRNN",
                                "model_params": {
                                    "kernel_initializer": "lecun_uniform",
                                    "epochs": 50,
                                    "batch_size": 8,
                                    "optimizer": "rmsprop",
                                    "loss": "Huber",
                                    "hidden_layer_sizes": [32, 64, 32],
                                    "rnn_type": "LSTM",
                                    "shape": 1,
                                },
                            },
                            "window_size": 10,
                            "input_dim": "univariate",
                            "output_dim": "forecast_length",
                            "normalize_window": False,
                            "max_windows": 5000,
                            "regression_type": None,
                        },
                        model_transform_dict={
                            "fillna": "ffill_mean_biased",
                            "transformations": {
                                "0": "MaxAbsScaler",
                                "1": "DifferencedTransformer",
                            },
                            "transformation_params": {"0": {}, "1": {}},
                        },
                        df_train=df,
                        forecast_length=12,
                        frequency="D",
                        prediction_interval=0.9,
                        random_seed=random_seed,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    self.tensorflow_rnn_runtime = (
                        self.tensorflow_rnn_runtime
                        + timeit.default_timer()
                        - start_time
                    )
                except Exception as e:
                    print(f"tensorflow failed with: {repr(e)}")

                try:
                    print("Beginning KerasCNN")
                    start_time = timeit.default_timer()
                    df_forecast = model_forecast(
                        model_name="WindowRegression",
                        model_param_dict={
                            "regression_model": {
                                "model": "KerasRNN",
                                "model_params": {
                                    "kernel_initializer": "glorot_normal",
                                    "epochs": 50,
                                    "batch_size": 8,
                                    "optimizer": "adam",
                                    "loss": "mae",
                                    "hidden_layer_sizes": [32, 32, 32],
                                    "rnn_type": "CNN",
                                    "shape": 1,
                                },
                            },
                            "window_size": 10,
                            "input_dim": "univariate",
                            "output_dim": "forecast_length",
                            "normalize_window": False,
                            "max_windows": 5000,
                            "regression_type": None,
                        },
                        model_transform_dict={
                            "fillna": "median",
                            "transformations": {
                                "0": "MaxAbsScaler",
                            },
                            "transformation_params": {"0": {}},
                        },
                        df_train=df,
                        forecast_length=12,
                        frequency="D",
                        prediction_interval=0.9,
                        random_seed=random_seed,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    self.tensorflow_cnn_runtime = (
                        self.tensorflow_cnn_runtime
                        + timeit.default_timer()
                        - start_time
                    )
                except Exception as e:
                    print(f"tensorflow CNN failed with: {repr(e)}")

                try:
                    print("Beginning GluonTS")
                    start_time = timeit.default_timer()
                    df_forecast = model_forecast(
                        model_name="GluonTS",
                        model_param_dict={
                            "gluon_model": "SFF",
                            "epochs": 40,
                            "learning_rate": 0.01,
                            "context_length": 10,
                            "regression_type": None,
                        },
                        model_transform_dict={
                            "fillna": "KNNImputer",
                            "transformations": {"0": "QuantileTransformer"},
                            "transformation_params": {
                                "0": {
                                    "output_distribution": "uniform",
                                    "n_quantiles": 100,
                                }
                            },
                        },
                        df_train=df,
                        forecast_length=12,
                        frequency="D",
                        prediction_interval=0.9,
                        random_seed=random_seed,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    self.gluonts_runtime = (
                        self.gluonts_runtime + timeit.default_timer() - start_time
                    )
                except Exception as e:
                    print(f"gluonts failed with: {repr(e)}")

                try:
                    print("Beginning Prophet")
                    start_time = timeit.default_timer()
                    df_forecast = model_forecast(  # noqa
                        model_name="FBProphet",
                        model_param_dict={"holiday": False, "regression_type": None},
                        model_transform_dict={
                            "fillna": "KNNImputer",
                            "transformations": {"0": "QuantileTransformer"},
                            "transformation_params": {
                                "0": {
                                    "output_distribution": "uniform",
                                    "n_quantiles": 100,
                                }
                            },
                        },
                        df_train=small_df,
                        forecast_length=12,
                        frequency="D",
                        prediction_interval=0.9,
                        random_seed=random_seed,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                    self.prophet_runtime = (
                        self.prophet_runtime + timeit.default_timer() - start_time
                    )
                except Exception as e:
                    print(f"prophet failed with: {repr(e)}")

        self.total_runtime = (
            self.avg_naive_runtime
            + self.sect_motif_runtime
            + self.nvar_runtime
            + self.datepart_trees_runtime
            + self.datepart_svm_runtime
            + self.multivariate_knn_runtime
            + self.theta_runtime
            + self.sklearn_mlp_runtime
            + self.arima_runtime
        ) / times
        self.results = {
            "version": self.version,
            "platform": self.platform,
            "node": self.node,
            "python_version": self.python_version,
            "n_jobs": n_jobs,
            "times": times,
            "avg_naive_runtime": self.avg_naive_runtime / times,
            "sect_motif_runtime": self.sect_motif_runtime / times,
            "nvar_runtime": self.nvar_runtime / times,
            "datepart_trees_runtime": self.datepart_trees_runtime / times,
            "datepart_svm_runtime": self.datepart_svm_runtime / times,
            "multivariate_knn_runtime": self.multivariate_knn_runtime / times,
            "theta_runtime": self.theta_runtime / times,
            "arima_runtime": self.arima_runtime / times,
            "sklearn_mlp_runtime": self.sklearn_mlp_runtime / times,
            "total_runtime": self.total_runtime,
            "tensorflow_rnn_runtime": self.tensorflow_rnn_runtime / times,
            "tensorflow_cnn_runtime": self.tensorflow_cnn_runtime / times,
            "gluonts_runtime": self.gluonts_runtime / times,
            "prophet_runtime": self.prophet_runtime / times,
        }
        return self


if __name__ == "__main__":
    import json
    import sys

    try:
        n_jobs = sys.argv[1]
        if str(n_jobs).isdigit():
            n_jobs = int(n_jobs)
    except Exception:
        n_jobs = "auto"
    try:
        times = int(sys.argv[2])
    except Exception:
        times = 3
    bench = Benchmark()
    bench = bench.run(n_jobs=n_jobs, times=times)
    print(json.dumps(bench.results, indent=1))
