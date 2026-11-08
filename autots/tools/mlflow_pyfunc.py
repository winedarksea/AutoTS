"""MLflow models-from-code file for registered AutoTS templates (see mlflow_registry).

Loaded only by MLflow (autots never imports it); the import guard only keeps module
scanners such as sphinx autodoc working where mlflow is not installed.

predict(model_input, params) refits the winning model on the given history and
forecasts; no search runs. model_input is wide history (one column per series) with a
DatetimeIndex, or with a date column (params date_col, default the first column) as
REST/JSON payloads lose the index.

params:
    forecast_length (int): overrides the logged forecast_length.
    prediction_interval (float): overrides the logged prediction_interval.
    date_col (str): date column when model_input has no DatetimeIndex.
    output (str): "long" (default) for long_form_results with upper/lower bounds, or
        "forecast" for the wide point forecast. Both return dates as a "datetime" column.
"""

import json

import pandas as pd

try:
    import mlflow
    from mlflow.pyfunc import PythonModel
except ImportError:
    mlflow = None
    PythonModel = object


class AutoTSPyfunc(PythonModel):
    def load_context(self, context):
        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            self.payload = json.load(f)

    @staticmethod
    def _wide_history(model_input, date_col=None) -> pd.DataFrame:
        df = pd.DataFrame(model_input)
        if isinstance(df.index, pd.DatetimeIndex) and date_col is None:
            return df
        date_col = date_col or df.columns[0]
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
        return df.apply(pd.to_numeric, errors="coerce")

    def predict(self, context, model_input, params=None):
        from autots.tools.mlflow import _build_autots_from_payload

        params = params or {}
        overrides = {
            key: params[key]
            for key in ["forecast_length", "prediction_interval"]
            if params.get(key) is not None
        }
        model = _build_autots_from_payload(
            self.payload, overrides, source="registered template"
        )
        if model.used_regressor_check:
            raise ValueError(
                "The registered best model uses a future_regressor, which the pyfunc "
                "does not accept; load it with mlflow_load_model and call fit_data/predict."
            )
        model.fit_data(self._wide_history(model_input, params.get("date_col")))
        prediction = model.predict()
        if params.get("output", "long") == "forecast":
            return prediction.forecast.rename_axis("datetime").reset_index()
        return prediction.long_form_results().reset_index()


if mlflow is not None:
    mlflow.models.set_model(AutoTSPyfunc())
