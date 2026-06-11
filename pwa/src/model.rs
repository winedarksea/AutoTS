//! Plain data structures and (de)serialization for chart/table/download.
//! Mirrors the JSON shapes the Python tools return (see llms.txt).

use serde_json::Value;

#[derive(Clone)]
pub struct SeriesData {
    pub name: String,
    pub values: Vec<f64>,
}

/// Wide-format data: a shared datetime axis plus one column per series.
#[derive(Clone, Default)]
pub struct WideData {
    pub datetime: Vec<String>,
    pub series: Vec<SeriesData>,
}

impl WideData {
    /// Parse the `json_wide` shape: {"datetime":[...], "<series>":[...], ...}.
    pub fn from_json_wide(v: &Value) -> Option<WideData> {
        let obj = v.as_object()?;
        let datetime = obj
            .get("datetime")?
            .as_array()?
            .iter()
            .map(|x| x.as_str().unwrap_or("").to_string())
            .collect();

        let mut series = Vec::new();
        for (key, val) in obj {
            if key == "datetime" {
                continue;
            }
            if let Some(arr) = val.as_array() {
                let values = arr.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect();
                series.push(SeriesData {
                    name: key.clone(),
                    values,
                });
            }
        }
        series.sort_by(|a, b| a.name.cmp(&b.name));
        Some(WideData { datetime, series })
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.series.iter().position(|s| s.name == name)
    }

    /// Select every series that has at least one real value, skipping any series
    /// that is entirely missing/NaN. NaN values within an otherwise-populated
    /// series are normal (AutoTS handles them during forecasting) and must not
    /// cause that series to be hidden. If every series is entirely missing, fall
    /// back to selecting the first one so the chart isn't left empty.
    pub fn initial_series_selection(&self) -> Vec<bool> {
        let selection: Vec<bool> = self
            .series
            .iter()
            .map(|series| series.values.iter().any(|value| value.is_finite()))
            .collect();
        if selection.iter().any(|&selected| selected) {
            selection
        } else {
            (0..self.series.len()).map(|index| index == 0).collect()
        }
    }
}

/// RFC4180: wrap a field in double-quotes if it contains comma, quote, or newline.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn wide_data_to_csv_with_overrides(
    data: &WideData,
    overrides: Option<&[Vec<Option<f64>>]>,
) -> String {
    let mut out = String::from("datetime");
    for s in &data.series {
        out.push(',');
        out.push_str(&csv_field(&s.name));
    }
    out.push('\n');

    for (r, dt) in data.datetime.iter().enumerate() {
        out.push_str(&csv_field(dt));
        for (c, s) in data.series.iter().enumerate() {
            out.push(',');
            let v = overrides
                .and_then(|all| all.get(c))
                .and_then(|col| col.get(r).copied().flatten())
                .unwrap_or_else(|| s.values.get(r).copied().unwrap_or(f64::NAN));
            if v.is_finite() {
                out.push_str(&format!("{v}"));
            }
        }
        out.push('\n');
    }
    out
}

/// Build a wide CSV (datetime + one column per series).
pub fn wide_data_to_csv(data: &WideData) -> String {
    wide_data_to_csv_with_overrides(data, None)
}

/// Build a wide forecast CSV, applying per-point overrides where present.
pub fn forecast_to_csv(fc: &WideData, overrides: &[Vec<Option<f64>>]) -> String {
    wide_data_to_csv_with_overrides(fc, Some(overrides))
}

/// Build a long forecast CSV with uncertainty bounds shifted by point adjustments.
pub fn forecast_with_bounds_to_long_csv(
    forecast: &WideData,
    upper: Option<&WideData>,
    lower: Option<&WideData>,
    overrides: &[Vec<Option<f64>>],
) -> String {
    let mut out = String::from("datetime,series,forecast,lower_forecast,upper_forecast\n");
    for (series_index, series) in forecast.series.iter().enumerate() {
        let effective_forecast = effective_values(forecast, overrides, series_index);
        let effective_lower =
            lower.map(|bounds| effective_bound_values(forecast, bounds, overrides, series_index));
        let effective_upper =
            upper.map(|bounds| effective_bound_values(forecast, bounds, overrides, series_index));

        for (row_index, datetime) in forecast.datetime.iter().enumerate() {
            out.push_str(&csv_field(datetime));
            out.push(',');
            out.push_str(&csv_field(&series.name));
            out.push(',');
            push_finite_value(&mut out, effective_forecast.get(row_index).copied());
            out.push(',');
            push_finite_value(
                &mut out,
                effective_lower
                    .as_ref()
                    .and_then(|values| values.get(row_index))
                    .copied(),
            );
            out.push(',');
            push_finite_value(
                &mut out,
                effective_upper
                    .as_ref()
                    .and_then(|values| values.get(row_index))
                    .copied(),
            );
            out.push('\n');
        }
    }
    out
}

fn push_finite_value(output: &mut String, value: Option<f64>) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        output.push_str(&value.to_string());
    }
}

/// Compact JSON for a params field, defaulting a missing/null value to `{}`
/// so the cell is always a valid object string for AutoTS `import_template`.
fn params_json(v: &Value) -> String {
    if v.is_null() {
        "{}".to_string()
    } else {
        serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Build a one-row AutoTS export template CSV from a model's parameters.
///
/// Columns match `auto_ts.template_cols_id`
/// (`ID,Model,ModelParameters,TransformationParameters,Ensemble`), so the file
/// loads back into a regular AutoTS run via `import_template`. The param fields
/// are JSON strings, RFC4180-quoted because they contain commas and quotes.
pub fn model_params_to_template_csv(
    id: &str,
    model_name: &str,
    model_parameters: &Value,
    transformation_parameters: &Value,
) -> String {
    let mut out = String::from("ID,Model,ModelParameters,TransformationParameters,Ensemble\n");
    out.push_str(&csv_field(id));
    out.push(',');
    out.push_str(&csv_field(model_name));
    out.push(',');
    out.push_str(&csv_field(&params_json(model_parameters)));
    out.push(',');
    out.push_str(&csv_field(&params_json(transformation_parameters)));
    out.push_str(",0\n");
    out
}

/// Effective forecast values for one series (override where set, else original).
pub fn effective_values(fc: &WideData, overrides: &[Vec<Option<f64>>], series: usize) -> Vec<f64> {
    let base = &fc.series[series].values;
    base.iter()
        .enumerate()
        .map(|(r, &orig)| {
            overrides
                .get(series)
                .and_then(|col| col.get(r).copied().flatten())
                .unwrap_or(orig)
        })
        .collect()
}

/// Shift a series' model bounds by the same per-point delta as its adjusted forecast.
pub fn effective_bound_values(
    forecast: &WideData,
    bounds: &WideData,
    overrides: &[Vec<Option<f64>>],
    series_index: usize,
) -> Vec<f64> {
    let Some(forecast_series) = forecast.series.get(series_index) else {
        return Vec::new();
    };
    let Some(bound_index) = bounds.index_of(&forecast_series.name) else {
        return Vec::new();
    };
    let bound_values = &bounds.series[bound_index].values;

    forecast_series
        .values
        .iter()
        .enumerate()
        .map(|(row_index, original_forecast)| {
            let adjustment_delta = overrides
                .get(series_index)
                .and_then(|series_overrides| series_overrides.get(row_index))
                .copied()
                .flatten()
                .map(|adjusted_forecast| adjusted_forecast - original_forecast)
                .unwrap_or(0.0);
            bound_values
                .get(row_index)
                .copied()
                .map(|bound| bound + adjustment_delta)
                .unwrap_or(f64::NAN)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        effective_bound_values, forecast_to_csv, forecast_with_bounds_to_long_csv,
        model_params_to_template_csv, wide_data_to_csv, SeriesData, WideData,
    };
    use serde_json::json;

    fn example_data() -> WideData {
        WideData {
            datetime: vec!["2026-01-01".into(), "2026-01-02".into()],
            series: vec![SeriesData {
                name: "sales,total".into(),
                values: vec![1.0, 2.0],
            }],
        }
    }

    #[test]
    fn initial_selection_skips_all_missing_series() {
        let data = WideData {
            datetime: vec!["2026-01-01".into()],
            series: vec![
                SeriesData {
                    name: "missing".into(),
                    values: vec![f64::NAN],
                },
                SeriesData {
                    name: "available".into(),
                    values: vec![4.0],
                },
            ],
        };

        assert_eq!(data.initial_series_selection(), vec![false, true]);
    }

    #[test]
    fn initial_selection_keeps_partially_missing_series() {
        let data = WideData {
            datetime: vec!["2026-01-01".into(), "2026-01-02".into()],
            series: vec![
                SeriesData {
                    name: "a".into(),
                    values: vec![1.0, f64::NAN],
                },
                SeriesData {
                    name: "b".into(),
                    values: vec![f64::NAN, 2.0],
                },
                SeriesData {
                    name: "all_missing".into(),
                    values: vec![f64::NAN, f64::NAN],
                },
            ],
        };

        // Series with at least one real value stay selected; only the
        // entirely-missing series is dropped.
        assert_eq!(
            data.initial_series_selection(),
            vec![true, true, false]
        );
    }

    #[test]
    fn initial_selection_falls_back_to_first_when_all_missing() {
        let data = WideData {
            datetime: vec!["2026-01-01".into()],
            series: vec![
                SeriesData {
                    name: "a".into(),
                    values: vec![f64::NAN],
                },
                SeriesData {
                    name: "b".into(),
                    values: vec![f64::NAN],
                },
            ],
        };

        assert_eq!(data.initial_series_selection(), vec![true, false]);
    }

    #[test]
    fn loaded_data_csv_preserves_wide_shape_and_quotes_headers() {
        assert_eq!(
            wide_data_to_csv(&example_data()),
            "datetime,\"sales,total\"\n2026-01-01,1\n2026-01-02,2\n"
        );
    }

    #[test]
    fn forecast_csv_applies_point_overrides() {
        assert_eq!(
            forecast_to_csv(&example_data(), &[vec![None, Some(3.5)]]),
            "datetime,\"sales,total\"\n2026-01-01,1\n2026-01-02,3.5\n"
        );
    }

    #[test]
    fn long_forecast_csv_includes_shifted_uncertainty_bounds() {
        let forecast = example_data();
        let lower = WideData {
            datetime: forecast.datetime.clone(),
            series: vec![SeriesData {
                name: "sales,total".into(),
                values: vec![0.5, 1.5],
            }],
        };
        let upper = WideData {
            datetime: forecast.datetime.clone(),
            series: vec![SeriesData {
                name: "sales,total".into(),
                values: vec![1.5, 2.5],
            }],
        };

        assert_eq!(
            forecast_with_bounds_to_long_csv(
                &forecast,
                Some(&upper),
                Some(&lower),
                &[vec![None, Some(3.5)]],
            ),
            "datetime,series,forecast,lower_forecast,upper_forecast\n\
             2026-01-01,\"sales,total\",1,0.5,1.5\n\
             2026-01-02,\"sales,total\",3.5,3,4\n"
        );
    }

    #[test]
    fn long_forecast_csv_leaves_missing_bounds_empty() {
        assert_eq!(
            forecast_with_bounds_to_long_csv(&example_data(), None, None, &[vec![None, Some(3.5)]],),
            "datetime,series,forecast,lower_forecast,upper_forecast\n\
             2026-01-01,\"sales,total\",1,,\n\
             2026-01-02,\"sales,total\",3.5,,\n"
        );
    }

    #[test]
    fn effective_bounds_preserve_interval_width_after_adjustment() {
        let forecast = example_data();
        let bounds = WideData {
            datetime: forecast.datetime.clone(),
            series: vec![SeriesData {
                name: "sales,total".into(),
                values: vec![1.5, 2.5],
            }],
        };
        assert_eq!(
            effective_bound_values(&forecast, &bounds, &[vec![None, Some(3.5)]], 0),
            vec![1.5, 4.0]
        );
    }

    #[test]
    fn template_csv_quotes_json_param_fields() {
        let csv =
            model_params_to_template_csv("1", "LastValueNaive", &json!({"window": 10}), &json!({}));
        assert_eq!(
            csv,
            "ID,Model,ModelParameters,TransformationParameters,Ensemble\n\
             1,LastValueNaive,\"{\"\"window\"\":10}\",{},0\n"
        );
    }

    #[test]
    fn template_csv_defaults_null_params_to_empty_object() {
        let csv = model_params_to_template_csv("1", "GLS", &json!(null), &json!(null));
        assert_eq!(
            csv,
            "ID,Model,ModelParameters,TransformationParameters,Ensemble\n1,GLS,{},{},0\n"
        );
    }
}
