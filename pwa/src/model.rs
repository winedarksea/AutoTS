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

#[cfg(test)]
mod tests {
    use super::{forecast_to_csv, wide_data_to_csv, SeriesData, WideData};

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
}
