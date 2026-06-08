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

/// Build a wide CSV (datetime + one column per series), applying per-point
/// overrides from drag/slider adjustments where present.
pub fn forecast_to_csv(fc: &WideData, overrides: &[Vec<Option<f64>>]) -> String {
    let mut out = String::from("datetime");
    for s in &fc.series {
        out.push(',');
        out.push_str(&csv_field(&s.name));
    }
    out.push('\n');

    for (r, dt) in fc.datetime.iter().enumerate() {
        out.push_str(&csv_field(dt));
        for (c, s) in fc.series.iter().enumerate() {
            out.push(',');
            let v = overrides
                .get(c)
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
