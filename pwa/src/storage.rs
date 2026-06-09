//! Rust bindings and JSON contracts for durable browser artifacts.

use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

pub const SCHEMA_VERSION: u64 = 1;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = autotsArtifactStore, js_name = putArtifact, catch)]
    async fn put_artifact_js(
        artifact_json: &str,
        protected_ids_json: &str,
    ) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsArtifactStore, js_name = listArtifacts, catch)]
    async fn list_artifacts_js() -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsArtifactStore, js_name = getArtifact, catch)]
    async fn get_artifact_js(id: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsArtifactStore, js_name = deleteArtifact, catch)]
    async fn delete_artifact_js(id: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsArtifactStore, js_name = storageSummary, catch)]
    async fn storage_summary_js() -> Result<JsValue, JsValue>;

    #[wasm_bindgen(
        js_namespace = autotsArtifactStore,
        js_name = requestPersistentStorage,
        catch
    )]
    async fn request_persistent_storage_js() -> Result<JsValue, JsValue>;
}

fn js_error(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

fn parse_json_result(value: JsValue) -> Result<Value, String> {
    let text = value
        .as_string()
        .ok_or_else(|| "Browser storage returned a non-string result".to_string())?;
    serde_json::from_str(&text).map_err(|error| error.to_string())
}

pub async fn put_artifact(artifact: Value, protected_ids: &[String]) -> Result<Value, String> {
    let protected_json = serde_json::to_string(protected_ids).map_err(|error| error.to_string())?;
    put_artifact_js(&artifact.to_string(), &protected_json)
        .await
        .map_err(js_error)
        .and_then(parse_json_result)
}

pub async fn list_artifacts() -> Result<Value, String> {
    list_artifacts_js()
        .await
        .map_err(js_error)
        .and_then(parse_json_result)
}

pub async fn get_artifact(id: &str) -> Result<Value, String> {
    get_artifact_js(id)
        .await
        .map_err(js_error)
        .and_then(parse_json_result)
}

pub async fn delete_artifact(id: &str) -> Result<Value, String> {
    delete_artifact_js(id)
        .await
        .map_err(js_error)
        .and_then(parse_json_result)
}

pub async fn storage_summary() -> Result<Value, String> {
    storage_summary_js()
        .await
        .map_err(js_error)
        .and_then(parse_json_result)
}

pub async fn request_persistent_storage() -> Result<bool, String> {
    request_persistent_storage_js()
        .await
        .map(|value| value.as_bool().unwrap_or(false))
        .map_err(js_error)
}

pub fn dataset_artifact(data: Value, metadata: Value, report: Option<Value>) -> Value {
    json!({
        "kind": "dataset",
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "data": {
            "wide": data,
            "report": report,
            "features": Value::Null,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn forecast_artifact(
    parent_id: &str,
    command: &str,
    forecast_length: i64,
    forecast: Value,
    upper: Option<Value>,
    lower: Option<Value>,
    model_parameters: Value,
    overrides: Value,
) -> Value {
    json!({
        "kind": "forecast",
        "schema_version": SCHEMA_VERSION,
        "parent_id": parent_id,
        "metadata": {
            "source": command,
            "forecast_length": forecast_length,
        },
        "data": {
            "forecast": forecast,
            "upper": upper,
            "lower": lower,
            "model_parameters": model_parameters,
            "overrides": overrides,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{dataset_artifact, forecast_artifact};
    use serde_json::json;

    #[test]
    fn forecast_snapshot_keeps_parent_and_adjustable_outputs() {
        let artifact = forecast_artifact(
            "dataset-1",
            "search_forecast",
            12,
            json!({"datetime": ["2026-01-01"], "sales": [4.0]}),
            None,
            None,
            json!({"model_name": "GLS"}),
            json!([[null]]),
        );
        assert_eq!(artifact["parent_id"], "dataset-1");
        assert_eq!(artifact["data"]["forecast"]["sales"][0], 4.0);
    }

    #[test]
    fn dataset_snapshot_reserves_feature_enrichment() {
        let artifact = dataset_artifact(
            json!({"datetime": ["2026-01-01"], "sales": [4.0]}),
            json!({"source": "test"}),
            None,
        );
        assert!(artifact["data"]["features"].is_null());
    }
}
