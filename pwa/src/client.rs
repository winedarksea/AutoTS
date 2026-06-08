//! Rust bindings to the language-neutral `autotsClient` JS facade.
//!
//! Everything platform-specific lives behind this seam: the rest of the app
//! only sees `call_tool(command, args) -> Result<Value, String>`. A TypeScript
//! frontend would bind to the exact same facade.

use serde_json::Value;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = autotsClient, js_name = initRuntime, catch)]
    async fn init_runtime_js(wheel_url: &str, pyodide_url: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsClient, js_name = callTool, catch)]
    async fn call_tool_js(command: &str, args_json: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = autotsClient, js_name = setProgressHandler)]
    fn set_progress_handler_js(cb: &JsValue);
}

fn jsval_str(v: JsValue) -> String {
    v.as_string().unwrap_or_else(|| format!("{v:?}"))
}

/// Boot the Pyodide runtime (uses URLs configured in index.html when blank).
pub async fn init_runtime() -> Result<(), String> {
    init_runtime_js("", "")
        .await
        .map(|_| ())
        .map_err(jsval_str)
}

/// Invoke a PWA command (preset or core tool) and parse the JSON result.
pub async fn call_tool(command: &str, args: Value) -> Result<Value, String> {
    let args_json = args.to_string();
    let res = call_tool_js(command, &args_json).await.map_err(jsval_str)?;
    let txt = res
        .as_string()
        .ok_or_else(|| "runtime returned a non-string result".to_string())?;
    let parsed: Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
    if let Some(err) = parsed.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string());
    }
    Ok(parsed)
}

/// Register a single progress handler; messages stream here during long calls.
pub fn set_progress_handler<F: FnMut(String) + 'static>(f: F) {
    let cb = Closure::wrap(Box::new(f) as Box<dyn FnMut(String)>);
    set_progress_handler_js(cb.as_ref().unchecked_ref());
    cb.forget();
}
