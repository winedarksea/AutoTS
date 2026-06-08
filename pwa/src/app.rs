//! AutoTS PWA — Leptos view layer.
//!
//! This layer is intentionally thin: it renders state and calls `client::call_tool`.
//! All forecasting/ingestion logic lives in Python (see autots.mcp.pyodide_api),
//! so this file could be replaced by a TypeScript app against the same facade.

use leptos::*;
use serde_json::{json, Value};
use wasm_bindgen::JsCast;

use crate::client::{call_tool, init_runtime, set_progress_handler};
use crate::model::{effective_values, forecast_to_csv, WideData};
use crate::svg;

type Overrides = Vec<Vec<Option<f64>>>;

// ---------------------------------------------------------------------------
// Async helpers (small, signal-driven; keep view handlers tiny)
// ---------------------------------------------------------------------------

async fn fetch_history(
    data_id: String,
    history: RwSignal<Option<WideData>>,
    sel: RwSignal<usize>,
    error: RwSignal<Option<String>>,
) {
    match call_tool(
        "get_data",
        json!({ "data_id": data_id, "output_format": "json_wide" }),
    )
    .await
    {
        Ok(v) => {
            if let Some(w) = WideData::from_json_wide(&v) {
                sel.set(0);
                history.set(Some(w));
            }
        }
        Err(e) => error.set(Some(e)),
    }
}

async fn fetch_features(data_id: String, features: RwSignal<Option<Value>>) {
    if let Ok(det) = call_tool("detect_features", json!({ "data_id": data_id })).await {
        if let Some(did) = det.get("detector_id").and_then(|x| x.as_str()) {
            if let Ok(f) =
                call_tool("get_detected_features", json!({ "detector_id": did })).await
            {
                features.set(f.get("detection_counts").cloned());
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_forecast(
    command: String,
    data_id: String,
    forecast_length: i64,
    forecast: RwSignal<Option<WideData>>,
    overrides: RwSignal<Overrides>,
    busy: RwSignal<bool>,
    error: RwSignal<Option<String>>,
    status: RwSignal<String>,
) {
    busy.set(true);
    error.set(None);
    status.set("Forecasting…".into());
    let res = call_tool(
        &command,
        json!({ "data_id": data_id, "forecast_length": forecast_length }),
    )
    .await;
    match res {
        Ok(v) => {
            if let Some(pid) = v.get("prediction_id").and_then(|x| x.as_str()) {
                match call_tool(
                    "get_forecast",
                    json!({ "prediction_id": pid, "output": "forecast", "format": "json_wide" }),
                )
                .await
                {
                    Ok(fv) => {
                        if let Some(w) = WideData::from_json_wide(&fv) {
                            let ov = w
                                .series
                                .iter()
                                .map(|s| vec![None; s.values.len()])
                                .collect();
                            overrides.set(ov);
                            forecast.set(Some(w));
                            status.set("Forecast ready — drag the sliders to adjust.".into());
                        }
                    }
                    Err(e) => error.set(Some(e)),
                }
            } else {
                error.set(Some("No prediction returned".into()));
            }
        }
        Err(e) => error.set(Some(e)),
    }
    busy.set(false);
}

fn download_csv(filename: &str, content: &str) -> Result<(), String> {
    let array = js_sys::Array::new();
    array.push(&wasm_bindgen::JsValue::from_str(content));
    let opts = web_sys::BlobPropertyBag::new();
    opts.set_type("text/csv");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&array, &opts)
        .map_err(|e| format!("{e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(|e| format!("{e:?}"))?;
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let a = doc
        .create_element("a")
        .map_err(|e| format!("{e:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "anchor cast failed".to_string())?;
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    web_sys::Url::revoke_object_url(&url).ok();
    Ok(())
}

fn finite_range(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    let span = (hi - lo).max(1.0);
    (lo - span, hi + span) // allow adjustment well beyond the original range
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn App() -> impl IntoView {
    let ready = create_rw_signal(false);
    let status = create_rw_signal(String::from("Booting Python runtime…"));
    let busy = create_rw_signal(false);
    let error = create_rw_signal::<Option<String>>(None);

    let data_id = create_rw_signal::<Option<String>>(None);
    let history = create_rw_signal::<Option<WideData>>(None);
    let report = create_rw_signal::<Option<Value>>(None);
    let features = create_rw_signal::<Option<Value>>(None);
    let forecast = create_rw_signal::<Option<WideData>>(None);
    let overrides = create_rw_signal::<Overrides>(Vec::new());
    let sel = create_rw_signal::<usize>(0);
    let forecast_length = create_rw_signal::<i64>(30);

    // Upload inputs
    let tab = create_rw_signal::<u8>(0); // 0 paste, 1 url, 2 file, 3 sample
    let paste_text = create_rw_signal(String::new());
    let url_text = create_rw_signal(String::new());
    let sample = create_rw_signal(String::from("monthly"));

    // Data manager
    let cache = create_rw_signal::<Option<Value>>(None);

    // Boot the runtime once.
    set_progress_handler(move |m| status.set(m));
    spawn_local(async move {
        match init_runtime().await {
            Ok(()) => {
                ready.set(true);
                status.set("Ready. Load some data to begin.".into());
            }
            Err(e) => {
                error.set(Some(format!("Runtime failed to start: {e}")));
                status.set("Runtime error".into());
            }
        }
    });

    // --- data-loaded continuation: fetch history + features ---------------
    let after_loaded = move |id: String| {
        data_id.set(Some(id.clone()));
        forecast.set(None);
        features.set(None);
        let id2 = id.clone();
        spawn_local(async move {
            fetch_history(id.clone(), history, sel, error).await;
            fetch_features(id2, features).await;
        });
    };

    // --- upload actions ----------------------------------------------------
    let load_sample = move |_| {
        let ds = sample.get();
        busy.set(true);
        error.set(None);
        status.set("Loading sample…".into());
        spawn_local(async move {
            match call_tool("load_sample_data", json!({ "dataset": ds })).await {
                Ok(v) => {
                    if let Some(id) = v.get("data_id").and_then(|x| x.as_str()) {
                        after_loaded(id.to_string());
                    }
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    let load_paste = move |_| {
        let text = paste_text.get();
        if text.trim().is_empty() {
            error.set(Some("Paste some CSV/TSV first".into()));
            return;
        }
        busy.set(true);
        error.set(None);
        status.set("Cleaning & detecting format…".into());
        spawn_local(async move {
            match call_tool("smart_load", json!({ "text": text })).await {
                Ok(v) => {
                    report.set(v.get("report").cloned());
                    if let Some(id) = v.get("data_id").and_then(|x| x.as_str()) {
                        after_loaded(id.to_string());
                    }
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    let load_url = move |_| {
        let u = url_text.get();
        if u.trim().is_empty() {
            error.set(Some("Enter a CSV URL first".into()));
            return;
        }
        busy.set(true);
        error.set(None);
        status.set("Fetching & detecting format…".into());
        spawn_local(async move {
            match call_tool("smart_load", json!({ "url": u })).await {
                Ok(v) => {
                    report.set(v.get("report").cloned());
                    if let Some(id) = v.get("data_id").and_then(|x| x.as_str()) {
                        after_loaded(id.to_string());
                    }
                }
                Err(e) => error.set(Some(e)),
            }
            busy.set(false);
        });
    };

    let on_file = move |ev: leptos::ev::Event| {
        let input: web_sys::HtmlInputElement = event_target(&ev);
        let Some(file) = input.files().and_then(|f| f.item(0)) else {
            return;
        };
        let name = file.name();
        busy.set(true);
        error.set(None);
        status.set("Reading file…".into());
        spawn_local(async move {
            let blob: web_sys::Blob = file.unchecked_into();
            match wasm_bindgen_futures::JsFuture::from(blob.text()).await {
                Ok(txt) => {
                    let text = txt.as_string().unwrap_or_default();
                    status.set("Cleaning & detecting format…".into());
                    match call_tool(
                        "smart_load",
                        json!({ "text": text, "filename": name }),
                    )
                    .await
                    {
                        Ok(v) => {
                            report.set(v.get("report").cloned());
                            if let Some(id) = v.get("data_id").and_then(|x| x.as_str()) {
                                after_loaded(id.to_string());
                            }
                        }
                        Err(e) => error.set(Some(e)),
                    }
                }
                Err(e) => error.set(Some(format!("{e:?}"))),
            }
            busy.set(false);
        });
    };

    // --- forecast actions --------------------------------------------------
    let do_forecast = move |command: &'static str| {
        let Some(id) = data_id.get() else {
            error.set(Some("Load data first".into()));
            return;
        };
        let fl = forecast_length.get();
        spawn_local(run_forecast(
            command.to_string(),
            id,
            fl,
            forecast,
            overrides,
            busy,
            error,
            status,
        ));
    };

    let refresh_cache = move |_| {
        spawn_local(async move {
            if let Ok(v) = call_tool("list_cache", json!({})).await {
                cache.set(Some(v));
            }
        });
    };

    let download = move |_| {
        if let Some(fc) = forecast.get() {
            let csv = forecast_to_csv(&fc, &overrides.get());
            if let Err(e) = download_csv("autots_forecast.csv", &csv) {
                error.set(Some(e));
            }
        }
    };

    // --- derived: combined chart for the selected series -------------------
    let chart_svg = move || {
        let s = sel.get();
        let hist = history.get();
        let fc = forecast.get();
        let ovr = overrides.get();
        let sel_name = hist
            .as_ref()
            .and_then(|h| h.series.get(s))
            .map(|x| x.name.clone());
        let hvals = hist
            .as_ref()
            .and_then(|h| h.series.get(s))
            .map(|x| x.values.clone());
        let fvals = fc.as_ref().map(|f| {
            let idx = sel_name
                .as_ref()
                .and_then(|n| f.index_of(n))
                .unwrap_or_else(|| s.min(f.series.len().saturating_sub(1)));
            effective_values(f, &ovr, idx)
        });
        svg::line_chart(hvals.as_deref(), fvals.as_deref())
    };

    view! {
        <header class="md-appbar">
            <span class="brand">"AutoTS"</span>
            <span class="muted">"forecasting for everyone"</span>
            <span class="spacer"></span>
            <a class="md-btn text" href="/llms.txt" target="_blank">"llms.txt"</a>
        </header>

        <main class="md-container">
            // Status / progress
            <div>
                <div class="md-status">{move || status.get()}</div>
                {move || busy.get().then(|| view! { <div class="md-progress"><div></div></div> })}
                {move || error.get().map(|e| view! { <p class="md-error">{e}</p> })}
            </div>

            // ---- Upload card ----
            <section class="md-card">
                <h2>"1 · Load your data"</h2>
                <div class="md-tabs" role="tablist">
                    <button class:active=move || tab.get() == 0 on:click=move |_| tab.set(0)>"Paste"</button>
                    <button class:active=move || tab.get() == 1 on:click=move |_| tab.set(1)>"URL"</button>
                    <button class:active=move || tab.get() == 2 on:click=move |_| tab.set(2)>"File"</button>
                    <button class:active=move || tab.get() == 3 on:click=move |_| tab.set(3)>"Sample"</button>
                </div>

                <div style="margin-top:16px">
                    {move || match tab.get() {
                        0 => view! {
                            <div class="md-field">
                                <label class="md-label">"Paste CSV or TSV (a date column + value columns)"</label>
                                <textarea
                                    placeholder="datetime,sales\n2024-01-01,100\n2024-01-02,102"
                                    on:input=move |ev| paste_text.set(event_target_value(&ev))
                                ></textarea>
                                <div class="md-btn-row">
                                    <button class="md-btn filled" disabled=move || !ready.get() || busy.get() on:click=load_paste>"Load pasted data"</button>
                                </div>
                            </div>
                        }.into_view(),
                        1 => view! {
                            <div class="md-field">
                                <label class="md-label">"CSV URL (e.g. a published Google Sheet)"</label>
                                <input type="url" placeholder="https://…/export?format=csv"
                                    on:input=move |ev| url_text.set(event_target_value(&ev)) />
                                <div class="md-btn-row">
                                    <button class="md-btn filled" disabled=move || !ready.get() || busy.get() on:click=load_url>"Load from URL"</button>
                                </div>
                            </div>
                        }.into_view(),
                        2 => view! {
                            <div class="md-field">
                                <label class="md-label">"Upload a CSV or Excel file"</label>
                                <input type="file" accept=".csv,.tsv,.txt,.xlsx,.xls"
                                    disabled=move || !ready.get() || busy.get()
                                    on:change=on_file />
                            </div>
                        }.into_view(),
                        _ => view! {
                            <div class="md-field">
                                <label class="md-label">"Or try a built-in sample dataset"</label>
                                <select on:change=move |ev| sample.set(event_target_value(&ev))>
                                    <option value="monthly">"monthly"</option>
                                    <option value="weekly">"weekly"</option>
                                    <option value="daily">"daily"</option>
                                    <option value="hourly">"hourly"</option>
                                    <option value="sine">"sine"</option>
                                </select>
                                <div class="md-btn-row">
                                    <button class="md-btn filled" disabled=move || !ready.get() || busy.get() on:click=load_sample>"Load sample"</button>
                                </div>
                                <p class="muted md-body">"Tip: load a sample, download it, and hand it to an LLM as a template for reformatting your own data."</p>
                            </div>
                        }.into_view(),
                    }}
                </div>

                {move || report.get().map(|r| view! {
                    <details class="md-expander" style="margin-top:12px">
                        <summary>"What we cleaned & detected"</summary>
                        <pre class="md-body">{serde_json::to_string_pretty(&r).unwrap_or_default()}</pre>
                    </details>
                })}
            </section>

            // ---- Data + features card ----
            {move || history.get().map(|h| {
                let names: Vec<String> = h.series.iter().map(|s| s.name.clone()).collect();
                view! {
                    <section class="md-card">
                        <h2>"2 · Your data"</h2>
                        {(names.len() > 1).then(|| {
                            let names = names.clone();
                            view! {
                                <div class="md-field" style="max-width:260px">
                                    <label class="md-label">"Series"</label>
                                    <select on:change=move |ev| {
                                        if let Ok(i) = event_target_value(&ev).parse::<usize>() { sel.set(i); }
                                    }>
                                        {names.into_iter().enumerate().map(|(i, n)| view! {
                                            <option value=i.to_string()>{n}</option>
                                        }).collect_view()}
                                    </select>
                                </div>
                            }
                        })}

                        <div class="md-chart" inner_html=chart_svg></div>

                        {move || features.get().map(|f| view! {
                            <div style="margin-top:8px">
                                <span class="md-label">"Detected features"</span>
                                <div>
                                    {feature_chips(&f, &history.get().and_then(|h| h.series.get(sel.get()).map(|s| s.name.clone())).unwrap_or_default())}
                                </div>
                            </div>
                        })}

                        <details class="md-expander" style="margin-top:12px">
                            <summary>"Data table (accessible / machine-readable)"</summary>
                            {move || history.get().map(|h| data_table(&h, sel.get()))}
                        </details>
                    </section>
                }
            })}

            // ---- Forecast controls ----
            {move || data_id.get().map(|_| view! {
                <section class="md-card">
                    <h2>"3 · Forecast"</h2>
                    <div class="md-btn-row">
                        <button class="md-btn filled" disabled=move || busy.get() on:click=move |_| do_forecast("make_forecast")>"Make forecast"</button>
                        <button class="md-btn tonal" disabled=move || busy.get() on:click=move |_| do_forecast("search_forecast")>"Search for best forecast"</button>
                        <button class="md-btn outlined" disabled=move || busy.get() on:click=move |_| do_forecast("search_all_night")>"Search all night"</button>
                    </div>
                    <details class="md-expander" style="margin-top:12px">
                        <summary>"Advanced options"</summary>
                        <div class="md-field" style="max-width:220px">
                            <label class="md-label">"Forecast length (periods)"</label>
                            <input type="number" min="1" max="365" prop:value=move || forecast_length.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(n) = event_target_value(&ev).parse::<i64>() { forecast_length.set(n.max(1)); }
                                } />
                        </div>
                    </details>
                </section>
            })}

            // ---- Forecast result + adjust + download ----
            {move || forecast.get().map(|_| view! {
                <section class="md-card">
                    <h2>"4 · Review, adjust & download"</h2>
                    <div class="md-chart" inner_html=chart_svg></div>

                    <details class="md-expander" style="margin-top:12px" open=true>
                        <summary>"Adjust forecast points"</summary>
                        {move || adjust_rows(forecast, overrides, history, sel)}
                    </details>

                    <div class="md-btn-row" style="margin-top:12px">
                        <button class="md-btn filled" on:click=download>"Download CSV"</button>
                    </div>
                </section>
            })}

            // ---- Data manager ----
            <section class="md-card">
                <h2>"Cached data & forecasts"</h2>
                <div class="md-btn-row">
                    <button class="md-btn outlined" on:click=refresh_cache>"Refresh"</button>
                </div>
                {move || cache.get().map(|c| view! {
                    <pre class="md-body">{serde_json::to_string_pretty(&c).unwrap_or_default()}</pre>
                })}
            </section>
        </main>
    }
}

// ---------------------------------------------------------------------------
// View helpers
// ---------------------------------------------------------------------------

fn feature_chips(counts: &Value, series_name: &str) -> View {
    let obj = match counts.get(series_name).and_then(|v| v.as_object()) {
        Some(o) => o.clone(),
        None => match counts.as_object().and_then(|m| m.values().next()).and_then(|v| v.as_object()) {
            Some(o) => o.clone(),
            None => return ().into_view(),
        },
    };
    obj.iter()
        .map(|(k, v)| {
            let label = format!("{k}: {v}");
            view! { <span class="md-chip">{label}</span> }
        })
        .collect_view()
}

fn data_table(data: &WideData, sel: usize) -> View {
    let Some(s) = data.series.get(sel) else {
        return ().into_view();
    };
    let name = s.name.clone();
    let rows: Vec<(String, f64)> = data
        .datetime
        .iter()
        .cloned()
        .zip(s.values.iter().copied())
        .collect();
    view! {
        <div class="md-table-wrap">
            <table class="md-table">
                <thead><tr><th>"datetime"</th><th>{name}</th></tr></thead>
                <tbody>
                    {rows.into_iter().map(|(d, v)| view! {
                        <tr><td>{d}</td><td>{format!("{v:.4}")}</td></tr>
                    }).collect_view()}
                </tbody>
            </table>
        </div>
    }
    .into_view()
}

fn adjust_rows(
    forecast: RwSignal<Option<WideData>>,
    overrides: RwSignal<Overrides>,
    history: RwSignal<Option<WideData>>,
    sel: RwSignal<usize>,
) -> View {
    let Some(fc) = forecast.get() else {
        return ().into_view();
    };
    // map selected history series name -> forecast series index
    let sel_name = history
        .get()
        .and_then(|h| h.series.get(sel.get()).map(|x| x.name.clone()));
    let idx = sel_name
        .as_ref()
        .and_then(|n| fc.index_of(n))
        .unwrap_or(0);
    let s = match fc.series.get(idx) {
        Some(s) => s.clone(),
        None => return ().into_view(),
    };
    let (lo, hi) = finite_range(&s.values);
    let step = ((hi - lo) / 200.0).max(1e-6);

    (0..s.values.len())
        .map(|r| {
            let orig = s.values[r];
            let dt = fc.datetime.get(r).cloned().unwrap_or_default();
            view! {
                <div class="md-adjust-row">
                    <span class="md-label">{dt}</span>
                    <input type="range"
                        min=lo.to_string() max=hi.to_string() step=step.to_string()
                        prop:value=move || {
                            overrides.get().get(idx).and_then(|c| c.get(r)).copied().flatten().unwrap_or(orig).to_string()
                        }
                        on:input=move |ev| {
                            let v = event_target_value(&ev).parse::<f64>().unwrap_or(orig);
                            overrides.update(|o| {
                                if let Some(col) = o.get_mut(idx) {
                                    if r < col.len() { col[r] = Some(v); }
                                }
                            });
                        } />
                    <span>{move || format!("{:.3}",
                        overrides.get().get(idx).and_then(|c| c.get(r)).copied().flatten().unwrap_or(orig))}</span>
                </div>
            }
        })
        .collect_view()
}
