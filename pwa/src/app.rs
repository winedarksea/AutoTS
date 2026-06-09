//! AutoTS PWA — Leptos view layer.
//!
//! This layer is intentionally thin: it renders state and calls `client::call_tool`.
//! All forecasting/ingestion logic lives in Python (see autots.mcp.pyodide_api),
//! so this file could be replaced by a TypeScript app against the same facade.

use std::collections::HashMap;

use leptos::*;
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};

use crate::client::{call_tool, init_runtime, set_progress_handler};
use crate::model::{effective_values, forecast_to_csv, wide_data_to_csv, WideData};
use crate::svg;

type Overrides = Vec<Vec<Option<f64>>>;

// ---------------------------------------------------------------------------
// Classical theme-toggle glyphs.
//
// Hand-drawn inline SVGs filled with the same metallic-bronze gradient as the
// corner emblem (the `--metal-bronze-*` tokens, so they track light/dark).
// SUN is a Helios radiate disc (shown to switch *to* light); MOON is a Selene
// crescent + star (shown to switch *to* dark). Decorative — the button carries
// an aria-label — so both are aria-hidden.
// ---------------------------------------------------------------------------

const SUN_SVG: &str = "<svg viewBox=\"0 0 24 24\" class=\"md-toggle-glyph\" aria-hidden=\"true\" \
    xmlns=\"http://www.w3.org/2000/svg\">\
    <defs><linearGradient id=\"bzsun\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\
    <stop offset=\"0\" stop-color=\"var(--metal-bronze-light)\"/>\
    <stop offset=\"0.5\" stop-color=\"var(--metal-bronze-mid)\"/>\
    <stop offset=\"1\" stop-color=\"var(--metal-bronze-dark)\"/></linearGradient></defs>\
    <g stroke=\"url(#bzsun)\" stroke-width=\"1.7\" stroke-linecap=\"round\">\
    <line x1=\"12\" y1=\"1.6\" x2=\"12\" y2=\"4.3\"/>\
    <line x1=\"12\" y1=\"19.7\" x2=\"12\" y2=\"22.4\"/>\
    <line x1=\"1.6\" y1=\"12\" x2=\"4.3\" y2=\"12\"/>\
    <line x1=\"19.7\" y1=\"12\" x2=\"22.4\" y2=\"12\"/>\
    <line x1=\"4.7\" y1=\"4.7\" x2=\"6.6\" y2=\"6.6\"/>\
    <line x1=\"17.4\" y1=\"17.4\" x2=\"19.3\" y2=\"19.3\"/>\
    <line x1=\"19.3\" y1=\"4.7\" x2=\"17.4\" y2=\"6.6\"/>\
    <line x1=\"6.6\" y1=\"17.4\" x2=\"4.7\" y2=\"19.3\"/></g>\
    <circle cx=\"12\" cy=\"12\" r=\"5.1\" fill=\"url(#bzsun)\" \
    stroke=\"var(--metal-bronze-dark)\" stroke-width=\"0.6\"/></svg>";

const MOON_SVG: &str = "<svg viewBox=\"0 0 24 24\" class=\"md-toggle-glyph\" aria-hidden=\"true\" \
    xmlns=\"http://www.w3.org/2000/svg\">\
    <defs><linearGradient id=\"bzmoon\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\
    <stop offset=\"0\" stop-color=\"var(--metal-bronze-light)\"/>\
    <stop offset=\"0.55\" stop-color=\"var(--metal-bronze-mid)\"/>\
    <stop offset=\"1\" stop-color=\"var(--metal-bronze-dark)\"/></linearGradient></defs>\
    <path d=\"M21 14.3A8.6 8.6 0 1 1 10.6 3.1 6.9 6.9 0 0 0 21 14.3Z\" \
    fill=\"url(#bzmoon)\" stroke=\"var(--metal-bronze-dark)\" stroke-width=\"0.6\"/>\
    <path d=\"M17.4 2.6l.62 1.66 1.66.62-1.66.62-.62 1.66-.62-1.66-1.66-.62 1.66-.62z\" \
    fill=\"url(#bzmoon)\"/></svg>";

/// An elegant card heading: a metallic-bronze Cinzel Roman numeral, a thin
/// bronze keyline, then the title. `numeral` may be empty for unnumbered cards.
fn section_header(numeral: &'static str, title: &'static str) -> impl IntoView {
    view! {
        <h2 class="md-section-head">
            {(!numeral.is_empty()).then(|| view! {
                <span class="md-section-num" aria-hidden="true">{numeral}</span>
            })}
            <span class="md-section-title">{title}</span>
        </h2>
    }
}

// ---------------------------------------------------------------------------
// Async helpers (small, signal-driven; keep view handlers tiny)
// ---------------------------------------------------------------------------

async fn fetch_history(
    data_id: String,
    history: RwSignal<Option<WideData>>,
    sel_set: RwSignal<Vec<bool>>,
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
                // Select only the first series by default.
                let n = w.series.len();
                sel_set.set((0..n).map(|i| i == 0).collect());
                history.set(Some(w));
            }
        }
        Err(e) => error.set(Some(e)),
    }
}

async fn fetch_features(
    data_id: String,
    features: RwSignal<Option<Value>>,
    feature_error: RwSignal<Option<String>>,
    detecting_features: RwSignal<bool>,
) {
    detecting_features.set(true);
    feature_error.set(None);
    match call_tool("detect_features", json!({ "data_id": data_id })).await {
        Ok(det) => {
            let Some(did) = det.get("detector_id").and_then(|x| x.as_str()) else {
                feature_error.set(Some("Feature detection returned no detector ID.".into()));
                detecting_features.set(false);
                return;
            };
            match call_tool("get_detected_features", json!({ "detector_id": did })).await {
                Ok(f) => features.set(f.get("detection_counts").cloned()),
                Err(e) => feature_error.set(Some(e)),
            }
        }
        Err(e) => feature_error.set(Some(e)),
    }
    detecting_features.set(false);
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

    // Effective light/dark theme (from a saved choice, else the OS preference).
    let theme = create_rw_signal(crate::theme::initial_theme());

    let data_id = create_rw_signal::<Option<String>>(None);
    let history = create_rw_signal::<Option<WideData>>(None);
    let report = create_rw_signal::<Option<Value>>(None);
    let features = create_rw_signal::<Option<Value>>(None);
    let feature_error = create_rw_signal::<Option<String>>(None);
    let detecting_features = create_rw_signal(false);
    let forecast = create_rw_signal::<Option<WideData>>(None);
    let overrides = create_rw_signal::<Overrides>(Vec::new());
    // sel_set: one bool per series in the loaded dataset (true = visible in chart).
    // Sized and reset by fetch_history when data loads.
    let sel_set = create_rw_signal::<Vec<bool>>(vec![true]);
    // adj_sel: which single series index to show in the adjust-forecast sliders.
    let adj_sel = create_rw_signal::<usize>(0);
    let forecast_length = create_rw_signal::<i64>(30);
    let forecast_history_points = create_rw_signal::<usize>(90);
    let confirm_loaded_data_delete = create_rw_signal(false);
    let deleting_loaded_data = create_rw_signal(false);

    // Upload inputs
    let tab = create_rw_signal::<u8>(0); // 0 paste, 1 url, 2 file, 3 sample, 4 live
    let paste_text = create_rw_signal(String::new());
    let url_text = create_rw_signal(String::new());
    let sample = create_rw_signal(String::from("monthly"));

    // Live-data inputs (declarative source model in LIVE_SOURCES)
    let mut init_enabled = HashMap::new();
    let mut init_values = HashMap::new();
    for src in LIVE_SOURCES {
        init_enabled.insert(src.id.to_string(), src.default_enabled);
        if let Some(k) = &src.key_field {
            init_values.insert(k.param.to_string(), k.default.to_string());
        }
        for f in src.fields {
            init_values.insert(f.param.to_string(), f.default.to_string());
        }
    }
    let live_enabled = create_rw_signal(init_enabled);
    let live_values = create_rw_signal(init_values);
    let live_sources_result = create_rw_signal::<Option<Value>>(None);
    let (init_start, init_end) = default_live_dates();
    let start_date = create_rw_signal(init_start);
    let end_date = create_rw_signal(init_end);

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
        history.set(None);
        forecast.set(None);
        features.set(None);
        feature_error.set(None);
        forecast_history_points.set(90);
        confirm_loaded_data_delete.set(false);
        sel_set.set(vec![true]);
        adj_sel.set(0);
        let id2 = id.clone();
        spawn_local(async move {
            fetch_history(id.clone(), history, sel_set, error).await;
            // Feature detection is optional enrichment. Starting it only after
            // history is loaded keeps the plot independent and avoids
            // concurrent calls into the single Pyodide runtime.
            spawn_local(async move {
                fetch_features(id2, features, feature_error, detecting_features).await;
            });
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
        let is_binary = name.ends_with(".xlsx") || name.ends_with(".xls");
        busy.set(true);
        error.set(None);
        status.set("Reading file…".into());
        spawn_local(async move {
            let blob: web_sys::Blob = file.unchecked_into();
            if is_binary {
                match wasm_bindgen_futures::JsFuture::from(blob.array_buffer()).await {
                    Ok(buf) => {
                        let bytes = js_sys::Uint8Array::new(&buf).to_vec();
                        use base64::Engine as _;
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        status.set("Cleaning & detecting format…".into());
                        match call_tool(
                            "smart_load",
                            json!({ "content_base64": b64, "filename": name }),
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
            } else {
                match wasm_bindgen_futures::JsFuture::from(blob.text()).await {
                    Ok(txt) => {
                        let text = txt.as_string().unwrap_or_default();
                        status.set("Cleaning & detecting format…".into());
                        match call_tool("smart_load", json!({ "text": text, "filename": name }))
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
            }
            busy.set(false);
        });
    };

    // True when at least one source is enabled and has any required key set.
    let live_any_ready = move || {
        let en = live_enabled.get();
        let va = live_values.get();
        LIVE_SOURCES
            .iter()
            .any(|s| live_source_ready(s, &en, &va))
    };

    let load_live = move |_| {
        let en = live_enabled.get();
        let va = live_values.get();
        let mut args = serde_json::Map::new();
        // Always emit every source-selecting param: real values when a source is
        // "ready", null otherwise, so a de-selected/incomplete source is skipped
        // deterministically (never silently re-enabled by a library default).
        for src in LIVE_SOURCES {
            let on = live_source_ready(src, &en, &va);
            for f in src.fields {
                let v = if on {
                    parse_live_field(f, va.get(f.param))
                } else {
                    Value::Null
                };
                args.insert(f.param.to_string(), v);
            }
            if let Some(k) = &src.key_field {
                let v = if on {
                    let s = va.get(k.param).map(|s| s.trim().to_string()).unwrap_or_default();
                    if s.is_empty() { Value::Null } else { Value::String(s) }
                } else {
                    Value::Null
                };
                args.insert(k.param.to_string(), v);
            }
        }
        args.insert("observation_start".into(), json!(start_date.get()));
        args.insert("observation_end".into(), json!(end_date.get()));
        // Deliberately gentle on these free APIs while staying tolerable in a UI.
        args.insert("sleep_seconds".into(), json!(3));

        busy.set(true);
        error.set(None);
        live_sources_result.set(None);
        status.set("Loading live data…".into());
        spawn_local(async move {
            match call_tool("load_live_data", Value::Object(args)).await {
                Ok(v) => {
                    live_sources_result.set(v.get("sources").cloned());
                    if let Some(id) = v.get("data_id").and_then(|x| x.as_str()) {
                        after_loaded(id.to_string());
                    } else if let Some(err) = v.get("error").and_then(|x| x.as_str()) {
                        error.set(Some(err.to_string()));
                    }
                }
                Err(e) => error.set(Some(e)),
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
            match call_tool("list_cache", json!({})).await {
                Ok(v) => cache.set(Some(v)),
                Err(e) => error.set(Some(e)),
            }
        });
    };

    let download_forecast = move |_| {
        if let Some(fc) = forecast.get() {
            let csv = forecast_to_csv(&fc, &overrides.get());
            if let Err(e) = download_csv("autots_forecast.csv", &csv) {
                error.set(Some(e));
            }
        }
    };

    let download_loaded_data = move |_| {
        if let Some(data) = history.get() {
            let csv = wide_data_to_csv(&data);
            if let Err(e) = download_csv("autots_loaded_data.csv", &csv) {
                error.set(Some(e));
            }
        }
    };

    let delete_loaded_data = move |_| {
        let Some(id) = data_id.get() else {
            return;
        };
        deleting_loaded_data.set(true);
        error.set(None);
        status.set("Deleting loaded data…".into());
        spawn_local(async move {
            match call_tool(
                "clear_cache",
                json!({ "object_id": id, "cache_type": "data" }),
            )
            .await
            {
                Ok(_) => {
                    data_id.set(None);
                    history.set(None);
                    report.set(None);
                    features.set(None);
                    feature_error.set(None);
                    forecast.set(None);
                    overrides.set(Vec::new());
                    cache.set(None);
                    confirm_loaded_data_delete.set(false);
                    status.set("Loaded data deleted.".into());
                }
                Err(e) => error.set(Some(e)),
            }
            deleting_loaded_data.set(false);
        });
    };

    // Keep the source-data chart independent from forecast state.
    let history_chart_svg = move || {
        let ss = sel_set.get();
        let hist = history.get();
        let hdates = hist.as_ref().map(|h| h.datetime.clone()).unwrap_or_default();
        if let Some(h) = &hist {
            let chart_series: Vec<svg::ChartSeries<'_>> = h
                .series
                .iter()
                .enumerate()
                .filter(|(i, _)| ss.get(*i).copied().unwrap_or(false))
                .map(|(color_idx, s)| svg::ChartSeries {
                    h: &s.values,
                    f: &[],
                    color_idx,
                })
                .collect();
            svg::line_chart(&chart_series, &hdates, &[])
        } else {
            String::new()
        }
    };

    let forecast_chart_svg = move || {
        let ss = sel_set.get();
        let hist = history.get();
        let fc = forecast.get();
        let ovr = overrides.get();
        let requested = forecast_history_points.get();

        let hdates = hist
            .as_ref()
            .map(|h| {
                let start = if requested == 0 {
                    0
                } else {
                    h.datetime.len().saturating_sub(requested)
                };
                h.datetime[start..].to_vec()
            })
            .unwrap_or_default();
        let fdates = fc.as_ref().map(|f| f.datetime.clone()).unwrap_or_default();

        // Build owned windowed history + forecast values per selected series.
        let mut series_data: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        if let Some(h) = &hist {
            for (color_idx, s) in h.series.iter().enumerate() {
                if !ss.get(color_idx).copied().unwrap_or(false) {
                    continue;
                }
                let start = if requested == 0 {
                    0
                } else {
                    s.values.len().saturating_sub(requested)
                };
                let h_vals = s.values[start..].to_vec();
                let f_vals = fc
                    .as_ref()
                    .map(|f| {
                        let fc_idx = f
                            .index_of(&s.name)
                            .unwrap_or_else(|| color_idx.min(f.series.len().saturating_sub(1)));
                        effective_values(f, &ovr, fc_idx)
                    })
                    .unwrap_or_default();
                series_data.push((color_idx, h_vals, f_vals));
            }
        }

        let chart_series: Vec<svg::ChartSeries<'_>> = series_data
            .iter()
            .map(|(color_idx, h_vals, f_vals)| svg::ChartSeries {
                h: h_vals,
                f: f_vals,
                color_idx: *color_idx,
            })
            .collect();
        svg::line_chart(&chart_series, &hdates, &fdates)
    };

    view! {
        <header class="md-appbar">
            <span class="metal-bronze md-emblem" aria-hidden="true"></span>
            <span class="brand">"AutoTS"</span>
            <span class="muted">"forecasting for everyone"</span>
            <span class="spacer"></span>
            <button
                class="md-theme-toggle"
                type="button"
                title=move || if theme.get() == crate::theme::Theme::Dark { "Switch to light theme" } else { "Switch to dark theme" }
                aria-label=move || if theme.get() == crate::theme::Theme::Dark { "Switch to light theme" } else { "Switch to dark theme" }
                on:click=move |_| {
                    let next = theme.get().toggled();
                    crate::theme::apply(next);
                    theme.set(next);
                }
            >
                <span
                    class="md-toggle-icon"
                    inner_html=move || if theme.get() == crate::theme::Theme::Dark { SUN_SVG } else { MOON_SVG }
                ></span>
            </button>
            <a class="md-btn text" href="/llms.txt" target="_blank">"llms.txt"</a>
        </header>

        <main class="md-container">
            // Status / progress
            <div>
                <div class="md-status">{move || status.get()}</div>
                {move || busy.get().then(|| match parse_progress(&status.get()) {
                    Some((i, total)) if total > 0 => {
                        let pct = (i.min(total) as f64 / total as f64 * 100.0).round();
                        view! { <div class="md-progress determinate"><div style=format!("width:{pct}%")></div></div> }.into_view()
                    }
                    _ => view! { <div class="md-progress"><div></div></div> }.into_view(),
                })}
                {move || error.get().map(|e| view! { <p class="md-error">{e}</p> })}
            </div>

            // ---- Upload card ----
            <section class="md-card">
                {section_header("I", "Load your data")}
                <div class="md-tabs" role="tablist">
                    <button class:active=move || tab.get() == 0 on:click=move |_| tab.set(0)>"Paste"</button>
                    <button class:active=move || tab.get() == 1 on:click=move |_| tab.set(1)>"URL"</button>
                    <button class:active=move || tab.get() == 2 on:click=move |_| tab.set(2)>"File"</button>
                    <button class:active=move || tab.get() == 3 on:click=move |_| tab.set(3)>"Sample"</button>
                    <button class:active=move || tab.get() == 4 on:click=move |_| tab.set(4)>"Live"</button>
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
                        3 => view! {
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
                        _ => view! {
                            <div>
                                <p class="md-body">"Pull recent daily data from free public APIs. Loading is intentionally slow to be kind to these services — watch the progress bar. Some sources may be blocked by your browser; you'll see exactly which below. Toggle off any source you don't want, and fill in API keys where a source says it needs one."</p>
                                <div class="md-date-row">
                                    <div class="md-field">
                                        <label class="md-label">"Start date"</label>
                                        <input type="date" prop:value=move || start_date.get()
                                            on:input=move |ev| start_date.set(event_target_value(&ev)) />
                                    </div>
                                    <div class="md-field">
                                        <label class="md-label">"End date"</label>
                                        <input type="date" prop:value=move || end_date.get()
                                            on:input=move |ev| end_date.set(event_target_value(&ev)) />
                                    </div>
                                </div>
                                <div class="md-sources">
                                    {LIVE_SOURCES.iter().map(|src| live_source_card(src, live_enabled, live_values)).collect_view()}
                                </div>
                                <div class="md-btn-row">
                                    <button class="md-btn filled"
                                        disabled=move || !ready.get() || busy.get() || !live_any_ready()
                                        on:click=load_live>"Load live data"</button>
                                </div>
                                {move || live_sources_result.get().map(|s| view! {
                                    <div style="margin-top:12px">
                                        <span class="md-label">"Source results"</span>
                                        {live_results_table(&s)}
                                    </div>
                                })}
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
                        {section_header("II", "Your data")}
                        {(names.len() > 1).then(|| {
                            let names = names.clone();
                            view! {
                                <div class="md-field">
                                    <label class="md-label">"Series (tap to toggle)"</label>
                                    <div class="md-series-chips">
                                        {names.into_iter().enumerate().map(|(i, n)| {
                                            let color_var = format!("--chip-color:var(--viz-s{})", i % 7);
                                            view! {
                                                <button
                                                    type="button"
                                                    class="md-series-chip"
                                                    class:active=move || sel_set.get().get(i).copied().unwrap_or(false)
                                                    style=color_var
                                                    on:click=move |_| sel_set.update(|v| {
                                                        let currently_on = v.get(i).copied().unwrap_or(false);
                                                        let n_on = v.iter().filter(|&&x| x).count();
                                                        if currently_on && n_on <= 1 {
                                                            return; // keep at least one selected
                                                        }
                                                        if let Some(b) = v.get_mut(i) {
                                                            *b = !*b;
                                                        }
                                                    })
                                                >
                                                    <span class="md-series-chip-swatch" aria-hidden="true"></span>
                                                    {n}
                                                </button>
                                            }
                                        }).collect_view()}
                                    </div>
                                </div>
                            }
                        })}

                        <div class="md-chart" inner_html=history_chart_svg></div>

                        {move || detecting_features.get().then(|| view! {
                            <p class="muted md-body">"Detecting features… The data plot is ready to use."</p>
                        })}
                        {move || feature_error.get().map(|e| view! {
                            <p class="md-warning md-body">"Feature detection was unavailable: " {e}</p>
                        })}
                        {move || features.get().map(|f| view! {
                            <div style="margin-top:8px">
                                <span class="md-label">"Detected features"</span>
                                <div>
                                    {feature_chips(&f, &history.get().and_then(|h| {
                                        let first = sel_set.get().iter().position(|&b| b).unwrap_or(0);
                                        h.series.get(first).map(|s| s.name.clone())
                                    }).unwrap_or_default())}
                                </div>
                            </div>
                        })}

                        <details class="md-expander" style="margin-top:12px">
                            <summary>"Data table (accessible / machine-readable)"</summary>
                            {move || history.get().map(|h| {
                                let first = sel_set.get().iter().position(|&b| b).unwrap_or(0);
                                data_table(&h, first)
                            })}
                        </details>

                        {move || confirm_loaded_data_delete.get().then(|| view! {
                            <div id="loaded-data-delete-confirmation" class="md-confirmation" role="alert">
                                <div>
                                    <strong>"Delete this loaded dataset?"</strong>
                                    <p class="md-body">"This removes it from the browser cache and closes its forecast controls. This action cannot be undone."</p>
                                </div>
                                <div class="md-btn-row">
                                    <button class="md-btn text"
                                        disabled=move || deleting_loaded_data.get()
                                        on:click=move |_| confirm_loaded_data_delete.set(false)>
                                        "Cancel"
                                    </button>
                                    <button class="md-btn filled error"
                                        disabled=move || deleting_loaded_data.get()
                                        on:click=delete_loaded_data>
                                        {move || if deleting_loaded_data.get() { "Deleting…" } else { "Delete data" }}
                                    </button>
                                </div>
                            </div>
                        })}

                        <div class="md-card-actions">
                            <button class="md-btn tonal"
                                disabled=move || deleting_loaded_data.get()
                                on:click=download_loaded_data>
                                "Download CSV"
                            </button>
                            <button class="md-btn text error"
                                aria-controls="loaded-data-delete-confirmation"
                                aria-expanded=move || confirm_loaded_data_delete.get().to_string()
                                disabled=move || deleting_loaded_data.get() || confirm_loaded_data_delete.get()
                                on:click=move |_| confirm_loaded_data_delete.set(true)>
                                "Delete data"
                            </button>
                        </div>
                    </section>
                }
            })}

            // ---- Forecast controls ----
            {move || data_id.get().map(|_| view! {
                <section class="md-card">
                    {section_header("III", "Forecast")}
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
                    {section_header("IV", "Review, adjust & download")}
                    <div class="md-chart-controls">
                        <div class="md-field">
                            <label class="md-label">"Actual history shown"</label>
                            <select prop:value=move || forecast_history_points.get().to_string()
                                on:change=move |ev| {
                                if let Ok(n) = event_target_value(&ev).parse::<usize>() {
                                    forecast_history_points.set(n);
                                }
                            }>
                                <option value="30">"30 periods"</option>
                                <option value="90">"90 periods"</option>
                                <option value="180">"180 periods"</option>
                                <option value="365">"365 periods"</option>
                                <option value="0">"All actuals"</option>
                            </select>
                        </div>
                        <div class="md-chart-legend" aria-label="Chart legend">
                            <span><i class="actuals"></i>"Actuals"</span>
                            <span><i class="forecast"></i>"Forecast"</span>
                        </div>
                    </div>
                    <div class="md-chart" inner_html=forecast_chart_svg></div>

                    <details class="md-expander" style="margin-top:12px">
                        <summary>"Adjust forecast points"</summary>
                        {move || adjust_rows(forecast, overrides, history, adj_sel)}
                    </details>

                    <div class="md-btn-row" style="margin-top:12px">
                        <button class="md-btn filled" on:click=download_forecast>"Download forecast CSV"</button>
                    </div>
                </section>
            })}

            // ---- Data manager ----
            <section class="md-card">
                {section_header("", "Cached data & forecasts")}
                <div class="md-btn-row">
                    <button class="md-btn outlined" on:click=refresh_cache>"Refresh"</button>
                </div>
                {move || cache.get().map(|c| cache_summary(&c))}
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
        None => match counts
            .as_object()
            .and_then(|m| m.values().next())
            .and_then(|v| v.as_object())
        {
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

fn cache_summary(cache: &Value) -> View {
    let Some(groups) = cache.as_object() else {
        return ().into_view();
    };
    if groups.is_empty() {
        return view! { <p class="muted md-body">"No cached items."</p> }.into_view();
    }

    groups
        .iter()
        .filter_map(|(group_name, entries)| {
            let entries = entries.as_array()?;
            let rows = entries
                .iter()
                .map(|entry| {
                    let id = entry
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let created = entry
                        .get("created_at")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let metadata = entry.get("metadata").cloned().unwrap_or_else(|| json!({}));
                    let source = metadata
                        .get("source")
                        .and_then(Value::as_str)
                        .unwrap_or("—")
                        .to_string();
                    let rows = metadata.get("rows").and_then(Value::as_u64);
                    let columns = metadata.get("columns").and_then(Value::as_u64);
                    let shape = match (rows, columns) {
                        (Some(r), Some(c)) => format!("{r} × {c}"),
                        _ => "—".into(),
                    };
                    let metadata_text = serde_json::to_string_pretty(&metadata).unwrap_or_default();
                    let id_title = id.clone();
                    view! {
                        <tr>
                            <td class="md-cache-id" title=id_title>{id}</td>
                            <td>{source}</td>
                            <td>{shape}</td>
                            <td>{created}</td>
                            <td>
                                <details class="md-expander">
                                    <summary>"Details"</summary>
                                    <pre class="md-body">{metadata_text}</pre>
                                </details>
                            </td>
                        </tr>
                    }
                })
                .collect_view();
            let title = group_name.replace('_', " ");
            Some(view! {
                <div class="md-cache-group">
                    <h3>{title}</h3>
                    <div class="md-table-wrap">
                        <table class="md-table md-cache-table">
                            <thead>
                                <tr>
                                    <th>"ID"</th>
                                    <th>"Source"</th>
                                    <th>"Shape"</th>
                                    <th>"Created"</th>
                                    <th>"More"</th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                </div>
            })
        })
        .collect_view()
}

fn adjust_rows(
    forecast: RwSignal<Option<WideData>>,
    overrides: RwSignal<Overrides>,
    history: RwSignal<Option<WideData>>,
    adj_sel: RwSignal<usize>,
) -> View {
    let Some(fc) = forecast.get() else {
        return ().into_view();
    };

    // Series picker — only shown when there are multiple forecast series.
    let series_picker = if fc.series.len() > 1 {
        let names: Vec<(usize, String)> = fc
            .series
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.name.clone()))
            .collect();
        view! {
            <div class="md-field" style="max-width:220px; margin-bottom:8px">
                <label class="md-label">"Adjust series"</label>
                <select
                    prop:value=move || adj_sel.get().to_string()
                    on:change=move |ev| {
                        if let Ok(n) = event_target_value(&ev).parse::<usize>() {
                            adj_sel.set(n);
                        }
                    }
                >
                    {names.into_iter().map(|(i, name)| view! {
                        <option value=i.to_string()>{name}</option>
                    }).collect_view()}
                </select>
            </div>
        }
        .into_view()
    } else {
        ().into_view()
    };

    // Map the history series name at adj_sel → forecast series index.
    let sel_name = history
        .get()
        .and_then(|h| h.series.get(adj_sel.get()).map(|x| x.name.clone()));
    let idx = sel_name
        .as_ref()
        .and_then(|n| fc.index_of(n))
        .unwrap_or_else(|| adj_sel.get().min(fc.series.len().saturating_sub(1)));
    let s = match fc.series.get(idx) {
        Some(s) => s.clone(),
        None => return ().into_view(),
    };
    let (lo, hi) = finite_range(&s.values);
    let step = ((hi - lo) / 200.0).max(1e-6);

    let rows = (0..s.values.len())
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
        .collect_view();

    view! { <div>{series_picker}{rows}</div> }.into_view()
}

// ---------------------------------------------------------------------------
// Live data sources (declarative)
// ---------------------------------------------------------------------------
//
// One descriptor per external API in `load_live_daily`. The Live tab renders
// these generically and builds the tool arguments from them, so adding a source
// is a data-only change here plus (optionally) a default in the Python loader.

#[derive(Clone, Copy)]
enum FieldKind {
    /// Comma-separated text -> JSON array of strings (None when empty).
    List,
    /// Integer -> JSON number (None when blank/invalid).
    Int,
    /// Free text -> JSON string (None when empty).
    Str,
}

struct LiveField {
    param: &'static str,
    label: &'static str,
    default: &'static str,
    kind: FieldKind,
}

struct LiveKey {
    param: &'static str,
    label: &'static str,
    default: &'static str,
    signup_url: &'static str,
    /// When true, an empty value makes the source incomplete (excluded). When
    /// false the key is optional and an empty value is sent as null (the loader
    /// falls back to a demo key).
    required: bool,
}

struct LiveSource {
    id: &'static str,
    label: &'static str,
    default_enabled: bool,
    key_field: Option<LiveKey>,
    fields: &'static [LiveField],
    note: Option<&'static str>,
}

const CORS_NOTE: &str =
    "Often blocked by the browser (CORS) or unavailable in-browser — off by default.";

static LIVE_SOURCES: &[LiveSource] = &[
    LiveSource {
        id: "fred",
        label: "FRED economic series",
        default_enabled: true,
        key_field: Some(LiveKey {
            param: "fred_key",
            label: "FRED API key",
            default: "",
            signup_url: "https://fred.stlouisfed.org/docs/api/api_key.html",
            required: true,
        }),
        fields: &[LiveField {
            param: "fred_series",
            label: "Series IDs (comma separated)",
            default: "DGS10,T5YIE,SP500,DCOILWTICO,DEXUSEU,WPU0911",
            kind: FieldKind::List,
        }],
        note: None,
    },
    LiveSource {
        id: "tickers",
        label: "Stock tickers (Yahoo Finance)",
        default_enabled: true,
        key_field: None,
        fields: &[LiveField {
            param: "tickers",
            label: "Tickers (comma separated)",
            default: "MSFT",
            kind: FieldKind::List,
        }],
        note: None,
    },
    LiveSource {
        id: "earthquake",
        label: "Earthquakes (USGS)",
        default_enabled: true,
        key_field: None,
        fields: &[
            LiveField {
                param: "earthquake_min_magnitude",
                label: "Minimum magnitude",
                default: "5",
                kind: FieldKind::Int,
            },
            LiveField {
                param: "earthquake_days",
                label: "Days of history",
                default: "180",
                kind: FieldKind::Int,
            },
        ],
        note: None,
    },
    LiveSource {
        id: "wikipedia",
        label: "Wikipedia pageviews",
        default_enabled: true,
        key_field: None,
        fields: &[
            LiveField {
                param: "wikipedia_pages",
                label: "Pages (comma separated, underscores for spaces)",
                default: "Microsoft_Office,List_of_highest-grossing_films",
                kind: FieldKind::List,
            },
            LiveField {
                param: "wiki_language",
                label: "Language code",
                default: "en",
                kind: FieldKind::Str,
            },
        ],
        note: None,
    },
    LiveSource {
        id: "nasa",
        label: "NASA space weather (DONKI)",
        default_enabled: true,
        key_field: Some(LiveKey {
            param: "nasa_api_key",
            label: "NASA API key",
            default: "DEMO_KEY",
            signup_url: "https://api.nasa.gov/",
            required: true,
        }),
        fields: &[],
        note: Some("Works with the prefilled DEMO_KEY, but it is heavily rate-limited."),
    },
    LiveSource {
        id: "gov",
        label: "US government web analytics",
        default_enabled: true,
        key_field: Some(LiveKey {
            param: "gsa_key",
            label: "GSA DAP API key (optional)",
            default: "",
            signup_url: "https://open.gsa.gov/api/dap/",
            required: false,
        }),
        fields: &[
            LiveField {
                param: "gov_domain_list",
                label: "Domains (comma separated)",
                default: "nasa.gov",
                kind: FieldKind::List,
            },
            LiveField {
                param: "gov_domain_limit",
                label: "Max records",
                default: "600",
                kind: FieldKind::Int,
            },
        ],
        note: Some("Slow. Without a key only the first domain is fetched (demo limit)."),
    },
    LiveSource {
        id: "severe",
        label: "Severe weather events (NOAA)",
        default_enabled: true,
        key_field: None,
        fields: &[LiveField {
            param: "weather_event_types",
            label: "Event types (URL-encoded, comma separated)",
            default: "%28Z%29+Winter+Weather,%28Z%29+Winter+Storm",
            kind: FieldKind::List,
        }],
        note: None,
    },
    LiveSource {
        id: "london",
        label: "London air quality",
        default_enabled: true,
        key_field: None,
        fields: &[
            LiveField {
                param: "london_air_stations",
                label: "Station codes (comma separated)",
                default: "CT3,SK8",
                kind: FieldKind::List,
            },
            LiveField {
                param: "london_air_species",
                label: "Species",
                default: "PM25",
                kind: FieldKind::Str,
            },
            LiveField {
                param: "london_air_days",
                label: "Days of history",
                default: "180",
                kind: FieldKind::Int,
            },
        ],
        note: None,
    },
    LiveSource {
        id: "weather",
        label: "NOAA weather stations",
        default_enabled: false,
        key_field: Some(LiveKey {
            param: "noaa_cdo_token",
            label: "NOAA CDO token",
            default: "",
            signup_url: "https://www.ncdc.noaa.gov/cdo-web/token",
            required: true,
        }),
        fields: &[
            LiveField {
                param: "weather_stations",
                label: "Station IDs (comma separated)",
                default: "USW00094846,USW00014925,USW00014771",
                kind: FieldKind::List,
            },
            LiveField {
                param: "weather_data_types",
                label: "Data types (comma separated)",
                default: "AWND,WSF2,TAVG,PRCP",
                kind: FieldKind::List,
            },
            LiveField {
                param: "weather_years",
                label: "Years of history",
                default: "5",
                kind: FieldKind::Int,
            },
        ],
        note: Some(CORS_NOTE),
    },
    LiveSource {
        id: "eia",
        label: "EIA electricity demand",
        default_enabled: false,
        key_field: Some(LiveKey {
            param: "eia_key",
            label: "EIA API key",
            default: "",
            signup_url: "https://www.eia.gov/opendata/register.php",
            required: true,
        }),
        fields: &[LiveField {
            param: "eia_respondents",
            label: "Respondents (comma separated)",
            default: "MISO,PJM,TVA,US48",
            kind: FieldKind::List,
        }],
        note: Some(CORS_NOTE),
    },
    LiveSource {
        id: "trends",
        label: "Google Trends",
        default_enabled: false,
        key_field: None,
        fields: &[
            LiveField {
                param: "trends_list",
                label: "Search terms (comma separated)",
                default: "forecasting,cycling,microsoft",
                kind: FieldKind::List,
            },
            LiveField {
                param: "trends_geo",
                label: "Geo (e.g. US)",
                default: "US",
                kind: FieldKind::Str,
            },
        ],
        note: Some(CORS_NOTE),
    },
    LiveSource {
        id: "caiso",
        label: "CAISO grid generation",
        default_enabled: false,
        key_field: None,
        fields: &[LiveField {
            param: "caiso_query",
            label: "Query name",
            default: "ENE_SLRS",
            kind: FieldKind::Str,
        }],
        note: Some(CORS_NOTE),
    },
];

/// Default date range for live pulls: end = yesterday, start = two years prior.
fn default_live_dates() -> (String, String) {
    let day_ms = 86_400_000.0;
    let now_ms = js_sys::Date::now();
    let end = js_sys::Date::new(&JsValue::from_f64(now_ms - day_ms));
    let start = js_sys::Date::new(&JsValue::from_f64(now_ms - day_ms));
    start.set_full_year(start.get_full_year() - 2);
    (fmt_js_date(&start), fmt_js_date(&end))
}

fn fmt_js_date(d: &js_sys::Date) -> String {
    format!(
        "{:04}-{:02}-{:02}",
        d.get_full_year() as i64,
        d.get_month() as i64 + 1,
        d.get_date() as i64,
    )
}

/// Parse a trailing "(i/total)" out of a progress message for a determinate bar.
fn parse_progress(s: &str) -> Option<(usize, usize)> {
    let open = s.rfind('(')?;
    let rest = &s[open + 1..];
    let close = rest.find(')')?;
    let (a, b) = rest[..close].split_once('/')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

fn parse_live_field(f: &LiveField, raw: Option<&String>) -> Value {
    let raw = raw.map(|s| s.as_str()).unwrap_or(f.default).trim();
    match f.kind {
        FieldKind::List => {
            let items: Vec<Value> = raw
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| Value::String(s.to_string()))
                .collect();
            if items.is_empty() {
                Value::Null
            } else {
                Value::Array(items)
            }
        }
        FieldKind::Int => raw.parse::<i64>().map(|n| json!(n)).unwrap_or(Value::Null),
        FieldKind::Str => {
            if raw.is_empty() {
                Value::Null
            } else {
                Value::String(raw.to_string())
            }
        }
    }
}

/// A source is "ready" (will be pulled) when enabled and any required key is set.
fn live_source_ready(
    src: &LiveSource,
    enabled: &HashMap<String, bool>,
    values: &HashMap<String, String>,
) -> bool {
    if !enabled.get(src.id).copied().unwrap_or(false) {
        return false;
    }
    if let Some(k) = &src.key_field {
        if k.required {
            let set = values
                .get(k.param)
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            if !set {
                return false;
            }
        }
    }
    true
}

fn live_source_card(
    src: &'static LiveSource,
    enabled: RwSignal<HashMap<String, bool>>,
    values: RwSignal<HashMap<String, String>>,
) -> View {
    let id = src.id;
    let chip = move || {
        if !enabled.get().get(id).copied().unwrap_or(false) {
            return view! { <span class="md-chip off">"Off"</span> };
        }
        if let Some(k) = &src.key_field {
            if k.required {
                let empty = values
                    .get()
                    .get(k.param)
                    .map(|s| s.trim().is_empty())
                    .unwrap_or(true);
                if empty {
                    return view! { <span class="md-chip warn">"Needs API key"</span> };
                }
            }
        }
        view! { <span class="md-chip ready">"Will load"</span> }
    };

    let key_view = src.key_field.as_ref().map(|k| {
        let param = k.param;
        let placeholder = if k.required { "required" } else { "optional" };
        view! {
            <div class="md-field">
                <label class="md-label">
                    {k.label}
                    <a class="md-info-icon" href=k.signup_url target="_blank" rel="noopener"
                       title="Get an API key (opens the sign-up page)">"ⓘ"</a>
                </label>
                <input type="text" value=k.default placeholder=placeholder
                    on:input=move |ev| values.update(|m| { m.insert(param.to_string(), event_target_value(&ev)); }) />
            </div>
        }
    });

    let fields_view = src
        .fields
        .iter()
        .map(|f| {
            let param = f.param;
            view! {
                <div class="md-field">
                    <label class="md-label">{f.label}</label>
                    <input type="text" value=f.default
                        on:input=move |ev| values.update(|m| { m.insert(param.to_string(), event_target_value(&ev)); }) />
                </div>
            }
        })
        .collect_view();

    let note = src
        .note
        .map(|n| view! { <p class="muted md-body">{n}</p> });

    view! {
        <details class="md-expander md-source-card">
            <summary>
                <span class="md-source-head">
                    <input type="checkbox"
                        prop:checked=move || enabled.get().get(id).copied().unwrap_or(false)
                        on:click=|ev| ev.stop_propagation()
                        on:change=move |ev| {
                            let c = event_target_checked(&ev);
                            enabled.update(|m| { m.insert(id.to_string(), c); });
                        } />
                    <span class="md-source-title">{src.label}</span>
                    {chip}
                </span>
            </summary>
            <div class="md-source-body">
                {key_view}
                {fields_view}
                {note}
            </div>
        </details>
    }
    .into_view()
}

fn live_results_table(sources: &Value) -> View {
    let Some(arr) = sources.as_array() else {
        return ().into_view();
    };
    if arr.is_empty() {
        return ().into_view();
    }
    let rows = arr
        .iter()
        .map(|s| {
            let name = s
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("—")
                .to_string();
            let ok = s.get("status").and_then(Value::as_str) == Some("ok");
            let series = s.get("series").and_then(Value::as_u64);
            let err = s.get("error").and_then(Value::as_str).map(str::to_string);
            let detail = if ok {
                let base = series
                    .map(|n| format!("{n} series"))
                    .unwrap_or_else(|| "loaded".into());
                match err {
                    Some(e) => format!("{base} (some items failed: {e})"),
                    None => base,
                }
            } else {
                err.unwrap_or_else(|| "failed".into())
            };
            let cls = if ok { "md-live-ok" } else { "md-live-fail" };
            let mark = if ok { "✓" } else { "✗" };
            view! {
                <li class=cls>
                    <span class="md-live-mark">{mark}</span>
                    <span class="md-live-name">{name}</span>
                    <span class="md-live-detail">{detail}</span>
                </li>
            }
        })
        .collect_view();
    view! { <ul class="md-live-results">{rows}</ul> }.into_view()
}
