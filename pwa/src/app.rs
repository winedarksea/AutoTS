//! AutoTS PWA — Leptos view layer.
//!
//! This layer is intentionally thin: it renders state and calls `client::call_tool`.
//! All forecasting/ingestion logic lives in Python (see autots.mcp.pyodide_api),
//! so this file could be replaced by a TypeScript app against the same facade.

use std::collections::HashMap;

use leptos::*;
use serde_json::{json, Value};
use wasm_bindgen::{JsCast, JsValue};

use crate::cache_view::{durable_cache_summary, storage_usage_view};
use crate::client::{
    call_tool, cancel_forecast, init_runtime, install_app, set_install_handler,
    set_lifecycle_handler, set_progress_handler,
};
use crate::dates::{date_span_days, fmt_date, infer_granularity};
use crate::job::JobState;
use crate::model::{
    effective_bound_values, effective_values, forecast_to_csv, forecast_with_bounds_to_long_csv,
    WideData,
};
use crate::storage;
use crate::svg::{self, ChartOutput, FeatureKind, FeatureKindSet, FeatureMarker};

type Overrides = Vec<Vec<Option<f64>>>;

/// Per-chart interaction state (pointer hover + drag-to-zoom). All fields are
/// `RwSignal`, so the struct is `Copy` and cheap to pass to view helpers.
#[derive(Clone, Copy)]
struct ChartUi {
    /// Rendered SVG string (recomputed by an effect from the chart inputs).
    svg: RwSignal<String>,
    /// Chart geometry for cursor hit-testing (Copy — cheap to read per move).
    geom: RwSignal<Option<svg::ChartGeom>>,
    /// Absolute axis index under the cursor (drives tooltip + crosshair).
    hover: RwSignal<Option<usize>>,
    /// Visible window (absolute, inclusive); `None` = full fit.
    zoom: RwSignal<Option<(usize, usize)>>,
    /// Drag-select endpoints in viewBox-x while brushing a zoom region.
    drag_start: RwSignal<Option<f64>>,
    drag_cur: RwSignal<Option<f64>>,
}

impl ChartUi {
    fn new() -> Self {
        ChartUi {
            svg: create_rw_signal(String::new()),
            geom: create_rw_signal(None),
            hover: create_rw_signal(None),
            zoom: create_rw_signal(None),
            drag_start: create_rw_signal(None),
            drag_cur: create_rw_signal(None),
        }
    }

    /// Apply a freshly built chart output to the SVG + geometry signals.
    fn set_output(&self, out: ChartOutput) {
        self.geom.set(Some(out.geom));
        self.svg.set(out.svg);
    }
}

/// One tooltip row: (palette color index, series name, value at the point).
type TtRow = (usize, String, f64);
type ChartTooltip = (String, Vec<TtRow>, Vec<FeatureKind>);

fn norm_date(s: &str) -> String {
    s.chars().take(10).collect()
}

/// Map normalized (YYYY-MM-DD) date -> first axis index, for feature matching.
fn date_index_map(dates: &[String]) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (i, d) in dates.iter().enumerate() {
        m.entry(norm_date(d)).or_insert(i);
    }
    m
}

/// Build detected-feature markers for one series by joining its feature dates to
/// the axis. `feats` is the full `get_detected_features` response. Unmatched
/// dates (sub-daily/irregular axes, or outside the data) are silently dropped.
fn markers_for_series(
    feats: &Value,
    name: &str,
    dmap: &HashMap<String, usize>,
    values: &[f64],
) -> Vec<FeatureMarker> {
    let mut out = Vec::new();
    let Some(sobj) = feats
        .get("features")
        .and_then(|f| f.get("series"))
        .and_then(|s| s.get(name))
    else {
        return out;
    };
    for kind in FeatureKind::ALL {
        let Some(arr) = sobj.get(kind.json_key()).and_then(|v| v.as_array()) else {
            continue;
        };
        for item in arr {
            // holiday_dates are bare strings; others are {date, ...} objects.
            let date = item.as_str().map(|s| s.to_string()).or_else(|| {
                item.get("date")
                    .and_then(|d| d.as_str())
                    .map(|s| s.to_string())
            });
            if let Some(date) = date {
                if let Some(&idx) = dmap.get(&norm_date(&date)) {
                    let value = values.get(idx).copied().unwrap_or(f64::NAN);
                    out.push(FeatureMarker { idx, kind, value });
                }
            }
        }
    }
    out
}

/// Return each detected feature kind once for the selected series at one axis
/// position. This uses the same marker join as the chart, so hover labels cannot
/// disagree with late-arriving feature markers.
fn feature_kinds_at_index(
    feats: &Value,
    selected_series: &[&crate::model::SeriesData],
    dates: &[String],
    idx: usize,
) -> Vec<FeatureKind> {
    let date_indices = date_index_map(dates);
    FeatureKind::ALL
        .into_iter()
        .filter(|kind| {
            selected_series.iter().any(|series| {
                markers_for_series(feats, &series.name, &date_indices, &series.values)
                    .iter()
                    .any(|marker| {
                        marker.idx == idx && marker.kind == *kind && marker.value.is_finite()
                    })
            })
        })
        .collect()
}

/// Cursor x (in the 920-wide viewBox) for a pointer event over `node`.
fn event_vbx(node: &NodeRef<html::Div>, client_x: i32) -> Option<f64> {
    let el = node.get()?;
    let rect = el.get_bounding_client_rect();
    let w = rect.width();
    if w <= 0.0 {
        return None;
    }
    Some((client_x as f64 - rect.left()) / w * svg::VB_W)
}

// ---------------------------------------------------------------------------
// Classical theme-toggle glyphs.
//
// Hand-drawn inline SVGs filled with the same metallic-bronze gradient as the
// corner emblem (the `--metal-bronze-*` tokens, so they track light/dark).
// SUN is a Vergina Sun — the ancient Macedonian 16-ray starburst, a central
// disc with a rosette boss (shown to switch *to* light); MOON is a Selene
// crescent + star (shown to switch *to* dark). Decorative — the button carries
// an aria-label — so both are aria-hidden.
// ---------------------------------------------------------------------------

// 16-point star polygon (alternating outer r=11 / inner r=4.6 about 12,12),
// the silhouette of a Vergina Sun's rays.
const SUN_SVG: &str = "<svg viewBox=\"0 0 24 24\" class=\"md-toggle-glyph\" aria-hidden=\"true\" \
    xmlns=\"http://www.w3.org/2000/svg\">\
    <defs><linearGradient id=\"bzsun\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\
    <stop offset=\"0\" stop-color=\"var(--metal-bronze-light)\"/>\
    <stop offset=\"0.5\" stop-color=\"var(--metal-bronze-mid)\"/>\
    <stop offset=\"1\" stop-color=\"var(--metal-bronze-dark)\"/></linearGradient></defs>\
    <polygon points=\"12,1 12.9,7.49 16.21,1.84 14.56,8.18 19.78,4.22 15.82,9.44 \
    22.16,7.79 16.51,11.1 23,12 16.51,12.9 22.16,16.21 15.82,14.56 19.78,19.78 \
    14.56,15.82 16.21,22.16 12.9,16.51 12,23 11.1,16.51 7.79,22.16 9.44,15.82 \
    4.22,19.78 8.18,14.56 1.84,16.21 7.49,12.9 1,12 7.49,11.1 1.84,7.79 8.18,9.44 \
    4.22,4.22 9.44,8.18 7.79,1.84 11.1,7.49\" \
    fill=\"url(#bzsun)\" stroke=\"var(--metal-bronze-dark)\" stroke-width=\"0.5\" \
    stroke-linejoin=\"round\"/>\
    <circle cx=\"12\" cy=\"12\" r=\"3.4\" fill=\"url(#bzsun)\" \
    stroke=\"var(--metal-bronze-dark)\" stroke-width=\"0.6\"/>\
    <circle cx=\"12\" cy=\"12\" r=\"1.4\" fill=\"none\" \
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

// Inline favicon SVG for the app-bar emblem. Surface and grid colors follow the
// active theme, while the established bronze gradient remains unchanged.
const EMBLEM_SVG: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" \
    viewBox=\"0 0 32 32\" width=\"22\" height=\"22\" aria-hidden=\"true\">\
    <defs>\
    <linearGradient id=\"emBorder\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\">\
    <stop offset=\"0\" stop-color=\"#5e4200\"/>\
    <stop offset=\"0.30\" stop-color=\"#c68a2e\"/>\
    <stop offset=\"0.55\" stop-color=\"#ffe2a8\"/>\
    <stop offset=\"0.78\" stop-color=\"#c68a2e\"/>\
    <stop offset=\"1\" stop-color=\"#5e4200\"/>\
    </linearGradient>\
    <linearGradient id=\"emBolt\" x1=\"0.85\" y1=\"0.1\" x2=\"0.15\" y2=\"0.9\">\
    <stop offset=\"0\" stop-color=\"#5e4200\"/>\
    <stop offset=\"0.32\" stop-color=\"#c68a2e\"/>\
    <stop offset=\"0.56\" stop-color=\"#ffe2a8\"/>\
    <stop offset=\"0.80\" stop-color=\"#c68a2e\"/>\
    <stop offset=\"1\" stop-color=\"#5e4200\"/>\
    </linearGradient>\
    </defs>\
    <rect x=\"2\" y=\"2\" width=\"28\" height=\"28\" rx=\"6\" \
    fill=\"var(--md-surface)\" stroke=\"url(#emBorder)\" stroke-width=\"1.5\"/>\
    <g stroke=\"var(--md-outline)\" stroke-opacity=\"0.35\" stroke-width=\"0.5\">\
    <line x1=\"5.5\" y1=\"10\" x2=\"26.5\" y2=\"10\"/>\
    <line x1=\"5.5\" y1=\"16\" x2=\"26.5\" y2=\"16\"/>\
    <line x1=\"5.5\" y1=\"22\" x2=\"26.5\" y2=\"22\"/>\
    </g>\
    <path d=\"M 26.5 7.32 L 18.19 11.84 L 20.72 14.05 L 13.79 16.45 L 16.07 18.56 \
    L 6.7 22.62 L 13.43 18.08 L 11.51 16.09 L 18.08 13.69 L 15.91 11.6 Z\" \
    fill=\"url(#emBolt)\" stroke=\"#5e4200\" stroke-width=\"0.3\" \
    stroke-opacity=\"0.25\" stroke-linejoin=\"round\"/>\
    </svg>";

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

/// Minimum drag distance (in viewBox-x units) to register as a zoom selection.
const MIN_BRUSH_VBX: f64 = 8.0;

/// A chart with its pointer-interaction overlay: hover tooltip + crosshair, and
/// drag-to-select zoom. The SVG is injected via `inner_html`; the tooltip,
/// crosshair, and selection rectangle are signal-driven Leptos elements stacked
/// over it. `tooltip` resolves an absolute axis index to a date + value rows.
fn chart_block(ui: ChartUi, tooltip: Callback<usize, Option<ChartTooltip>>) -> impl IntoView {
    let node = create_node_ref::<html::Div>();

    let crosshair = move || {
        let idx = ui.hover.get()?;
        let g = ui.geom.get()?;
        let pct = g.vbx_of_abs(idx) / svg::VB_W * 100.0;
        Some(view! { <div class="md-chart-crosshair" style=format!("left:{pct:.3}%")></div> })
    };
    let tip = move || {
        let idx = ui.hover.get()?;
        let g = ui.geom.get()?;
        let (date, rows, detected_feature_kinds) = tooltip.call(idx)?;
        let pct = g.vbx_of_abs(idx) / svg::VB_W * 100.0;
        let rows_view = rows
            .into_iter()
            .map(|(ci, name, val)| {
                view! {
                    <div class="md-tt-row">
                        <span class="md-tt-swatch" style=format!("background:var(--viz-s{})", ci % 7)></span>
                        <span class="md-tt-name">{name}</span>
                        <span class="md-tt-val">{svg::fmt_value(val)}</span>
                    </div>
                }
            })
            .collect_view();
        let detected_features_view = detected_feature_kinds
            .into_iter()
            .map(|kind| {
                view! {
                    <div class="md-tt-feature">{kind.tooltip_label()}</div>
                }
            })
            .collect_view();
        Some(view! {
            <div class="md-chart-tooltip" style=format!("left:{pct:.3}%")>
                <div class="md-tt-date">{date}</div>
                {rows_view}
                {detected_features_view}
            </div>
        })
    };
    let brush = move || {
        let s = ui.drag_start.get()?;
        let c = ui.drag_cur.get()?;
        let left = s.min(c) / svg::VB_W * 100.0;
        let width = (c - s).abs() / svg::VB_W * 100.0;
        Some(view! {
            <div class="md-chart-brush" style=format!("left:{left:.3}%;width:{width:.3}%")></div>
        })
    };

    view! {
        <div class="md-chart-wrap">
            <div
                class="md-chart"
                node_ref=node
                inner_html=move || ui.svg.get()
                on:pointerdown=move |ev: leptos::ev::PointerEvent| {
                    if let Some(vbx) = event_vbx(&node, ev.client_x()) {
                        ui.drag_start.set(Some(vbx));
                        ui.drag_cur.set(Some(vbx));
                        ui.hover.set(None);
                        if let Some(el) = node.get() {
                            let _ = el.set_pointer_capture(ev.pointer_id());
                        }
                    }
                }
                on:pointermove=move |ev: leptos::ev::PointerEvent| {
                    let Some(vbx) = event_vbx(&node, ev.client_x()) else { return };
                    if ui.drag_start.get_untracked().is_some() {
                        ui.drag_cur.set(Some(vbx));
                    } else if let Some(g) = ui.geom.get_untracked() {
                        ui.hover.set(Some(g.index_at_vbx(vbx)));
                    }
                }
                on:pointerup=move |_ev: leptos::ev::PointerEvent| {
                    if let (Some(s), Some(c)) =
                        (ui.drag_start.get_untracked(), ui.drag_cur.get_untracked())
                    {
                        if (c - s).abs() >= MIN_BRUSH_VBX {
                            if let Some(g) = ui.geom.get_untracked() {
                                let a = g.index_at_vbx(s.min(c));
                                let b = g.index_at_vbx(s.max(c));
                                if b > a {
                                    ui.zoom.set(Some((a, b)));
                                }
                            }
                        }
                    }
                    ui.drag_start.set(None);
                    ui.drag_cur.set(None);
                }
                on:pointerleave=move |_| {
                    ui.hover.set(None);
                }
            ></div>
            {crosshair}
            {brush}
            {tip}
        </div>
    }
}

/// Interactive "Detected features" legend: one toggle chip per feature kind,
/// showing its glyph + count and flipping its marker visibility on the charts.
fn feature_legend(feats: &Value, series_name: &str, features_on: RwSignal<FeatureKindSet>) -> View {
    let counts = feats
        .get("detection_counts")
        .and_then(|c| c.get(series_name))
        .cloned()
        .unwrap_or(Value::Null);
    FeatureKind::ALL
        .into_iter()
        .map(|kind| {
            let count = counts
                .get(kind.count_key())
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let glyph = svg::legend_glyph(kind);
            let label = format!("{}: {}", kind.label(), count);
            view! {
                <button
                    type="button"
                    class="md-feat-chip"
                    class:active=move || features_on.get().get(kind)
                    disabled=count == 0
                    on:click=move |_| features_on.update(|s| *s = s.with_toggled(kind))
                >
                    <span class="md-feat-glyph" inner_html=glyph></span>
                    {label}
                </button>
            }
        })
        .collect_view()
}

async fn fetch_features(
    data_id: String,
    features: RwSignal<Option<Value>>,
    feature_error: RwSignal<Option<String>>,
    detecting_features: RwSignal<bool>,
) -> Option<Value> {
    detecting_features.set(true);
    feature_error.set(None);
    match call_tool("detect_features", json!({ "data_id": data_id })).await {
        Ok(det) => {
            let Some(did) = det.get("detector_id").and_then(|x| x.as_str()) else {
                feature_error.set(Some("Feature detection returned no detector ID.".into()));
                detecting_features.set(false);
                return None;
            };
            match call_tool("get_detected_features", json!({ "detector_id": did })).await {
                // Keep the whole response: `detection_counts` drives the legend,
                // `features` drives the per-point markers.
                Ok(f) => {
                    features.set(Some(f.clone()));
                    detecting_features.set(false);
                    return Some(f);
                }
                Err(e) => feature_error.set(Some(e)),
            }
        }
        Err(e) => feature_error.set(Some(e)),
    }
    detecting_features.set(false);
    None
}

async fn fetch_forecast_output_value(pid: &str, output: &str) -> Option<(Value, WideData)> {
    let value = call_tool(
        "get_forecast",
        json!({ "prediction_id": pid, "output": output, "format": "json_wide" }),
    )
    .await
    .ok()?;
    let wide = WideData::from_json_wide(&value)?;
    Some((value, wide))
}

async fn reload_artifacts(
    artifacts: RwSignal<Option<Value>>,
    storage_summary: RwSignal<Option<Value>>,
    error: RwSignal<Option<String>>,
) {
    match storage::list_artifacts().await {
        Ok(value) => artifacts.set(Some(value)),
        Err(message) => error.set(Some(message)),
    }
    if let Ok(value) = storage::storage_summary().await {
        storage_summary.set(Some(value));
    }
}

async fn restore_runtime_dataset(artifact: &Value) -> Result<String, String> {
    let data = artifact
        .pointer("/data/wide")
        .cloned()
        .ok_or_else(|| "Stored dataset has no wide data snapshot".to_string())?;
    let metadata = artifact
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let result = call_tool(
        "restore_data_snapshot",
        json!({ "data": data, "metadata": metadata }),
    )
    .await?;
    result
        .get("data_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "Restored dataset returned no runtime ID".to_string())
}

/// Fetch a prediction's model parameters via `get_model_params`.
async fn fetch_model_params(pid: &str) -> Result<Value, String> {
    call_tool("get_model_params", json!({ "prediction_id": pid })).await
}

#[allow(clippy::too_many_arguments)]
async fn run_forecast(
    command: String,
    data_id: String,
    forecast_length: i64,
    additional_arguments: Option<Value>,
    forecast: RwSignal<Option<WideData>>,
    upper: RwSignal<Option<WideData>>,
    lower: RwSignal<Option<WideData>>,
    overrides: RwSignal<Overrides>,
    active_dataset_artifact_id: String,
    active_forecast_artifact_id: RwSignal<Option<String>>,
    artifacts: RwSignal<Option<Value>>,
    storage_summary: RwSignal<Option<Value>>,
    busy: RwSignal<bool>,
    job_state: RwSignal<JobState>,
    error: RwSignal<Option<String>>,
    status: RwSignal<String>,
) {
    busy.set(true);
    job_state.set(JobState::Running);
    error.set(None);
    status.set("Forecasting…".into());
    let mut arguments = json!({
        "data_id": data_id,
        "forecast_length": forecast_length,
    });
    if let Some(Value::Object(additional)) = additional_arguments {
        if let Some(arguments_object) = arguments.as_object_mut() {
            arguments_object.extend(additional);
        }
    }
    let res = call_tool(&command, arguments).await;
    match res {
        Ok(v) => {
            if let Some(pid) = v.get("prediction_id").and_then(|x| x.as_str()) {
                match fetch_forecast_output_value(pid, "forecast").await {
                    Some((forecast_value, w)) => {
                        let ov = w
                            .series
                            .iter()
                            .map(|s| vec![None; s.values.len()])
                            .collect();
                        // Prediction-interval bounds for the uncertainty band.
                        let upper_result = fetch_forecast_output_value(pid, "upper_forecast").await;
                        let lower_result = fetch_forecast_output_value(pid, "lower_forecast").await;
                        upper.set(upper_result.as_ref().map(|(_, wide)| wide.clone()));
                        lower.set(lower_result.as_ref().map(|(_, wide)| wide.clone()));
                        overrides.set(ov);
                        forecast.set(Some(w));
                        let model_parameters =
                            fetch_model_params(pid).await.unwrap_or_else(|_| json!({}));
                        let artifact = storage::forecast_artifact(
                            &active_dataset_artifact_id,
                            &command,
                            forecast_length,
                            forecast_value,
                            upper_result.map(|(value, _)| value),
                            lower_result.map(|(value, _)| value),
                            model_parameters,
                            json!(overrides.get()),
                        );
                        match storage::put_artifact(
                            artifact,
                            std::slice::from_ref(&active_dataset_artifact_id),
                        )
                        .await
                        {
                            Ok(saved) => {
                                active_forecast_artifact_id.set(
                                    saved
                                        .pointer("/artifact/id")
                                        .and_then(Value::as_str)
                                        .map(str::to_string),
                                );
                                reload_artifacts(artifacts, storage_summary, error).await;
                                status.set(
                                    "Forecast ready and saved — drag the sliders to adjust.".into(),
                                );
                            }
                            Err(message) => {
                                error.set(Some(format!(
                                    "Forecast is available for this session but could not be saved: {message}"
                                )));
                                status.set("Forecast ready.".into());
                            }
                        }
                    }
                    None => error.set(Some("Forecast could not be read".into())),
                }
            } else {
                error.set(Some("No prediction returned".into()));
            }
        }
        Err(e) => {
            if !matches!(
                job_state.get_untracked(),
                JobState::Cancelling | JobState::Restarting
            ) {
                error.set(Some(e));
                job_state.set(JobState::Failed);
            }
        }
    }
    busy.set(false);
    if job_state.get_untracked() == JobState::Running {
        job_state.set(JobState::Ready);
    }
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

/// Half the browser-local time remaining until 07:00 tomorrow, in whole minutes.
fn overnight_generation_timeout_minutes() -> i64 {
    let now = js_sys::Date::new_0();
    let target = js_sys::Date::new_0();
    target.set_date(target.get_date() + 1);
    target.set_hours(7);
    target.set_minutes(0);
    target.set_seconds(0);
    target.set_milliseconds(0);
    (((target.get_time() - now.get_time()) / 60_000.0) * 0.5)
        .floor()
        .max(1.0) as i64
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

#[component]
pub fn App() -> impl IntoView {
    let ready = create_rw_signal(false);
    let status = create_rw_signal(String::from("Booting Python runtime…"));
    let busy = create_rw_signal(false);
    let job_state = create_rw_signal(JobState::Restarting);
    let error = create_rw_signal::<Option<String>>(None);
    let offline_ready = create_rw_signal(false);
    let install_available = create_rw_signal(false);
    let app_installed = create_rw_signal(false);

    // Effective light/dark theme (from a saved choice, else the OS preference).
    let theme = create_rw_signal(crate::theme::initial_theme());

    let data_id = create_rw_signal::<Option<String>>(None);
    let active_dataset_artifact_id = create_rw_signal::<Option<String>>(None);
    let active_forecast_artifact_id = create_rw_signal::<Option<String>>(None);
    let history = create_rw_signal::<Option<WideData>>(None);
    let report = create_rw_signal::<Option<Value>>(None);
    let features = create_rw_signal::<Option<Value>>(None);
    let feature_error = create_rw_signal::<Option<String>>(None);
    let detecting_features = create_rw_signal(false);
    let forecast = create_rw_signal::<Option<WideData>>(None);
    // Prediction-interval bounds (for the optional uncertainty band).
    let upper = create_rw_signal::<Option<WideData>>(None);
    let lower = create_rw_signal::<Option<WideData>>(None);
    let overrides = create_rw_signal::<Overrides>(Vec::new());
    // sel_set: one bool per series in the loaded dataset (true = visible in chart).
    // Sized and reset by fetch_history when data loads.
    let sel_set = create_rw_signal::<Vec<bool>>(vec![true]);
    // adj_sel: which single series index to show in the adjust-forecast sliders.
    let adj_sel = create_rw_signal::<usize>(0);
    let forecast_length = create_rw_signal::<i64>(30);
    let prediction_interval = create_rw_signal::<f64>(0.9);
    // Chart interaction state (one per chart) + shared display toggles.
    let history_ui = ChartUi::new();
    let forecast_ui = ChartUi::new();
    let features_on = create_rw_signal(FeatureKindSet::none());
    let show_bands = create_rw_signal(false);
    let series_menu_open = create_rw_signal(false);

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
    let live_end_max = fmt_js_date(&js_sys::Date::new_0());

    // Data manager
    let cache = create_rw_signal::<Option<Value>>(None);
    let storage_summary = create_rw_signal::<Option<Value>>(None);

    // Boot the runtime once.
    set_progress_handler(move |m| status.set(m));
    set_lifecycle_handler(move |message| {
        let next = JobState::from_lifecycle_message(&message);
        job_state.set(next);
        if next == JobState::Ready {
            ready.set(true);
        }
    });
    set_install_handler(move |state_json| {
        if let Ok(state) = serde_json::from_str::<Value>(&state_json) {
            offline_ready.set(
                state
                    .get("offlineReady")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            );
            install_available.set(
                state
                    .get("installAvailable")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            );
            app_installed.set(
                state
                    .get("installed")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            );
        }
    });
    spawn_local(async move {
        let _ = storage::request_persistent_storage().await;
        reload_artifacts(cache, storage_summary, error).await;
    });
    spawn_local(async move {
        match init_runtime().await {
            Ok(()) => {
                ready.set(true);
                job_state.set(JobState::Ready);
                status.set("Ready. Load some data to begin.".into());
            }
            Err(e) => {
                error.set(Some(format!("Runtime failed to start: {e}")));
                job_state.set(JobState::Failed);
                status.set("Runtime error".into());
            }
        }
    });

    // --- data-loaded continuation: fetch history + features ---------------
    let after_loaded = move |id: String| {
        data_id.set(Some(id.clone()));
        active_dataset_artifact_id.set(None);
        active_forecast_artifact_id.set(None);
        history.set(None);
        forecast.set(None);
        upper.set(None);
        lower.set(None);
        features.set(None);
        feature_error.set(None);
        features_on.set(FeatureKindSet::none());
        show_bands.set(false);
        history_ui.zoom.set(None);
        forecast_ui.zoom.set(None);
        sel_set.set(vec![true]);
        adj_sel.set(0);
        let id2 = id.clone();
        spawn_local(async move {
            let snapshot = call_tool(
                "get_data",
                json!({ "data_id": id.clone(), "output_format": "json_wide" }),
            )
            .await;
            let Ok(snapshot) = snapshot else {
                error.set(snapshot.err());
                return;
            };
            let Some(wide) = WideData::from_json_wide(&snapshot) else {
                error.set(Some("Loaded data could not be read".into()));
                return;
            };
            let series_count = wide.series.len();
            let row_count = wide.datetime.len();
            sel_set.set(wide.initial_series_selection());
            history.set(Some(wide));

            let artifact = storage::dataset_artifact(
                snapshot,
                json!({
                    "source": "pwa_load",
                    "rows": row_count,
                    "columns": series_count,
                }),
                report.get_untracked(),
            );
            match storage::put_artifact(artifact, &[]).await {
                Ok(saved) => {
                    active_dataset_artifact_id.set(
                        saved
                            .pointer("/artifact/id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    );
                    reload_artifacts(cache, storage_summary, error).await;
                }
                Err(message) => error.set(Some(format!(
                    "Data is loaded for this session but could not be saved: {message}"
                ))),
            }
            // Feature detection is optional enrichment. Starting it only after
            // history is loaded keeps the plot independent and avoids
            // concurrent calls into the single Pyodide runtime.
            spawn_local(async move {
                if let Some(feature_value) =
                    fetch_features(id2, features, feature_error, detecting_features).await
                {
                    if let Some(artifact_id) = active_dataset_artifact_id.get_untracked() {
                        if let Ok(mut artifact) = storage::get_artifact(&artifact_id).await {
                            if let Some(data) =
                                artifact.get_mut("data").and_then(Value::as_object_mut)
                            {
                                data.insert("features".into(), feature_value);
                                let _ = storage::put_artifact(
                                    artifact,
                                    std::slice::from_ref(&artifact_id),
                                )
                                .await;
                                reload_artifacts(cache, storage_summary, error).await;
                            }
                        }
                    }
                }
            });
        });
    };

    // --- upload actions ----------------------------------------------------
    let load_sample = move |_| {
        let ds = sample.get();
        report.set(None);
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
        let lower_name = name.to_ascii_lowercase();
        if lower_name.ends_with(".xls") {
            error.set(Some(
                "Legacy .xls workbooks are not supported. Save the file as .xlsx or CSV.".into(),
            ));
            input.set_value("");
            return;
        }
        let is_binary = lower_name.ends_with(".xlsx");
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
        LIVE_SOURCES.iter().any(|s| live_source_ready(s, &en, &va))
    };

    let load_live = {
        let live_end_max = live_end_max.clone();
        move |_| {
            let en = live_enabled.get();
            let va = live_values.get();
            let observation_start = start_date.get();
            let observation_end = clamp_iso_date_max(end_date.get(), &live_end_max);
            let Some((history_days, history_years)) =
                live_history_parameters(&observation_start, &observation_end)
            else {
                error.set(Some(
                    "Start date must be a valid date on or before the end date.".into(),
                ));
                return;
            };
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
                        let s = va
                            .get(k.param)
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        if s.is_empty() {
                            Value::Null
                        } else {
                            Value::String(s)
                        }
                    } else {
                        Value::Null
                    };
                    args.insert(k.param.to_string(), v);
                }
            }
            args.insert("observation_start".into(), json!(observation_start));
            args.insert("observation_end".into(), json!(observation_end));
            args.insert("earthquake_days".into(), json!(history_days));
            args.insert("london_air_days".into(), json!(history_days));
            args.insert("weather_years".into(), json!(history_years));
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
        }
    };

    // --- forecast actions --------------------------------------------------
    let do_forecast = move |command: &'static str| {
        let Some(id) = data_id.get() else {
            error.set(Some("Load data first".into()));
            return;
        };
        let Some(dataset_artifact_id) = active_dataset_artifact_id.get() else {
            error.set(Some("Wait for the loaded dataset to finish saving.".into()));
            return;
        };
        let fl = forecast_length.get();
        let additional_arguments = match command {
            "search_forecast" => Some(json!({
                "autots_params": {
                    "prediction_interval": prediction_interval.get(),
                },
            })),
            "search_all_night" => Some(json!({
                "autots_params": {
                    "prediction_interval": prediction_interval.get(),
                    "generation_timeout": overnight_generation_timeout_minutes(),
                    "max_generations": 1_000_000,
                },
            })),
            _ => None,
        };
        spawn_local(run_forecast(
            command.to_string(),
            id,
            fl,
            additional_arguments,
            forecast,
            upper,
            lower,
            overrides,
            dataset_artifact_id,
            active_forecast_artifact_id,
            cache,
            storage_summary,
            busy,
            job_state,
            error,
            status,
        ));
    };

    let refresh_cache = move |_| {
        spawn_local(async move { reload_artifacts(cache, storage_summary, error).await });
    };

    let cancel_running_forecast = move |_| {
        if !job_state.get_untracked().is_forecasting() {
            return;
        }
        job_state.set(JobState::Cancelling);
        status.set("Cancelling forecast…".into());
        error.set(None);
        spawn_local(async move {
            match cancel_forecast().await {
                Ok(()) => {
                    job_state.set(JobState::Restarting);
                    status.set("Forecast cancelled. Restarting local compute…".into());
                    if let Some(artifact_id) = active_dataset_artifact_id.get_untracked() {
                        match storage::get_artifact(&artifact_id).await {
                            Ok(artifact) => match restore_runtime_dataset(&artifact).await {
                                Ok(runtime_id) => {
                                    data_id.set(Some(runtime_id));
                                    job_state.set(JobState::Ready);
                                    status
                                        .set("Forecast cancelled. Local compute is ready.".into());
                                }
                                Err(message) => {
                                    job_state.set(JobState::Failed);
                                    error.set(Some(format!(
                                        "Forecast was cancelled, but the active data could not be restored: {message}"
                                    )));
                                }
                            },
                            Err(message) => {
                                job_state.set(JobState::Failed);
                                error.set(Some(message));
                            }
                        }
                    } else {
                        job_state.set(JobState::Ready);
                        status.set("Forecast cancelled. Local compute is ready.".into());
                    }
                }
                Err(message) => {
                    job_state.set(JobState::Failed);
                    error.set(Some(format!("Could not restart local compute: {message}")));
                }
            }
            busy.set(false);
        });
    };

    let download_forecast = move |_| {
        if let Some(fc) = forecast.get() {
            let csv = if show_bands.get() {
                forecast_with_bounds_to_long_csv(
                    &fc,
                    upper.get().as_ref(),
                    lower.get().as_ref(),
                    &overrides.get(),
                )
            } else {
                forecast_to_csv(&fc, &overrides.get())
            };
            if let Err(e) = download_csv("autots_forecast.csv", &csv) {
                error.set(Some(e));
            }
        }
    };

    // True when the actively adjusted series carries a manual override.
    let has_adjustments = move || {
        overrides
            .get()
            .get(adj_sel.get())
            .map(|col| col.iter().any(Option::is_some))
            .unwrap_or(false)
    };

    // Restore only the actively selected series to its original values.
    let reset_adjustments = move |_| {
        let active_series = adj_sel.get();
        overrides.update(|all| {
            if let Some(series_overrides) = all.get_mut(active_series) {
                series_overrides.fill(None);
            }
        });
    };

    // Clears every "active" signal tied to the loaded dataset and its forecast.
    // Reused when the active dataset is deleted from the cached-data view.
    let teardown_active = Callback::new(move |_| {
        data_id.set(None);
        active_dataset_artifact_id.set(None);
        active_forecast_artifact_id.set(None);
        history.set(None);
        report.set(None);
        features.set(None);
        feature_error.set(None);
        forecast.set(None);
        upper.set(None);
        lower.set(None);
        overrides.set(Vec::new());
        history_ui.zoom.set(None);
        forecast_ui.zoom.set(None);
    });

    let open_cached_artifact = Callback::new(move |artifact_id: String| {
        error.set(None);
        status.set("Opening saved artifact…".into());
        spawn_local(async move {
            let result: Result<(), String> = async {
                let artifact = storage::get_artifact(&artifact_id).await?;
                let kind = artifact.get("kind").and_then(Value::as_str).unwrap_or("");
                let dataset_artifact = if kind == "forecast" {
                    let parent_id = artifact
                        .get("parent_id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "Saved forecast has no input dataset".to_string())?;
                    storage::get_artifact(parent_id).await?
                } else {
                    artifact.clone()
                };
                let runtime_id = restore_runtime_dataset(&dataset_artifact).await?;
                let history_value = dataset_artifact
                    .pointer("/data/wide")
                    .cloned()
                    .ok_or_else(|| "Saved dataset has no data snapshot".to_string())?;
                let history_wide = WideData::from_json_wide(&history_value)
                    .ok_or_else(|| "Saved dataset is invalid".to_string())?;
                let parent_id = dataset_artifact
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Saved dataset has no ID".to_string())?
                    .to_string();

                data_id.set(Some(runtime_id));
                active_dataset_artifact_id.set(Some(parent_id));
                sel_set.set(history_wide.initial_series_selection());
                history.set(Some(history_wide));
                report.set(
                    dataset_artifact
                        .pointer("/data/report")
                        .filter(|value| !value.is_null())
                        .cloned(),
                );
                features.set(
                    dataset_artifact
                        .pointer("/data/features")
                        .filter(|value| !value.is_null())
                        .cloned(),
                );
                feature_error.set(None);
                history_ui.zoom.set(None);
                forecast_ui.zoom.set(None);

                if kind == "forecast" {
                    forecast.set(
                        artifact
                            .pointer("/data/forecast")
                            .and_then(WideData::from_json_wide),
                    );
                    upper.set(
                        artifact
                            .pointer("/data/upper")
                            .filter(|value| !value.is_null())
                            .and_then(WideData::from_json_wide),
                    );
                    lower.set(
                        artifact
                            .pointer("/data/lower")
                            .filter(|value| !value.is_null())
                            .and_then(WideData::from_json_wide),
                    );
                    overrides.set(
                        artifact
                            .pointer("/data/overrides")
                            .cloned()
                            .and_then(|value| serde_json::from_value(value).ok())
                            .unwrap_or_default(),
                    );
                    active_forecast_artifact_id.set(Some(artifact_id.clone()));
                    status.set("Saved forecast opened.".into());
                } else {
                    forecast.set(None);
                    upper.set(None);
                    lower.set(None);
                    overrides.set(Vec::new());
                    active_forecast_artifact_id.set(None);
                    status.set("Saved dataset opened.".into());
                }
                Ok(())
            }
            .await;
            if let Err(message) = result {
                error.set(Some(message));
            }
        });
    });

    let delete_cached_artifact = Callback::new(move |artifact_id: String| {
        error.set(None);
        spawn_local(async move {
            match storage::delete_artifact(&artifact_id).await {
                Ok(result) => {
                    let deleted = result
                        .get("deleted_ids")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let active_dataset_deleted = active_dataset_artifact_id
                        .get_untracked()
                        .map(|active_id| {
                            deleted
                                .iter()
                                .any(|value| value.as_str() == Some(&active_id))
                        })
                        .unwrap_or(false);
                    let active_forecast_deleted = active_forecast_artifact_id
                        .get_untracked()
                        .map(|active_id| {
                            deleted
                                .iter()
                                .any(|value| value.as_str() == Some(&active_id))
                        })
                        .unwrap_or(false);
                    if active_dataset_deleted {
                        teardown_active.call(());
                    } else if active_forecast_deleted {
                        forecast.set(None);
                        upper.set(None);
                        lower.set(None);
                        overrides.set(Vec::new());
                        active_forecast_artifact_id.set(None);
                    }
                    status.set(format!("Deleted {} saved artifact(s).", deleted.len()));
                    reload_artifacts(cache, storage_summary, error).await;
                }
                Err(message) => error.set(Some(message)),
            }
        });
    });

    // Point adjustments are user artifacts too. Persist each settled signal
    // update so a tab refresh cannot silently return to the unadjusted values.
    create_effect(move |_| {
        let Some(forecast_artifact_id) = active_forecast_artifact_id.get() else {
            return;
        };
        let current_overrides = overrides.get();
        spawn_local(async move {
            if let Ok(mut artifact) = storage::get_artifact(&forecast_artifact_id).await {
                if let Some(data) = artifact.get_mut("data").and_then(Value::as_object_mut) {
                    data.insert("overrides".into(), json!(current_overrides));
                    let protected = active_dataset_artifact_id
                        .get_untracked()
                        .into_iter()
                        .collect::<Vec<_>>();
                    let _ = storage::put_artifact(artifact, &protected).await;
                }
            }
        });
    });

    // --- "Your data" chart: rebuilt whenever its inputs change. ----------
    create_effect(move |_| {
        let Some(h) = history.get() else {
            history_ui.svg.set(String::new());
            history_ui.geom.set(None);
            return;
        };
        let ss = sel_set.get();
        let feats = features.get();
        let gran = infer_granularity(&h.datetime);
        let dmap = date_index_map(&h.datetime);
        let enabled = features_on.get();
        let zoom = history_ui.zoom.get();

        let visible: Vec<(usize, &crate::model::SeriesData)> = h
            .series
            .iter()
            .enumerate()
            .filter(|(i, _)| ss.get(*i).copied().unwrap_or(false))
            .collect();
        let series: Vec<svg::ChartSeries<'_>> = visible
            .iter()
            .map(|(color_idx, s)| svg::ChartSeries {
                h: &s.values,
                f: &[],
                upper: None,
                lower: None,
                color_idx: *color_idx,
            })
            .collect();
        let markers: Vec<Vec<FeatureMarker>> = visible
            .iter()
            .map(|(_, s)| {
                feats
                    .as_ref()
                    .map(|f| markers_for_series(f, &s.name, &dmap, &s.values))
                    .unwrap_or_default()
            })
            .collect();
        let out = svg::line_chart_ex(
            &series,
            &h.datetime,
            &[],
            zoom,
            false,
            &markers,
            enabled,
            gran,
        );
        history_ui.set_output(out);
    });

    // --- forecast chart: history + forecast + optional band & markers. ----
    create_effect(move |_| {
        let (Some(h), Some(fc)) = (history.get(), forecast.get()) else {
            forecast_ui.svg.set(String::new());
            forecast_ui.geom.set(None);
            return;
        };
        let ss = sel_set.get();
        let ovr = overrides.get();
        let feats = features.get();
        let up = upper.get();
        let lo = lower.get();
        let bands = show_bands.get();
        let enabled = features_on.get();
        let zoom = forecast_ui.zoom.get();
        let gran = infer_granularity(&fc.datetime);
        let dmap = date_index_map(&h.datetime);

        // Owned per-series history/forecast/bound value vectors.
        struct SeriesVals {
            color_idx: usize,
            name: String,
            h: Vec<f64>,
            f: Vec<f64>,
            up: Vec<f64>,
            lo: Vec<f64>,
        }
        let mut data: Vec<SeriesVals> = Vec::new();
        for (color_idx, s) in h.series.iter().enumerate() {
            if !ss.get(color_idx).copied().unwrap_or(false) {
                continue;
            }
            let fc_idx = fc
                .index_of(&s.name)
                .unwrap_or_else(|| color_idx.min(fc.series.len().saturating_sub(1)));
            data.push(SeriesVals {
                color_idx,
                name: s.name.clone(),
                h: s.values.clone(),
                f: effective_values(&fc, &ovr, fc_idx),
                up: up
                    .as_ref()
                    .map(|bounds| effective_bound_values(&fc, bounds, &ovr, fc_idx))
                    .unwrap_or_default(),
                lo: lo
                    .as_ref()
                    .map(|bounds| effective_bound_values(&fc, bounds, &ovr, fc_idx))
                    .unwrap_or_default(),
            });
        }

        let series: Vec<svg::ChartSeries<'_>> = data
            .iter()
            .map(|d| svg::ChartSeries {
                h: &d.h,
                f: &d.f,
                upper: (!d.up.is_empty()).then_some(d.up.as_slice()),
                lower: (!d.lo.is_empty()).then_some(d.lo.as_slice()),
                color_idx: d.color_idx,
            })
            .collect();
        let markers: Vec<Vec<FeatureMarker>> = data
            .iter()
            .map(|d| {
                feats
                    .as_ref()
                    .map(|f| markers_for_series(f, &d.name, &dmap, &d.h))
                    .unwrap_or_default()
            })
            .collect();
        let out = svg::line_chart_ex(
            &series,
            &h.datetime,
            &fc.datetime,
            zoom,
            bands,
            &markers,
            enabled,
            gran,
        );
        forecast_ui.set_output(out);
    });

    // Default the forecast view to the last ~90 history points + all forecast
    // when a new forecast arrives (drag/reset takes over from there).
    create_effect(move |_| {
        if let (Some(h), Some(fc)) = (history.get(), forecast.get()) {
            let total = h.datetime.len() + fc.datetime.len();
            if total > 0 {
                let start = h.datetime.len().saturating_sub(90);
                forecast_ui.zoom.set(Some((start, total - 1)));
            }
        }
    });

    // Cached or newly generated forecasts may have fewer series than the prior
    // selection; keep the controlled adjustment picker on a valid forecast index.
    create_effect(move |_| {
        if let Some(fc) = forecast.get() {
            if adj_sel.get() >= fc.series.len() {
                adj_sel.set(0);
            }
        }
    });

    // Tooltip resolvers: absolute axis index -> (date label, value rows).
    let history_tooltip = Callback::new(move |idx: usize| {
        let h = history.get_untracked()?;
        let ss = sel_set.get_untracked();
        let selected_series: Vec<&crate::model::SeriesData> = h
            .series
            .iter()
            .enumerate()
            .filter(|(series_index, _)| ss.get(*series_index).copied().unwrap_or(false))
            .map(|(_, series)| series)
            .collect();
        let date = h
            .datetime
            .get(idx)
            .map(|d| fmt_date(d, infer_granularity(&h.datetime)))?;
        let rows: Vec<TtRow> = selected_series
            .iter()
            .filter_map(|series| {
                let color_index = h.index_of(&series.name)?;
                series
                    .values
                    .get(idx)
                    .map(|value| (color_index, series.name.clone(), *value))
            })
            .collect();
        let detected_feature_kinds = features
            .get()
            .as_ref()
            .map(|feature_data| {
                feature_kinds_at_index(feature_data, &selected_series, &h.datetime, idx)
            })
            .unwrap_or_default();
        Some((date, rows, detected_feature_kinds))
    });
    let forecast_tooltip = Callback::new(move |idx: usize| {
        let h = history.get_untracked()?;
        let fc = forecast.get_untracked()?;
        let ovr = overrides.get_untracked();
        let ss = sel_set.get_untracked();
        let nh = h.datetime.len();
        let selected_series: Vec<(usize, &crate::model::SeriesData)> = h
            .series
            .iter()
            .enumerate()
            .filter(|(series_index, _)| ss.get(*series_index).copied().unwrap_or(false))
            .collect();
        let gran = infer_granularity(&fc.datetime);
        let date = if idx < nh {
            fmt_date(h.datetime.get(idx)?, gran)
        } else {
            fmt_date(fc.datetime.get(idx - nh)?, gran)
        };
        let rows: Vec<TtRow> = selected_series
            .iter()
            .filter_map(|(color_index, series)| {
                let v = if idx < nh {
                    series.values.get(idx).copied()
                } else {
                    let fc_idx = fc.index_of(&series.name)?;
                    effective_values(&fc, &ovr, fc_idx).get(idx - nh).copied()
                }?;
                Some((*color_index, series.name.clone(), v))
            })
            .collect();
        let selected_history_series: Vec<&crate::model::SeriesData> =
            selected_series.iter().map(|(_, series)| *series).collect();
        let detected_feature_kinds = if idx < nh {
            features
                .get()
                .as_ref()
                .map(|feature_data| {
                    feature_kinds_at_index(feature_data, &selected_history_series, &h.datetime, idx)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Some((date, rows, detected_feature_kinds))
    });
    let install_pwa = move |_| {
        spawn_local(async move {
            match install_app().await {
                Ok(true) => status.set("AutoTS installed for offline use.".into()),
                Ok(false) => {}
                Err(message) => error.set(Some(format!("Install failed: {message}"))),
            }
        });
    };

    view! {
        <header class="md-appbar">
            <span class="md-emblem" inner_html=EMBLEM_SVG></span>
            <span class="brand">"AutoTS"</span>
            <span class="muted">"forecasting for everyone"</span>
            <span class="spacer"></span>
            {move || (
                offline_ready.get() && install_available.get() && !app_installed.get()
            ).then(|| view! {
                <button class="md-btn tonal" type="button" on:click=install_pwa>
                    "Install"
                </button>
            })}
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
        </header>

        <main class="md-container">
            // Status / progress
            <div aria-live="polite" aria-busy=move || job_state.get().blocks_data_loading().to_string()>
                <div class="md-status-row">
                    <div class="md-status">{move || status.get()}</div>
                    {move || job_state.get().is_forecasting().then(|| view! {
                        <button class="md-btn text error" type="button"
                            on:click=cancel_running_forecast>
                            "Cancel forecast"
                        </button>
                    })}
                </div>
                {move || (busy.get() || job_state.get().blocks_data_loading()).then(|| match parse_progress(&status.get()) {
                    Some((i, total)) if total > 0 => {
                        let pct = (i.min(total) as f64 / total as f64 * 100.0).round();
                        view! { <div class="md-progress determinate"><div style=format!("width:{pct}%")></div></div> }.into_view()
                    }
                    _ => view! { <div class="md-progress"><div></div></div> }.into_view(),
                })}
                {move || error.get().map(|e| view! { <p class="md-error">{e}</p> })}
                <details class="md-compute-status">
                    <summary>"Local compute status"</summary>
                    <dl>
                        <dt>"Worker"</dt><dd>"AutoTS Forecast Worker"</dd>
                        <dt>"State"</dt><dd>{move || format!("{:?}", job_state.get())}</dd>
                        <dt>"Offline"</dt><dd>{move || if offline_ready.get() { "ready" } else { "preparing" }}</dd>
                    </dl>
                </details>
                <span id="forecast-blocked-reason" class="sr-only">
                    "Blocked by ongoing forecast"
                </span>
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
                                    <button class="md-btn filled"
                                        disabled=move || !ready.get() || busy.get() || job_state.get().blocks_data_loading()
                                        title=move || if job_state.get().blocks_data_loading() { "Blocked by ongoing forecast" } else { "Load pasted data" }
                                        aria-describedby=move || job_state.get().blocks_data_loading().then_some("forecast-blocked-reason")
                                        on:click=load_paste>"Load pasted data"</button>
                                </div>
                            </div>
                        }.into_view(),
                        1 => view! {
                            <div class="md-field">
                                <label class="md-label">"CSV URL (e.g. a published Google Sheet)"</label>
                                <input type="url" placeholder="https://…/export?format=csv"
                                    on:input=move |ev| url_text.set(event_target_value(&ev)) />
                                <div class="md-btn-row">
                                    <button class="md-btn filled"
                                        disabled=move || !ready.get() || busy.get() || job_state.get().blocks_data_loading()
                                        title=move || if job_state.get().blocks_data_loading() { "Blocked by ongoing forecast" } else { "Load from URL" }
                                        aria-describedby=move || job_state.get().blocks_data_loading().then_some("forecast-blocked-reason")
                                        on:click=load_url>"Load from URL"</button>
                                </div>
                            </div>
                        }.into_view(),
                        2 => view! {
                            <div class="md-field">
                                <label class="md-label">"Upload a CSV or .xlsx file"</label>
                                <input type="file" accept=".csv,.tsv,.txt,.xlsx"
                                    disabled=move || !ready.get() || busy.get() || job_state.get().blocks_data_loading()
                                    title=move || if job_state.get().blocks_data_loading() { "Blocked by ongoing forecast" } else { "Upload a data file" }
                                    aria-describedby=move || job_state.get().blocks_data_loading().then_some("forecast-blocked-reason")
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
                                    <span class="md-disabled-explanation"
                                        tabindex=move || if job_state.get().blocks_data_loading() { "0" } else { "-1" }
                                        title=move || if job_state.get().blocks_data_loading() { "Blocked by ongoing forecast" } else { "Load sample" }>
                                        <button class="md-btn filled"
                                            disabled=move || !ready.get() || busy.get() || job_state.get().blocks_data_loading()
                                            aria-describedby=move || job_state.get().blocks_data_loading().then_some("forecast-blocked-reason")
                                            on:click=load_sample>"Load sample"</button>
                                    </span>
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
                                        <input type="date" max=live_end_max.clone() prop:value=move || end_date.get()
                                            on:input={
                                                let live_end_max = live_end_max.clone();
                                                move |ev| end_date.set(clamp_iso_date_max(event_target_value(&ev), &live_end_max))
                                            } />
                                    </div>
                                </div>
                                <div class="md-sources">
                                    {LIVE_SOURCES.iter().map(|src| live_source_card(src, live_enabled, live_values)).collect_view()}
                                </div>
                                <div class="md-btn-row">
                                    <button class="md-btn filled"
                                        disabled=move || !ready.get() || busy.get() || job_state.get().blocks_data_loading() || !live_any_ready()
                                        title=move || if job_state.get().blocks_data_loading() { "Blocked by ongoing forecast" } else { "Load live data" }
                                        aria-describedby=move || job_state.get().blocks_data_loading().then_some("forecast-blocked-reason")
                                        on:click=load_live.clone()>"Load live data"</button>
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
                        {chart_block(history_ui, history_tooltip)}

                        {(names.len() > 1).then(|| {
                            let names = names.clone();
                            view! {
                                <div class="md-field md-series-select">
                                    <div class="md-dropdown">
                                        <button
                                            type="button"
                                            class="md-btn outlined md-dropdown-toggle"
                                            class:open=move || series_menu_open.get()
                                            on:click=move |_| series_menu_open.update(|o| *o = !*o)
                                        >
                                            {move || {
                                                let v = sel_set.get();
                                                let on = v.iter().filter(|&&b| b).count();
                                                format!("{on} of {} series", v.len())
                                            }}
                                            <span class="md-dropdown-caret" aria-hidden="true">"▾"</span>
                                        </button>
                                        {move || series_menu_open.get().then(|| {
                                            let names = names.clone();
                                            view! {
                                                <div class="md-dropdown-panel">
                                                    <div class="md-dropdown-actions">
                                                        <button type="button" class="md-btn text"
                                                            on:click=move |_| sel_set.update(|v| v.iter_mut().for_each(|b| *b = true))>
                                                            "Select all"
                                                        </button>
                                                        <button type="button" class="md-btn text"
                                                            on:click=move |_| sel_set.update(|v| v.iter_mut().for_each(|b| *b = false))>
                                                            "Deselect all"
                                                        </button>
                                                    </div>
                                                    <div class="md-dropdown-list">
                                                        {names.into_iter().enumerate().map(|(i, n)| view! {
                                                            <label class="md-dropdown-item">
                                                                <input type="checkbox"
                                                                    prop:checked=move || sel_set.get().get(i).copied().unwrap_or(false)
                                                                    on:change=move |_| sel_set.update(|v| {
                                                                        let currently_on = v.get(i).copied().unwrap_or(false);
                                                                        let n_on = v.iter().filter(|&&x| x).count();
                                                                        if currently_on && n_on <= 1 {
                                                                            return; // keep at least one selected
                                                                        }
                                                                        if let Some(b) = v.get_mut(i) { *b = !*b; }
                                                                    }) />
                                                                <span class="md-series-chip-swatch"
                                                                    style=format!("background:var(--viz-s{})", i % 7)
                                                                    aria-hidden="true"></span>
                                                                {n}
                                                            </label>
                                                        }).collect_view()}
                                                    </div>
                                                </div>
                                            }
                                        })}
                                    </div>
                                </div>
                            }
                        })}

                        <div class="md-chart-controls" style="margin-top:8px">
                            {move || history_ui.zoom.get().is_some().then(|| view! {
                                <button type="button" class="md-btn text"
                                    on:click=move |_| history_ui.zoom.set(None)>
                                    "Reset zoom"
                                </button>
                            })}
                        </div>

                        {move || detecting_features.get().then(|| view! {
                            <p class="muted md-body">"Detecting features… The data plot is ready to use."</p>
                        })}
                        {move || feature_error.get().map(|e| view! {
                            <p class="md-warning md-body">"Feature detection was unavailable: " {e}</p>
                        })}
                        {move || features.get().map(|f| {
                            let series_name = history.get().and_then(|h| {
                                let first = sel_set.get().iter().position(|&b| b).unwrap_or(0);
                                h.series.get(first).map(|s| s.name.clone())
                            }).unwrap_or_default();
                            view! {
                                <div style="margin-top:8px">
                                    <div class="md-feat-legend">
                                        {feature_legend(&f, &series_name, features_on)}
                                    </div>
                                </div>
                            }
                        })}

                        <details class="md-expander" style="margin-top:12px">
                            <summary>"Data table"</summary>
                            {move || history.get().map(|h| {
                                let first = sel_set.get().iter().position(|&b| b).unwrap_or(0);
                                data_table(&h, first)
                            })}
                        </details>
                    </section>
                }
            })}

            // ---- Forecast controls ----
            {move || data_id.get().map(|_| view! {
                <section class="md-card">
                    {section_header("III", "Forecast")}
                    <div class="md-btn-row">
                        <button class="md-btn forecast-duration-short"
                            disabled=move || busy.get() || job_state.get() != JobState::Ready || active_dataset_artifact_id.get().is_none()
                            on:click=move |_| do_forecast("forecast_fast")>"Make forecast"</button>
                        <button class="md-btn forecast-duration-medium"
                            disabled=move || busy.get() || job_state.get() != JobState::Ready || active_dataset_artifact_id.get().is_none()
                            on:click=move |_| do_forecast("search_forecast")>"Search for best forecast"</button>
                        <button class="md-btn forecast-duration-long"
                            disabled=move || busy.get() || job_state.get() != JobState::Ready || active_dataset_artifact_id.get().is_none()
                            on:click=move |_| do_forecast("search_all_night")>"Search all night"</button>
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
                        <div class="md-field" style="max-width:220px; margin-top:12px">
                            <label class="md-label">"Prediction interval"</label>
                            <input type="number" min="0.01" max="0.99" step="0.01"
                                prop:value=move || prediction_interval.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                        prediction_interval.set(value.clamp(0.01, 0.99));
                                    }
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
                        <button
                            type="button"
                            class="md-toggle-chip"
                            class:active=move || show_bands.get()
                            disabled=move || upper.get().is_none() || lower.get().is_none()
                            on:click=move |_| show_bands.update(|b| *b = !*b)
                        >
                            <span class="md-toggle-chip-mark" aria-hidden="true"></span>
                            "Uncertainty bounds"
                        </button>
                        {move || forecast_ui.zoom.get().is_some().then(|| view! {
                            <button type="button" class="md-btn text"
                                on:click=move |_| forecast_ui.zoom.set(None)>
                                "Reset zoom"
                            </button>
                        })}
                        <span class="spacer"></span>
                        <div class="md-chart-legend" aria-label="Chart legend">
                            <span><i class="actuals"></i>"Actuals"</span>
                            <span><i class="forecast"></i>"Forecast"</span>
                        </div>
                    </div>
                    {chart_block(forecast_ui, forecast_tooltip)}

                    <details class="md-expander" style="margin-top:12px">
                        <summary>"Adjust forecast points"</summary>
                        {move || adjust_rows(forecast, overrides, adj_sel)}
                    </details>

                    <div class="md-btn-row" style="margin-top:12px">
                        <button class="md-btn filled" on:click=download_forecast>"Download forecast CSV"</button>
                        <button class="md-btn text"
                            disabled=move || !has_adjustments()
                            title="Restore the original, un-adjusted forecast"
                            on:click=reset_adjustments>
                            "Reset adjustments"
                        </button>
                    </div>
                </section>
            })}

            // ---- Data manager ----
            <section class="md-card">
                {section_header("", "Cached data & forecasts")}
                <div class="md-btn-row">
                    <button class="md-btn outlined" on:click=refresh_cache>"Refresh"</button>
                </div>
                {move || storage_summary.get().map(storage_usage_view)}
                {move || cache.get().map(|c| {
                    durable_cache_summary(
                        c,
                        active_dataset_artifact_id,
                        job_state,
                        open_cached_artifact,
                        delete_cached_artifact,
                        error,
                    )
                })}
            </section>
        </main>

        <footer class="md-footer">
            <a href="/llms.txt" target="_blank" rel="noopener">"llms.txt"</a>
        </footer>
    }
}

// ---------------------------------------------------------------------------
// View helpers
// ---------------------------------------------------------------------------

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

#[cfg(any())]
mod legacy_runtime_cache_view {
    use super::*;

    /// Which per-row actions a cached group supports, plus the `clear_cache`
    /// `cache_type` it maps to. (`list_cache` group keys are pluralised; see
    /// `CACHE_SUMMARY_KEYS` in cache.py.)
    #[derive(Clone, Copy, PartialEq)]
    enum CacheKind {
        Data,
        Prediction,
        Other,
    }

    fn group_to_cache_type(group: &str) -> (&'static str, CacheKind) {
        match group {
            "data" => ("data", CacheKind::Data),
            "predictions" => ("prediction", CacheKind::Prediction),
            "autots_models" => ("autots", CacheKind::Other),
            "feature_detectors" => ("feature_detector", CacheKind::Other),
            "event_risk" => ("event_risk", CacheKind::Other),
            _ => ("", CacheKind::Other),
        }
    }

    /// Render a prediction's fetched model parameters as a compact definition list,
    /// truncating long JSON for display (the downloaded template keeps the full text).
    fn render_model_params(v: &Value) -> View {
        let name = v
            .get("model_name")
            .and_then(Value::as_str)
            .unwrap_or("—")
            .to_string();
        let mp = serde_json::to_string_pretty(
            &v.get("model_parameters").cloned().unwrap_or(Value::Null),
        )
        .unwrap_or_default();
        let tp = serde_json::to_string_pretty(
            &v.get("transformation_parameters")
                .cloned()
                .unwrap_or(Value::Null),
        )
        .unwrap_or_default();
        view! {
            <dl class="md-cache-params">
                <dt>"Model"</dt><dd>{name}</dd>
                <dt>"Model parameters"</dt><dd>{truncate_chars(&mp, 1200)}</dd>
                <dt>"Transformation parameters"</dt><dd>{truncate_chars(&tp, 1200)}</dd>
            </dl>
        }
        .into_view()
    }

    fn cache_summary(
        cache: Value,
        cache_sig: RwSignal<Option<Value>>,
        error: RwSignal<Option<String>>,
        status: RwSignal<String>,
        active_data_id: RwSignal<Option<String>>,
        teardown_active: Callback<()>,
    ) -> View {
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
            let (cache_type, kind) = group_to_cache_type(group_name);
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

                    // Per-row interaction state.
                    let confirm = create_rw_signal(false);
                    let busy = create_rw_signal(false);
                    let params = create_rw_signal::<Option<Value>>(None);
                    let tried = create_rw_signal(false);

                    // --- Action icon buttons (vary by cache kind) ---
                    let mut actions: Vec<View> = Vec::new();
                    if kind == CacheKind::Data {
                        let did = id.clone();
                        actions.push(view! {
                            <button class="md-icon-btn" type="button"
                                title="Download CSV" aria-label="Download CSV"
                                on:click=move |_| {
                                    let did = did.clone();
                                    spawn_local(async move { download_cached_data_csv(did, error).await });
                                }>
                                <span class="md-icon-glyph" inner_html=DOWNLOAD_SVG></span>
                            </button>
                        }.into_view());
                    }
                    if kind == CacheKind::Prediction {
                        let fid = id.clone();
                        actions.push(view! {
                            <button class="md-icon-btn" type="button"
                                title="Download forecast CSV" aria-label="Download forecast CSV"
                                on:click=move |_| {
                                    let fid = fid.clone();
                                    spawn_local(async move { download_cached_forecast_csv(fid, error).await });
                                }>
                                <span class="md-icon-glyph" inner_html=DOWNLOAD_SVG></span>
                            </button>
                        }.into_view());
                        let tid = id.clone();
                        actions.push(view! {
                            <button class="md-icon-btn" type="button"
                                title="Download model template" aria-label="Download model template"
                                on:click=move |_| {
                                    let tid = tid.clone();
                                    spawn_local(async move { download_model_template(tid, error).await });
                                }>
                                <span class="md-icon-glyph" inner_html=TEMPLATE_SVG></span>
                            </button>
                        }.into_view());
                    }
                    // Delete is offered for any group with a known cache type.
                    if !cache_type.is_empty() {
                        actions.push(view! {
                            <button class="md-icon-btn danger" type="button"
                                title="Delete" aria-label="Delete"
                                disabled=move || busy.get()
                                on:click=move |_| confirm.set(true)>
                                <span class="md-icon-glyph" inner_html=TRASH_SVG></span>
                            </button>
                        }.into_view());
                    }

                    // Inline delete confirmation.
                    let confirm_id = id.clone();
                    let confirm_view = move || {
                        let id = confirm_id.clone();
                        confirm.get().then(move || {
                            let del_id = id.clone();
                            view! {
                                <div class="md-cache-confirm" role="alert">
                                    <span>"Delete this cached item? This cannot be undone."</span>
                                    <button class="md-btn text" disabled=move || busy.get()
                                        on:click=move |_| confirm.set(false)>"Cancel"</button>
                                    <button class="md-btn filled error" disabled=move || busy.get()
                                        on:click=move |_| {
                                            let id = del_id.clone();
                                            busy.set(true);
                                            error.set(None);
                                            spawn_local(async move {
                                                match call_tool(
                                                    "clear_cache",
                                                    json!({ "object_id": id, "cache_type": cache_type }),
                                                ).await {
                                                    Ok(_) => {
                                                        if active_data_id.get().as_deref() == Some(id.as_str()) {
                                                            teardown_active.call(());
                                                        }
                                                        status.set("Cached item deleted.".into());
                                                        confirm.set(false);
                                                        reload_cache(cache_sig, error).await;
                                                    }
                                                    Err(e) => error.set(Some(e)),
                                                }
                                                busy.set(false);
                                            });
                                        }>
                                        {move || if busy.get() { "Deleting…" } else { "Delete" }}
                                    </button>
                                </div>
                            }
                        })
                    };

                    // Details: metadata always; model parameters lazily for predictions.
                    let pid = id.clone();
                    let is_prediction = kind == CacheKind::Prediction;
                    view! {
                        <tr>
                            <td class="md-cache-id" title=id_title>{id.clone()}</td>
                            <td>{source}</td>
                            <td>{shape}</td>
                            <td>{created}</td>
                            <td>
                                <details class="md-expander"
                                    on:toggle=move |_| {
                                        if is_prediction && !tried.get() {
                                            tried.set(true);
                                            let pid = pid.clone();
                                            spawn_local(async move {
                                                if let Ok(v) = fetch_model_params(&pid).await {
                                                    params.set(Some(v));
                                                }
                                            });
                                        }
                                    }>
                                    <summary>"Details"</summary>
                                    {move || params.get().map(|v| render_model_params(&v))}
                                    <pre class="md-body">{metadata_text.clone()}</pre>
                                </details>
                            </td>
                            <td>
                                <div class="md-cache-actions">{actions}</div>
                                {confirm_view}
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
                                    <th>"Actions"</th>
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
}

fn adjust_rows(
    forecast: RwSignal<Option<WideData>>,
    overrides: RwSignal<Overrides>,
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
                    on:change=move |ev| {
                        if let Ok(n) = event_target_value(&ev).parse::<usize>() {
                            adj_sel.set(n);
                        }
                    }
                >
                    {names.into_iter().map(|(i, name)| view! {
                        <option
                            value=i.to_string()
                            prop:selected=move || adj_sel.get() == i
                        >
                            {name}
                        </option>
                    }).collect_view()}
                </select>
            </div>
        }
        .into_view()
    } else {
        ().into_view()
    };

    // `adj_sel` is always a forecast-series index, matching this picker.
    let idx = adj_sel.get().min(fc.series.len().saturating_sub(1));
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
        default_enabled: false,
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
        default_enabled: false,
        key_field: None,
        fields: &[LiveField {
            param: "tickers",
            label: "Tickers (comma separated)",
            default: "MSFT",
            kind: FieldKind::List,
        }],
        note: Some(
            "Unavailable in-browser: current yfinance requires curl-cffi, which has no Pyodide wheel.",
        ),
    },
    LiveSource {
        id: "earthquake",
        label: "Earthquakes (USGS)",
        default_enabled: true,
        key_field: None,
        fields: &[LiveField {
            param: "earthquake_min_magnitude",
            label: "Minimum magnitude",
            default: "5",
            kind: FieldKind::Int,
        }],
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
        default_enabled: false,
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
        default_enabled: false,
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
                default: "KC1,BX2",
                kind: FieldKind::List,
            },
            LiveField {
                param: "london_air_species",
                label: "Species",
                default: "PM25",
                kind: FieldKind::Str,
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
        ],
        note: Some(CORS_NOTE),
    },
    LiveSource {
        id: "eia",
        label: "EIA electricity demand",
        default_enabled: true,
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
        note: None,
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

fn clamp_iso_date_max(date: String, max: &str) -> String {
    if date.as_str() > max {
        max.to_string()
    } else {
        date
    }
}

/// Translate the shared calendar range into legacy per-source history units.
fn live_history_parameters(start: &str, end: &str) -> Option<(i64, i64)> {
    let history_days = date_span_days(start, end)?;
    if history_days < 0 {
        return None;
    }
    // NOAA and CAISO internally use 360-day request chunks. Round upward so
    // their pull fully covers the selected range; Python trims the final frame.
    let history_years = (history_days.max(1) + 359) / 360;
    Some((history_days, history_years))
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

    let note = src.note.map(|n| view! { <p class="muted md-body">{n}</p> });

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
