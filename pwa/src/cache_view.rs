//! Durable Cached View rendering and artifact-only row actions.

use leptos::*;
use serde_json::Value;
use wasm_bindgen::JsCast;

use crate::job::JobState;
use crate::model::{forecast_to_csv, model_params_to_template_csv, wide_data_to_csv, WideData};
use crate::storage;

fn download_csv(filename: &str, content: &str) -> Result<(), String> {
    let array = js_sys::Array::new();
    array.push(&wasm_bindgen::JsValue::from_str(content));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("text/csv");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&array, &options)
        .map_err(|error| format!("{error:?}"))?;
    let url =
        web_sys::Url::create_object_url_with_blob(&blob).map_err(|error| format!("{error:?}"))?;
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or("No browser document")?;
    let anchor = document
        .create_element("a")
        .map_err(|error| format!("{error:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "Anchor cast failed".to_string())?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();
    web_sys::Url::revoke_object_url(&url).ok();
    Ok(())
}

pub fn storage_usage_view(summary: Value) -> View {
    let used = summary
        .get("total_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0) as f64
        / (1024.0 * 1024.0);
    let limit = summary
        .get("byte_limit")
        .and_then(Value::as_u64)
        .unwrap_or(200 * 1024 * 1024) as f64
        / (1024.0 * 1024.0);
    let count = summary
        .get("artifact_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    view! {
        <p class="muted md-body md-storage-summary">
            {format!("{count} saved artifacts · {used:.1} MiB of {limit:.0} MiB · up to 25 forecasts and 25 standalone datasets")}
        </p>
    }
    .into_view()
}

pub fn durable_cache_summary(
    artifacts: Value,
    active_dataset_artifact_id: RwSignal<Option<String>>,
    job_state: RwSignal<JobState>,
    open_artifact: Callback<String>,
    delete_artifact: Callback<String>,
    error: RwSignal<Option<String>>,
) -> View {
    let Some(entries) = artifacts.as_array() else {
        return ().into_view();
    };
    if entries.is_empty() {
        return view! { <p class="muted md-body">"No saved artifacts."</p> }.into_view();
    }

    let all_entries = entries.clone();
    let rows = entries
        .iter()
        .map(|entry| {
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let kind = entry
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            let source = entry
                .pointer("/metadata/source")
                .and_then(Value::as_str)
                .unwrap_or("—")
                .to_string();
            let created = entry
                .get("created_at")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let size_kib = entry
                .get("payload_size")
                .and_then(Value::as_u64)
                .unwrap_or(0) as f64
                / 1024.0;
            let dependency_count = if kind == "dataset" {
                all_entries
                    .iter()
                    .filter(|candidate| {
                        candidate.get("kind").and_then(Value::as_str) == Some("forecast")
                            && candidate.get("parent_id").and_then(Value::as_str)
                                == Some(id.as_str())
                    })
                    .count()
            } else {
                0
            };
            let confirm = create_rw_signal(false);

            let open_id = id.clone();
            let download_id = id.clone();
            let template_id = id.clone();
            let delete_id = id.clone();
            let delete_title_id = delete_id.clone();
            let is_dataset = kind == "dataset";
            let is_forecast = kind == "forecast";

            view! {
                <tr>
                    <td><span class="md-chip">{kind.clone()}</span></td>
                    <td class="md-cache-id" title=id.clone()>{id.clone()}</td>
                    <td>{source}</td>
                    <td>{created}</td>
                    <td>{format!("{size_kib:.1} KiB")}</td>
                    <td>
                        <div class="md-cache-actions">
                            <button class="md-btn text" type="button"
                                disabled=move || job_state.get().blocks_data_loading()
                                title=move || if job_state.get().blocks_data_loading() {
                                    "Blocked by ongoing forecast"
                                } else {
                                    "Open saved artifact"
                                }
                                on:click=move |_| open_artifact.call(open_id.clone())>
                                "Open"
                            </button>
                            <button class="md-btn text" type="button"
                                on:click=move |_| {
                                    let artifact_id = download_id.clone();
                                    spawn_local(async move {
                                        let result: Result<(), String> = async {
                                            let artifact = storage::get_artifact(&artifact_id).await?;
                                            if is_dataset {
                                                let wide = artifact.pointer("/data/wide")
                                                    .and_then(WideData::from_json_wide)
                                                    .ok_or_else(|| "Saved dataset is invalid".to_string())?;
                                                download_csv(
                                                    &format!("autots_data_{artifact_id}.csv"),
                                                    &wide_data_to_csv(&wide),
                                                )
                                            } else {
                                                let wide = artifact.pointer("/data/forecast")
                                                    .and_then(WideData::from_json_wide)
                                                    .ok_or_else(|| "Saved forecast is invalid".to_string())?;
                                                let overrides: Vec<Vec<Option<f64>>> = artifact.pointer("/data/overrides")
                                                    .cloned()
                                                    .and_then(|value| serde_json::from_value(value).ok())
                                                    .unwrap_or_default();
                                                download_csv(
                                                    &format!("autots_forecast_{artifact_id}.csv"),
                                                    &forecast_to_csv(&wide, &overrides),
                                                )
                                            }
                                        }.await;
                                        if let Err(message) = result {
                                            error.set(Some(message));
                                        }
                                    });
                                }>
                                "Download"
                            </button>
                            {is_forecast.then(|| view! {
                                <button class="md-btn text" type="button"
                                    on:click=move |_| {
                                        let artifact_id = template_id.clone();
                                        spawn_local(async move {
                                            let result: Result<(), String> = async {
                                                let artifact = storage::get_artifact(&artifact_id).await?;
                                                let parameters = artifact.pointer("/data/model_parameters")
                                                    .cloned().unwrap_or(Value::Null);
                                                let model_name = parameters.get("model_name")
                                                    .and_then(Value::as_str).unwrap_or("Unknown");
                                                let csv = model_params_to_template_csv(
                                                    &artifact_id,
                                                    model_name,
                                                    &parameters.get("model_parameters").cloned().unwrap_or(Value::Null),
                                                    &parameters.get("transformation_parameters").cloned().unwrap_or(Value::Null),
                                                );
                                                download_csv(
                                                    &format!("autots_template_{artifact_id}.csv"),
                                                    &csv,
                                                )
                                            }.await;
                                            if let Err(message) = result {
                                                error.set(Some(message));
                                            }
                                        });
                                    }>
                                    "Template"
                                </button>
                            })}
                            <button class="md-btn text error" type="button"
                                disabled=move || {
                                    is_dataset
                                        && job_state.get().blocks_data_loading()
                                        && active_dataset_artifact_id.get().as_deref() == Some(delete_title_id.as_str())
                                }
                                title=move || {
                                    if is_dataset
                                        && job_state.get().blocks_data_loading()
                                        && active_dataset_artifact_id.get().as_deref() == Some(delete_id.as_str())
                                    {
                                        "Active forecast input cannot be deleted while forecasting"
                                    } else {
                                        "Delete saved artifact"
                                    }
                                }
                                on:click=move |_| confirm.set(true)>
                                "Delete"
                            </button>
                        </div>
                        {move || confirm.get().then(|| {
                            let confirmed_id = id.clone();
                            view! {
                                <div class="md-cache-confirm" role="alert">
                                    <span>
                                        {if dependency_count > 0 {
                                            format!("Delete this dataset and its {dependency_count} linked forecast(s)?")
                                        } else {
                                            "Delete this saved artifact? This cannot be undone.".to_string()
                                        }}
                                    </span>
                                    <button class="md-btn text" type="button"
                                        on:click=move |_| confirm.set(false)>"Cancel"</button>
                                    <button class="md-btn filled error" type="button"
                                        on:click=move |_| {
                                            confirm.set(false);
                                            delete_artifact.call(confirmed_id.clone());
                                        }>
                                        "Delete"
                                    </button>
                                </div>
                            }
                        })}
                    </td>
                </tr>
            }
        })
        .collect_view();

    view! {
        <div class="md-table-wrap md-cache-group">
            <table class="md-table md-cache-table">
                <thead>
                    <tr>
                        <th>"Type"</th>
                        <th>"ID"</th>
                        <th>"Source"</th>
                        <th>"Created"</th>
                        <th>"Size"</th>
                        <th>"Actions"</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    }
    .into_view()
}
