//! Responsive session and local-compute status presentation.

use leptos::*;

use crate::job::JobState;

#[component]
pub fn SessionStatusPanel(
    status: RwSignal<String>,
    busy: RwSignal<bool>,
    job_state: RwSignal<JobState>,
    offline_ready: RwSignal<bool>,
    error: RwSignal<Option<String>>,
    cancel_forecast: Callback<()>,
) -> impl IntoView {
    let expanded = create_rw_signal(false);
    let cancellation_requested = create_rw_signal(false);

    create_effect(move |_| {
        let state = job_state.get();
        if error.get().is_some() {
            expanded.set(true);
            cancellation_requested.set(false);
        } else if cancellation_requested.get() && state == JobState::Ready {
            expanded.set(false);
            cancellation_requested.set(false);
        }
    });

    let cancel = move |_| {
        cancellation_requested.set(true);
        cancel_forecast.call(());
    };

    view! {
        <aside
            class="md-session-panel"
            class:expanded=move || expanded.get()
            aria-live="polite"
            aria-busy=move || job_state.get().blocks_data_loading().to_string()
            aria-label="Session status"
        >
            <div class="md-session-summary">
                <div class="md-session-heading">
                    <span class="md-session-title">"Session"</span>
                    <span
                        class=move || format!(
                            "md-session-state-indicator {}",
                            job_state.get().status_role_class()
                        )
                        aria-hidden="true"
                    ></span>
                    <span class="md-session-state">
                        {move || job_state.get().display_label()}
                    </span>
                </div>
                <button
                    class="md-session-toggle"
                    type="button"
                    aria-expanded=move || expanded.get().to_string()
                    aria-controls="session-status-details"
                    aria-label=move || if expanded.get() {
                        "Collapse session details"
                    } else {
                        "Expand session details"
                    }
                    on:click=move |_| expanded.update(|value| *value = !*value)
                >
                    <span class="md-session-toggle-glyph" aria-hidden="true">"⌃"</span>
                </button>
            </div>

            <p class="md-session-activity">{move || status.get()}</p>

            {move || (busy.get() || job_state.get().blocks_data_loading()).then(|| {
                match parse_progress(&status.get()) {
                    Some((current, total)) if total > 0 => {
                        let percent =
                            (current.min(total) as f64 / total as f64 * 100.0).round();
                        view! {
                            <div
                                class="md-progress determinate"
                                role="progressbar"
                                aria-label="Session progress"
                                aria-valuemin="0"
                                aria-valuemax="100"
                                aria-valuenow=percent.to_string()
                            >
                                <div style=format!("width:{percent}%")></div>
                            </div>
                        }
                        .into_view()
                    }
                    _ => view! {
                        <div class="md-progress" role="progressbar" aria-label="Session progress">
                            <div></div>
                        </div>
                    }
                    .into_view(),
                }
            })}

            <div id="session-status-details" class="md-session-details">
                {move || error.get().map(|message| view! {
                    <p class="md-error md-session-error">{message}</p>
                })}

                <dl class="md-session-facts">
                    <dt>"Worker"</dt>
                    <dd>"AutoTS Forecast Worker"</dd>
                    <dt>"State"</dt>
                    <dd>{move || job_state.get().display_label()}</dd>
                    <dt>"Offline"</dt>
                    <dd>{move || if offline_ready.get() { "ready" } else { "preparing" }}</dd>
                </dl>

                {move || job_state.get().is_forecasting().then(|| view! {
                    <button class="md-btn text error md-session-cancel" type="button" on:click=cancel>
                        "Cancel forecast"
                    </button>
                })}
            </div>
        </aside>
    }
}

/// Parse a trailing "(current/total)" progress marker.
fn parse_progress(message: &str) -> Option<(usize, usize)> {
    let open = message.rfind('(')?;
    let remainder = &message[open + 1..];
    let close = remainder.find(')')?;
    let (current, total) = remainder[..close].split_once('/')?;
    Some((current.trim().parse().ok()?, total.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::parse_progress;

    #[test]
    fn parses_trailing_progress_marker() {
        assert_eq!(parse_progress("Forecasting model (3/12)"), Some((3, 12)));
        assert_eq!(parse_progress("Forecasting"), None);
        assert_eq!(parse_progress("Forecasting (bad/12)"), None);
    }
}
