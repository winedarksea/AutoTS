//! Responsive session and local-compute status presentation.

use leptos::*;

use crate::job::JobState;

#[component]
pub fn SessionStatusPanel(
    status: RwSignal<String>,
    busy: RwSignal<bool>,
    job_state: RwSignal<JobState>,
    error: RwSignal<Option<String>>,
    cancel_forecast: Callback<()>,
) -> impl IntoView {
    view! {
        <aside
            class="md-session-panel"
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

            {move || error.get().map(|message| view! {
                <p class="md-error md-session-error">{message}</p>
            })}

            {move || job_state.get().is_forecasting().then(|| view! {
                <button
                    class="md-btn text error md-session-cancel"
                    type="button"
                    on:click=move |_| cancel_forecast.call(())
                >
                    "Cancel forecast"
                </button>
            })}
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
