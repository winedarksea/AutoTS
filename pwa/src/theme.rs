//! Light/dark theme handling.
//!
//! The active theme is expressed as a `data-theme` attribute on `<html>` and
//! persisted in `localStorage`. CSS in `styles/md3.css` reads `data-theme` (and
//! falls back to `prefers-color-scheme` when no explicit choice is stored). A
//! tiny inline script in `index.html` applies the saved choice before WASM
//! mounts to avoid a flash of the wrong theme; this module keeps it in sync once
//! the app is running.

const STORAGE_KEY: &str = "autots-theme";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theme {
    Light,
    Dark,
}

impl Theme {
    fn as_str(self) -> &'static str {
        match self {
            Theme::Light => "light",
            Theme::Dark => "dark",
        }
    }

    fn from_str(s: &str) -> Option<Theme> {
        match s {
            "light" => Some(Theme::Light),
            "dark" => Some(Theme::Dark),
            _ => None,
        }
    }

    pub fn toggled(self) -> Theme {
        match self {
            Theme::Light => Theme::Dark,
            Theme::Dark => Theme::Light,
        }
    }
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

fn document_element() -> Option<web_sys::Element> {
    web_sys::window()?.document()?.document_element()
}

/// Override the OS-based favicon media queries after a manual theme choice.
fn apply_favicon_theme(theme: Theme) {
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return;
    };
    let (light_media, dark_media) = match theme {
        Theme::Light => ("all", "not all"),
        Theme::Dark => ("not all", "all"),
    };
    if let Some(light_icon) = document.get_element_by_id("autots-favicon-light") {
        let _ = light_icon.set_attribute("media", light_media);
    }
    if let Some(dark_icon) = document.get_element_by_id("autots-favicon-dark") {
        let _ = dark_icon.set_attribute("media", dark_media);
    }
}

/// Whether the OS currently prefers a dark color scheme.
fn os_prefers_dark() -> bool {
    web_sys::window()
        .and_then(|w| w.match_media("(prefers-color-scheme: dark)").ok().flatten())
        .map(|mql| mql.matches())
        .unwrap_or(false)
}

/// The effective theme at startup: an explicit stored choice if present,
/// otherwise whatever the OS prefers.
pub fn initial_theme() -> Theme {
    if let Some(stored) = local_storage()
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .and_then(|v| Theme::from_str(&v))
    {
        return stored;
    }
    if os_prefers_dark() {
        Theme::Dark
    } else {
        Theme::Light
    }
}

/// Apply a theme to `<html>` and persist the choice.
pub fn apply(theme: Theme) {
    if let Some(el) = document_element() {
        let _ = el.set_attribute("data-theme", theme.as_str());
    }
    apply_favicon_theme(theme);
    if let Some(storage) = local_storage() {
        let _ = storage.set_item(STORAGE_KEY, theme.as_str());
    }
}
