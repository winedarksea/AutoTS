mod app;
mod client;
mod model;
mod svg;
mod theme;

use leptos::*;

fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(app::App);
}
