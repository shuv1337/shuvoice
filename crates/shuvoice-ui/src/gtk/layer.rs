//! Shared layer-shell + CSS helpers.

use gtk4::prelude::*;
use gtk4::{CssProvider, STYLE_PROVIDER_PRIORITY_APPLICATION};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

/// Whether the compositor supports wlr-layer-shell.
pub fn layer_shell_supported() -> bool {
    gtk4_layer_shell::is_supported()
}

/// Install application CSS for the default display.
pub fn install_css(css: &str) {
    let provider = CssProvider::new();
    provider.load_from_string(css);
    if let Some(display) = gtk4::gdk::Display::default() {
        gtk4::style_context_add_provider_for_display(
            &display,
            &provider,
            STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

/// Empty input region so pointer events pass through.
pub fn make_click_through(window: &impl IsA<gtk4::Window>) {
    let window = window.as_ref();
    window.connect_realize(|window| {
        if let Some(surface) = window.surface() {
            let region = cairo::Region::create();
            surface.set_input_region(Some(&region));
        }
    });
}

fn base_layer(
    window: &impl IsA<gtk4::Window>,
    layer: Layer,
    namespace: &str,
    keyboard: KeyboardMode,
) {
    if !layer_shell_supported() {
        tracing::error!("Layer shell not supported — not on a wlroots compositor?");
        return;
    }
    let window = window.as_ref();
    window.init_layer_shell();
    window.set_layer(layer);
    window.set_keyboard_mode(keyboard);
    window.set_exclusive_zone(-1);
    window.set_namespace(Some(namespace));
}

/// STT caption: OVERLAY, keyboard NONE, bottom anchor.
pub fn apply_caption_layer_policy(window: &impl IsA<gtk4::Window>, bottom_margin: i32) {
    base_layer(
        window,
        Layer::Overlay,
        crate::CAPTION_NAMESPACE,
        KeyboardMode::None,
    );
    if layer_shell_supported() {
        let window = window.as_ref();
        window.set_anchor(Edge::Bottom, true);
        window.set_margin(Edge::Bottom, bottom_margin);
    }
    make_click_through(window);
}

/// TTS: OVERLAY, keyboard ON_DEMAND, bottom anchor + offset margin.
pub fn apply_tts_layer_policy(window: &impl IsA<gtk4::Window>, bottom_margin: i32) {
    base_layer(
        window,
        Layer::Overlay,
        crate::TTS_NAMESPACE,
        KeyboardMode::OnDemand,
    );
    if layer_shell_supported() {
        let window = window.as_ref();
        window.set_anchor(Edge::Bottom, true);
        window.set_margin(Edge::Bottom, bottom_margin);
    }
}

/// Splash: OVERLAY, keyboard NONE, no anchors (centered), click-through.
pub fn apply_splash_layer_policy(window: &impl IsA<gtk4::Window>) {
    base_layer(
        window,
        Layer::Overlay,
        crate::SPLASH_NAMESPACE,
        KeyboardMode::None,
    );
    make_click_through(window);
}

/// Wizard: TOP, keyboard ON_DEMAND.
pub fn apply_wizard_layer_policy(window: &impl IsA<gtk4::Window>) {
    base_layer(
        window,
        Layer::Top,
        crate::WIZARD_NAMESPACE,
        KeyboardMode::OnDemand,
    );
}

/// Release wizard keyboard grab before destroy.
pub fn release_wizard_keyboard(window: &impl IsA<gtk4::Window>) {
    if layer_shell_supported() {
        window.as_ref().set_keyboard_mode(KeyboardMode::None);
    }
}
