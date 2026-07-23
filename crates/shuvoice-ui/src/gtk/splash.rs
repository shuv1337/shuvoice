//! Branded splash overlay GTK surface.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use glib::ControlFlow;
use gtk4::prelude::*;
use gtk4::{Align, Box as GtkBox, Label, Orientation, Picture, ProgressBar, Window};

use super::layer::{apply_splash_layer_policy, install_css};
use crate::branding::find_logo;
use crate::splash::{SPLASH_PULSE_INTERVAL_MS, SplashVm, splash_css};

struct SplashInner {
    window: Option<Window>,
    status: Option<Label>,
    progress: Option<ProgressBar>,
    vm: SplashVm,
    pulse_source: Option<glib::SourceId>,
    shown_at: Option<Instant>,
}

/// Layer-shell splash shown while models load.
#[derive(Clone)]
pub struct SplashOverlay {
    inner: Rc<RefCell<SplashInner>>,
}

impl SplashOverlay {
    pub fn new(app: &gtk4::Application, vm: SplashVm) -> Self {
        install_css(splash_css());

        let window = Window::new();
        window.set_application(Some(app));
        window.add_css_class("splash-window");
        apply_splash_layer_policy(&window);

        let box_widget = GtkBox::new(Orientation::Vertical, 8);
        box_widget.add_css_class("splash-box");
        box_widget.set_halign(Align::Center);
        box_widget.set_valign(Align::Center);

        if let Some(logo_path) = find_logo(discover_repo_root().as_deref()) {
            let picture = Picture::for_filename(&logo_path);
            picture.set_can_shrink(true);
            picture.set_alternative_text(Some("ShuVoice logo"));
            picture.set_size_request(300, -1);
            picture.set_halign(Align::Center);
            box_widget.append(&picture);
        } else {
            let title = Label::new(Some("ShuVoice"));
            title.add_css_class("splash-title");
            title.set_halign(Align::Center);
            box_widget.append(&title);
        }

        let status = Label::new(Some(&vm.status));
        status.add_css_class("splash-status");
        status.set_halign(Align::Center);
        box_widget.append(&status);

        let progress = ProgressBar::new();
        progress.add_css_class("splash-progress");
        progress.set_show_text(true);
        progress.set_text(Some(&vm.progress_text));
        progress.set_fraction(vm.progress_fraction.unwrap_or(0.0));
        progress.set_halign(Align::Fill);
        progress.set_hexpand(true);
        box_widget.append(&progress);

        window.set_child(Some(&box_widget));

        let overlay = Self {
            inner: Rc::new(RefCell::new(SplashInner {
                window: Some(window.clone()),
                status: Some(status),
                progress: Some(progress),
                vm,
                pulse_source: None,
                shown_at: None,
            })),
        };

        {
            let this = overlay.clone();
            window.connect_realize(move |_| {
                let mut inner = this.inner.borrow_mut();
                if inner.shown_at.is_none() {
                    let now = Instant::now();
                    inner.shown_at = Some(now);
                    inner.vm.on_realize(now);
                }
            });
        }

        window.present();
        overlay
    }

    pub fn set_status(&self, text: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.set_status(text);
        if let Some(status) = &inner.status {
            status.set_text(text);
        }
    }

    pub fn set_progress(&self, fraction: Option<f64>, text: Option<&str>) {
        let need_pulse;
        {
            let mut inner = self.inner.borrow_mut();
            inner.vm.set_progress(fraction, text);
            if let Some(t) = text
                && !t.is_empty()
                && let Some(status) = &inner.status
            {
                status.set_text(t);
            }
            if let Some(progress) = &inner.progress {
                progress.set_show_text(true);
                progress.set_text(Some(&inner.vm.progress_text));
                if let Some(f) = inner.vm.progress_fraction {
                    progress.set_fraction(f);
                }
            }
            need_pulse = inner.vm.pulsing;
            if !need_pulse && let Some(id) = inner.pulse_source.take() {
                id.remove();
            }
        }
        if need_pulse {
            self.ensure_pulse();
        }
    }

    fn ensure_pulse(&self) {
        if self.inner.borrow().pulse_source.is_some() {
            return;
        }
        let this = self.clone();
        let id = glib::timeout_add_local(
            Duration::from_millis(u64::from(SPLASH_PULSE_INTERVAL_MS)),
            move || {
                let inner = this.inner.borrow();
                if !inner.vm.pulsing || inner.vm.destroyed {
                    return ControlFlow::Break;
                }
                if let Some(progress) = &inner.progress {
                    progress.pulse();
                }
                ControlFlow::Continue
            },
        );
        self.inner.borrow_mut().pulse_source = Some(id);
    }

    pub fn dismiss(self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(id) = inner.pulse_source.take() {
            id.remove();
        }
        inner.vm.dismiss();
        if let Some(window) = inner.window.take() {
            window.set_visible(false);
            window.destroy();
        }
        inner.status = None;
        inner.progress = None;
    }

    pub fn shown_instant(&self) -> Option<Instant> {
        self.inner.borrow().shown_at
    }
}

fn discover_repo_root() -> Option<std::path::PathBuf> {
    // Best-effort: walk up from CWD looking for docs/assets/branding.
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..6 {
        if dir.join("docs/assets/branding").is_dir() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

impl Drop for SplashInner {
    fn drop(&mut self) {
        if let Some(id) = self.pulse_source.take() {
            id.remove();
        }
    }
}
