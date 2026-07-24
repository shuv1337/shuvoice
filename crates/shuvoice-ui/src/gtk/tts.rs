//! Interactive TTS overlay GTK surface.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use gtk4::prelude::*;
use gtk4::{Align, Box as GtkBox, Button, DropDown, Label, Orientation, StringList, Window};

use super::layer::{apply_tts_layer_policy, install_css};
use crate::channel::UiEventSender;
use crate::protocol::UiEvent;
use crate::tts_overlay::{
    SPEED_UNSUPPORTED_TOOLTIP, TtsOverlayState, TtsVm, VoiceInfo, tts_overlay_css,
};

struct TtsInner {
    window: Window,
    status_label: Label,
    preview_label: Label,
    pause_btn: Button,
    slower_btn: Button,
    faster_btn: Button,
    speed_label: Label,
    voice_store: StringList,
    voice_dropdown: DropDown,
    vm: TtsVm,
    auto_hide_source: Option<glib::SourceId>,
    event_tx: UiEventSender,
    updating_voices: bool,
}

/// Interactive TTS transport / voice / speed overlay.
#[derive(Clone)]
pub struct TtsOverlay {
    inner: Rc<RefCell<TtsInner>>,
}

impl TtsOverlay {
    pub fn new(app: &gtk4::Application, vm: TtsVm, event_tx: UiEventSender) -> Self {
        install_css(tts_overlay_css());

        let window = Window::new();
        window.set_application(Some(app));
        apply_tts_layer_policy(&window, vm.layer_bottom_margin());

        let root = GtkBox::new(Orientation::Vertical, 10);
        root.add_css_class("tts-overlay-box");

        let status_label = Label::new(Some("🔈 Idle"));
        status_label.add_css_class("tts-status-label");
        status_label.set_halign(Align::Start);
        root.append(&status_label);

        let preview_label = Label::new(None);
        preview_label.add_css_class("tts-preview-label");
        preview_label.set_halign(Align::Start);
        preview_label.set_wrap(true);
        preview_label.set_max_width_chars(56);
        root.append(&preview_label);

        let transport = GtkBox::new(Orientation::Horizontal, 8);
        let pause_btn = Button::with_label("⏸ Pause");
        pause_btn.add_css_class("tts-control-btn");
        let restart_btn = Button::with_label("⟲ Restart");
        restart_btn.add_css_class("tts-control-btn");
        let stop_btn = Button::with_label("■ Stop");
        stop_btn.add_css_class("tts-control-btn");
        transport.append(&pause_btn);
        transport.append(&restart_btn);
        transport.append(&stop_btn);
        root.append(&transport);

        let settings = GtkBox::new(Orientation::Horizontal, 8);
        let speed_box = GtkBox::new(Orientation::Horizontal, 6);
        let slower_btn = Button::with_label("−");
        slower_btn.add_css_class("tts-control-btn");
        let speed_label = Label::new(None);
        speed_label.add_css_class("tts-preview-label");
        speed_label.set_valign(Align::Center);
        let faster_btn = Button::with_label("+");
        faster_btn.add_css_class("tts-control-btn");
        speed_box.append(&slower_btn);
        speed_box.append(&speed_label);
        speed_box.append(&faster_btn);
        settings.append(&speed_box);

        let voice_store = StringList::new(&["Default"]);
        let voice_dropdown = DropDown::new(Some(voice_store.clone()), gtk4::Expression::NONE);
        voice_dropdown.set_hexpand(true);
        settings.append(&voice_dropdown);
        root.append(&settings);

        window.set_child(Some(&root));
        window.set_visible(false);

        let overlay = Self {
            inner: Rc::new(RefCell::new(TtsInner {
                window,
                status_label,
                preview_label,
                pause_btn: pause_btn.clone(),
                slower_btn: slower_btn.clone(),
                faster_btn: faster_btn.clone(),
                speed_label,
                voice_store,
                voice_dropdown: voice_dropdown.clone(),
                vm,
                auto_hide_source: None,
                event_tx: event_tx.clone(),
                updating_voices: false,
            })),
        };

        {
            let this = overlay.clone();
            pause_btn.connect_clicked(move |_| {
                let state = this.inner.borrow().vm.state;
                let tx = this.inner.borrow().event_tx.clone();
                if state == TtsOverlayState::Paused {
                    let _ = tx.send(UiEvent::TtsResume);
                } else {
                    let _ = tx.send(UiEvent::TtsPause);
                }
            });
        }
        {
            let tx = event_tx.clone();
            restart_btn.connect_clicked(move |_| {
                let _ = tx.send(UiEvent::TtsRestart);
            });
        }
        {
            let this = overlay.clone();
            stop_btn.connect_clicked(move |_| {
                let _ = this.inner.borrow().event_tx.send(UiEvent::TtsStop);
                this.hide();
            });
        }
        {
            let this = overlay.clone();
            slower_btn.connect_clicked(move |_| {
                this.step_speed(-1);
            });
        }
        {
            let this = overlay.clone();
            faster_btn.connect_clicked(move |_| {
                this.step_speed(1);
            });
        }
        {
            let this = overlay.clone();
            voice_dropdown.connect_selected_notify(move |dd| {
                if this.inner.borrow().updating_voices {
                    return;
                }
                let idx = dd.selected() as usize;
                let maybe = {
                    let mut inner = this.inner.borrow_mut();
                    inner.vm.select_voice_index(idx)
                };
                if let Some(id) = maybe {
                    let _ = this
                        .inner
                        .borrow()
                        .event_tx
                        .send(UiEvent::TtsVoiceSelected { voice_id: id });
                }
            });
        }

        overlay.render();
        overlay.inner.borrow().window.present();
        overlay.inner.borrow().window.set_visible(false);
        overlay
    }

    fn render(&self) {
        let inner = self.inner.borrow();
        inner.status_label.set_text(&inner.vm.status_label());
        inner.preview_label.set_text(&inner.vm.display_preview());
        inner.pause_btn.set_label(inner.vm.pause_button_label());
        inner.speed_label.set_text(&inner.vm.speed_label());
        inner.slower_btn.set_sensitive(inner.vm.slower_enabled());
        inner.faster_btn.set_sensitive(inner.vm.faster_enabled());
        if inner.vm.speed_supported {
            inner.slower_btn.set_tooltip_text(Some(
                "Restart current speech slower using provider-native synthesis",
            ));
            inner.faster_btn.set_tooltip_text(Some(
                "Restart current speech faster using provider-native synthesis",
            ));
            inner.speed_label.set_tooltip_text(Some(
                "Speed changes apply on the next utterance, or restart the current one from the beginning",
            ));
        } else {
            inner
                .slower_btn
                .set_tooltip_text(Some(SPEED_UNSUPPORTED_TOOLTIP));
            inner
                .faster_btn
                .set_tooltip_text(Some(SPEED_UNSUPPORTED_TOOLTIP));
            inner
                .speed_label
                .set_tooltip_text(Some(SPEED_UNSUPPORTED_TOOLTIP));
        }
    }

    fn clear_auto_hide(&self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(id) = inner.auto_hide_source.take() {
            id.remove();
        }
        inner.vm.pending_auto_hide = false;
    }

    fn schedule_auto_hide(&self) {
        self.clear_auto_hide();
        let delay = self.inner.borrow().vm.auto_hide_delay_ms();
        match delay {
            None => self.hide(),
            Some(ms) => {
                let weak = Rc::downgrade(&self.inner);
                let id = glib::timeout_add_local_once(Duration::from_millis(ms), move || {
                    let Some(inner_rc) = weak.upgrade() else {
                        return;
                    };
                    let mut inner = inner_rc.borrow_mut();
                    inner.auto_hide_source = None;
                    inner.vm.hide();
                    inner.window.set_visible(false);
                });
                self.inner.borrow_mut().auto_hide_source = Some(id);
            }
        }
    }

    pub fn show(&self) {
        self.clear_auto_hide();
        let mut inner = self.inner.borrow_mut();
        inner.vm.show();
        inner.window.set_visible(true);
    }

    pub fn hide(&self) {
        self.clear_auto_hide();
        let mut inner = self.inner.borrow_mut();
        inner.vm.hide();
        inner.window.set_visible(false);
        drop(inner);
        self.render();
    }

    pub fn set_state(&self, state: TtsOverlayState, preview: &str, error: Option<&str>) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.vm.set_state(state, preview, error);
        }
        self.render();
        if state == TtsOverlayState::Idle {
            // ensure visible then schedule hide
            self.inner.borrow().window.set_visible(true);
            self.schedule_auto_hide();
        } else {
            self.show();
        }
    }

    pub fn set_speed(&self, speed: f64) {
        {
            let mut inner = self.inner.borrow_mut();
            let _ = inner.vm.apply_speed(speed, false);
        }
        self.render();
    }

    fn step_speed(&self, steps: i32) {
        let emitted = {
            let mut inner = self.inner.borrow_mut();
            inner.vm.step_speed(steps)
        };
        self.render();
        if let Some(speed) = emitted {
            let _ = self
                .inner
                .borrow()
                .event_tx
                .send(UiEvent::TtsSpeedChanged { speed });
        }
    }

    pub fn set_voices(&self, voices: Vec<VoiceInfo>, selected: Option<&str>) {
        let labels = {
            let mut inner = self.inner.borrow_mut();
            inner.updating_voices = true;
            inner.vm.set_voices(voices, selected);
            inner.vm.voice_dropdown_labels()
        };

        {
            let inner = self.inner.borrow();
            let store = &inner.voice_store;
            while store.n_items() > 0 {
                store.remove(0);
            }
            if labels.is_empty() {
                store.append("Voice: default");
            } else {
                for label in &labels {
                    store.append(label);
                }
            }
            let selected_idx = inner
                .vm
                .voices
                .iter()
                .position(|v| v.id == inner.vm.selected_voice_id)
                .unwrap_or(0) as u32;
            if inner.vm.voices.is_empty() {
                inner.voice_dropdown.set_selected(0);
                inner.voice_dropdown.set_sensitive(false);
            } else {
                inner.voice_dropdown.set_sensitive(true);
                inner.voice_dropdown.set_selected(selected_idx);
            }
        }
        self.inner.borrow_mut().updating_voices = false;
    }

    pub fn vm(&self) -> TtsVm {
        self.inner.borrow().vm.clone()
    }
}

impl Drop for TtsInner {
    fn drop(&mut self) {
        if let Some(id) = self.auto_hide_source.take() {
            id.remove();
        }
    }
}
