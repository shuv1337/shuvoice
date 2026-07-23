//! GTK caption overlay surface.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use gtk4::accessible::Property as AccProp;
use gtk4::prelude::*;
use gtk4::{AccessibleRole, Align, Box as GtkBox, Image, Label, Orientation, Window};

use super::layer::{apply_caption_layer_policy, install_css};
use crate::caption::{CaptionVm, OVERLAY_STATES, OverlayState, OverlayStateExt, caption_css};

struct CaptionInner {
    window: Window,
    box_widget: GtkBox,
    icon: Image,
    label: Label,
    debug_label: Option<Label>,
    vm: CaptionVm,
    flash_source: Option<glib::SourceId>,
}

/// Transparent STT caption overlay.
#[derive(Clone)]
pub struct CaptionOverlay {
    inner: Rc<RefCell<CaptionInner>>,
}

impl CaptionOverlay {
    pub fn new(app: &gtk4::Application, vm: CaptionVm) -> Self {
        install_css(&caption_css(&vm.style));

        let window = Window::new();
        window.set_application(Some(app));
        apply_caption_layer_policy(&window, vm.style.bottom_margin);

        let box_widget = GtkBox::new(Orientation::Horizontal, 12);
        box_widget.add_css_class("caption-box");

        let icon_size = (vm.style.font_size as f64 * 1.2) as i32;
        let icon = Image::from_icon_name("microphone-sensitivity-high-symbolic");
        icon.set_pixel_size(icon_size);
        icon.set_valign(Align::Center);
        icon.add_css_class("recording-icon");
        icon.set_tooltip_text(Some("Microphone active"));
        box_widget.append(&icon);

        let text_box = GtkBox::new(Orientation::Vertical, 6);
        let label = Label::new(None);
        label.add_css_class("caption-label");
        label.set_wrap(true);
        label.set_max_width_chars(60);
        label.set_halign(Align::Start);
        label.set_accessible_role(AccessibleRole::Status);
        label.update_property(&[AccProp::Label("")]);
        text_box.append(&label);

        let debug_label = if vm.debug_enabled {
            let dl = Label::new(None);
            dl.add_css_class("caption-debug-label");
            dl.set_wrap(true);
            dl.set_max_width_chars(80);
            dl.set_halign(Align::Start);
            dl.set_xalign(0.0);
            text_box.append(&dl);
            Some(dl)
        } else {
            None
        };

        box_widget.append(&text_box);
        window.set_child(Some(&box_widget));
        window.set_visible(false);

        let overlay = Self {
            inner: Rc::new(RefCell::new(CaptionInner {
                window,
                box_widget,
                icon,
                label,
                debug_label,
                vm,
                flash_source: None,
            })),
        };
        overlay.apply_state_visuals(OverlayState::Listening);
        overlay.inner.borrow().window.present();
        overlay.inner.borrow().window.set_visible(false);
        overlay
    }

    fn apply_state_visuals(&self, state: OverlayState) {
        let inner = self.inner.borrow();
        for s in OVERLAY_STATES {
            inner.box_widget.remove_css_class(s.css_class());
        }
        inner.box_widget.add_css_class(state.css_class());
        inner.icon.set_icon_name(Some(state.icon_name()));
        inner.icon.set_tooltip_text(Some(state.status_text()));
        inner
            .icon
            .update_property(&[AccProp::Label(state.status_text())]);
    }

    pub fn set_text(&self, text: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.set_text(text);
        inner.label.set_text(text);
        inner.label.update_property(&[AccProp::Label(text)]);
        if inner.vm.visible {
            inner.window.set_visible(true);
        }
    }

    pub fn set_debug_text(&self, text: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.set_debug_text(text);
        if let Some(dl) = &inner.debug_label {
            dl.set_text(&inner.vm.debug_text);
        }
        if inner.vm.visible {
            inner.window.set_visible(true);
        }
    }

    pub fn set_state(&self, state: OverlayState) {
        self.inner.borrow_mut().vm.set_state(state);
        self.apply_state_visuals(state);
    }

    pub fn show(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.show();
        inner.window.set_visible(true);
    }

    pub fn hide(&self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(id) = inner.flash_source.take() {
            id.remove();
        }
        inner.vm.hide();
        inner.label.set_text("");
        if let Some(dl) = &inner.debug_label {
            dl.set_text("");
        }
        inner.window.set_visible(false);
        drop(inner);
        self.apply_state_visuals(OverlayState::Listening);
    }

    pub fn flash_error(&self, text: &str, token: u64, secs: u32) {
        {
            let mut inner = self.inner.borrow_mut();
            if let Some(id) = inner.flash_source.take() {
                id.remove();
            }
            inner.vm.flash_error(text, token);
            inner.label.set_text(text);
            inner.label.update_property(&[AccProp::Label(text)]);
            inner.window.set_visible(true);
        }
        self.apply_state_visuals(OverlayState::Error);

        let weak = Rc::downgrade(&self.inner);
        let source =
            glib::timeout_add_local_once(Duration::from_secs(u64::from(secs)), move || {
                let Some(inner_rc) = weak.upgrade() else {
                    return;
                };
                // Reconstruct a handle-like clear without requiring Self.
                let should_hide = {
                    let mut inner = inner_rc.borrow_mut();
                    inner.flash_source = None;
                    inner.vm.clear_flash_if_token(token, true)
                };
                if should_hide {
                    let mut inner = inner_rc.borrow_mut();
                    inner.vm.hide();
                    inner.label.set_text("");
                    if let Some(dl) = &inner.debug_label {
                        dl.set_text("");
                    }
                    inner.window.set_visible(false);
                    for s in OVERLAY_STATES {
                        inner.box_widget.remove_css_class(s.css_class());
                    }
                    inner
                        .box_widget
                        .add_css_class(OverlayState::Listening.css_class());
                    inner
                        .icon
                        .set_icon_name(Some(OverlayState::Listening.icon_name()));
                }
            });
        self.inner.borrow_mut().flash_source = Some(source);
    }

    pub fn clear_flash_if_token(&self, token: u64, allow_hide: bool) {
        let should_hide = {
            let mut inner = self.inner.borrow_mut();
            inner.flash_source = None;
            inner.vm.clear_flash_if_token(token, allow_hide)
        };
        if should_hide {
            self.hide();
        }
    }

    pub fn vm(&self) -> CaptionVm {
        self.inner.borrow().vm.clone()
    }
}

impl Drop for CaptionInner {
    fn drop(&mut self) {
        if let Some(id) = self.flash_source.take() {
            id.remove();
        }
    }
}
