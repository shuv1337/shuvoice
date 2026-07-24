//! Welcome wizard GTK application shell (UX layer over headless [`WizardVm`]).

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use glib::ControlFlow;
use gtk4::accessible::Property as AccProp;
use gtk4::prelude::*;
use gtk4::{
    Align, Application, ApplicationWindow, Box as GtkBox, Button, CheckButton, DropDown, Entry,
    Justification, Label, Orientation, Picture, ProgressBar, Stack, StackTransitionType,
};

use super::layer::{apply_wizard_layer_policy, install_css, release_wizard_keyboard};
use super::wizard_helpers::{
    TtsControlVisibility, keybind_auto_add_sensitive, keybind_index, sherpa_controls_visible,
    sherpa_profile_index, should_show_done_back, speed_preset_index, tts_backend_index,
};
use crate::branding::find_logo;
use crate::channel::{UiCmdReceiver, UiEventSender};
use crate::protocol::{UiCmd, UiEvent};
use crate::wizard::{
    ASR_BACKENDS, DEFAULT_KOKORO_TTS_BASE_URL, FINAL_INJECTION_MODES, KEYBIND_PRESETS,
    SHERPA_PROFILE_OPTIONS, TTS_BACKENDS, TTS_PLAYBACK_SPEED_PRESET_IDS, TYPING_TEXT_CASE_MODES,
    WIZARD_APPLICATION_ID, WizardPageId, WizardVm, wizard_css,
};

/// Held GTK widgets that need later sync / visibility toggles.
struct WizardWidgets {
    window: Option<ApplicationWindow>,
    stack: Stack,
    // ASR
    sherpa_profile_title: Label,
    sherpa_profile_dd: DropDown,
    sherpa_profile_desc: Label,
    sherpa_provider_title: Label,
    sherpa_provider_dd: DropDown,
    sherpa_provider_desc: Label,
    // Keybind
    keybind_dd: DropDown,
    auto_add_keybind: CheckButton,
    // TTS
    tts_provider_dd: DropDown,
    tts_voice_label: Label,
    tts_voice_entry: Entry,
    tts_voice_help: Label,
    kokoro_url_label: Label,
    kokoro_url_entry: Entry,
    kokoro_url_help: Label,
    local_setup_title: Label,
    local_setup_dd: DropDown,
    local_setup_desc: Label,
    local_path_label: Label,
    local_path_entry: Entry,
    local_path_help: Label,
    local_auto_voice_title: Label,
    local_auto_voice_dd: DropDown,
    local_auto_voice_desc: Label,
    melotts_device_title: Label,
    melotts_device_dd: DropDown,
    melotts_device_desc: Label,
    speed_dd: DropDown,
    tts_error: Label,
    // Done
    summary_label: Label,
    finish_status: Label,
    download_progress: ProgressBar,
    download_note: Label,
    cancel_btn: Button,
    back_from_done_btn: Button,
    retry_btn: Button,
    launch_btn: Button,
}

struct WizardInner {
    vm: WizardVm,
    widgets: Option<WizardWidgets>,
    event_tx: Option<UiEventSender>,
    download_pulse: Option<glib::SourceId>,
    cmd_pump: Option<glib::SourceId>,
    /// Suppress notify loops while programmatically selecting dropdowns.
    syncing: bool,
    closed_emitted: bool,
}

/// First-run guided setup wizard (`Gtk.Application`).
#[derive(Clone)]
pub struct WelcomeWizard {
    app: Application,
    inner: Rc<RefCell<WizardInner>>,
}

struct TtsPage {
    page: GtkBox,
    provider_dd: DropDown,
    voice_label: Label,
    voice_entry: Entry,
    voice_help: Label,
    kokoro_url_label: Label,
    kokoro_url_entry: Entry,
    kokoro_url_help: Label,
    local_setup_title: Label,
    local_setup_dd: DropDown,
    local_setup_desc: Label,
    local_path_label: Label,
    local_path_entry: Entry,
    local_path_help: Label,
    local_auto_voice_title: Label,
    local_auto_voice_dd: DropDown,
    local_auto_voice_desc: Label,
    melotts_device_title: Label,
    melotts_device_dd: DropDown,
    melotts_device_desc: Label,
    speed_dd: DropDown,
    error_label: Label,
}

struct DonePage {
    page: GtkBox,
    summary_label: Label,
    finish_status: Label,
    download_progress: ProgressBar,
    download_note: Label,
    cancel_btn: Button,
    back_btn: Button,
    retry_btn: Button,
    launch_btn: Button,
}

impl WelcomeWizard {
    pub fn new(force_reconfigure: bool) -> Self {
        let app = Application::builder()
            .application_id(WIZARD_APPLICATION_ID)
            .build();
        let inner = Rc::new(RefCell::new(WizardInner {
            vm: WizardVm::new(force_reconfigure),
            widgets: None,
            event_tx: None,
            download_pulse: None,
            cmd_pump: None,
            syncing: false,
            closed_emitted: false,
        }));
        let wizard = Self { app, inner };

        {
            let wizard_clone = wizard.clone();
            wizard.app.connect_activate(move |app| {
                wizard_clone.build_ui(app);
            });
        }
        {
            let wizard_clone = wizard.clone();
            wizard.app.connect_shutdown(move |_| {
                wizard_clone.on_shutdown();
            });
        }

        wizard
    }

    pub fn set_event_sender(&self, tx: UiEventSender) {
        self.inner.borrow_mut().event_tx = Some(tx);
    }

    pub fn application(&self) -> Application {
        self.app.clone()
    }

    pub fn completed(&self) -> bool {
        self.inner.borrow().vm.completed
    }

    pub fn vm(&self) -> WizardVm {
        self.inner.borrow().vm.clone()
    }

    pub fn run(&self) -> i32 {
        i32::from(self.app.run_with_args::<&str>(&[]))
    }

    fn emit(&self, event: UiEvent) {
        if let Some(tx) = &self.inner.borrow().event_tx {
            let _ = tx.send(event);
        }
    }

    fn emit_closed_once(&self, completed: bool) {
        let mut inner = self.inner.borrow_mut();
        if inner.closed_emitted {
            return;
        }
        inner.closed_emitted = true;
        drop(inner);
        self.emit(UiEvent::WizardClosed { completed });
    }

    fn build_ui(&self, app: &Application) {
        install_css(wizard_css());

        let window = ApplicationWindow::new(app);
        apply_wizard_layer_policy(&window);
        window.add_css_class("wizard-window");
        window.set_title(Some("ShuVoice setup"));
        window.update_property(&[AccProp::Label("ShuVoice setup wizard")]);

        {
            let this = self.clone();
            window.connect_close_request(move |_| {
                // WM close / Esc path — not a successful launch.
                this.emit_closed_once(false);
                this.app.quit();
                glib::Propagation::Stop
            });
        }

        let stack = Stack::new();
        stack.set_transition_type(StackTransitionType::SlideLeftRight);
        stack.set_transition_duration(200);
        stack.update_property(&[AccProp::Label("Setup steps")]);

        stack.add_named(&self.build_welcome_page(), Some("welcome"));

        let (
            asr_page,
            sherpa_profile_title,
            sherpa_profile_dd,
            sherpa_profile_desc,
            sherpa_provider_title,
            sherpa_provider_dd,
            sherpa_provider_desc,
        ) = self.build_asr_page();
        stack.add_named(&asr_page, Some("asr"));

        let (keybind_page, keybind_dd, auto_add_keybind) = self.build_keybind_page();
        stack.add_named(&keybind_page, Some("keybind"));

        let tts = self.build_tts_page();
        stack.add_named(&tts.page, Some("tts"));

        let done = self.build_done_page();
        stack.add_named(&done.page, Some("done"));

        window.set_child(Some(&stack));

        self.inner.borrow_mut().widgets = Some(WizardWidgets {
            window: Some(window.clone()),
            stack,
            sherpa_profile_title,
            sherpa_profile_dd,
            sherpa_profile_desc,
            sherpa_provider_title,
            sherpa_provider_dd,
            sherpa_provider_desc,
            keybind_dd,
            auto_add_keybind,
            tts_provider_dd: tts.provider_dd,
            tts_voice_label: tts.voice_label,
            tts_voice_entry: tts.voice_entry,
            tts_voice_help: tts.voice_help,
            kokoro_url_label: tts.kokoro_url_label,
            kokoro_url_entry: tts.kokoro_url_entry,
            kokoro_url_help: tts.kokoro_url_help,
            local_setup_title: tts.local_setup_title,
            local_setup_dd: tts.local_setup_dd,
            local_setup_desc: tts.local_setup_desc,
            local_path_label: tts.local_path_label,
            local_path_entry: tts.local_path_entry,
            local_path_help: tts.local_path_help,
            local_auto_voice_title: tts.local_auto_voice_title,
            local_auto_voice_dd: tts.local_auto_voice_dd,
            local_auto_voice_desc: tts.local_auto_voice_desc,
            melotts_device_title: tts.melotts_device_title,
            melotts_device_dd: tts.melotts_device_dd,
            melotts_device_desc: tts.melotts_device_desc,
            speed_dd: tts.speed_dd,
            tts_error: tts.error_label,
            summary_label: done.summary_label,
            finish_status: done.finish_status,
            download_progress: done.download_progress,
            download_note: done.download_note,
            cancel_btn: done.cancel_btn,
            back_from_done_btn: done.back_btn,
            retry_btn: done.retry_btn,
            launch_btn: done.launch_btn,
        });

        // Apply initial visibility / selection from VM (before user input).
        self.sync_sherpa_controls();
        self.sync_auto_add_checkbox();
        self.sync_tts_controls();
        self.sync_done_actions();

        window.present();
    }

    // -- Pages ----------------------------------------------------------------

    fn build_welcome_page(&self) -> GtkBox {
        let page = GtkBox::new(Orientation::Vertical, 4);
        page.add_css_class("wizard-page");
        page.set_halign(Align::Center);
        page.set_valign(Align::Center);

        if let Some(logo) = find_logo(discover_repo_root().as_deref()) {
            let picture = Picture::for_filename(logo);
            picture.set_can_shrink(true);
            picture.set_alternative_text(Some("ShuVoice logo"));
            picture.set_size_request(320, -1);
            picture.set_halign(Align::Center);
            picture.set_margin_bottom(12);
            page.append(&picture);
        } else {
            page.append(&title_label("ShuVoice"));
        }

        let sub = Label::new(Some("Streaming speech-to-text for Hyprland"));
        sub.add_css_class("wizard-subtitle");
        sub.set_halign(Align::Center);
        page.append(&sub);

        let desc = Label::new(Some(
            "Let’s set up a few things before you start.\nThis will only take a moment.",
        ));
        desc.add_css_class("wizard-desc");
        desc.set_halign(Align::Center);
        desc.set_justify(Justification::Center);
        desc.set_margin_bottom(24);
        page.append(&desc);

        let btn = primary_button("Get Started");
        btn.update_property(&[AccProp::Label("Get Started")]);
        btn.set_tooltip_text(Some("Continue to speech recognition setup"));
        let this = self.clone();
        btn.connect_clicked(move |_| this.navigate(WizardPageId::Asr));
        btn.set_halign(Align::Center);
        page.append(&btn);
        page
    }

    fn build_asr_page(&self) -> (GtkBox, Label, DropDown, Label, Label, DropDown, Label) {
        let page = GtkBox::new(Orientation::Vertical, 4);
        page.add_css_class("wizard-page");
        page.set_halign(Align::Center);
        page.set_valign(Align::Center);
        page.append(&title_label("Speech Recognition Engine"));

        let sub = Label::new(Some("Choose the ASR backend that fits your hardware."));
        sub.add_css_class("wizard-subtitle");
        page.append(&sub);

        let mut group: Option<CheckButton> = None;
        for (id, label, desc) in ASR_BACKENDS {
            let radio = CheckButton::with_label(label);
            radio.add_css_class("wizard-radio");
            radio.set_tooltip_text(Some(*desc));
            set_accessible_description(&radio, desc);
            if let Some(g) = &group {
                radio.set_group(Some(g));
            } else {
                group = Some(radio.clone());
            }
            if *id == self.inner.borrow().vm.asr_backend {
                radio.set_active(true);
            }
            let this = self.clone();
            let backend = (*id).to_string();
            radio.connect_toggled(move |btn| {
                if btn.is_active() {
                    this.inner.borrow_mut().vm.set_asr_backend(backend.clone());
                    this.sync_sherpa_controls();
                }
            });
            page.append(&radio);
            let d = Label::new(Some(desc));
            d.add_css_class("wizard-radio-desc");
            d.set_halign(Align::Start);
            d.set_wrap(true);
            page.append(&d);
        }

        let profile_labels: Vec<&str> = SHERPA_PROFILE_OPTIONS.iter().map(|(_, l, _)| *l).collect();
        let profile_dd = DropDown::from_strings(&profile_labels);
        profile_dd.add_css_class("wizard-dropdown");
        profile_dd.set_selected(sherpa_profile_index(
            self.inner.borrow().vm.sherpa_profile_id(),
        ));
        profile_dd.update_property(&[AccProp::Label("Sherpa profile")]);
        let profile_desc = Label::new(Some(
            SHERPA_PROFILE_OPTIONS
                [sherpa_profile_index(self.inner.borrow().vm.sherpa_profile_id()) as usize]
                .2,
        ));
        profile_desc.add_css_class("wizard-radio-desc");
        profile_desc.set_halign(Align::Start);
        profile_desc.set_wrap(true);
        {
            let this = self.clone();
            let desc_label = profile_desc.clone();
            profile_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let idx = dd.selected() as usize;
                if let Some((id, _l, d)) = SHERPA_PROFILE_OPTIONS.get(idx) {
                    this.inner.borrow_mut().vm.set_sherpa_profile(id);
                    desc_label.set_text(d);
                    dd.set_tooltip_text(Some(*d));
                    set_accessible_description(dd, d);
                }
            });
        }
        let profile_title = Label::new(Some("Sherpa profile"));
        profile_title.add_css_class("wizard-subtitle");
        profile_title.set_halign(Align::Start);
        profile_title.set_margin_top(8);
        page.append(&profile_title);
        page.append(&profile_dd);
        page.append(&profile_desc);

        let provider_labels = ["CPU", "GPU (CUDA)"];
        let provider_dd = DropDown::from_strings(&provider_labels);
        provider_dd.add_css_class("wizard-dropdown");
        provider_dd.set_selected(if self.inner.borrow().vm.sherpa_provider == "cuda" {
            1
        } else {
            0
        });
        provider_dd.update_property(&[AccProp::Label("Compute device")]);
        let provider_desc = Label::new(Some(
            "CPU is the reliable default for Parakeet instant mode.",
        ));
        provider_desc.add_css_class("wizard-radio-desc");
        provider_desc.set_halign(Align::Start);
        provider_desc.set_wrap(true);
        {
            let this = self.clone();
            let desc_label = provider_desc.clone();
            provider_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let provider = if dd.selected() == 1 { "cuda" } else { "cpu" };
                this.inner.borrow_mut().vm.sherpa_provider = provider.into();
                let d = if provider == "cuda" {
                    "Higher throughput when CUDA runtime is available."
                } else {
                    "CPU is the reliable default for Parakeet instant mode."
                };
                desc_label.set_text(d);
                dd.set_tooltip_text(Some(d));
            });
        }
        let provider_title = Label::new(Some("Compute device"));
        provider_title.add_css_class("wizard-subtitle");
        provider_title.set_halign(Align::Start);
        provider_title.set_margin_top(8);
        page.append(&provider_title);
        page.append(&provider_dd);
        page.append(&provider_desc);

        page.append(&self.nav_row(
            Some(WizardPageId::Welcome),
            Some(WizardPageId::Keybind),
            "Next",
        ));

        (
            page,
            profile_title,
            profile_dd,
            profile_desc,
            provider_title,
            provider_dd,
            provider_desc,
        )
    }

    fn build_keybind_page(&self) -> (GtkBox, DropDown, CheckButton) {
        let page = GtkBox::new(Orientation::Vertical, 4);
        page.add_css_class("wizard-page");
        page.set_halign(Align::Center);
        page.set_valign(Align::Center);
        page.append(&title_label("Push-to-Talk & Text Input"));

        let labels: Vec<&str> = KEYBIND_PRESETS.iter().map(|p| p.label).collect();
        let dd = DropDown::from_strings(&labels);
        dd.add_css_class("wizard-dropdown");
        dd.set_selected(keybind_index(&self.inner.borrow().vm.keybind));
        dd.update_property(&[AccProp::Label("Push-to-talk keybind")]);
        {
            let this = self.clone();
            dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some(p) = KEYBIND_PRESETS.get(i) {
                    this.inner.borrow_mut().vm.set_keybind(p.id);
                    dd.set_tooltip_text(Some(p.description));
                    set_accessible_description(dd, p.description);
                    this.sync_auto_add_checkbox();
                }
            });
        }
        let kt = Label::new(Some("Push-to-talk keybind"));
        kt.add_css_class("wizard-subtitle");
        kt.set_halign(Align::Start);
        page.append(&kt);
        page.append(&dd);

        let auto =
            CheckButton::with_label("Try to add this keybind to Hyprland config automatically");
        auto.add_css_class("wizard-radio");
        auto.set_active(self.inner.borrow().vm.auto_add_keybind);
        auto.set_tooltip_text(Some(
            "Only applies when the selected key is not already used by another bind.",
        ));
        set_accessible_description(
            &auto,
            "Only applies when the selected key is not already used by another bind.",
        );
        {
            let this = self.clone();
            auto.connect_toggled(move |btn| {
                if this.inner.borrow().syncing {
                    return;
                }
                let mut inner = this.inner.borrow_mut();
                if btn.is_sensitive() {
                    inner.vm.auto_add_keybind = btn.is_active();
                    inner.vm.auto_add_last_non_custom = btn.is_active();
                }
            });
        }
        page.append(&auto);

        let inj_labels: Vec<&str> = FINAL_INJECTION_MODES.iter().map(|(_, l, _)| *l).collect();
        let inj_dd = DropDown::from_strings(&inj_labels);
        inj_dd.add_css_class("wizard-dropdown");
        let inj_idx = FINAL_INJECTION_MODES
            .iter()
            .position(|(id, ..)| *id == self.inner.borrow().vm.typing_final_injection_mode)
            .unwrap_or(0) as u32;
        inj_dd.set_selected(inj_idx);
        inj_dd.update_property(&[AccProp::Label("Final text injection")]);
        {
            let this = self.clone();
            inj_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _l, d)) = FINAL_INJECTION_MODES.get(i) {
                    this.inner.borrow_mut().vm.typing_final_injection_mode = (*id).into();
                    dd.set_tooltip_text(Some(*d));
                    set_accessible_description(dd, d);
                }
            });
        }
        let it = Label::new(Some("Final text injection"));
        it.add_css_class("wizard-subtitle");
        it.set_halign(Align::Start);
        it.set_margin_top(12);
        page.append(&it);
        page.append(&inj_dd);

        let case_labels: Vec<&str> = TYPING_TEXT_CASE_MODES.iter().map(|(_, l, _)| *l).collect();
        let case_dd = DropDown::from_strings(&case_labels);
        case_dd.add_css_class("wizard-dropdown");
        let case_idx = TYPING_TEXT_CASE_MODES
            .iter()
            .position(|(id, ..)| *id == self.inner.borrow().vm.typing_text_case)
            .unwrap_or(0) as u32;
        case_dd.set_selected(case_idx);
        case_dd.update_property(&[AccProp::Label("Text case")]);
        {
            let this = self.clone();
            case_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _l, d)) = TYPING_TEXT_CASE_MODES.get(i) {
                    this.inner.borrow_mut().vm.typing_text_case = (*id).into();
                    dd.set_tooltip_text(Some(*d));
                    set_accessible_description(dd, d);
                }
            });
        }
        let ct = Label::new(Some("Text case"));
        ct.add_css_class("wizard-subtitle");
        ct.set_halign(Align::Start);
        ct.set_margin_top(12);
        page.append(&ct);
        page.append(&case_dd);

        page.append(&self.nav_row(Some(WizardPageId::Asr), Some(WizardPageId::Tts), "Next"));
        (page, dd, auto)
    }

    fn build_tts_page(&self) -> TtsPage {
        let page = GtkBox::new(Orientation::Vertical, 4);
        page.add_css_class("wizard-page");
        page.set_halign(Align::Center);
        page.set_valign(Align::Center);
        page.append(&title_label("Text-to-Speech"));

        let sub = Label::new(Some(
            "Choose the TTS provider and default voice for read-aloud shortcuts.",
        ));
        sub.add_css_class("wizard-subtitle");
        page.append(&sub);

        let labels: Vec<&str> = TTS_BACKENDS.iter().map(|(_, l, _)| *l).collect();
        let provider_dd = DropDown::from_strings(&labels);
        provider_dd.add_css_class("wizard-dropdown");
        provider_dd.set_selected(tts_backend_index(&self.inner.borrow().vm.tts_backend));
        provider_dd.update_property(&[AccProp::Label("TTS provider")]);
        {
            let this = self.clone();
            provider_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _l, d)) = TTS_BACKENDS.get(i) {
                    this.inner.borrow_mut().vm.set_tts_backend(*id);
                    // Voice entry follows backend default.
                    let voice = this.inner.borrow().vm.tts_voice_id.clone();
                    if let Some(w) = &this.inner.borrow().widgets {
                        w.tts_voice_entry.set_text(&voice);
                    }
                    dd.set_tooltip_text(Some(*d));
                    set_accessible_description(dd, d);
                    this.sync_tts_controls();
                }
            });
        }
        let pt = Label::new(Some("TTS provider"));
        pt.add_css_class("wizard-subtitle");
        pt.set_halign(Align::Start);
        page.append(&pt);
        page.append(&provider_dd);

        // Local setup mode
        let local_setup_options = [
            (
                "automatic",
                "Automatic setup (recommended)",
                "Install Piper if needed and download a curated voice into the managed directory.",
            ),
            (
                "manual",
                "Use existing local path",
                "Point ShuVoice at a Piper .onnx model file or a directory of .onnx voices.",
            ),
        ];
        let local_setup_labels: Vec<&str> =
            local_setup_options.iter().map(|(_, l, _)| *l).collect();
        let local_setup_dd = DropDown::from_strings(&local_setup_labels);
        local_setup_dd.add_css_class("wizard-dropdown");
        local_setup_dd.set_selected(if self.inner.borrow().vm.tts_local_setup_mode == "manual" {
            1
        } else {
            0
        });
        local_setup_dd.update_property(&[AccProp::Label("Local Piper setup mode")]);
        let local_setup_desc = Label::new(Some(local_setup_options[0].2));
        local_setup_desc.add_css_class("wizard-radio-desc");
        local_setup_desc.set_halign(Align::Start);
        local_setup_desc.set_wrap(true);
        {
            let this = self.clone();
            let desc = local_setup_desc.clone();
            local_setup_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _l, d)) = local_setup_options.get(i) {
                    this.inner.borrow_mut().vm.tts_local_setup_mode = (*id).into();
                    desc.set_text(d);
                    this.sync_tts_controls();
                }
            });
        }
        let local_setup_title = Label::new(Some("Setup mode"));
        local_setup_title.add_css_class("wizard-subtitle");
        local_setup_title.set_halign(Align::Start);
        local_setup_title.set_margin_top(8);
        page.append(&local_setup_title);
        page.append(&local_setup_dd);
        page.append(&local_setup_desc);

        // Curated local voice (simple fixed list matching common Piper stems)
        let curated = [
            ("en_US-amy-medium", "Amy (US, medium)"),
            ("en_US-lessac-medium", "Lessac (US, medium)"),
            ("en_GB-alba-medium", "Alba (GB, medium)"),
        ];
        let curated_labels: Vec<&str> = curated.iter().map(|(_, l)| *l).collect();
        let local_auto_voice_dd = DropDown::from_strings(&curated_labels);
        local_auto_voice_dd.add_css_class("wizard-dropdown");
        let auto_idx = curated
            .iter()
            .position(|(id, _)| *id == self.inner.borrow().vm.tts_local_auto_voice_id)
            .unwrap_or(0) as u32;
        local_auto_voice_dd.set_selected(auto_idx);
        local_auto_voice_dd.update_property(&[AccProp::Label("Curated Local Piper voice")]);
        {
            let this = self.clone();
            local_auto_voice_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _)) = curated.get(i) {
                    this.inner.borrow_mut().vm.tts_local_auto_voice_id = (*id).into();
                }
            });
        }
        let local_auto_voice_title = Label::new(Some("Curated voice"));
        local_auto_voice_title.add_css_class("wizard-subtitle");
        local_auto_voice_title.set_halign(Align::Start);
        local_auto_voice_title.set_margin_top(8);
        let local_auto_voice_desc = Label::new(Some(
            "Wizard finish will request automatic Piper setup for this voice (when automation is available).",
        ));
        local_auto_voice_desc.add_css_class("wizard-radio-desc");
        local_auto_voice_desc.set_halign(Align::Start);
        local_auto_voice_desc.set_wrap(true);
        page.append(&local_auto_voice_title);
        page.append(&local_auto_voice_dd);
        page.append(&local_auto_voice_desc);

        // Local path
        let local_path_label = Label::new(Some("Local model path"));
        local_path_label.add_css_class("wizard-subtitle");
        local_path_label.set_halign(Align::Start);
        local_path_label.set_margin_top(8);
        let local_path_entry = Entry::new();
        local_path_entry.add_css_class("wizard-entry");
        local_path_entry.set_placeholder_text(Some("/path/to/piper-model.onnx"));
        local_path_entry.set_text(&self.inner.borrow().vm.tts_local_model_path);
        local_path_entry.update_property(&[AccProp::Label("Local Piper model path")]);
        {
            let this = self.clone();
            local_path_entry.connect_changed(move |e| {
                if this.inner.borrow().syncing {
                    return;
                }
                this.inner.borrow_mut().vm.tts_local_model_path = e.text().to_string();
                this.inner.borrow_mut().vm.tts_config_error = None;
            });
        }
        let local_path_help = Label::new(Some(
            "Point to a Piper .onnx model file or a directory containing .onnx voices.",
        ));
        local_path_help.add_css_class("wizard-radio-desc");
        local_path_help.set_halign(Align::Start);
        local_path_help.set_wrap(true);
        page.append(&local_path_label);
        page.append(&local_path_entry);
        page.append(&local_path_help);

        // Kokoro base URL
        let kokoro_url_label = Label::new(Some("Kokoro base URL"));
        kokoro_url_label.add_css_class("wizard-subtitle");
        kokoro_url_label.set_halign(Align::Start);
        kokoro_url_label.set_margin_top(8);
        let kokoro_url_entry = Entry::new();
        kokoro_url_entry.add_css_class("wizard-entry");
        kokoro_url_entry.set_placeholder_text(Some(DEFAULT_KOKORO_TTS_BASE_URL));
        kokoro_url_entry.set_text(&self.inner.borrow().vm.tts_kokoro_base_url);
        kokoro_url_entry.update_property(&[AccProp::Label("Kokoro OpenAI-compatible base URL")]);
        {
            let this = self.clone();
            kokoro_url_entry.connect_changed(move |e| {
                if this.inner.borrow().syncing {
                    return;
                }
                let text = e.text();
                this.inner.borrow_mut().vm.tts_kokoro_base_url = if text.trim().is_empty() {
                    DEFAULT_KOKORO_TTS_BASE_URL.into()
                } else {
                    text.to_string()
                };
                this.inner.borrow_mut().vm.tts_config_error = None;
            });
        }
        let kokoro_url_help = Label::new(Some(
            "Base URL for your Kokoro OpenAI-compatible API, for example http://localhost:8880/v1",
        ));
        kokoro_url_help.add_css_class("wizard-radio-desc");
        kokoro_url_help.set_halign(Align::Start);
        kokoro_url_help.set_wrap(true);
        page.append(&kokoro_url_label);
        page.append(&kokoro_url_entry);
        page.append(&kokoro_url_help);

        // MeloTTS device
        let melo_opts = [
            ("auto", "Auto", "CUDA if available, otherwise CPU."),
            ("cpu", "CPU", "Reliable; no GPU setup."),
            (
                "cuda",
                "GPU (CUDA)",
                "Faster when NVIDIA CUDA is available.",
            ),
        ];
        let melo_labels: Vec<&str> = melo_opts.iter().map(|(_, l, _)| *l).collect();
        let melotts_device_dd = DropDown::from_strings(&melo_labels);
        melotts_device_dd.add_css_class("wizard-dropdown");
        let melo_idx = melo_opts
            .iter()
            .position(|(id, ..)| *id == self.inner.borrow().vm.tts_melotts_device)
            .unwrap_or(0) as u32;
        melotts_device_dd.set_selected(melo_idx);
        melotts_device_dd.update_property(&[AccProp::Label("MeloTTS compute device")]);
        {
            let this = self.clone();
            melotts_device_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some((id, _, d)) = melo_opts.get(i) {
                    this.inner.borrow_mut().vm.tts_melotts_device = (*id).into();
                    dd.set_tooltip_text(Some(*d));
                }
            });
        }
        let melotts_device_title = Label::new(Some("Compute device"));
        melotts_device_title.add_css_class("wizard-subtitle");
        melotts_device_title.set_halign(Align::Start);
        melotts_device_title.set_margin_top(8);
        let melotts_device_desc = Label::new(Some(melo_opts[0].2));
        melotts_device_desc.add_css_class("wizard-radio-desc");
        melotts_device_desc.set_halign(Align::Start);
        melotts_device_desc.set_wrap(true);
        page.append(&melotts_device_title);
        page.append(&melotts_device_dd);
        page.append(&melotts_device_desc);

        // Voice ID entry (all backends except local automatic)
        let voice_label = Label::new(Some("Default voice ID"));
        voice_label.add_css_class("wizard-subtitle");
        voice_label.set_halign(Align::Start);
        voice_label.set_margin_top(8);
        let voice_entry = Entry::new();
        voice_entry.add_css_class("wizard-entry");
        voice_entry.set_text(&self.inner.borrow().vm.tts_voice_id);
        voice_entry.update_property(&[AccProp::Label("Default TTS voice ID")]);
        {
            let this = self.clone();
            voice_entry.connect_changed(move |e| {
                if this.inner.borrow().syncing {
                    return;
                }
                this.inner.borrow_mut().vm.tts_voice_id = e.text().to_string();
                this.inner.borrow_mut().vm.tts_config_error = None;
            });
        }
        let voice_help = Label::new(Some("Provider-specific voice id or name."));
        voice_help.add_css_class("wizard-radio-desc");
        voice_help.set_halign(Align::Start);
        voice_help.set_wrap(true);
        page.append(&voice_label);
        page.append(&voice_entry);
        page.append(&voice_help);

        // Speed — initialize from VM *before* connecting notify
        let speed_labels: Vec<String> = TTS_PLAYBACK_SPEED_PRESET_IDS
            .iter()
            .map(|id| format!("{id}×"))
            .collect();
        let speed_refs: Vec<&str> = speed_labels.iter().map(String::as_str).collect();
        let speed_dd = DropDown::from_strings(&speed_refs);
        speed_dd.add_css_class("wizard-dropdown");
        let speed_idx = speed_preset_index(self.inner.borrow().vm.tts_playback_speed);
        speed_dd.set_selected(speed_idx);
        // Ensure VM matches selected preset (no spurious notify needed).
        {
            let id = TTS_PLAYBACK_SPEED_PRESET_IDS[speed_idx as usize];
            self.inner.borrow_mut().vm.set_tts_playback_speed_preset(id);
        }
        speed_dd.update_property(&[AccProp::Label("Default playback speed")]);
        speed_dd.set_tooltip_text(Some("Default synthesis speed for TTS (0.5×–2.0× presets)."));
        {
            let this = self.clone();
            speed_dd.connect_selected_notify(move |dd| {
                if this.inner.borrow().syncing {
                    return;
                }
                let i = dd.selected() as usize;
                if let Some(id) = TTS_PLAYBACK_SPEED_PRESET_IDS.get(i) {
                    this.inner.borrow_mut().vm.set_tts_playback_speed_preset(id);
                }
            });
        }
        let st = Label::new(Some("Default playback speed"));
        st.add_css_class("wizard-subtitle");
        st.set_halign(Align::Start);
        st.set_margin_top(8);
        page.append(&st);
        page.append(&speed_dd);

        let error_label = Label::new(None);
        error_label.add_css_class("wizard-radio-desc");
        error_label.set_halign(Align::Start);
        error_label.set_visible(false);
        error_label.set_wrap(true);
        error_label.update_property(&[AccProp::Label("TTS configuration error")]);
        page.append(&error_label);

        page.append(&self.nav_row(
            Some(WizardPageId::Keybind),
            Some(WizardPageId::Done),
            "Finish",
        ));

        TtsPage {
            page,
            provider_dd,
            voice_label,
            voice_entry,
            voice_help,
            kokoro_url_label,
            kokoro_url_entry,
            kokoro_url_help,
            local_setup_title,
            local_setup_dd,
            local_setup_desc,
            local_path_label,
            local_path_entry,
            local_path_help,
            local_auto_voice_title,
            local_auto_voice_dd,
            local_auto_voice_desc,
            melotts_device_title,
            melotts_device_dd,
            melotts_device_desc,
            speed_dd,
            error_label,
        }
    }

    fn build_done_page(&self) -> DonePage {
        let page = GtkBox::new(Orientation::Vertical, 4);
        page.add_css_class("wizard-page");
        page.set_halign(Align::Center);
        page.set_valign(Align::Center);
        page.append(&title_label("You’re All Set!"));

        let sub = Label::new(Some("Here’s a summary of your settings:"));
        sub.add_css_class("wizard-subtitle");
        page.append(&sub);

        let summary_label = Label::new(None);
        summary_label.add_css_class("wizard-summary");
        summary_label.set_halign(Align::Center);
        summary_label.update_property(&[AccProp::Label("Configuration summary")]);
        page.append(&summary_label);

        let finish_status = Label::new(None);
        finish_status.add_css_class("wizard-desc");
        finish_status.set_halign(Align::Center);
        finish_status.set_visible(false);
        finish_status.set_margin_top(8);
        finish_status.set_wrap(true);
        finish_status.update_property(&[AccProp::Label("Finish status")]);
        page.append(&finish_status);

        let download_progress = ProgressBar::new();
        download_progress.set_show_text(true);
        download_progress.set_visible(false);
        download_progress.set_margin_top(6);
        download_progress.update_property(&[AccProp::Label("Model setup progress")]);
        page.append(&download_progress);

        let download_note = Label::new(Some(
            "Note: extraction can pause for 10–60s on slower disks. Please keep this window open.",
        ));
        download_note.add_css_class("wizard-desc");
        download_note.set_visible(false);
        page.append(&download_note);

        let cancel_btn = plain_button("Cancel download");
        cancel_btn.set_visible(false);
        cancel_btn.set_halign(Align::Center);
        cancel_btn.set_tooltip_text(Some("Cancel in-progress model setup if supported"));
        {
            let this = self.clone();
            cancel_btn.connect_clicked(move |_| {
                this.emit(UiEvent::WizardCancelDownload);
            });
        }
        page.append(&cancel_btn);

        let actions = GtkBox::new(Orientation::Horizontal, 16);
        actions.set_halign(Align::Center);
        actions.set_margin_top(16);

        let back_btn = plain_button("Back");
        back_btn.set_tooltip_text(Some("Return to TTS settings to change selections"));
        {
            let this = self.clone();
            back_btn.connect_clicked(move |_| {
                this.clear_finish_state_for_retry();
                this.navigate(WizardPageId::Tts);
            });
        }
        actions.append(&back_btn);

        let retry_btn = plain_button("Retry setup");
        retry_btn.set_tooltip_text(Some("Retry configuration write and model setup"));
        {
            let this = self.clone();
            retry_btn.connect_clicked(move |_| {
                this.clear_finish_state_for_retry();
                this.inner.borrow_mut().vm.begin_finish();
                this.sync_done_actions();
                this.emit(UiEvent::WizardFinishRequested);
            });
        }
        actions.append(&retry_btn);

        let launch_btn = primary_button("Launch ShuVoice");
        launch_btn.set_margin_top(0);
        launch_btn.set_visible(false);
        launch_btn.set_tooltip_text(Some("Close the wizard and start ShuVoice"));
        {
            let this = self.clone();
            launch_btn.connect_clicked(move |_| {
                this.inner.borrow_mut().vm.mark_launched();
                this.emit(UiEvent::WizardLaunch);
                this.emit_closed_once(true);
                this.app.quit();
            });
        }
        actions.append(&launch_btn);
        page.append(&actions);

        DonePage {
            page,
            summary_label,
            finish_status,
            download_progress,
            download_note,
            cancel_btn,
            back_btn,
            retry_btn,
            launch_btn,
        }
    }

    // -- Sync helpers ---------------------------------------------------------

    fn sync_sherpa_controls(&self) {
        let inner = self.inner.borrow();
        let Some(w) = &inner.widgets else {
            return;
        };
        let visible = sherpa_controls_visible(&inner.vm.asr_backend);
        let widgets = [
            w.sherpa_profile_title.clone().upcast::<gtk4::Widget>(),
            w.sherpa_profile_dd.clone().upcast(),
            w.sherpa_profile_desc.clone().upcast(),
            w.sherpa_provider_title.clone().upcast(),
            w.sherpa_provider_dd.clone().upcast(),
            w.sherpa_provider_desc.clone().upcast(),
        ];
        drop(inner);
        for widget in widgets {
            widget.set_visible(visible);
        }
    }

    fn sync_auto_add_checkbox(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.sync_auto_add_keybind_state();
        let sensitive = keybind_auto_add_sensitive(&inner.vm.keybind);
        let active = inner.vm.auto_add_keybind;
        let tooltip = if !sensitive {
            "Automatic Hyprland edit is unavailable for the Custom keybind preset."
        } else {
            "Only applies when the selected key is not already used by another bind."
        };
        let Some(w) = &inner.widgets else {
            return;
        };
        let checkbox = w.auto_add_keybind.clone();
        let keybind_dd = w.keybind_dd.clone();
        let key_idx = keybind_index(&inner.vm.keybind);
        inner.syncing = true;
        drop(inner);
        checkbox.set_sensitive(sensitive);
        checkbox.set_active(active && sensitive);
        checkbox.set_tooltip_text(Some(tooltip));
        if keybind_dd.selected() != key_idx {
            keybind_dd.set_selected(key_idx);
        }
        self.inner.borrow_mut().syncing = false;
    }

    fn sync_tts_controls(&self) {
        let mut inner = self.inner.borrow_mut();
        let Some(w) = &inner.widgets else {
            return;
        };
        let vis = TtsControlVisibility::for_backend(
            &inner.vm.tts_backend,
            &inner.vm.tts_local_setup_mode,
        );
        let backend = inner.vm.tts_backend.clone();
        let speed = inner.vm.tts_playback_speed;
        let err = inner.vm.tts_config_error.clone();
        let kokoro_empty = w.kokoro_url_entry.text().is_empty();

        // Clone widget handles so we can drop the borrow before GTK calls.
        let voice_label = w.tts_voice_label.clone();
        let voice_entry = w.tts_voice_entry.clone();
        let voice_help = w.tts_voice_help.clone();
        let kokoro_url_label = w.kokoro_url_label.clone();
        let kokoro_url_entry = w.kokoro_url_entry.clone();
        let kokoro_url_help = w.kokoro_url_help.clone();
        let local_setup_title = w.local_setup_title.clone();
        let local_setup_dd = w.local_setup_dd.clone();
        let local_setup_desc = w.local_setup_desc.clone();
        let local_path_label = w.local_path_label.clone();
        let local_path_entry = w.local_path_entry.clone();
        let local_path_help = w.local_path_help.clone();
        let local_auto_voice_title = w.local_auto_voice_title.clone();
        let local_auto_voice_dd = w.local_auto_voice_dd.clone();
        let local_auto_voice_desc = w.local_auto_voice_desc.clone();
        let melotts_device_title = w.melotts_device_title.clone();
        let melotts_device_dd = w.melotts_device_dd.clone();
        let melotts_device_desc = w.melotts_device_desc.clone();
        let speed_dd = w.speed_dd.clone();
        let provider_dd = w.tts_provider_dd.clone();
        let tts_error = w.tts_error.clone();

        inner.syncing = true;
        drop(inner);

        voice_label.set_visible(vis.voice_entry);
        voice_entry.set_visible(vis.voice_entry);
        voice_help.set_visible(vis.voice_entry);
        kokoro_url_label.set_visible(vis.kokoro_url);
        kokoro_url_entry.set_visible(vis.kokoro_url);
        kokoro_url_help.set_visible(vis.kokoro_url);
        local_setup_title.set_visible(vis.local_setup_mode);
        local_setup_dd.set_visible(vis.local_setup_mode);
        local_setup_desc.set_visible(vis.local_setup_mode);
        local_path_label.set_visible(vis.local_path);
        local_path_entry.set_visible(vis.local_path);
        local_path_help.set_visible(vis.local_path);
        local_auto_voice_title.set_visible(vis.local_auto_voice);
        local_auto_voice_dd.set_visible(vis.local_auto_voice);
        local_auto_voice_desc.set_visible(vis.local_auto_voice);
        melotts_device_title.set_visible(vis.melotts_device);
        melotts_device_dd.set_visible(vis.melotts_device);
        melotts_device_desc.set_visible(vis.melotts_device);

        match backend.as_str() {
            "openai" => {
                voice_entry.set_placeholder_text(Some("onyx"));
                voice_help.set_text("Examples: onyx, nova, shimmer, alloy, sage");
            }
            "elevenlabs" => {
                voice_entry.set_placeholder_text(Some("ElevenLabs voice id"));
                voice_help.set_text("Paste an ElevenLabs voice ID, or keep the default.");
            }
            "melotts" => {
                voice_entry.set_placeholder_text(Some("EN-US"));
                voice_help.set_text("MeloTTS voices: EN-US, EN-BR, EN-INDIA, EN-AU, EN-Newest");
            }
            "kokoro" => {
                voice_entry.set_placeholder_text(Some("af_heart"));
                voice_help.set_text("Enter a Kokoro voice ID (for example af_heart).");
                if kokoro_empty {
                    kokoro_url_entry.set_text(DEFAULT_KOKORO_TTS_BASE_URL);
                }
            }
            "local" => {
                voice_entry.set_placeholder_text(Some("Optional voice/model stem"));
                voice_help
                    .set_text("Leave blank to use the first discovered .onnx model automatically.");
            }
            _ => {}
        }

        let idx = speed_preset_index(speed);
        if speed_dd.selected() != idx {
            speed_dd.set_selected(idx);
        }
        let pidx = tts_backend_index(&backend);
        if provider_dd.selected() != pidx {
            provider_dd.set_selected(pidx);
        }
        if let Some(err) = err {
            tts_error.set_text(&err);
            tts_error.set_visible(true);
        } else {
            tts_error.set_visible(false);
        }

        self.inner.borrow_mut().syncing = false;
    }

    fn sync_done_actions(&self) {
        let inner = self.inner.borrow();
        let Some(w) = &inner.widgets else {
            return;
        };
        let show_launch = inner.vm.launch_visible;
        let in_progress = inner.vm.finish_in_progress;
        let show_back = should_show_done_back(in_progress, show_launch);
        w.back_from_done_btn.set_visible(show_back || !show_launch);
        w.back_from_done_btn.set_sensitive(!in_progress);
        w.retry_btn.set_visible(show_back);
        w.retry_btn.set_sensitive(!in_progress);
        w.launch_btn.set_visible(show_launch);
        w.launch_btn.set_sensitive(show_launch);
        w.cancel_btn
            .set_visible(in_progress && inner.vm.download_visible);
    }

    fn clear_finish_state_for_retry(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.finish_in_progress = false;
        inner.vm.launch_visible = false;
        inner.vm.download_visible = false;
        inner.vm.cancel_download_visible = false;
        inner.vm.finish_status.clear();
        if let Some(id) = inner.download_pulse.take() {
            id.remove();
        }
        if let Some(w) = &inner.widgets {
            w.finish_status.set_visible(false);
            w.finish_status.set_text("");
            w.download_progress.set_visible(false);
            w.download_note.set_visible(false);
            w.cancel_btn.set_visible(false);
            w.launch_btn.set_visible(false);
        }
        drop(inner);
        self.sync_done_actions();
    }

    // -- Navigation -----------------------------------------------------------

    fn nav_row(
        &self,
        back: Option<WizardPageId>,
        next: Option<WizardPageId>,
        next_label: &str,
    ) -> GtkBox {
        let row = GtkBox::new(Orientation::Horizontal, 16);
        row.set_halign(Align::Center);
        row.set_margin_top(20);
        if let Some(page) = back {
            let btn = plain_button("Back");
            btn.set_tooltip_text(Some("Go to previous step"));
            let this = self.clone();
            btn.connect_clicked(move |_| this.navigate(page));
            row.append(&btn);
        }
        if let Some(page) = next {
            let btn = primary_button(next_label);
            btn.set_tooltip_text(Some(if page == WizardPageId::Done {
                "Validate TTS settings and apply configuration"
            } else {
                "Go to next step"
            }));
            let this = self.clone();
            btn.connect_clicked(move |_| {
                if page == WizardPageId::Done {
                    let ok = {
                        let mut inner = this.inner.borrow_mut();
                        inner.vm.validate_tts_selection().is_ok()
                    };
                    if !ok {
                        this.sync_tts_controls();
                        return;
                    }
                    this.inner.borrow_mut().vm.begin_finish();
                    this.navigate(WizardPageId::Done);
                    this.sync_done_actions();
                    this.emit(UiEvent::WizardFinishRequested);
                    return;
                }
                this.navigate(page);
            });
            row.append(&btn);
        }
        row
    }

    fn navigate(&self, page: WizardPageId) {
        {
            let mut inner = self.inner.borrow_mut();
            inner.vm.navigate(page);
            if let Some(w) = &inner.widgets {
                w.stack.set_visible_child_name(page.as_str());
                if page == WizardPageId::Done {
                    w.summary_label.set_text(&inner.vm.summary);
                    w.finish_status.set_text(&inner.vm.finish_status);
                    w.finish_status
                        .set_visible(!inner.vm.finish_status.is_empty());
                }
            }
        }
        if page == WizardPageId::Asr {
            self.sync_sherpa_controls();
        }
        if page == WizardPageId::Keybind {
            self.sync_auto_add_checkbox();
        }
        if page == WizardPageId::Tts {
            self.sync_tts_controls();
        }
        if page == WizardPageId::Done {
            self.sync_done_actions();
        }
        self.emit(UiEvent::WizardPageChanged { page });
    }

    // -- External commands ----------------------------------------------------

    pub fn apply_cmd(&self, cmd: UiCmd) {
        match cmd {
            UiCmd::WizardNavigate { page } => self.navigate(page),
            UiCmd::WizardSetStatus { text } => {
                let mut inner = self.inner.borrow_mut();
                inner.vm.finish_status = text.clone();
                if let Some(w) = &inner.widgets {
                    w.finish_status.set_text(&text);
                    w.finish_status.set_visible(true);
                }
            }
            UiCmd::WizardSetProgress { fraction, text } => {
                self.set_download_progress(fraction, &text);
            }
            UiCmd::WizardDownloadFinished {
                status_text,
                show_launch,
            } => {
                let mut inner = self.inner.borrow_mut();
                if let Some(id) = inner.download_pulse.take() {
                    id.remove();
                }
                inner.vm.finish_status = status_text.clone();
                inner.vm.finish_in_progress = false;
                inner.vm.launch_visible = show_launch;
                inner.vm.download_visible = true;
                if let Some(w) = &inner.widgets {
                    w.finish_status.set_text(&status_text);
                    w.finish_status.set_visible(true);
                    w.cancel_btn.set_visible(false);
                    w.download_note.set_visible(false);
                    w.download_progress
                        .set_fraction(if show_launch { 1.0 } else { 0.0 });
                    w.download_progress.set_text(Some(if show_launch {
                        "Setup finished"
                    } else {
                        "Setup incomplete — use Back or Retry"
                    }));
                    w.download_progress.set_visible(true);
                }
                drop(inner);
                self.sync_done_actions();
            }
            UiCmd::WizardClose => {
                self.emit_closed_once(self.completed());
                self.app.quit();
            }
            _ => {}
        }
    }

    fn set_download_progress(&self, fraction: Option<f64>, text: &str) {
        let mut inner = self.inner.borrow_mut();
        inner.vm.download_visible = true;
        inner.vm.download_text = text.into();
        inner.vm.download_fraction = fraction;
        if let Some(w) = &inner.widgets {
            w.download_progress.set_visible(true);
            w.download_progress.set_text(Some(text));
            if inner.vm.finish_in_progress {
                w.cancel_btn.set_visible(true);
            }
        }
        match fraction {
            None => {
                if inner.download_pulse.is_none()
                    && let Some(w) = &inner.widgets
                {
                    let progress = w.download_progress.clone();
                    let id = glib::timeout_add_local(Duration::from_millis(120), move || {
                        progress.pulse();
                        ControlFlow::Continue
                    });
                    inner.download_pulse = Some(id);
                }
            }
            Some(f) => {
                if let Some(id) = inner.download_pulse.take() {
                    id.remove();
                }
                if let Some(w) = &inner.widgets {
                    w.download_progress.set_fraction(f.clamp(0.0, 1.0));
                }
            }
        }
        drop(inner);
        self.sync_done_actions();
    }

    /// Attach a non-blocking command pump for external finish/download updates.
    pub fn attach_cmd_pump(&self, cmd_rx: UiCmdReceiver) {
        // Replace any existing pump.
        {
            let mut inner = self.inner.borrow_mut();
            if let Some(id) = inner.cmd_pump.take() {
                id.remove();
            }
        }
        let this = self.clone();
        let id = glib::timeout_add_local(Duration::from_millis(16), move || {
            loop {
                match cmd_rx.try_recv() {
                    Ok(cmd) => this.apply_cmd(cmd),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return ControlFlow::Break;
                    }
                }
            }
            ControlFlow::Continue
        });
        self.inner.borrow_mut().cmd_pump = Some(id);
    }

    /// Detach pumps/timers (also called on shutdown).
    pub fn detach_sources(&self) {
        let mut inner = self.inner.borrow_mut();
        if let Some(id) = inner.download_pulse.take() {
            id.remove();
        }
        if let Some(id) = inner.cmd_pump.take() {
            id.remove();
        }
    }

    fn on_shutdown(&self) {
        self.detach_sources();
        // If the user never clicked Launch, this is a cancel/close.
        if !self.completed() {
            self.emit_closed_once(false);
        }
        let mut inner = self.inner.borrow_mut();
        if let Some(widgets) = inner.widgets.as_mut()
            && let Some(window) = widgets.window.take()
        {
            release_wizard_keyboard(&window);
            window.set_visible(false);
            window.destroy();
        }
    }
}

impl Drop for WelcomeWizard {
    fn drop(&mut self) {
        // Best-effort source cleanup if the app never shut down cleanly.
        self.detach_sources();
    }
}

// -- Small builders -----------------------------------------------------------

fn title_label(text: &str) -> Label {
    let title = Label::new(Some(text));
    title.add_css_class("wizard-title");
    title.set_halign(Align::Center);
    title
}

fn primary_button(label: &str) -> Button {
    let btn = Button::with_label(label);
    btn.add_css_class("wizard-btn");
    btn.add_css_class("wizard-btn-primary");
    btn
}

fn plain_button(label: &str) -> Button {
    let btn = Button::with_label(label);
    btn.add_css_class("wizard-btn");
    btn
}

fn set_accessible_description(widget: &impl IsA<gtk4::Widget>, description: &str) {
    widget
        .as_ref()
        .update_property(&[AccProp::Description(description)]);
}

fn discover_repo_root() -> Option<std::path::PathBuf> {
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
