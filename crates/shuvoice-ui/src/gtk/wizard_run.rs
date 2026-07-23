//! GTK entrypoint that drives WelcomeWizard + finish controller on the main thread.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use glib::ControlFlow;
use gtk4::prelude::*;

use super::wizard::WelcomeWizard;
use crate::channel::UiBus;
use crate::protocol::{UiCmd, UiEvent};
use crate::wizard_controller::{
    DeferredModelSetup, ModelSetupHook, ModelSetupStatus, WizardFinishReport, finish_wizard,
};

/// Errors launching the GTK wizard (no panic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WizardUiLaunchError {
    /// `gtk::init` failed (no display / missing GDK backend).
    GtkInit(String),
    /// Could not spawn the finish worker and refused to block the GTK thread.
    WorkerSpawn(String),
}

impl std::fmt::Display for WizardUiLaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GtkInit(msg) | Self::WorkerSpawn(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for WizardUiLaunchError {}

/// Run the GTK wizard and return whether the user completed (Launch).
///
/// Finish always runs off the GTK thread. If the worker cannot be spawned, the
/// UI shows an actionable error and leaves finish unlocked for Retry — it does
/// **not** call `finish_wizard` on the main thread.
pub fn run_welcome_wizard_gtk(
    force_reconfigure: bool,
    model_hook: Option<Box<dyn ModelSetupHook>>,
) -> Result<bool, WizardUiLaunchError> {
    if let Err(err) = gtk4::init() {
        return Err(WizardUiLaunchError::GtkInit(format!(
            "Failed to initialize GTK (is DISPLAY/WAYLAND_DISPLAY set?): {err}"
        )));
    }

    let wizard = WelcomeWizard::new(force_reconfigure);
    let bus = UiBus::new();
    let (cmd_tx, cmd_rx, event_tx, event_rx) = bus.split();
    wizard.set_event_sender(event_tx);

    let completed = Rc::new(RefCell::new(false));
    let cancel_flag = Arc::new(Mutex::new(false));
    let finish_started = Arc::new(AtomicBool::new(false));
    let hook_slot: Arc<Mutex<Option<Box<dyn ModelSetupHook>>>> = Arc::new(Mutex::new(Some(
        model_hook.unwrap_or_else(|| Box::new(DeferredModelSetup)),
    )));

    // Command pump: attached on activate; WelcomeWizard owns the SourceId and
    // removes it on shutdown/detach_sources.
    {
        let wizard_for_activate = wizard.clone();
        let cmd_rx_slot = Mutex::new(Some(cmd_rx));
        wizard.application().connect_activate(move |_| {
            if let Ok(mut guard) = cmd_rx_slot.lock()
                && let Some(rx) = guard.take()
            {
                wizard_for_activate.attach_cmd_pump(rx);
            }
        });
    }

    // Event pump SourceId — removed when the timeout returns Break or on Drop via
    // a shared flag checked each tick after quit.
    let event_pump_alive = Arc::new(AtomicBool::new(true));
    {
        let wizard = wizard.clone();
        let completed = Rc::clone(&completed);
        let cancel_flag = Arc::clone(&cancel_flag);
        let finish_started = Arc::clone(&finish_started);
        let cmd_tx = cmd_tx.clone();
        let hook_slot = Arc::clone(&hook_slot);
        let event_pump_alive = Arc::clone(&event_pump_alive);

        glib::timeout_add_local(Duration::from_millis(16), move || {
            if !event_pump_alive.load(Ordering::SeqCst) {
                return ControlFlow::Break;
            }

            loop {
                match event_rx.try_recv() {
                    Ok(UiEvent::WizardFinishRequested) => {
                        if finish_started.swap(true, Ordering::SeqCst) {
                            continue;
                        }
                        if let Ok(mut g) = cancel_flag.lock() {
                            *g = false;
                        }

                        let vm = wizard.vm();
                        let _ = cmd_tx.send(UiCmd::WizardSetStatus {
                            text: "Applying settings…".into(),
                        });
                        let _ = cmd_tx.send(UiCmd::WizardSetProgress {
                            fraction: Some(0.05),
                            text: "Writing configuration…".into(),
                        });

                        let cmd_tx_w = cmd_tx.clone();
                        let cancel_w = Arc::clone(&cancel_flag);
                        let hook_slot_w = Arc::clone(&hook_slot);
                        let finish_started_w = Arc::clone(&finish_started);
                        let vm_worker = vm.clone();

                        let spawn_result = std::thread::Builder::new()
                            .name("wizard-finish".into())
                            .spawn(move || {
                                let mut owned_hook: Box<dyn ModelSetupHook> = {
                                    let mut guard =
                                        hook_slot_w.lock().unwrap_or_else(|e| e.into_inner());
                                    guard
                                        .take()
                                        .unwrap_or_else(|| Box::new(DeferredModelSetup))
                                };

                                let cmd_progress = cmd_tx_w.clone();
                                let mut progress = move |fraction: Option<f64>, message: &str| {
                                    let _ = cmd_progress.send(UiCmd::WizardSetProgress {
                                        fraction,
                                        text: message.to_string(),
                                    });
                                };
                                let cancel_flag = Arc::clone(&cancel_w);
                                let mut cancel =
                                    move || cancel_flag.lock().map(|g| *g).unwrap_or(false);

                                let report = finish_wizard(
                                    &vm_worker,
                                    owned_hook.as_mut(),
                                    Some(&mut progress),
                                    Some(&mut cancel),
                                );

                                match report {
                                    Ok(report) => {
                                        let _ = cmd_tx_w.send(UiCmd::WizardDownloadFinished {
                                            status_text: report.status_text(),
                                            show_launch: true,
                                        });
                                    }
                                    Err(err) => {
                                        finish_started_w.store(false, Ordering::SeqCst);
                                        let _ = cmd_tx_w.send(UiCmd::WizardDownloadFinished {
                                            status_text: format!(
                                                "⚠ Wizard finish failed: {err}\nUse Back or Retry setup."
                                            ),
                                            show_launch: false,
                                        });
                                    }
                                }

                                if let Ok(mut guard) = hook_slot_w.lock() {
                                    *guard = Some(owned_hook);
                                }
                            });

                        if let Err(err) = spawn_result {
                            // Never block the GTK main thread on finish work.
                            finish_started.store(false, Ordering::SeqCst);
                            let _ = cmd_tx.send(UiCmd::WizardDownloadFinished {
                                status_text: format!(
                                    "⚠ Could not start background setup worker ({err}).\n\
                                     Use Retry setup. Finish is never run on the UI thread."
                                ),
                                show_launch: false,
                            });
                        }
                    }
                    Ok(UiEvent::WizardCancelDownload) => {
                        if let Ok(mut g) = cancel_flag.lock() {
                            *g = true;
                        }
                        // Unlock finish so Retry works after cancel settles.
                        finish_started.store(false, Ordering::SeqCst);
                        let _ = cmd_tx.send(UiCmd::WizardSetProgress {
                            fraction: None,
                            text: "Cancelling…".into(),
                        });
                        let _ = cmd_tx.send(UiCmd::WizardSetStatus {
                            text: "ℹ Cancel requested — use Retry setup if needed.".into(),
                        });
                    }
                    Ok(UiEvent::WizardLaunch) | Ok(UiEvent::WizardClosed { completed: true }) => {
                        *completed.borrow_mut() = true;
                        event_pump_alive.store(false, Ordering::SeqCst);
                    }
                    Ok(UiEvent::WizardClosed { completed: false }) => {
                        event_pump_alive.store(false, Ordering::SeqCst);
                    }
                    Ok(_) => {}
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        event_pump_alive.store(false, Ordering::SeqCst);
                        return ControlFlow::Break;
                    }
                }
            }
            ControlFlow::Continue
        });
    }

    let _code = wizard.run();
    // Ensure pumps/timers are gone even if shutdown ordering was odd.
    event_pump_alive.store(false, Ordering::SeqCst);
    wizard.detach_sources();

    Ok(*completed.borrow() || wizard.completed())
}

/// Convenience: GTK wizard with deferred model setup.
pub fn run_welcome_wizard_gtk_deferred(
    force_reconfigure: bool,
) -> Result<bool, WizardUiLaunchError> {
    run_welcome_wizard_gtk(force_reconfigure, None)
}

#[allow(dead_code)]
fn _use_types(_: WizardFinishReport, _: ModelSetupStatus) {}
