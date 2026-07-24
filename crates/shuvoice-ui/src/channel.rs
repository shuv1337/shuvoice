//! Thread-safe UI command channel (headless-friendly).

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};

use crate::protocol::{UiCmd, UiEvent};

/// Bidirectional UI bus used by the app actor and GTK host.
#[derive(Debug)]
pub struct UiBus {
    cmd_tx: Sender<UiCmd>,
    cmd_rx: Receiver<UiCmd>,
    event_tx: Sender<UiEvent>,
    event_rx: Receiver<UiEvent>,
}

impl Default for UiBus {
    fn default() -> Self {
        Self::new()
    }
}

impl UiBus {
    pub fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        Self {
            cmd_tx,
            cmd_rx,
            event_tx,
            event_rx,
        }
    }

    pub fn split(self) -> (UiCmdSender, UiCmdReceiver, UiEventSender, UiEventReceiver) {
        (
            UiCmdSender(self.cmd_tx),
            UiCmdReceiver(self.cmd_rx),
            UiEventSender(self.event_tx),
            UiEventReceiver(self.event_rx),
        )
    }
}

#[derive(Debug, Clone)]
pub struct UiCmdSender(Sender<UiCmd>);

impl UiCmdSender {
    pub fn send(&self, cmd: UiCmd) -> Result<(), mpsc::SendError<UiCmd>> {
        self.0.send(cmd)
    }
}

#[derive(Debug)]
pub struct UiCmdReceiver(Receiver<UiCmd>);

impl UiCmdReceiver {
    pub fn try_recv(&self) -> Result<UiCmd, TryRecvError> {
        self.0.try_recv()
    }

    pub fn recv(&self) -> Result<UiCmd, mpsc::RecvError> {
        self.0.recv()
    }

    pub fn into_inner(self) -> Receiver<UiCmd> {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct UiEventSender(Sender<UiEvent>);

impl UiEventSender {
    pub fn send(&self, event: UiEvent) -> Result<(), mpsc::SendError<UiEvent>> {
        self.0.send(event)
    }
}

#[derive(Debug)]
pub struct UiEventReceiver(Receiver<UiEvent>);

impl UiEventReceiver {
    pub fn try_recv(&self) -> Result<UiEvent, TryRecvError> {
        self.0.try_recv()
    }

    pub fn recv(&self) -> Result<UiEvent, mpsc::RecvError> {
        self.0.recv()
    }
}
