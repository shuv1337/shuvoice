//! NeMo / Moonshine backends via [`shuvoice_worker_proto`].
//!
//! Native Rust does **not** run NeMo or useful-moonshine. These backends are
//! thin hosts around [`WorkerClient`] / [`WorkerProcess`] / [`WorkerSupervisor`].

mod client;
mod mock_server;

pub use client::{WorkerAsrBackend, WorkerAttach, WorkerBackendKind};
pub use mock_server::spawn_mock_worker;

pub use shuvoice_worker_proto::{
    ClientOptions, ControlMessage, DEFAULT_LOAD_TIMEOUT, DEFAULT_RPC_TIMEOUT, FramedConnection,
    PROTOCOL_VERSION, WorkerClient, WorkerManifest, WorkerProcess, WorkerSpawnConfig,
    WorkerSupervisor, accept_handshake,
};
