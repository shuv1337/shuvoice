use std::collections::BTreeMap;
use std::time::Duration;

use shuvoice_core::Config;

use crate::control::{ControlCmd, send_cmd};
use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};

pub async fn execute(config: &Config, json_output: bool) -> ExitStatus {
    let socket = config.control_socket.as_deref();
    let timeout = Some(Duration::from_millis(1500));
    let mut payload: BTreeMap<&str, String> = BTreeMap::new();

    payload.insert(
        "status",
        match send_cmd(ControlCmd::Status, socket, timeout) {
            Ok(v) => v,
            Err(e) => format!("ERROR: {e}"),
        },
    );
    payload.insert(
        "metrics",
        match send_cmd(ControlCmd::Metrics, socket, timeout) {
            Ok(v) => v,
            Err(e) => format!("ERROR: {e}"),
        },
    );
    payload.insert(
        "debug_status",
        match send_cmd(ControlCmd::DebugStatus, socket, timeout) {
            Ok(v) => v,
            Err(e) => format!("ERROR: {e}"),
        },
    );

    if json_output {
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{text}"),
            Err(err) => {
                eprintln!("ERROR: {err}");
                return ExitStatus::code(EXIT_FAILURE);
            }
        }
    } else {
        for (key, value) in &payload {
            println!("{key}: {value}");
        }
    }

    if payload
        .get("status")
        .map(|s| s.starts_with("ERROR"))
        .unwrap_or(true)
    {
        ExitStatus::code(EXIT_FAILURE)
    } else {
        ExitStatus::code(EXIT_SUCCESS)
    }
}
