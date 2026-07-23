use crate::control::{ControlCmd, run_control};
use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};

pub async fn execute(command: ControlCmd, socket: Option<&str>, wait_sec: f64) -> ExitStatus {
    // Blocking control client; fine for short CLI RPCs.
    match run_control(command, socket, wait_sec) {
        Ok(response) => {
            println!("{response}");
            ExitStatus::code(EXIT_SUCCESS)
        }
        Err(err) => {
            eprintln!("ERROR: {err}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}
