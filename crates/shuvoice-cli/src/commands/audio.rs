use shuvoice_core::DeviceRef;

use crate::error::{EXIT_FAILURE, EXIT_SUCCESS, ExitStatus};

/// List capture devices.
pub fn list_devices() -> ExitStatus {
    match list_devices_impl() {
        Ok(lines) => {
            println!("Audio devices:");
            for line in lines {
                println!("{line}");
            }
            ExitStatus::code(EXIT_SUCCESS)
        }
        Err(err) => {
            eprintln!("ERROR: Could not list audio devices: {err}");
            ExitStatus::code(EXIT_FAILURE)
        }
    }
}

fn list_devices_impl() -> Result<Vec<String>, String> {
    #[cfg(feature = "audio")]
    {
        list_devices_cpal()
    }
    #[cfg(not(feature = "audio"))]
    {
        Err("audio feature disabled (rebuild with --features audio / default features)".into())
    }
}

/// Validate the configured capture device when the `audio` feature is enabled.
///
/// - `None` → default device must resolve
/// - `DeviceRef::Name` → exact name match among input devices
/// - `DeviceRef::Index` → index into the enumerated input device list
pub fn validate_configured_input(device: &Option<DeviceRef>) -> Result<String, String> {
    #[cfg(feature = "audio")]
    {
        validate_configured_input_cpal(device)
    }
    #[cfg(not(feature = "audio"))]
    {
        let detail = match device {
            None => "default".into(),
            Some(DeviceRef::Index(i)) => format!("index {i}"),
            Some(DeviceRef::Name(n)) => n.clone(),
        };
        Ok(format!(
            "{detail} (audio feature disabled; not validated at preflight)"
        ))
    }
}

#[cfg(feature = "audio")]
fn list_devices_cpal() -> Result<Vec<String>, String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| format!("query input devices: {e}"))?;

    let mut lines = Vec::new();
    for (idx, device) in devices.enumerate() {
        let name = device
            .description()
            .ok()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|| device.to_string());
        let (channels, default_sr) = match device.default_input_config() {
            Ok(cfg) => (cfg.channels(), f64::from(cfg.sample_rate())),
            Err(_) => (0, 0.0),
        };
        if channels == 0 {
            continue;
        }
        lines.push(format!(
            "[{idx}] {name} (in={channels}, default_sr={default_sr})"
        ));
    }
    if lines.is_empty() {
        return Err("no input devices found".into());
    }
    Ok(lines)
}

#[cfg(feature = "audio")]
fn validate_configured_input_cpal(device: &Option<DeviceRef>) -> Result<String, String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let devices: Vec<_> = host
        .input_devices()
        .map_err(|e| format!("query input devices: {e}"))?
        .collect();

    match device {
        None => {
            let dev = host
                .default_input_device()
                .ok_or_else(|| "no default input device available".to_string())?;
            let name = dev
                .description()
                .ok()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|| dev.to_string());
            let cfg = dev
                .default_input_config()
                .map_err(|e| format!("default input config: {e}"))?;
            if cfg.channels() == 0 {
                return Err(format!("default input device '{name}' has zero channels"));
            }
            Ok(format!(
                "default ({name} @ {}Hz, ch={})",
                cfg.sample_rate(),
                cfg.channels()
            ))
        }
        Some(DeviceRef::Index(idx)) => {
            if *idx < 0 {
                return Err(format!("audio_device index {idx} is negative"));
            }
            let i = *idx as usize;
            let dev = devices.get(i).ok_or_else(|| {
                format!(
                    "audio_device index {idx} out of range ({} devices)",
                    devices.len()
                )
            })?;
            let name = dev
                .description()
                .ok()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|| dev.to_string());
            let cfg = dev
                .default_input_config()
                .map_err(|e| format!("input config for '{name}': {e}"))?;
            if cfg.channels() == 0 {
                return Err(format!("input device '{name}' has zero input channels"));
            }
            Ok(format!(
                "index {idx} ({name} @ {}Hz, ch={})",
                cfg.sample_rate(),
                cfg.channels()
            ))
        }
        Some(DeviceRef::Name(want)) => {
            for dev in &devices {
                let name = dev
                    .description()
                    .ok()
                    .map(|d| d.name().to_string())
                    .unwrap_or_else(|| dev.to_string());
                if name == *want {
                    let cfg = dev
                        .default_input_config()
                        .map_err(|e| format!("input config for '{name}': {e}"))?;
                    if cfg.channels() == 0 {
                        return Err(format!("input device '{name}' has zero input channels"));
                    }
                    return Ok(format!(
                        "{name} @ {}Hz, ch={}",
                        cfg.sample_rate(),
                        cfg.channels()
                    ));
                }
            }
            Err(format!("configured audio input device not found: {want}"))
        }
    }
}
