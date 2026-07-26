//! Dump the capture configs cpal advertises for the default input device.
//!
//! Diagnostic for channel-layout selection in `audio::cpal_source`.

fn main() {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        println!("no default input device");
        return;
    };
    let name = device
        .description()
        .ok()
        .map(|d| d.name().to_string())
        .unwrap_or_default();
    println!("device: {name}");

    match device.default_input_config() {
        Ok(c) => println!(
            "default_input_config: channels={} rate={} fmt={:?}",
            c.channels(),
            c.sample_rate(),
            c.sample_format()
        ),
        Err(e) => println!("default_input_config error: {e}"),
    }

    match device.supported_input_configs() {
        Ok(configs) => {
            for (i, c) in configs.enumerate() {
                println!(
                    "  [{i}] channels={} rate={}..{} fmt={:?}",
                    c.channels(),
                    c.min_sample_rate(),
                    c.max_sample_rate(),
                    c.sample_format()
                );
            }
        }
        Err(e) => println!("supported_input_configs error: {e}"),
    }
}
