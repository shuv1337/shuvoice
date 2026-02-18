# Building a streaming STT overlay for Hyprland with Nemotron 0.6B

**NVIDIA's Nemotron Speech Streaming 0.6B model can power a real-time speech-to-text overlay on Hyprland using approximately 2–3 GB of VRAM on an RTX 5080, with end-to-end latency as low as 160ms.** The complete pipeline chains five components: PipeWire audio capture via `sounddevice`, streaming inference through NeMo's `conformer_stream_step()`, a GTK4 + gtk4-layer-shell transparent overlay, text injection via `wtype`, and push-to-talk hotkeys through `python-evdev`. Every piece has proven, working implementations in the Linux speech-to-text ecosystem. This report provides the exact code, import paths, and integration patterns needed to build each component.

## Nemotron streaming inference: the core engine

The model loads through NeMo's unified ASR API and streams via a cache-aware FastConformer encoder paired with an RNN-T decoder. The critical architecture detail: **the model expects native 1120ms audio chunks (17,920 samples at 16kHz)**, not smaller fragments. Sending undersized chunks causes the RNNT decoder to stall. You buffer incoming audio to this chunk size before each inference step.

```python
import torch
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)
model.eval().cuda()

# Set latency: right_context choices are 0 (80ms), 1 (160ms), 6 (560ms), 13 (1120ms)
right_context = 1  # 160ms — best for interactive use
model.encoder.set_default_att_context_size([70, right_context])
model.encoder.setup_streaming_params()
```

The streaming loop maintains three cache tensors between steps. Each call to `conformer_stream_step()` processes one chunk of mel-spectrogram features and returns the cumulative transcription:

```python
cache_last_channel, cache_last_time, cache_last_channel_len = \
    model.encoder.get_initial_cache_state(batch_size=1)

num_features = model.preprocessor.featurizer.feat_out  # 80 mel bins
pre_encode_size = model.encoder.streaming_cfg.pre_encode_cache_size[1]  # 9
cache_pre_encode = torch.zeros((1, num_features, pre_encode_size), device=model.device)

previous_hypotheses = None
pred_out_stream = None

for step_num, audio_chunk in enumerate(audio_stream):
    audio_tensor = audio_chunk.unsqueeze(0).to(model.device)
    audio_len = torch.tensor([audio_tensor.shape[1]], device=model.device)

    processed_signal, processed_signal_length = model.preprocessor(
        input_signal=audio_tensor, length=audio_len
    )
    processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
    processed_signal_length += cache_pre_encode.shape[-1]
    cache_pre_encode = processed_signal[:, :, -pre_encode_size:].clone()

    (pred_out_stream, transcribed_texts, cache_last_channel,
     cache_last_time, cache_last_channel_len, previous_hypotheses
    ) = model.conformer_stream_step(
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
        keep_all_outputs=False,
        previous_hypotheses=previous_hypotheses,
        previous_pred_out=pred_out_stream,
        drop_extra_pre_encoded=0 if step_num == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded,
        return_transcription=True,
    )
    current_text = transcribed_texts[0]
```

The key imports are `nemo.collections.asr` for the model and `nemo.collections.asr.parts.utils.streaming_utils.CacheAwareStreamingAudioBuffer` for the higher-level file-based streaming API. Requirements: **NeMo ≥ 2.0.0** (recommended 2.6.0), **PyTorch ≥ 2.0** (recommended 2.4+), **CUDA 12.x**, Python 3.10–3.12. The 600M parameter model consumes roughly **1.2 GB for weights in FP16** plus cache overhead, totaling **2–3 GB VRAM** for single-stream inference — well within the RTX 5080's 16 GB. Set `pad_and_drop_preencoded=False` for best accuracy (~1.8% WER on LibriSpeech).

## GTK4 layer-shell overlay with transparent background

The overlay uses `gtk4-layer-shell` to create an always-on-top, click-through, transparent caption window anchored to the bottom-center of the screen. One critical implementation detail: **you must load `libgtk4-layer-shell.so` via `ctypes.CDLL` before any `gi` imports**, or the Wayland linking order will be wrong and initialization will fail silently.

```python
from ctypes import CDLL
CDLL('libgtk4-layer-shell.so')  # MUST come first

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gtk4LayerShell', '1.0')
from gi.repository import Gtk, Gdk, GLib
from gi.repository import Gtk4LayerShell as LayerShell
```

Window setup requires calling `init_for_window()` before the window is presented, then configuring layer, anchoring, and exclusive zone:

```python
window = Gtk.Window(application=app)
LayerShell.init_for_window(window)
LayerShell.set_layer(window, LayerShell.Layer.OVERLAY)       # Above everything
LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.NONE)  # Click-through
LayerShell.set_exclusive_zone(window, -1)                     # Don't push windows
LayerShell.set_namespace(window, "stt-overlay")
LayerShell.set_anchor(window, LayerShell.Edge.BOTTOM, True)   # Bottom-center
LayerShell.set_margin(window, LayerShell.Edge.BOTTOM, 60)
# NOT anchoring LEFT or RIGHT → window auto-centers horizontally
```

Transparency comes from GTK4's native RGBA support on Wayland — just apply CSS with `background-color: transparent` on the window and a semi-opaque background on the label container:

```python
css_provider = Gtk.CssProvider()
css_provider.load_from_string("""
    window { background-color: transparent; }
    .caption-box {
        background-color: rgba(0, 0, 0, 0.75);
        border-radius: 16px;
        padding: 16px 28px;
    }
    .caption-label { color: white; font-size: 22px; font-weight: bold; }
""")
Gtk.StyleContext.add_provider_for_display(
    Gdk.Display.get_default(), css_provider,
    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
)
```

Thread-safe label updates from the ASR thread use `GLib.idle_add()`, which schedules a callback on the GTK main loop. The callback must return `False` to execute only once. This pattern eliminates flickering because GTK batches rendering with its frame clock:

```python
def update_label(text):
    label.set_text(text)
    return False  # GLib.SOURCE_REMOVE

# From ASR thread:
GLib.idle_add(update_label, transcribed_text)
```

Install on Arch with `pacman -S gtk4 gtk4-layer-shell python-gobject`. This works on all wlroots-based compositors (Hyprland, Sway) but not GNOME Wayland. Check at runtime with `LayerShell.is_supported()`.

## PipeWire audio capture feeds 16kHz PCM to the model

The `sounddevice` library is the best choice for real-time microphone capture with PipeWire on Arch Linux. It uses PortAudio under the hood, which connects to PipeWire through the `pipewire-alsa` compatibility layer. When you request a 16kHz sample rate, **PipeWire handles resampling from the hardware's native 48kHz transparently** — no manual conversion needed.

The callback + queue pattern is the standard approach. The audio callback runs in a PortAudio thread, copies each chunk into a `queue.Queue`, and a separate ASR thread consumes it:

```python
import queue
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100ms at 16kHz — optimal for low-latency VAD
audio_queue = queue.Queue(maxsize=100)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata[:, 0].copy())  # Must copy — buffer is reused

stream = sd.InputStream(
    samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
    channels=1, dtype='float32',
    callback=audio_callback, latency='low'
)
```

The ASR thread then accumulates 100ms chunks until it has a full **1120ms native chunk** (roughly 11–12 chunks) before calling `conformer_stream_step()`. This bridging between the 100ms audio capture granularity and the 1120ms model chunk size is the key integration point:

```python
def asr_thread():
    buffer = []
    NATIVE_CHUNK = 17920  # 1120ms × 16kHz
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)
        total_samples = sum(len(c) for c in buffer)
        if total_samples >= NATIVE_CHUNK:
            audio = np.concatenate(buffer)[:NATIVE_CHUNK]
            tensor = torch.from_numpy(audio)
            # Feed to conformer_stream_step() ...
            buffer = [np.concatenate(buffer)[NATIVE_CHUNK:]]  # Keep remainder
```

Install with `pacman -S pipewire pipewire-audio pipewire-alsa portaudio` and `pip install sounddevice numpy`. An alternative lightweight option is `pasimple`, which talks directly to PipeWire's PulseAudio layer without PortAudio, but it only provides raw bytes requiring manual conversion.

## Text injection and the partial-hypothesis replacement problem

For typing transcribed text into the focused Hyprland window, **`wtype` is the primary tool** — it uses the `zwp_virtual_keyboard_v1` Wayland protocol, requires no daemon, and handles Unicode natively. The critical design decision is how to handle streaming partial hypotheses: the model continuously refines its transcription, so you need to erase previous partial text and replace it.

The most reliable approach combines two strategies. For **streaming partial updates**, batch backspace deletion and new text into a single `wtype` call to avoid timing issues between separate subprocess invocations:

```python
import subprocess

class StreamingTyper:
    def __init__(self):
        self.last_partial_len = 0

    def update_partial(self, new_text: str):
        args = ["wtype"]
        for _ in range(self.last_partial_len):
            args.extend(["-k", "BackSpace"])
        if new_text:
            args.extend(["--", new_text])
        subprocess.run(args, check=True)
        self.last_partial_len = len(new_text)

    def commit_final(self, final_text: str):
        """Use clipboard for final text — more reliable for longer strings."""
        args = ["wtype"]
        for _ in range(self.last_partial_len):
            args.extend(["-k", "BackSpace"])
        subprocess.run(args, check=True)
        subprocess.run(["wl-copy", "--", final_text], check=True)
        subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], check=True)
        self.last_partial_len = 0
```

The **clipboard approach** (`wl-copy` + `wtype Ctrl+V`) is the proven pattern used by hyprwhspr, hyprvoice, and sotto for final transcription output. It avoids character-by-character timing issues and handles any text length or Unicode content. For rapid sequential `wtype` calls, each invocation creates and destroys a virtual keyboard object, which can cause race conditions; batching operations into single calls (using wtype's `-s` sleep flag between stages if needed) mitigates this.

## Push-to-talk hotkeys via evdev and Hyprland IPC

Global hotkey capture on Wayland requires reading input devices at the kernel level, since Wayland provides no global keyboard shortcut protocol for third-party apps. **`python-evdev`** reads `/dev/input/event*` devices directly and is the approach used by whisper-overlay, sotto, and omarchy-speech-to-text.

```python
import asyncio
import evdev
from evdev import InputDevice, ecodes

HOTKEY = ecodes.KEY_RIGHTCTRL

async def hotkey_listener(device_path, on_press, on_release):
    device = InputDevice(device_path)
    async for event in device.async_read_loop():
        if event.type == ecodes.EV_KEY and event.code == HOTKEY:
            if event.value == 1:
                await on_press()
            elif event.value == 0:
                await on_release()

def find_keyboard():
    for path in evdev.list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities()
        if ecodes.EV_KEY in caps:
            keys = caps[ecodes.EV_KEY]
            if ecodes.KEY_A in keys and ecodes.KEY_ENTER in keys:
                return dev.path
    raise RuntimeError("No keyboard found")
```

The user must be in the `input` group (`sudo usermod -aG input $USER`). For users who prefer not to grant input-group permissions, **Hyprland IPC** provides an alternative: bind keys in `hyprland.conf` to execute your script, and use `bindr` for key-release detection to implement push-to-talk:

```
bind = , F9, exec, /path/to/start-recording.sh
bindr = , F9, exec, /path/to/stop-recording.sh
```

The hybrid tap/hold pattern from hyprwhspr is worth implementing: if the key is held longer than **300ms**, treat it as push-to-talk (stop on release); if tapped briefly, toggle recording on/off. This gives users flexibility without separate keybinds.

## Packaging for Arch Linux and the AUR

The recommended packaging strategy depends on system `python-pytorch-cuda` from the Arch `extra` repository rather than bundling PyTorch, and uses HuggingFace's auto-download mechanism for the model. Key PKGBUILD patterns:

```bash
depends=('python-pytorch-cuda' 'python-sounddevice' 'python-evdev'
         'python-huggingface-hub' 'gtk4' 'gtk4-layer-shell' 'python-gobject'
         'wtype' 'wl-clipboard' 'portaudio')
optdepends=('ydotool: fallback text injection')
makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')
```

The NeMo toolkit itself is the heaviest dependency — `nemo_toolkit[asr]` pulls in the full ASR stack. For the PKGBUILD, install via pip into the package directory during `package()`. **Do not bundle the 0.6B model in the package.** Instead, auto-download on first run to `~/.cache/huggingface/hub/` (HuggingFace's default, respecting `HF_HOME`). Provide a `--download-model` CLI flag for explicit pre-download.

For XDG compliance, store configuration in `$XDG_CONFIG_HOME/your-app/` and any runtime data in `$XDG_DATA_HOME/your-app/`. The model itself lives in the HuggingFace cache which is already XDG-aware.

## Conclusion

The complete architecture chains five independently testable components: `sounddevice` captures 100ms audio chunks at 16kHz → a buffer accumulates these into 1120ms native chunks → `conformer_stream_step()` produces streaming transcription with ~160ms algorithmic latency → `GLib.idle_add()` pushes text to the GTK4 layer-shell overlay → `wtype` injects finalized text via backspace-and-retype or clipboard paste. The most underappreciated detail across this entire stack is the **chunk size mismatch**: the model demands 1120ms chunks while good audio capture operates at 100ms granularity, making the accumulation buffer the critical integration point. Existing projects like whisper-overlay (server-client, dual-model) and sotto (single-binary, whisper.cpp) validate that this GTK4 + evdev + wtype architecture works reliably on Hyprland in production. The 0.6B model's **2–3 GB VRAM footprint** leaves ample headroom on a 16 GB RTX 5080 for concurrent GPU workloads.