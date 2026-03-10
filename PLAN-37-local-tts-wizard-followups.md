## PLAN-37: Local TTS Wizard Follow-ups

### Goal
Finish the last mile for ShuVoice local Piper TTS now that wizard support, config persistence, and sample-rate-aware playback are wired in.

### Current state
- Local Piper can now be selected in the setup wizard.
- Wizard persists `tts_local_model_path` and local voice selection.
- TTS playback/preflight use backend-declared sample rates.
- Full test suite is passing.

### Next steps

#### 1. Manual QA the real wizard flow
- [x] Run `shuvoice wizard`
- [x] Select `Local Piper` in the TTS section
- [x] Enter a real `.onnx` model path or directory containing Piper voices
- [x] Finish setup and verify `~/.config/shuvoice/config.toml` contains:
  - [x] `tts_backend = "local"`
  - [x] `tts_local_model_path = "..."`
  - [x] sensible `tts_default_voice_id`
- [x] Trigger `tts_speak` and confirm audio plays correctly (latency=0.84s, plays through correctly)
- [x] Verify behavior both with:
  - [x] a single `.onnx` file (works: 1 voice discovered, synthesis OK)
  - [x] a directory of multiple `.onnx` voices (works: 2 voices discovered)

#### 2. Verify sample-rate behavior against real Piper models
- [x] Confirm `.onnx.json` sidecar sample rate is detected correctly (22050 Hz from sidecar)
- [x] Confirm fallback behavior is acceptable when sidecar metadata is missing (falls back to 22050 Hz with warning)
- [x] Check preflight output for local TTS device/sample-rate reporting (`local deps OK (2 voices, sample_rate=22050Hz)`)

#### 3. Polish wizard UX if needed
- [ ] Decide whether the text-entry model path is good enough
- [ ] If not, add a file/directory chooser for local model selection
- [ ] Consider adding clearer inline guidance for:
  - [ ] single-model file mode
  - [ ] voice-directory mode
  - [ ] automatic first-voice selection

#### 4. Optional setup improvements
- [ ] Decide whether ShuVoice should help users install Piper or acquire models
- [ ] If yes, scope a small follow-up for setup guidance or automation
- [ ] If no, keep the current "existing local model required" workflow and document it clearly

#### 5. Bugs found and fixed during QA
- [x] `piper-tts` binary name not recognized (AUR installs as `piper-tts`, not `piper`) — added multi-name resolution
- [x] `synthesize_stream` ValueError: `communicate()` tried to flush already-closed stdin — switched to `wait()` + `stderr.read()`

### Validation
- [x] `uv run ruff check shuvoice tests`
- [x] `uv run pytest` (504 passed, 2 skipped)
- [x] Real local Piper speech works from wizard-generated config

### Relevant files
- `shuvoice/tts_local.py`
- `shuvoice/tts_base.py`
- `shuvoice/tts_player.py`
- `shuvoice/app.py`
- `shuvoice/cli/commands/preflight.py`
- `shuvoice/wizard/__init__.py`
- `shuvoice/wizard_state.py`
- `shuvoice/wizard/actions.py`
- `shuvoice/waybar/format.py`
- `README.md`
- `examples/config.toml`
- `AGENTS.md`
