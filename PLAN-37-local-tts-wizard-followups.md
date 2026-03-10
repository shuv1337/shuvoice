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
- [ ] Run `shuvoice wizard`
- [ ] Select `Local Piper` in the TTS section
- [ ] Enter a real `.onnx` model path or directory containing Piper voices
- [ ] Finish setup and verify `~/.config/shuvoice/config.toml` contains:
  - [ ] `tts_backend = "local"`
  - [ ] `tts_local_model_path = "..."`
  - [ ] sensible `tts_default_voice_id`
- [ ] Trigger `tts_speak` and confirm audio plays correctly
- [ ] Verify behavior both with:
  - [ ] a single `.onnx` file
  - [ ] a directory of multiple `.onnx` voices

#### 2. Verify sample-rate behavior against real Piper models
- [ ] Confirm `.onnx.json` sidecar sample rate is detected correctly
- [ ] Confirm fallback behavior is acceptable when sidecar metadata is missing
- [ ] Check preflight output for local TTS device/sample-rate reporting

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

### Validation
- [ ] `uv run ruff check shuvoice tests`
- [ ] `uv run pytest`
- [ ] Real local Piper speech works from wizard-generated config

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
