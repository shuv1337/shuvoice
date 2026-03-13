# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## MeloTTS Venv

- **Location**: `~/.local/share/shuvoice/melotts-venv/` (managed by setup automation)
- **Python version**: 3.12 (confirmed working; system Python 3.14 is NOT compatible)
- **Key packages**: melotts, torch, unidic
- **Size**: ~9GB (mostly PyTorch)
- **Models**: Downloaded to HuggingFace cache (`~/.cache/huggingface/hub/models--myshell-ai--MeloTTS-*`)
- **Model sizes**: ~200MB per language model

## Python Versions

- System Python: 3.14 (Arch Linux)
- Project venv: 3.12+ recommended
- MeloTTS venv: must use 3.12 (compatibility issues with 3.13+)
- uv provides Python 3.12.12 for venv creation

## Test MeloTTS Venv

A test venv exists at `/tmp/melotts-test-venv` from feasibility testing. Workers should NOT depend on it — the production path uses `~/.local/share/shuvoice/melotts-venv/`.
