# Contributing to ShuVoice

Thanks for your interest in contributing to ShuVoice!

Need logo assets for docs or release notes? See `docs/BRANDING.md`.

## Development setup

1. Clone the repository.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Install project dependencies (creates a virtualenv automatically):

```bash
uv sync --dev
```

For ASR backend-specific work, install one of:

```bash
uv sync --extra asr-nemo
uv sync --extra asr-sherpa
uv sync --extra asr-moonshine
```

## Recommended local checks

Run these before opening a pull request:

```bash
uv run ruff check shuvoice tests
uv run ruff format --check shuvoice tests
uv run pytest -m "not gui" -v
```

If your change touches GTK/UI behavior and your environment supports it, also run:

```bash
uv run pytest -m gui -v
```

## Commit and PR expectations

- Keep commits focused and descriptive.
- Include tests for behavior changes when practical.
- Update documentation for any user-facing change.
- Ensure no generated artifacts are committed (for example: `build/`, `dist/`, coverage outputs, cache directories).

## Reporting issues

Please open an issue with:

- steps to reproduce
- expected behavior
- actual behavior
- logs and environment details (OS, Python version, backend used)

## Code of Conduct

By participating in this project, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
