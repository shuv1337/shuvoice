# Contributing to ShuVoice

Thanks for your interest in contributing to ShuVoice!

## Development setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install project dependencies:

```bash
pip install -e .[dev]
```

For ASR backend-specific work, install one of:

```bash
pip install -e .[asr-nemo]
pip install -e .[asr-sherpa]
pip install -e .[asr-moonshine]
```

## Recommended local checks

Run these before opening a pull request:

```bash
ruff check shuvoice tests
ruff format --check shuvoice tests
pytest -m "not gui" -v
```

If your change touches GTK/UI behavior and your environment supports it, also run:

```bash
pytest -m gui -v
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
