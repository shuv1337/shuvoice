# AUR staging artifacts

This directory stores AUR-related packaging artifacts maintained alongside the
ShuVoice repo.

## Subdirectories

- `python-sherpa-onnx-bin/` — binary-wheel provider package files.
- `sherpa-onnx-gcc15-fix/` — upstream patch proposal for source-build GCC 15 failures.

## Maintainer workflow

- Validate package changes locally (`makepkg -sf`, `namcap`, smoke import checks).
- Regenerate `.SRCINFO` after every `PKGBUILD` change.
- Copy the package files into their corresponding AUR git repos and push.
