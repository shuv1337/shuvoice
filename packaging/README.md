# Packaging and AUR workflow

This directory contains Arch Linux packaging files for ShuVoice.

## Files

- `PKGBUILD` — AUR package recipe (`shuvoice-git`)
- `.SRCINFO` — generated metadata required by AUR (must match `PKGBUILD`)
- `systemd/user/shuvoice.service` — installed user service unit

## Rust package build

The AUR package builds the full Rust **desktop** feature set from the locked
workspace:

```bash
cargo fetch --locked --target x86_64-unknown-linux-gnu
cargo build --release --locked -p shuvoice-cli --features desktop \
  --bin shuvoice --bin shuvoice-waybar
```

Installed artifacts:

| Path | Contents |
|------|----------|
| `/usr/bin/shuvoice` | Desktop application / CLI |
| `/usr/bin/shuvoice-waybar` | Waybar helper |
| `/usr/lib/systemd/user/shuvoice.service` | User unit (`ExecStart=/usr/bin/shuvoice`, `RestartPreventExitStatus=78`, `RUST_LOG=info`) |
| `/usr/lib/shuvoice/workers/` | Optional worker packages (`shuvoice_worker_proto`, `nemo_asr`, `moonshine_asr`, `melotts`) — `.py` source only |
| `/usr/share/doc/shuvoice-git/` | README, example config, branding/screenshot assets |
| `/usr/share/licenses/shuvoice-git/LICENSE` | MIT license |

Worker discovery matches the runtime contract: env `SHUVOICE_WORKERS_DIR`, then
`/usr/lib/shuvoice/workers`, then `/usr/libexec/shuvoice/workers`.

### Dependencies

- **depends**: GTK4 + layer-shell, ALSA/PipeWire audio stack, `wtype`, `wl-clipboard`
- **optdepends**: `python` (worker interpreter), `uv` (preferred for isolated worker venvs via setup), `piper-tts`, `xdotool`, `ydotool`
- **makedepends**: `git`, `cargo`/`rust`, `pkgconf`, `clang`, GTK/layer-shell, `alsa-lib`, `pipewire` (static sherpa-onnx uses upstream prebuilts; no system `sherpa-onnx` package required)

Python is **not** a hard dependency of the application shell. Native Sherpa is
linked statically into the Rust binary. Optional NeMo/Moonshine/MeloTTS engines
run only as out-of-process workers. ML packages install into **isolated XDG data
venvs** through `shuvoice setup --install-missing` (prefer `uv`); system
`python-pytorch` packages are not declared because they do not feed those venvs.

## Update checklist

1. Edit `PKGBUILD` (`pkgrel`, deps, install paths, etc.).
2. Validate package metadata:
   ```bash
   cd packaging
   namcap PKGBUILD
   ```
3. Build and validate package:
   ```bash
   makepkg -sf
   namcap shuvoice-git-*.pkg.tar.zst
   ```
4. Regenerate `.SRCINFO` (do not hand-edit):
   ```bash
   makepkg --printsrcinfo > .SRCINFO
   ```
5. Push `PKGBUILD` + `.SRCINFO` to AUR repo:
   ```bash
   git clone ssh://aur@aur.archlinux.org/shuvoice-git.git /tmp/aur-shuvoice-git
   cp PKGBUILD .SRCINFO /tmp/aur-shuvoice-git/
   cd /tmp/aur-shuvoice-git
   git add PKGBUILD .SRCINFO
   git commit -m "Update shuvoice-git"
   git push origin master
   ```

## Packaged runtime validation (post-install)

```bash
# Install the Rust package
yay -S --needed shuvoice-git

systemctl --user daemon-reload
systemctl --user enable --now shuvoice.service
systemctl --user status shuvoice.service --no-pager
shuvoice control status

shuvoice setup --skip-model-download --skip-preflight
shuvoice preflight
shuvoice config validate
```

Dependency failure behavior (no restart storm):

```bash
# Simulate a hard dependency/config failure that exits 78
systemctl --user show -p ExecMainStatus -p NRestarts shuvoice.service
# Expect RestartPreventExitStatus=78 to block restart loops when the binary
# exits with status 78.
```

Worker tree check:

```bash
ls /usr/lib/shuvoice/workers/shuvoice_worker_proto/__init__.py
ls /usr/lib/shuvoice/workers/nemo_asr/__main__.py
ls /usr/lib/shuvoice/workers/moonshine_asr/__main__.py
ls /usr/lib/shuvoice/workers/melotts/__main__.py
# Optional engines still need their own Python venv + packages via setup.
```

Branding assets (packaged docs tree):

```bash
ls /usr/share/doc/shuvoice-git/docs/assets/branding/
ls /usr/share/doc/shuvoice-git/docs/assets/screenshots/
```

## Notes

- `shuvoice-git` intentionally tracks the latest git commit (VCS package).
- Desktop feature set is the packaging default (`--features desktop`).
- Release builds use `--locked` against the repo `Cargo.lock`.
- Workspace release LTO is controlled by `Cargo.toml`; PKGBUILD sets `options=('!lto')` so pacman does not double-apply LTO.
