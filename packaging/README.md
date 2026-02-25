# Packaging and AUR workflow

This directory contains Arch Linux packaging files for ShuVoice.

## Files

- `PKGBUILD` — AUR package recipe (`shuvoice-git`)
- `.SRCINFO` — generated metadata required by AUR (must match `PKGBUILD`)
- `systemd/user/shuvoice.service` — installed user service unit
- `aur/` — staging files for related AUR packages and upstream patches

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
4. Regenerate `.SRCINFO`:
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
# Install with preferred Sherpa runtime provider
yay -S --needed python-sherpa-onnx-bin shuvoice-git

systemctl --user daemon-reload
systemctl --user enable --now shuvoice.service
systemctl --user status shuvoice.service --no-pager
shuvoice control status

shuvoice setup --skip-model-download --skip-preflight
shuvoice preflight
```

Dependency failure behavior (no restart storm):

```bash
sudo pacman -Rns --noconfirm python-sherpa-onnx-bin
systemctl --user restart shuvoice.service
systemctl --user show -p ExecMainStatus -p NRestarts shuvoice.service
# Expect ExecMainStatus=78 and restarts blocked by RestartPreventExitStatus=78

# Recover
yay -S --needed python-sherpa-onnx-bin
systemctl --user restart shuvoice.service
```

Branding path check in packaged context:

```bash
cd /tmp
/usr/bin/python - <<'PY'
from shuvoice.splash import _find_logo
from shuvoice.wizard.ui import find_logo
print(_find_logo())
print(find_logo())
PY
```

## Notes

- `shuvoice-git` intentionally tracks the latest git commit (VCS package).
- `shuvoice-git` keeps a hard runtime dependency on `python-sherpa-onnx`.
  - Preferred provider for end users: `python-sherpa-onnx-bin`
    (`provides=('python-sherpa-onnx')`).
  - Source provider (`python-sherpa-onnx` from split `sherpa-onnx`) remains
    compatible once upstream GCC 15 fixes are merged.
- AUR staging files for the binary provider live in:
  `packaging/aur/python-sherpa-onnx-bin/`.
- NeMo and Moonshine remain optional runtime backend stacks.
