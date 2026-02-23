# Packaging and AUR workflow

This directory contains Arch Linux packaging files for ShuVoice.

## Files

- `PKGBUILD` — AUR package recipe (`shuvoice-git`)
- `.SRCINFO` — generated metadata required by AUR (must match `PKGBUILD`)
- `systemd/user/shuvoice.service` — installed user service unit

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

## Notes

- `shuvoice-git` intentionally tracks the latest git commit (VCS package).
- ASR backends are optional runtime dependencies. Base package installs the app,
  service unit, docs, and CLI entry points.
