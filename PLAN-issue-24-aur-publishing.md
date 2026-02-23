# PLAN: Publish ShuVoice to AUR

**Issue**: [#24 — Publish ShuVoice to AUR (Arch User Repository)](https://github.com/shuv1337/shuvoice/issues/24)

---

## Context

ShuVoice already ships an Arch-oriented `packaging/PKGBUILD` for the
`shuvoice-git` package, a systemd user unit, and documents Arch system packages
in `README.md`. However:

- The PKGBUILD has never been validated in a clean chroot.
- No `.SRCINFO` has been generated.
- No AUR repository has been created.
- No SSH key is configured for AUR pushes (`~/.ssh/aur` does not exist).
- No git tags exist (current version is `0.1.0` from `pyproject.toml`).
- The PKGBUILD has `python-pytorch-cuda` as a hard `depends` entry — this
  forces NVIDIA GPU on every installer, but ShuVoice supports CPU-only backends
  (Sherpa CPU, Moonshine).
- The PKGBUILD does not mention the `shuvoice-waybar` entry point (installed
  automatically by `python -m installer` from `pyproject.toml [project.scripts]`,
  but not documented in the package).
- `devtools` is not installed on this system (needed for `extra-x86_64-build`
  clean-chroot validation).

---

## Goals

1. Publish a working `shuvoice-git` package to the AUR.
2. Ensure the PKGBUILD passes `namcap` and builds cleanly in a chroot.
3. Add AUR install instructions to `README.md`.
4. Establish a repeatable workflow for future PKGBUILD updates.

## Non-goals

- **Release-tagged `shuvoice` package**: There are no git tags yet and the
  project is at `0.1.0` pre-release. A stable `shuvoice` AUR package that
  tracks versioned tarballs is deferred until a proper release tagging workflow
  exists. Starting with `shuvoice-git` only is the standard AUR practice for
  projects without tagged releases.
- Automated CI-driven AUR publishing (can be added later).
- Packaging ASR model weights inside the AUR package.
- Providing a binary package (`shuvoice-git-bin`).

---

## Current-State Findings

### `packaging/PKGBUILD`

| Aspect | Current state | Issue |
|---|---|---|
| `pkgname` | `shuvoice-git` | ✅ Correct for VCS package |
| `pkgver()` | `0.1.0.r<count>.g<hash>` | ✅ Standard VCS versioning |
| `source` | `git+https://github.com/shuv1337/shuvoice.git` | ✅ |
| `depends` | includes `python-pytorch-cuda` | ⚠️ Should be `optdepends` — forces NVIDIA GPU |
| `depends` | includes `python-tomli` | ⚠️ Unnecessary for Python ≥3.11 (tomllib is stdlib). Only needed as optdepends for 3.10 |
| `optdepends` | lists `python-nemo-toolkit` | ✅ Appropriate as optional |
| `optdepends` | missing `sherpa-onnx`, `moonshine` mention | ⚠️ Users won't know about alternatives |
| `makedepends` | `git python-build python-installer python-hatchling` | ✅ |
| `license` | `MIT` | ⚠️ Should use SPDX format: `license=('MIT')` is fine but `license` file not installed |
| `package()` | installs wheel, README, example config, systemd unit | ✅ Good coverage |
| `package()` | missing LICENSE install | ⚠️ Arch guidelines recommend including license |
| `sha256sums` | `SKIP` | ✅ Correct for VCS source |
| `.SRCINFO` | Does not exist | ❌ Required for AUR submission |

### `pyproject.toml` (relevant sections)

- `[project.scripts]`: `shuvoice` and `shuvoice-waybar` — both get installed
  by `python -m installer`, so the package will ship both binaries to
  `/usr/bin/` automatically.
- `version = "0.1.0"` — no dynamic versioning.
- Build system: `hatchling`.

### `packaging/systemd/user/shuvoice.service`

- `ExecStart=/usr/bin/shuvoice` — matches wheel install path. ✅

### AUR account / SSH

- No `~/.ssh/aur` key pair found. Must be created and registered at
  `https://aur.archlinux.org/account/` before pushing.

### System tooling

- `makepkg` 7.1.0 available. ✅
- `devtools` not installed (needed for `extra-x86_64-build`).

---

## Package Scope Recommendation

**`shuvoice-git` only** (for now).

Rationale:
- No git tags or release tarballs exist.
- Project is at `0.1.0`, still pre-release.
- `-git` packages are the AUR standard for active, untagged projects.
- A release-tracking `shuvoice` package can be added when a tag-based release
  workflow is established (potential follow-up issue).

---

## Proposed Approach

### Phase 1: Fix PKGBUILD issues

Move `python-pytorch-cuda` and `python-tomli` to `optdepends`. Add missing
`optdepends` entries for all three ASR backends. Install LICENSE file. These
changes make the package installable on CPU-only systems and comply with Arch
packaging guidelines.

### Phase 2: Local validation

Build in a clean chroot (`extra-x86_64-build` via `devtools`) and run `namcap`
on both PKGBUILD and the built package. Fix any issues found.

### Phase 3: AUR submission

Generate `.SRCINFO`, create the AUR git repo, push, and verify the package
page appears correctly.

### Phase 4: Documentation

Add AUR installation section to `README.md` and update `AGENTS.md` if needed.

---

## Task Breakdown

### Phase 1 — PKGBUILD fixes

- [ ] **1.1** Move `python-pytorch-cuda` from `depends` to `optdepends` with
  description: `'python-pytorch-cuda: GPU acceleration for NeMo/Sherpa CUDA backends'`

- [ ] **1.2** Move `python-tomli` from `depends` to `optdepends` with
  description: `'python-tomli: TOML config parsing for Python < 3.11'`
  (Python ≥3.11 has `tomllib` in stdlib; Arch system Python is 3.12+)

- [ ] **1.3** Expand `optdepends` to document all backend options clearly:
  ```
  optdepends=(
    'python-pytorch-cuda: GPU acceleration for NeMo and Sherpa CUDA backends'
    'python-tomli: TOML config parsing for Python < 3.11'
    'python-nemo-toolkit: NeMo ASR backend (pip: nemo-toolkit[asr])'
    'sherpa-onnx: Sherpa ONNX ASR backend (pip: sherpa-onnx)'
    'python-useful-moonshine-onnx: Moonshine ASR backend (pip: useful-moonshine-onnx)'
    'ydotool: alternative text injection utility'
    'espeak-ng: TTS engine for round-trip testing'
  )
  ```
  Note: `sherpa-onnx` and `python-useful-moonshine-onnx` are not in official
  repos — include pip hints in the description strings so users know.

- [ ] **1.4** Add LICENSE file installation to `package()`:
  ```bash
  install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
  ```

- [ ] **1.5** Update `pkgdesc` to be more descriptive:
  ```
  pkgdesc="Streaming speech-to-text overlay for Hyprland/Wayland with pluggable ASR backends"
  ```

- [ ] **1.6** Verify `license=('MIT')` field. The current format is acceptable
  per current `makepkg` but confirm no `namcap` warnings. (SPDX identifier
  `license=('MIT')` is the modern form and already used.)

- [ ] **1.7** Add `provides=('shuvoice')` and `conflicts=('shuvoice')` to the
  PKGBUILD so a future release-tracking `shuvoice` package won't conflict.

### Phase 2 — Local validation

- [ ] **2.1** Install `devtools` if not present:
  ```bash
  sudo pacman -S devtools
  ```

- [ ] **2.2** Install `namcap` if not present:
  ```bash
  sudo pacman -S namcap
  ```

- [ ] **2.3** Run `namcap` on the PKGBUILD:
  ```bash
  cd packaging && namcap PKGBUILD
  ```
  Fix any errors/warnings (dependency issues, missing fields, etc.).

- [ ] **2.4** Test `makepkg` in a temporary directory (non-chroot, quick sanity):
  ```bash
  TMPDIR=$(mktemp -d)
  cp packaging/PKGBUILD "$TMPDIR/"
  cd "$TMPDIR" && makepkg -s --noconfirm
  ```

- [ ] **2.5** Run `namcap` on the built package:
  ```bash
  namcap "$TMPDIR"/shuvoice-git-*.pkg.tar.zst
  ```
  Fix any issues (missing deps, incorrect permissions, dangling symlinks).

- [ ] **2.6** *(Stretch)* Clean-chroot build with `devtools`:
  ```bash
  cd packaging && extra-x86_64-build
  ```
  This is the gold standard for AUR validation — catches any undeclared
  `makedepends` or `depends`. May require initial chroot setup time.

- [ ] **2.7** Verify installed file layout from the built package:
  ```bash
  tar -tf shuvoice-git-*.pkg.tar.zst | grep -E '(bin/|systemd/|share/doc|share/licenses)'
  ```
  Expected files:
  - `usr/bin/shuvoice`
  - `usr/bin/shuvoice-waybar`
  - `usr/lib/systemd/user/shuvoice.service`
  - `usr/share/doc/shuvoice-git/README.md`
  - `usr/share/doc/shuvoice-git/config.toml.example`
  - `usr/share/licenses/shuvoice-git/LICENSE`

- [ ] **2.8** Test install the package and verify functionality:
  ```bash
  sudo pacman -U shuvoice-git-*.pkg.tar.zst
  shuvoice --help
  shuvoice --preflight  # (will fail on missing ASR deps, but binary should run)
  shuvoice-waybar --help 2>/dev/null || shuvoice-waybar status
  systemctl --user cat shuvoice.service
  sudo pacman -R shuvoice-git  # clean up
  ```

### Phase 3 — AUR submission

- [ ] **3.1** Set up AUR SSH key (if not already done):
  ```bash
  ssh-keygen -t ed25519 -f ~/.ssh/aur -C "aur@shuvoice"
  # Add to ~/.ssh/config:
  #   Host aur.archlinux.org
  #     IdentityFile ~/.ssh/aur
  #     User aur
  # Copy public key to AUR account: https://aur.archlinux.org/account/
  cat ~/.ssh/aur.pub
  ```

- [ ] **3.2** Verify AUR SSH access:
  ```bash
  ssh -T aur@aur.archlinux.org
  # Expected: "Interactive shell is disabled."
  ```

- [ ] **3.3** Generate `.SRCINFO` from the validated PKGBUILD:
  ```bash
  cd packaging && makepkg --printsrcinfo > .SRCINFO
  ```

- [ ] **3.4** Clone the (empty) AUR package repo:
  ```bash
  git clone ssh://aur@aur.archlinux.org/shuvoice-git.git /tmp/aur-shuvoice-git
  ```
  (This creates a new AUR package if it doesn't exist on first push.)

- [ ] **3.5** Copy PKGBUILD and .SRCINFO into the AUR repo:
  ```bash
  cp packaging/PKGBUILD /tmp/aur-shuvoice-git/PKGBUILD
  cp packaging/.SRCINFO /tmp/aur-shuvoice-git/.SRCINFO
  ```

- [ ] **3.6** Commit and push to AUR:
  ```bash
  cd /tmp/aur-shuvoice-git
  git add PKGBUILD .SRCINFO
  git commit -m "Initial upload: shuvoice-git 0.1.0"
  git push origin master
  ```

- [ ] **3.7** Verify the AUR page is live:
  ```
  https://aur.archlinux.org/packages/shuvoice-git
  ```
  Check: description, dependencies, maintainer, license all display correctly.

### Phase 4 — Documentation updates

- [ ] **4.1** Add AUR install section to `README.md` (after the "System
  packages" section or as a new top-level "Installation" section):
  ```markdown
  ## Installation (AUR)

  ShuVoice is available on the AUR as [`shuvoice-git`](https://aur.archlinux.org/packages/shuvoice-git):

  ```bash
  # Using yay
  yay -S shuvoice-git

  # Using paru
  paru -S shuvoice-git
  ```

  After installation, enable the systemd user service:

  ```bash
  systemctl --user enable --now shuvoice.service
  ```

  ASR backends are optional dependencies — install your preferred backend
  separately (see [Configuration](#configuration) for backend details).
  ```

- [ ] **4.2** Store `.SRCINFO` in the main repo at `packaging/.SRCINFO` for
  reference (and add a comment that it must be regenerated before AUR pushes):
  ```bash
  # packaging/.SRCINFO is a reference copy.
  # Regenerate before AUR push: cd packaging && makepkg --printsrcinfo > .SRCINFO
  ```

- [ ] **4.3** Add a `packaging/README.md` documenting the AUR update workflow:
  - How to regenerate `.SRCINFO`
  - How to push updates to the AUR repo
  - Checklist: bump `pkgrel`, re-test, regenerate `.SRCINFO`, push

- [ ] **4.4** Update `AGENTS.md` if any packaging paths, commands, or
  dependencies changed during this work.

---

## Validation Checklist

### Local (quick)

```bash
# Lint PKGBUILD
cd packaging && namcap PKGBUILD

# Build package
TMPDIR=$(mktemp -d)
cp PKGBUILD "$TMPDIR/" && cd "$TMPDIR" && makepkg -sf --noconfirm

# Lint built package
namcap shuvoice-git-*.pkg.tar.zst

# Check file layout
tar -tf shuvoice-git-*.pkg.tar.zst | grep -E '(bin/|systemd/|licenses/|doc/)'

# Test install
sudo pacman -U shuvoice-git-*.pkg.tar.zst --noconfirm
shuvoice --help
shuvoice-waybar status 2>&1 | head -5
systemctl --user cat shuvoice.service
sudo pacman -R shuvoice-git --noconfirm
```

### Clean chroot (thorough)

```bash
cd packaging && extra-x86_64-build
# Confirms all makedepends/depends are declared correctly
```

### AUR page verification

```bash
# After push:
curl -s "https://aur.archlinux.org/rpc/?v=5&type=info&arg[]=shuvoice-git" | python -m json.tool
# Check: Name, Version, Description, Maintainer, License, NumVotes (0 initially)
```

### End-to-end install from AUR (post-publish)

```bash
# On a separate Arch system or clean VM:
yay -S shuvoice-git --noconfirm
shuvoice --help
shuvoice --preflight
systemctl --user enable shuvoice.service
systemctl --user start shuvoice.service
systemctl --user status shuvoice.service
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `python-pytorch-cuda` not available in all Arch setups | Package fails to install on non-NVIDIA or CPU-only systems | Move to `optdepends` (Task 1.1) |
| ASR Python deps not in official Arch repos | Users can't install NeMo/Sherpa/Moonshine via pacman | Document pip/uv install paths in `optdepends` descriptions and README |
| `hatchling` build fails in chroot due to missing build isolation deps | `makepkg` fails | `--no-isolation` flag is already used; verify `makedepends` includes `python-hatchling` |
| AUR naming conflict with future `shuvoice` release package | Confusion when release package is added | Add `provides=('shuvoice')` and `conflicts=('shuvoice')` now (Task 1.7) |
| `.SRCINFO` drifts out of sync with PKGBUILD | AUR shows stale metadata | Document regeneration workflow in `packaging/README.md` (Task 4.3) |
| SSH key not registered / wrong AUR account | Push rejected | Verify SSH access before attempting push (Task 3.2) |

---

## Rollback Strategy

- **AUR package removal**: If the published package is broken, either:
  - Push a fixed PKGBUILD + `.SRCINFO` immediately, or
  - Disown/delete the AUR package via the web interface at
    `https://aur.archlinux.org/packages/shuvoice-git` (Package Actions →
    Disown / Request Deletion).
- **PKGBUILD revert in main repo**: `git revert` the PKGBUILD changes.
  The AUR repo is separate — revert there independently if needed.

---

## Definition of Done

1. ✅ `packaging/PKGBUILD` passes `namcap` with no errors.
2. ✅ Package builds successfully in a clean environment (chroot or fresh
   `makepkg -s`).
3. ✅ Built package contains: `shuvoice` binary, `shuvoice-waybar` binary,
   systemd user unit, LICENSE, README, example config.
4. ✅ `shuvoice --help` and `shuvoice --preflight` work from installed package.
5. ✅ `.SRCINFO` generated and committed to AUR repo.
6. ✅ `https://aur.archlinux.org/packages/shuvoice-git` is live and shows
   correct metadata.
7. ✅ `README.md` includes AUR installation instructions.
8. ✅ `packaging/README.md` documents the AUR update workflow.
9. ✅ `AGENTS.md` updated if any packaging paths or commands changed.
