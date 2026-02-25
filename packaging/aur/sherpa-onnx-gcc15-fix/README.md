# sherpa-onnx AUR GCC 15 source-build fix (staging)

This folder tracks the proposed upstream AUR fix for the GCC 15
`-Werror=format-security` interaction seen in `sherpa-onnx` source builds.

## Contents

- `PKGBUILD.diff` — minimal patch for the upstream `sherpa-onnx` AUR PKGBUILD.

## Problem summary

On current Arch toolchains (GCC 15), the source build can fail while compiling
`espeak-ng` third-party code due to warning flag interaction where
`-Werror=format-security` becomes fatal in subbuild paths that also manipulate
format warnings.

## Patch strategy

The patch:

- keeps global hardening flags intact,
- adds only scoped `-Wno-error=format-security` for this package build,
- applies the same scoped C/C++ flags to both the CMake binary build and the
  Python-wheel build path.

## Reproduction checklist (clean chroot)

```bash
# Example tooling; adjust to your chroot setup
git clone https://aur.archlinux.org/sherpa-onnx.git /tmp/sherpa-onnx
cd /tmp/sherpa-onnx

# Build in clean environment and capture logs
extra-x86_64-build 2>&1 | tee build.log
```

If chroot tooling is unavailable, use:

```bash
makepkg -srf 2>&1 | tee build.log
```

## Upstream submission checklist

1. Reproduce failure with current upstream PKGBUILD.
2. Apply `PKGBUILD.diff`.
3. Rebuild in clean chroot.
4. Post AUR maintainer comment with:
   - failing log excerpt,
   - successful post-patch log excerpt,
   - attached diff.

Suggested AUR comment summary:

> GCC 15 source builds fail in third-party espeak-ng with format-security
> warnings promoted to errors. Proposed PKGBUILD patch adds scoped
> `-Wno-error=format-security` (keeps hardening, limits scope to this package
> build) and unblocks both binary and python wheel build paths.
