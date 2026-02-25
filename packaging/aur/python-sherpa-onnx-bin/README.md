# python-sherpa-onnx-bin (AUR helper package)

This folder tracks the AUR packaging files for a binary-wheel provider of
`python-sherpa-onnx`.

The package installs both upstream wheels:

- `sherpa-onnx-core` (shared runtime libraries)
- `sherpa-onnx` (Python wrapper bindings)

Why it exists:

- `shuvoice-git` depends on `python-sherpa-onnx`.
- Source-build `sherpa-onnx` can fail on some Arch toolchain combinations.
- `python-sherpa-onnx-bin` provides a fast prebuilt-wheel path while keeping the
  dependency contract stable (`provides=('python-sherpa-onnx')`).

## Files

- `PKGBUILD`
- `.SRCINFO`

## Local validation

```bash
cd packaging/aur/python-sherpa-onnx-bin
makepkg -sf
namcap PKGBUILD
namcap *.pkg.tar.zst

# Verify import/version after install
/usr/bin/python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"

# Rootless verification against built package tree
PYTHONPATH="pkg/python-sherpa-onnx-bin/usr/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages" \
  /usr/bin/python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"
```

## Update procedure for a new sherpa-onnx release

1. Pick target version (`pkgver`) that has wheels for supported ABIs.
2. Pull wheel metadata from PyPI JSON:

```bash
python - <<'PY'
import json, urllib.request
version = "1.12.26"  # change me

targets = {
    "sherpa-onnx": [
        "cp312-manylinux2014_x86_64.manylinux_2_17_x86_64",
        "cp313-manylinux2014_x86_64.manylinux_2_17_x86_64",
        "cp314-manylinux2014_x86_64.manylinux_2_17_x86_64",
        "cp312-manylinux2014_aarch64.manylinux_2_17_aarch64",
        "cp313-manylinux2014_aarch64.manylinux_2_17_aarch64",
        "cp314-manylinux2014_aarch64.manylinux_2_17_aarch64",
    ],
    "sherpa-onnx-core": [
        "py3-none-manylinux2014_x86_64",
        "py3-none-manylinux2014_aarch64",
    ],
}

for project, wanted in targets.items():
    print(f"[{project}]")
    data = json.load(
        urllib.request.urlopen(f"https://pypi.org/pypi/{project}/{version}/json", timeout=10)
    )
    for item in data["urls"]:
        fn = item["filename"]
        if any(tag in fn for tag in wanted):
            print(fn)
            print("  url:", item["url"])
            print("  sha256:", item["digests"]["sha256"])
    print()
PY
```

3. Update wheel filenames, URLs, and checksums in `PKGBUILD`.
4. Regenerate `.SRCINFO`:

```bash
makepkg --printsrcinfo > .SRCINFO
```

5. Re-run local validation commands.
6. Push `PKGBUILD` + `.SRCINFO` to the AUR repo `python-sherpa-onnx-bin`.
