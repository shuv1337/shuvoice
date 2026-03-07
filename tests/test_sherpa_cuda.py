from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from shuvoice import sherpa_cuda


def _make_fake_sherpa_layout(tmp_path: Path) -> Path:
    site_packages = tmp_path / "site-packages"
    sherpa_lib = site_packages / "sherpa_onnx" / "lib"
    sherpa_lib.mkdir(parents=True, exist_ok=True)
    (sherpa_lib / "libonnxruntime_providers_cuda.so").write_text("cuda")
    (sherpa_lib / "libonnxruntime_providers_shared.so").write_text("shared")
    (sherpa_lib / "libonnxruntime.so").write_text("ort")

    nvidia = site_packages / "nvidia"
    mapping = {
        "cublas": ("libcublasLt.so.12", "libcublas.so.12"),
        "cuda_runtime": ("libcudart.so.12",),
        "cufft": ("libcufft.so.11",),
        "curand": ("libcurand.so.10",),
        "cudnn": ("libcudnn.so.9",),
    }
    for package, names in mapping.items():
        lib_dir = nvidia / package / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (lib_dir / name).write_text(name)

    return sherpa_lib


def test_ensure_cuda_compat_libs_links_required_sonames(tmp_path: Path):
    sherpa_lib = _make_fake_sherpa_layout(tmp_path)

    ok, detail = sherpa_cuda.ensure_cuda_compat_libs(sherpa_lib)

    assert ok is True
    assert "linked CUDA compat libs" in detail
    for soname in sherpa_cuda.REQUIRED_CUDA_LIBS:
        path = sherpa_lib / soname
        assert path.exists()
        assert path.is_symlink()


def test_cuda_provider_runtime_status_detects_missing_runtime_symbols(tmp_path: Path, monkeypatch):
    sherpa_lib = _make_fake_sherpa_layout(tmp_path)

    monkeypatch.setattr(sherpa_cuda.shutil, "which", lambda exe: "/usr/bin/ldd" if exe == "ldd" else None)
    monkeypatch.setattr(
        sherpa_cuda.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="libcublasLt.so.12 => not found\n",
            stderr="",
        ),
    )

    ok, detail = sherpa_cuda.cuda_provider_runtime_status(sherpa_lib)

    assert ok is False
    assert "not found" in detail


def test_prepare_cuda_runtime_patches_rpath_and_validates(tmp_path: Path, monkeypatch):
    sherpa_lib = _make_fake_sherpa_layout(tmp_path)

    monkeypatch.setattr(sherpa_cuda.shutil, "which", lambda exe: f"/usr/bin/{exe}")

    calls: list[list[str]] = []

    def fake_run(argv, check=False, capture_output=False, text=False):
        calls.append(list(argv))
        if argv[0] == "/usr/bin/patchelf":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] == "/usr/bin/ldd":
            return SimpleNamespace(returncode=0, stdout="all good\n", stderr="")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(sherpa_cuda.subprocess, "run", fake_run)

    ok, detail = sherpa_cuda.prepare_cuda_runtime(sherpa_lib)

    assert ok is True
    assert "patched RUNPATH" in detail
    assert any(cmd[0] == "/usr/bin/patchelf" for cmd in calls)
    assert any(cmd[0] == "/usr/bin/ldd" for cmd in calls)
