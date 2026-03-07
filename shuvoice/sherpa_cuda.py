"""Sherpa CUDA runtime repair helpers.

Handles two common CUDA-runtime issues for GPU-enabled sherpa-onnx wheels:
- provider libraries present but missing RUNPATH to resolve sibling libs
- required CUDA sonames absent from the sherpa lib dir even though matching
  compat/runtime libs exist elsewhere (for example under site-packages/nvidia)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REQUIRED_CUDA_LIBS: tuple[str, ...] = (
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudart.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcudnn.so.9",
)
_PATCH_RPATH_LIBS: tuple[str, ...] = (
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime.so",
)


def sherpa_lib_dir() -> Path | None:
    try:
        import sherpa_onnx  # noqa: PLC0415
    except Exception:
        return None

    lib_dir = Path(sherpa_onnx.__file__).resolve().parent / "lib"
    if lib_dir.is_dir():
        return lib_dir
    return None


def _site_packages_root(lib_dir: Path) -> Path:
    return lib_dir.resolve().parent.parent


def _candidate_dirs(lib_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    site_packages = _site_packages_root(lib_dir)

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.is_dir():
        for child in sorted(nvidia_root.iterdir()):
            lib_path = child / "lib"
            if lib_path.is_dir():
                candidates.append(lib_path)

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_name, "").strip()
        if value:
            base = Path(value)
            for candidate in (base / "lib64", base / "targets" / "x86_64-linux" / "lib"):
                if candidate.is_dir():
                    candidates.append(candidate)

    for candidate in (
        Path("/opt/cuda/lib64"),
        Path("/opt/cuda/targets/x86_64-linux/lib"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
    ):
        if candidate.is_dir():
            candidates.append(candidate)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _find_exact_lib(lib_dir: Path, soname: str) -> Path | None:
    direct = lib_dir / soname
    if direct.exists():
        return direct

    for candidate_dir in _candidate_dirs(lib_dir):
        candidate = candidate_dir / soname
        if candidate.exists():
            return candidate
    return None


def ensure_cuda_compat_libs(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    linked: list[str] = []
    missing: list[str] = []
    for soname in REQUIRED_CUDA_LIBS:
        target = _find_exact_lib(resolved_lib_dir, soname)
        if target is None:
            missing.append(soname)
            continue

        destination = resolved_lib_dir / soname
        if destination.exists():
            continue

        destination.symlink_to(target)
        linked.append(soname)

    if missing:
        return False, "missing required CUDA libs: " + ", ".join(missing)
    if linked:
        return True, "linked CUDA compat libs: " + ", ".join(linked)
    return True, "CUDA compat libs already present"


def patch_sherpa_rpaths(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    patchelf = shutil.which("patchelf")
    if patchelf is None:
        return False, "patchelf not available"

    patched: list[str] = []
    for name in _PATCH_RPATH_LIBS:
        path = resolved_lib_dir / name
        if not path.exists():
            continue
        proc = subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
            return False, f"failed to patch {name}: {detail}"
        patched.append(name)

    if not patched:
        return False, "no sherpa runtime libraries needed RPATH patching"
    return True, "patched RUNPATH for: " + ", ".join(patched)


def cuda_provider_runtime_status(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    provider_lib = resolved_lib_dir / "libonnxruntime_providers_cuda.so"
    if not provider_lib.exists():
        return False, f"missing CUDA provider library under {resolved_lib_dir}"

    ldd = shutil.which("ldd")
    if ldd is None:
        return True, f"found CUDA provider libraries under {resolved_lib_dir}"

    proc = subprocess.run(
        [ldd, str(provider_lib)],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part).strip()
    for line in output.splitlines():
        normalized = line.strip().lower()
        if "not found" in normalized:
            return False, normalized
        if "version `" in line and "not found" in line:
            return False, line.strip()

    if proc.returncode != 0:
        return False, output or f"ldd failed with exit code {proc.returncode}"

    return True, f"found CUDA provider libraries under {resolved_lib_dir}"


def prepare_cuda_runtime(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    compat_ok, compat_detail = ensure_cuda_compat_libs(resolved_lib_dir)
    if not compat_ok:
        return False, compat_detail

    patch_ok, patch_detail = patch_sherpa_rpaths(resolved_lib_dir)
    status_ok, status_detail = cuda_provider_runtime_status(resolved_lib_dir)
    if status_ok:
        if patch_ok:
            return True, f"{compat_detail}; {patch_detail}; {status_detail}"
        return True, f"{compat_detail}; {status_detail}"

    if patch_ok:
        return False, f"{compat_detail}; {patch_detail}; {status_detail}"
    return False, f"{compat_detail}; {status_detail}"
