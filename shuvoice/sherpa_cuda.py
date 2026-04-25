"""Sherpa CUDA runtime repair helpers.

Handles two common CUDA-runtime issues for GPU-enabled sherpa-onnx wheels:
- provider libraries present but missing RUNPATH to resolve sibling libs
- required CUDA sonames absent from the sherpa lib dir even though matching
  compat/runtime libs exist elsewhere (for example under site-packages/nvidia)
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

REQUIRED_CUDA_LIBS: tuple[str, ...] = (
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudart.so.12",
    "libcufft.so.11",
    "libcurand.so.10",
    "libcudnn.so.9",
)
_REQUIRED_IMPORT_RUNTIME_LIBS: tuple[str, ...] = (
    "libonnxruntime.so",
)
_PATCH_RPATH_LIBS: tuple[str, ...] = (
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime.so",
)
_ONNXRUNTIME_GPU_PROVIDER_LIBS: tuple[str, ...] = (
    "libonnxruntime_providers_cuda.so",
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_tensorrt.so",
)


def _module_root(module: Any) -> Path | None:
    module_file = getattr(module, "__file__", None)
    if module_file:
        return Path(module_file).resolve().parent

    spec = getattr(module, "__spec__", None)
    if spec is not None:
        search_locations = getattr(spec, "submodule_search_locations", None)
        if search_locations:
            for location in search_locations:
                if location:
                    return Path(location).resolve()

        origin = getattr(spec, "origin", None)
        if origin:
            return Path(origin).resolve().parent

    return None



def sherpa_lib_dir() -> Path | None:
    try:
        import sherpa_onnx
    except Exception:
        sherpa_onnx = None

    module_root = _module_root(sherpa_onnx) if sherpa_onnx is not None else None
    if module_root is None:
        spec = importlib.util.find_spec("sherpa_onnx")
        if spec is not None:
            search_locations = getattr(spec, "submodule_search_locations", None)
            if search_locations:
                for location in search_locations:
                    if location:
                        module_root = Path(location).resolve()
                        break
            if module_root is None:
                origin = getattr(spec, "origin", None)
                if origin:
                    module_root = Path(origin).resolve().parent

    if module_root is None:
        return None

    lib_dir = module_root / "lib"
    if lib_dir.is_dir():
        return lib_dir
    return None


def _site_packages_root(lib_dir: Path) -> Path:
    return lib_dir.resolve().parent.parent


def _onnxruntime_capi_dir(lib_dir: Path) -> Path | None:
    site_packages = _site_packages_root(lib_dir)
    capi_dir = site_packages / "onnxruntime" / "capi"
    if capi_dir.is_dir():
        return capi_dir
    return None


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

    for root in (Path("/usr/lib"), Path("/usr/lib64"), Path("/usr/local/lib")):
        for sherpa_path in root.glob("python*/site-packages/sherpa_onnx/lib"):
            if sherpa_path.is_dir():
                candidates.append(sherpa_path)
        if root.is_dir():
            candidates.append(root)

    for candidate in (
        Path("/opt/cuda/lib64"),
        Path("/opt/cuda/targets/x86_64-linux/lib"),
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


def _ensure_exact_libs(
    required_libs: tuple[str, ...],
    *,
    lib_dir: Path,
    label: str,
) -> tuple[bool, str]:
    linked: list[str] = []
    missing: list[str] = []
    for soname in required_libs:
        target = _find_exact_lib(lib_dir, soname)
        if target is None:
            missing.append(soname)
            continue

        destination = lib_dir / soname
        if destination.exists():
            continue

        destination.symlink_to(target)
        linked.append(soname)

    if missing:
        return False, f"missing required {label} libs: " + ", ".join(missing)
    if linked:
        return True, f"linked {label} libs: " + ", ".join(linked)
    return True, f"{label} libs already present"


def ensure_import_runtime_libs(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    destination = resolved_lib_dir / "libonnxruntime.so"
    if destination.exists():
        return True, "import runtime libs already present"

    capi_dir = _onnxruntime_capi_dir(resolved_lib_dir)
    if capi_dir is not None:
        source = capi_dir / "libonnxruntime.so"
        if not source.exists():
            try:
                import onnxruntime  # type: ignore

                versioned_source = capi_dir / f"libonnxruntime.so.{onnxruntime.__version__}"
                if versioned_source.exists():
                    source = versioned_source
            except Exception:
                pass
        if not source.exists():
            versioned = sorted(capi_dir.glob("libonnxruntime.so.*"))
            if versioned:
                source = versioned[-1]
        if source.exists():
            shutil.copy2(source, destination)
            return True, "copied import runtime libs from onnxruntime-gpu"

    return _ensure_exact_libs(
        _REQUIRED_IMPORT_RUNTIME_LIBS,
        lib_dir=resolved_lib_dir,
        label="import runtime",
    )


def ensure_onnxruntime_gpu_provider_libs(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    capi_dir = _onnxruntime_capi_dir(resolved_lib_dir)
    if capi_dir is None:
        required_present = all(
            (resolved_lib_dir / name).exists() for name in _PATCH_RPATH_LIBS[:2]
        )
        if required_present:
            return True, "ONNX Runtime GPU provider libs already present"
        return False, "onnxruntime-gpu capi directory not found"

    copied: list[str] = []
    for name in _ONNXRUNTIME_GPU_PROVIDER_LIBS:
        source = capi_dir / name
        destination = resolved_lib_dir / name
        if not source.exists() or destination.exists():
            continue
        shutil.copy2(source, destination)
        copied.append(name)

    if copied:
        return True, "copied ONNX Runtime GPU provider libs: " + ", ".join(copied)
    return True, "ONNX Runtime GPU provider libs already present"


def ensure_cuda_compat_libs(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    compat_ok, compat_detail = _ensure_exact_libs(
        REQUIRED_CUDA_LIBS,
        lib_dir=resolved_lib_dir,
        label="CUDA compat",
    )
    if not compat_ok:
        return False, compat_detail

    lower_alias = resolved_lib_dir / "libcublaslt.so.12"
    if not lower_alias.exists() and (resolved_lib_dir / "libcublasLt.so.12").exists():
        lower_alias.symlink_to("libcublasLt.so.12")
        return True, compat_detail + "; linked CUDA compat alias: libcublaslt.so.12"

    return True, compat_detail


def patch_sherpa_rpaths(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    patchelf = shutil.which("patchelf")
    if patchelf is None:
        return False, "patchelf not available"

    paths: list[Path] = []
    seen: set[Path] = set()
    for name in _PATCH_RPATH_LIBS:
        path = resolved_lib_dir / name
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for path in sorted(resolved_lib_dir.glob("_sherpa_onnx*.so")):
        if path not in seen:
            paths.append(path)
            seen.add(path)

    patched: list[str] = []
    for path in paths:
        proc = subprocess.run(
            [patchelf, "--set-rpath", "$ORIGIN", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
            return False, f"failed to patch {path.name}: {detail}"
        patched.append(path.name)

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


def prepare_import_runtime(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    runtime_ok, runtime_detail = ensure_import_runtime_libs(resolved_lib_dir)
    if not runtime_ok:
        return False, runtime_detail

    patch_ok, patch_detail = patch_sherpa_rpaths(resolved_lib_dir)
    if patch_ok:
        return True, f"{runtime_detail}; {patch_detail}"
    return True, runtime_detail


def prepare_cuda_runtime(lib_dir: Path | None = None) -> tuple[bool, str]:
    resolved_lib_dir = lib_dir or sherpa_lib_dir()
    if resolved_lib_dir is None:
        return False, "sherpa_onnx lib directory not found"

    runtime_ok, runtime_detail = prepare_import_runtime(resolved_lib_dir)
    if not runtime_ok:
        return False, runtime_detail

    provider_ok, provider_detail = ensure_onnxruntime_gpu_provider_libs(resolved_lib_dir)
    if not provider_ok:
        return False, f"{runtime_detail}; {provider_detail}"

    compat_ok, compat_detail = ensure_cuda_compat_libs(resolved_lib_dir)
    if not compat_ok:
        return False, f"{runtime_detail}; {provider_detail}; {compat_detail}"

    patch_ok, patch_detail = patch_sherpa_rpaths(resolved_lib_dir)
    status_ok, status_detail = cuda_provider_runtime_status(resolved_lib_dir)
    details = f"{runtime_detail}; {provider_detail}; {compat_detail}"
    if patch_ok:
        details += f"; {patch_detail}"
    if status_ok:
        return True, f"{details}; {status_detail}"
    return False, f"{details}; {status_detail}"
