#!/usr/bin/env python3
"""Run an OpenCut AI GPU service natively on the host GPU -- cross-platform.

Why this exists
---------------
Docker cannot expose an AMD ROCm GPU to containers on Windows (and ROCm-in-WSL2
is unsupported for most consumer Radeon cards). So on Windows the GPU-bound AI
services have to run natively on the host. This launcher does that on
**Windows, Linux and macOS** from one file.

It handles every torch-using service -- turboquant, image, tts and speaker -- and
on Linux it's just an alternative to the Docker path. It creates an isolated
venv per service, installs the right PyTorch build (ROCm/CUDA/CPU) plus the
service deps, and starts the FastAPI app against your local GPU. The rest of the
stack (Postgres, Redis, Ollama, web, ai-backend) keeps running in Docker -- see
scripts/install.ps1 / install.sh and docker-compose.native-all.yml, which point
the Dockerised ai-backend at these host-native services.

Examples
--------
    # Linux (AMD ROCm) -- torch installed from the ROCm wheel index automatically
    python scripts/run-native.py --service turboquant
    python scripts/run-native.py --service image

    # Windows (AMD ROCm) -- zero-config: the matching torch wheels from AMD's
    # Radeon repo (repo.radeon.com) are downloaded and installed automatically
    # for your Python version. Just run:
    python scripts\\run-native.py --service tts
    # Override only if AMD republishes: --torch-wheel <url[,url2]>, or set
    # ROCM_WINDOWS_TORCH_WHEELS / ROCM_WINDOWS_REL / ROCM_WINDOWS_TORCH_VER.

    # NVIDIA / explicit index override (any platform):
    python scripts/run-native.py --service image --torch-index https://download.pytorch.org/whl/cu128

    # Skip the install step on subsequent runs:
    python scripts/run-native.py --service turboquant --no-install
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-service launch config. `torch` lists the wheels that must come from the
# selected backend index (so coqui-tts / speechbrain don't drag a CUDA build in
# from PyPI). `skip` is the prefixes stripped from requirements.txt before the
# PyPI install, because they're installed separately (torch) or are GPU-vendor
# specific (bitsandbytes = 4-bit CUDA/HIP, turboquant-gpu = NVIDIA cuTile).
# torch is pinned to the 2.9 series (~=2.9.0) so every backend matches the AMD
# Windows ROCm wheels (torch 2.9.1). `wants_codec` adds torchcodec (best-effort)
# for coqui-tts 0.27.x audio decoding on torch 2.9.
SERVICES: dict[str, dict] = {
    "turboquant": {
        "dir": "services/turboquant-service",
        "port": "8430",
        "torch": ["torch~=2.9.0"],
        "skip": ("torch", "bitsandbytes", "turboquant-gpu"),
        "wants_triton": True,
        "wants_bnb": True,
        "wants_codec": False,
    },
    "image": {
        "dir": "services/image-service",
        "port": "8423",
        "torch": ["torch~=2.9.0"],
        "skip": ("torch", "torchaudio", "torchvision", "torchcodec"),
        "wants_triton": False,
        "wants_bnb": False,
        "wants_codec": False,
    },
    "tts": {
        "dir": "services/tts-service",
        "port": "8422",
        "torch": ["torch~=2.9.0", "torchaudio~=2.9.0"],
        "skip": ("torch", "torchaudio", "torchvision", "torchcodec"),
        "wants_triton": False,
        "wants_bnb": False,
        "wants_codec": True,
    },
    "speaker": {
        "dir": "services/speaker-service",
        "port": "8424",
        "torch": ["torch~=2.9.0", "torchaudio~=2.9.0"],
        "skip": ("torch", "torchaudio", "torchvision", "torchcodec"),
        "wants_triton": False,
        "wants_bnb": False,
        "wants_codec": False,
    },
}

# Default PyTorch ROCm wheel index for Linux (rocm6.4 -> torch 2.9). Windows ROCm
# wheels are NOT on download.pytorch.org -- they live on AMD's Radeon repo
# (repo.radeon.com). We construct those wheel URLs automatically for the running
# Python version so the Windows ROCm path is zero-config (no manual wheel hunting
# or ROCM_WINDOWS_TORCH_WHEELS needed). The release/version pieces are pinned to a
# known-good set matching the torch 2.9 series used across the services; each is
# overridable via env in case AMD republishes under a new path.
DEFAULT_LINUX_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.4"

WINDOWS_ROCM_REPO = os.environ.get("ROCM_WINDOWS_REPO", "https://repo.radeon.com/rocm/windows")
WINDOWS_ROCM_REL = os.environ.get("ROCM_WINDOWS_REL", "rocm-rel-7.2.1")
WINDOWS_ROCM_TORCH_VER = os.environ.get("ROCM_WINDOWS_TORCH_VER", "2.9.1")
WINDOWS_ROCM_LOCAL_TAG = os.environ.get("ROCM_WINDOWS_LOCAL_TAG", "rocm7.2.1")

# AMD's Windows torch wheels are built against a *split* ROCm runtime: torch's
# _rocm_init does `import rocm_sdk`, which lives in separate wheels next to torch
# on repo.radeon.com (rocm_sdk_core = the python module + runtime, plus the
# libraries wheel = the actual HIP/BLAS/etc DLLs). They are NOT on PyPI, so the
# --no-deps torch install (which we use so pip can't swap in a CPU build) skips
# them -- without them `import torch` dies with "ModuleNotFoundError: No module
# named 'rocm_sdk'". We install them explicitly before torch. The wheels are
# python-version-agnostic (py3-none-win_amd64) and tagged with the ROCm release
# version (e.g. 7.2.1). Override the whole list with ROCM_WINDOWS_SDK_WHEELS
# (comma-separated URLs) or the version with ROCM_WINDOWS_SDK_VER if AMD
# republishes under a different path.
WINDOWS_ROCM_SDK_VER = os.environ.get("ROCM_WINDOWS_SDK_VER", "7.2.1")
WINDOWS_ROCM_SDK_PKGS = ("rocm_sdk_core", "rocm_sdk_libraries_custom")

# bitsandbytes on PyPI is CUDA-only, so on Windows ROCm we install a community
# AMD/ROCm build instead (github.com/0xDELUXA/bitsandbytes_win_rocm). The wheel
# filename only varies by cpXY; the GPU arch and ROCm version are in the release
# TAG. We auto-pick the release from the detected GPU: RDNA (consumer Radeon,
# gfx10xx-12xx; covers all RDNA, py3.11-3.13) vs CDNA (Instinct, gfx9xx). Force a
# tag with ROCM_WINDOWS_BNB_TAG, or pass an exact wheel URL via --bnb-wheel /
# ROCM_WINDOWS_BNB_WHEEL, if needed.
WINDOWS_BNB_REPO = os.environ.get(
    "ROCM_WINDOWS_BNB_REPO", "https://github.com/0xDELUXA/bitsandbytes_win_rocm"
)
WINDOWS_BNB_VER = os.environ.get("ROCM_WINDOWS_BNB_VER", "0.50.0.dev0")
WINDOWS_BNB_TAG_RDNA = "0.50.0.dev0-py3-rocm7-win_amd64_rdna"
WINDOWS_BNB_TAG_CDNA = "0.50.0.dev0-py3-rocm7-win_amd64_all"
# Set ROCM_WINDOWS_BNB_TAG to force a specific release (skips auto-detection).
WINDOWS_BNB_TAG = os.environ.get("ROCM_WINDOWS_BNB_TAG")

# torch's own runtime deps -- pip --no-deps (used for the Windows ROCm wheels, per
# AMD's guide) skips these, so we install them explicitly afterwards.
TORCH_RUNTIME_DEPS = [
    "filelock", "typing-extensions", "sympy", "networkx", "jinja2", "fsspec", "numpy",
]

WINDOWS_ROCM_GUIDE = (
    "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/"
    "installrad/windows/install-pytorch.html"
)


def log(msg: str) -> None:
    print(f"[run-native] {msg}", flush=True)


def venv_python(venv_dir: Path) -> Path:
    """Path to the python executable inside a venv, per-platform."""
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str], **kw) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd, **kw)


def torch_has_gpu(py: Path) -> tuple[bool, str]:
    """Return (gpu_available, description) by probing torch inside the venv."""
    probe = (
        "import torch,json;"
        "hip=getattr(torch.version,'hip',None);"
        "print(json.dumps({'ok':torch.cuda.is_available(),"
        "'hip':hip,'cuda':torch.version.cuda,'ver':torch.__version__}))"
    )
    try:
        out = subprocess.check_output([str(py), "-c", probe], text=True).strip()
        info = json.loads(out.splitlines()[-1])
        vendor = "AMD/ROCm" if info.get("hip") else ("NVIDIA/CUDA" if info.get("cuda") else "CPU")
        return bool(info.get("ok")), f"torch {info.get('ver')} ({vendor})"
    except Exception as exc:  # torch not installed yet, or probe failed
        return False, f"torch not importable ({exc})"


def _windows_rocm_wheels(cli_value: str | None) -> list[str]:
    """Collect Windows ROCm torch wheel URLs from the flag or environment."""
    raw = cli_value or os.environ.get("ROCM_WINDOWS_TORCH_WHEELS", "")
    return [w.strip() for w in raw.split(",") if w.strip()]


def _torch_pkg_names(svc: dict) -> list[str]:
    """Bare distribution names for the torch wheels a service needs.

    e.g. ["torch~=2.9.0", "torchaudio~=2.9.0"] -> ["torch", "torchaudio"].
    """
    return [re.split(r"[~=<>!\s]", spec, maxsplit=1)[0] for spec in svc["torch"]]


def _windows_rocm_sdk_wheels() -> list[str]:
    """ROCm SDK runtime wheel URLs (override list, else AMD's default layout)."""
    raw = os.environ.get("ROCM_WINDOWS_SDK_WHEELS", "")
    override = [w.strip() for w in raw.split(",") if w.strip()]
    if override:
        return override
    base = f"{WINDOWS_ROCM_REPO}/{WINDOWS_ROCM_REL}"
    return [
        f"{base}/{name}-{WINDOWS_ROCM_SDK_VER}-py3-none-win_amd64.whl"
        for name in WINDOWS_ROCM_SDK_PKGS
    ]


def _install_windows_rocm_sdk(py: Path) -> None:
    """Install AMD's ROCm SDK runtime wheels that the Windows torch wheels need.

    torch's _rocm_init does `import rocm_sdk`; that module (and the HIP runtime
    DLLs) ship in separate wheels on repo.radeon.com that --no-deps skips. Without
    them `import torch` fails with ModuleNotFoundError: No module named
    'rocm_sdk'. Idempotent: skipped if rocm_sdk_core is already present.
    """
    if _installed_version(py, "rocm_sdk_core"):
        log("ROCm SDK runtime (rocm_sdk_core) already installed -- leaving it alone.")
        return
    wheels = _windows_rocm_sdk_wheels()
    log("Installing AMD ROCm SDK runtime wheels (provide the 'rocm_sdk' module):")
    for w in wheels:
        log(f"  {w}")
    for wheel in wheels:
        run([str(py), "-m", "pip", "install", "--no-cache-dir", "--no-deps", wheel])


def _auto_windows_rocm_wheels(svc: dict) -> list[str]:
    """Build AMD Radeon Windows ROCm wheel URLs for the running interpreter.

    Mirrors AMD's filename scheme, e.g. for Python 3.12:
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/
        torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl
    """
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    base = f"{WINDOWS_ROCM_REPO}/{WINDOWS_ROCM_REL}"
    return [
        f"{base}/{name}-{WINDOWS_ROCM_TORCH_VER}+{WINDOWS_ROCM_LOCAL_TAG}"
        f"-{py_tag}-{py_tag}-win_amd64.whl"
        for name in _torch_pkg_names(svc)
    ]


def _detect_amd_gfx(py: Path) -> str | None:
    """Probe the installed torch for the AMD GPU arch (e.g. 'gfx1100'), or None."""
    probe = (
        "import torch;"
        "p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None;"
        "print(getattr(p,'gcnArchName','') if p is not None else '')"
    )
    try:
        out = subprocess.check_output(
            [str(py), "-c", probe], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return out or None
    except Exception:
        return None


def _windows_bnb_tag(py: Path) -> str:
    """Pick the bitsandbytes release tag matching the detected GPU architecture.

    RDNA (consumer Radeon, gfx10xx-12xx) uses the 'rdna' build; CDNA/Instinct
    (gfx9xx) uses the 'all' build. ROCM_WINDOWS_BNB_TAG forces a specific tag.
    """
    if WINDOWS_BNB_TAG:
        return WINDOWS_BNB_TAG
    gfx = _detect_amd_gfx(py)
    m = re.search(r"gfx([0-9a-f]+)", (gfx or "").lower())
    arch = m.group(1) if m else ""
    if arch.startswith("9"):  # gfx9xx == CDNA / GCN data-center parts
        log(f"Detected AMD GPU {gfx} (CDNA) -> bitsandbytes 'all' build.")
        return WINDOWS_BNB_TAG_CDNA
    if gfx:
        log(f"Detected AMD GPU {gfx} (RDNA) -> bitsandbytes 'rdna' build.")
    else:
        log("Could not detect AMD GPU arch via torch -> defaulting to bitsandbytes 'rdna' build.")
    return WINDOWS_BNB_TAG_RDNA


def _windows_rocm_bnb_wheel(py: Path) -> str:
    """AMD/ROCm bitsandbytes wheel URL (0xDELUXA/bitsandbytes_win_rocm) for the
    running Python + detected GPU arch, e.g. for Python 3.12 on RDNA:
      https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases/download/
        0.50.0.dev0-py3-rocm7-win_amd64_rdna/
        bitsandbytes-0.50.0.dev0-cp312-cp312-win_amd64.whl
    """
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return (
        f"{WINDOWS_BNB_REPO}/releases/download/{_windows_bnb_tag(py)}/"
        f"bitsandbytes-{WINDOWS_BNB_VER}-{py_tag}-{py_tag}-win_amd64.whl"
    )


def _install_wheels_no_deps(py: Path, wheels: list[str]) -> None:
    """Install torch wheel URL(s) with --no-deps, then torch's runtime deps.

    --no-deps stops pip from replacing the ROCm build with a CPU wheel from PyPI,
    so we add torch's own runtime deps explicitly afterwards (per AMD's guide).
    """
    for wheel in wheels:
        run([str(py), "-m", "pip", "install", "--no-cache-dir", "--no-deps", wheel])
    run([str(py), "-m", "pip", "install", *TORCH_RUNTIME_DEPS])


def _installed_version(py: Path, pkg: str) -> str | None:
    """Return the version of `pkg` installed in the venv, or None if absent."""
    code = f"import importlib.metadata as m;print(m.version({pkg!r}))"
    try:
        out = subprocess.check_output(
            [str(py), "-c", code], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip() or None
    except Exception:
        return None


def _write_torch_constraints(py: Path, names: list[str], dest: Path) -> Path | None:
    """Pin the installed torch package(s) to their exact versions in a pip
    constraints file.

    Without this, installing the service deps lets pip's resolver replace the
    ROCm/CUDA torch build with a CPU wheel from PyPI -- transitive deps like
    accelerate (torch>=2.0) and bitsandbytes (torch>=2.3) are enough to trigger
    it even though we strip the direct torch line from requirements. Pinning the
    exact installed versions (e.g. torch==2.9.1+rocm7.2.1) forces pip to keep the
    build that's already present.
    """
    lines = []
    for name in names:
        ver = _installed_version(py, name)
        if ver:
            lines.append(f"{name}=={ver}")
    if not lines:
        return None
    dest.write_text("\n".join(lines) + "\n")
    return dest


def install_torch(py: Path, svc: dict, torch_index: str | None, torch_wheels: list[str]) -> None:
    """Install the right PyTorch build for the platform into the venv."""
    pkgs = svc["torch"]  # e.g. ["torch"] or ["torch", "torchaudio"]
    index = torch_index or os.environ.get("TORCH_INDEX_URL")
    already, desc = torch_has_gpu(py)

    if torch_wheels:
        # Explicit AMD Radeon Windows wheels (flag / ROCM_WINDOWS_TORCH_WHEELS).
        log(f"Installing torch from explicit wheel URL(s): {len(torch_wheels)} wheel(s)")
        if IS_WINDOWS:
            _install_windows_rocm_sdk(py)
        _install_wheels_no_deps(py, torch_wheels)
    elif index:
        log(f"Installing {' '.join(pkgs)} from index: {index}")
        run([str(py), "-m", "pip", "install", *pkgs, "--index-url", index])
    elif IS_LINUX:
        log(f"Installing {' '.join(pkgs)} from ROCm wheel index: {DEFAULT_LINUX_TORCH_INDEX}")
        run([str(py), "-m", "pip", "install", *pkgs, "--index-url", DEFAULT_LINUX_TORCH_INDEX])
    elif already:
        log(f"Using pre-installed {desc} (no torch index/wheel given) -- leaving it alone.")
    elif IS_WINDOWS:
        # Zero-config ROCm: construct AMD's Radeon wheel URLs for this Python and
        # install them automatically (no manual wheel hunting / env var needed).
        auto = _auto_windows_rocm_wheels(svc)
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        log(f"Auto-installing AMD Radeon Windows ROCm torch for Python {pyver}:")
        for w in auto:
            log(f"  {w}")
        try:
            _install_windows_rocm_sdk(py)
            _install_wheels_no_deps(py, auto)
        except subprocess.CalledProcessError:
            log(
                "Automatic ROCm torch install failed -- AMD may not publish wheels "
                f"for Python {pyver}, or the release path changed.\n"
                f"See AMD's guide: {WINDOWS_ROCM_GUIDE}\n"
                "Then either install a supported Python (3.10-3.13), or override the\n"
                "release/version via ROCM_WINDOWS_REL / ROCM_WINDOWS_TORCH_VER /\n"
                "ROCM_WINDOWS_LOCAL_TAG / ROCM_WINDOWS_SDK_VER, or pass exact URLs with\n"
                "--torch-wheel <url[,url2]> / ROCM_WINDOWS_TORCH_WHEELS (and the SDK\n"
                "runtime wheels via ROCM_WINDOWS_SDK_WHEELS)."
            )
            raise
    else:
        log(
            "No GPU-enabled torch found and no --torch-index/--torch-wheel given.\n"
            f"On Windows, PyTorch for ROCm is installed automatically; otherwise see\n"
            f"  {WINDOWS_ROCM_GUIDE}\n"
            "or pass --torch-index <url> for your platform's build."
        )


def install(py: Path, svc: dict, service_dir: Path, torch_index: str | None,
            torch_wheels: list[str], bnb_wheel: str | None = None) -> None:
    """Install torch + the service deps into the venv."""
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # --- PyTorch -----------------------------------------------------------
    install_torch(py, svc, torch_index, torch_wheels)
    # Remember whether we ended up with a GPU build, so we can detect (and undo)
    # a dependency install silently swapping it for a CPU wheel further down.
    had_gpu_torch, _ = torch_has_gpu(py)

    # Pin the just-installed torch build so none of the dependency installs below
    # can swap it for a CPU wheel from PyPI while resolving transitive deps.
    constraints = _write_torch_constraints(
        py, _torch_pkg_names(svc), service_dir / "constraints.torch.txt"
    )
    cflags = ["-c", str(constraints)] if constraints else []

    try:
        # --- torchcodec (tts on torch 2.9, best-effort) --------------------
        # coqui-tts >=0.27.4 decodes audio via torchcodec on torch 2.9. --no-deps
        # so it can't pull a CPU torch over the backend build; non-fatal if
        # unavailable (coqui-tts falls back to soundfile).
        if svc.get("wants_codec"):
            try:
                run([str(py), "-m", "pip", "install", "--no-deps", "torchcodec>=0.8.0", *cflags])
            except subprocess.CalledProcessError:
                log("torchcodec unavailable for this backend -- coqui-tts falls back to soundfile.")

        # --- Triton (turboquant torch.compile path only) -------------------
        if svc.get("wants_triton"):
            if IS_WINDOWS:
                try:
                    run([str(py), "-m", "pip", "install", "triton-windows", *cflags])
                except subprocess.CalledProcessError:
                    log("triton-windows install failed -- torch.compile will fall back to eager.")
            else:
                log("Triton provided by the torch ROCm wheels (pytorch-triton-rocm).")

        # --- Service deps (minus torch + GPU-vendor lines) -----------------
        skip = tuple(s.lower() for s in svc["skip"])
        req = (service_dir / "requirements.txt").read_text().splitlines()
        filtered = [
            ln for ln in req
            if ln.strip() and not ln.strip().lower().startswith(skip)
        ]
        tmp = service_dir / "requirements.native.txt"
        tmp.write_text("\n".join(filtered) + "\n")
        try:
            run([str(py), "-m", "pip", "install", "-r", str(tmp), *cflags])
        finally:
            tmp.unlink(missing_ok=True)

        # --- bitsandbytes (turboquant only, best-effort) -------------------
        # PyPI bitsandbytes is CUDA-only. On Windows we install a community
        # AMD/ROCm build (0xDELUXA/bitsandbytes_win_rocm) so 4-bit works on
        # Radeon; elsewhere we use the PyPI wheel (NVIDIA, or Linux ROCm builds).
        # If the install fails app.py degrades to bf16 automatically.
        if svc.get("wants_bnb"):
            url = bnb_wheel or (_windows_rocm_bnb_wheel(py) if IS_WINDOWS else None)
            have = _installed_version(py, "bitsandbytes")
            try:
                if url:
                    # --no-deps so it can't drag a CPU torch in; deps are present.
                    if bnb_wheel or have != WINDOWS_BNB_VER:
                        log(f"Installing AMD/ROCm bitsandbytes from wheel: {url}")
                        run([str(py), "-m", "pip", "install", "--no-deps", url, *cflags])
                    else:
                        log(f"AMD/ROCm bitsandbytes {have} already installed -- leaving it alone.")
                else:
                    run([str(py), "-m", "pip", "install", "bitsandbytes", *cflags])
            except subprocess.CalledProcessError:
                log(
                    "bitsandbytes install failed -- model will load in bf16 (no 4-bit)."
                    + (
                        "\nWindows: no ROCm wheel for your Python/GPU at the default "
                        "release. Set ROCM_WINDOWS_BNB_TAG to the release matching your "
                        "GPU arch (RDNA2/RDNA/CDNA) or ROCM_WINDOWS_BNB_WHEEL to an "
                        "exact wheel URL. See " + WINDOWS_BNB_REPO + "/releases"
                        if IS_WINDOWS else ""
                    )
                )
    finally:
        if constraints:
            constraints.unlink(missing_ok=True)

    # --- Re-assert the GPU torch build last --------------------------------
    # The constraints pin above normally keeps the ROCm/CUDA build in place, but
    # it's not bulletproof (e.g. if the pin file couldn't be written, or a stray
    # transitive pin slips through). If we started with a GPU build and a
    # dependency install replaced it with a CPU wheel, reinstall it last so the
    # GPU build is the final state -- nothing runs after this to clobber it.
    if had_gpu_torch:
        ok, desc = torch_has_gpu(py)
        if not ok:
            log(f"GPU torch was replaced during dependency install ({desc}) -- "
                "reinstalling the GPU build last.")
            install_torch(py, svc, torch_index, torch_wheels)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an OpenCut AI GPU service natively on the host GPU.")
    ap.add_argument("--service", default=os.environ.get("NATIVE_SERVICE", "turboquant"),
                    choices=sorted(SERVICES.keys()),
                    help="Which GPU service to run natively (default turboquant).")
    ap.add_argument("--port", default=None, help="Override the service port.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                    help="auto|cpu|cuda|rocm|mps (default auto)")
    ap.add_argument("--venv", default=None, help="venv path (default per-service .venv-native).")
    ap.add_argument("--torch-index", default=None,
                    help="Override the pip index URL used to install torch.")
    ap.add_argument("--torch-wheel", default=None,
                    help="Comma-separated torch wheel URL(s) (AMD Radeon Windows ROCm wheels).")
    ap.add_argument("--bnb-wheel", default=os.environ.get("ROCM_WINDOWS_BNB_WHEEL") or None,
                    help="bitsandbytes wheel URL (AMD/ROCm build). Defaults to the "
                         "0xDELUXA/bitsandbytes_win_rocm release on Windows.")
    ap.add_argument("--no-install", action="store_true", help="Skip dependency install.")
    args = ap.parse_args()

    svc = SERVICES[args.service]
    service_dir = REPO_ROOT / svc["dir"]
    port = args.port or svc["port"]
    venv_dir = Path(args.venv) if args.venv else (service_dir / ".venv-native")
    py = venv_python(venv_dir)
    torch_wheels = _windows_rocm_wheels(args.torch_wheel)

    log(f"Service: {args.service}  ->  {service_dir}  (port {port})")

    if not py.exists():
        log(f"Creating venv at {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if not args.no_install:
        install(py, svc, service_dir, args.torch_index, torch_wheels, args.bnb_wheel)

    ok, desc = torch_has_gpu(py)
    log(f"GPU check: {desc} -- torch.cuda.is_available()={ok}")
    if not ok:
        log("WARNING: no GPU detected by torch; the service will run on CPU.")

    # --- Launch ------------------------------------------------------------
    env = os.environ.copy()
    env.setdefault("DEVICE", args.device)
    env.setdefault("TURBOQUANT_COMPILE", "1")
    # Per-service compute-mode override env (auto|cpu|cuda) honored by app.py.
    for var in ("IMAGE_DEVICE", "TTS_DEVICE", "SPEAKER_DEVICE"):
        env.setdefault(var, args.device)
    env.setdefault("HF_HOME", str(service_dir / "models"))
    env.setdefault("TRANSFORMERS_CACHE", str(service_dir / "models"))
    (service_dir / "models").mkdir(exist_ok=True)

    log(f"Starting {args.service}-service on {args.host}:{port} (DEVICE={env['DEVICE']})")
    cmd = [str(py), "-m", "uvicorn", "app:app", "--host", args.host, "--port", str(port)]
    try:
        run(cmd, cwd=str(service_dir), env=env)
    except KeyboardInterrupt:
        log("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
