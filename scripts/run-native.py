#!/usr/bin/env python3
"""Run an OpenCut AI GPU service natively on the host GPU — cross-platform.

Why this exists
---------------
Docker cannot expose an AMD ROCm GPU to containers on Windows (and ROCm-in-WSL2
is unsupported for most consumer Radeon cards). So on Windows the GPU-bound AI
services have to run natively on the host. This launcher does that on
**Windows, Linux and macOS** from one file.

It handles every torch-using service — turboquant, image, tts and speaker — and
on Linux it's just an alternative to the Docker path. It creates an isolated
venv per service, installs the right PyTorch build (ROCm/CUDA/CPU) plus the
service deps, and starts the FastAPI app against your local GPU. The rest of the
stack (Postgres, Redis, Ollama, web, ai-backend) keeps running in Docker — see
scripts/install.ps1 / install.sh and docker-compose.native-all.yml, which point
the Dockerised ai-backend at these host-native services.

Examples
--------
    # Linux (AMD ROCm) — torch installed from the ROCm wheel index automatically
    python scripts/run-native.py --service turboquant
    python scripts/run-native.py --service image

    # Windows (AMD ROCm) — install torch from AMD's Radeon wheels (repo.radeon.com).
    # Pass the wheel URL(s) for your Python version, comma-separated:
    python scripts\\run-native.py --service tts ^
      --torch-wheel https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl,https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl
    # ...or set ROCM_WINDOWS_TORCH_WHEELS once and reuse it.

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
SERVICES: dict[str, dict] = {
    "turboquant": {
        "dir": "services/turboquant-service",
        "port": "8430",
        "torch": ["torch"],
        "skip": ("torch", "bitsandbytes", "turboquant-gpu"),
        "wants_triton": True,
        "wants_bnb": True,
    },
    "image": {
        "dir": "services/image-service",
        "port": "8423",
        "torch": ["torch"],
        "skip": ("torch", "torchaudio", "torchvision"),
        "wants_triton": False,
        "wants_bnb": False,
    },
    "tts": {
        "dir": "services/tts-service",
        # coqui-tts 0.24.2 requires torch<2.6; pin 2.5.1 (on cpu/cu124/rocm6.2).
        "port": "8422",
        "torch": ["torch==2.5.1", "torchaudio==2.5.1"],
        "skip": ("torch", "torchaudio", "torchvision"),
        "wants_triton": False,
        "wants_bnb": False,
    },
    "speaker": {
        "dir": "services/speaker-service",
        "port": "8424",
        "torch": ["torch", "torchaudio"],
        "skip": ("torch", "torchaudio", "torchvision"),
        "wants_triton": False,
        "wants_bnb": False,
    },
}

# Default PyTorch ROCm wheel index for Linux. Windows ROCm wheels are NOT on
# download.pytorch.org — Windows users install torch via AMD's Radeon wheels
# (repo.radeon.com), passed with --torch-wheel / ROCM_WINDOWS_TORCH_WHEELS.
DEFAULT_LINUX_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.2"

# torch's own runtime deps — pip --no-deps (used for the Windows ROCm wheels, per
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


def install_torch(py: Path, svc: dict, torch_index: str | None, torch_wheels: list[str]) -> None:
    """Install the right PyTorch build for the platform into the venv."""
    pkgs = svc["torch"]  # e.g. ["torch"] or ["torch", "torchaudio"]
    index = torch_index or os.environ.get("TORCH_INDEX_URL")
    already, desc = torch_has_gpu(py)

    if torch_wheels:
        # AMD Radeon Windows wheels: install with --no-deps so pip can't replace
        # the ROCm build with a CPU wheel from PyPI, then add torch's runtime deps.
        log(f"Installing torch from explicit wheel URL(s): {len(torch_wheels)} wheel(s)")
        for wheel in torch_wheels:
            run([str(py), "-m", "pip", "install", "--no-cache-dir", "--no-deps", wheel])
        run([str(py), "-m", "pip", "install", *TORCH_RUNTIME_DEPS])
    elif index:
        log(f"Installing {' '.join(pkgs)} from index: {index}")
        run([str(py), "-m", "pip", "install", *pkgs, "--index-url", index])
    elif IS_LINUX:
        log(f"Installing {' '.join(pkgs)} from ROCm wheel index: {DEFAULT_LINUX_TORCH_INDEX}")
        run([str(py), "-m", "pip", "install", *pkgs, "--index-url", DEFAULT_LINUX_TORCH_INDEX])
    elif already:
        log(f"Using pre-installed {desc} (no torch index/wheel given) — leaving it alone.")
    else:
        log(
            "No GPU-enabled torch found and no --torch-index/--torch-wheel given.\n"
            "On Windows, install PyTorch for ROCm from AMD's Radeon wheels:\n"
            f"  {WINDOWS_ROCM_GUIDE}\n"
            "Then re-run with --no-install, or pass the wheel URL(s) via\n"
            "  --torch-wheel <url[,url2]>   (e.g. torch + torchaudio), or set\n"
            "  ROCM_WINDOWS_TORCH_WHEELS=<url[,url2]> in the environment."
        )


def install(py: Path, svc: dict, service_dir: Path, torch_index: str | None,
            torch_wheels: list[str]) -> None:
    """Install torch + the service deps into the venv."""
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # --- PyTorch -----------------------------------------------------------
    install_torch(py, svc, torch_index, torch_wheels)

    # --- Triton (turboquant torch.compile path only) -----------------------
    if svc.get("wants_triton"):
        if IS_WINDOWS:
            try:
                run([str(py), "-m", "pip", "install", "triton-windows"])
            except subprocess.CalledProcessError:
                log("triton-windows install failed — torch.compile will fall back to eager.")
        else:
            log("Triton provided by the torch ROCm wheels (pytorch-triton-rocm).")

    # --- Service deps (minus torch + GPU-vendor lines) ---------------------
    skip = tuple(s.lower() for s in svc["skip"])
    req = (service_dir / "requirements.txt").read_text().splitlines()
    filtered = [
        ln for ln in req
        if ln.strip() and not ln.strip().lower().startswith(skip)
    ]
    tmp = service_dir / "requirements.native.txt"
    tmp.write_text("\n".join(filtered) + "\n")
    try:
        run([str(py), "-m", "pip", "install", "-r", str(tmp)])
    finally:
        tmp.unlink(missing_ok=True)

    # --- bitsandbytes (turboquant only, best-effort) -----------------------
    # The PyPI wheel is CUDA-only; on ROCm hosts it may lack GPU kernels, in
    # which case app.py degrades to bf16 automatically. We still try so NVIDIA
    # and any ROCm-enabled bnb builds get 4-bit.
    if svc.get("wants_bnb"):
        try:
            run([str(py), "-m", "pip", "install", "bitsandbytes"])
        except subprocess.CalledProcessError:
            log("bitsandbytes install failed — model will load in bf16 (no 4-bit).")


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
    ap.add_argument("--no-install", action="store_true", help="Skip dependency install.")
    args = ap.parse_args()

    svc = SERVICES[args.service]
    service_dir = REPO_ROOT / svc["dir"]
    port = args.port or svc["port"]
    venv_dir = Path(args.venv) if args.venv else (service_dir / ".venv-native")
    py = venv_python(venv_dir)
    torch_wheels = _windows_rocm_wheels(args.torch_wheel)

    log(f"Service: {args.service}  →  {service_dir}  (port {port})")

    if not py.exists():
        log(f"Creating venv at {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if not args.no_install:
        install(py, svc, service_dir, args.torch_index, torch_wheels)

    ok, desc = torch_has_gpu(py)
    log(f"GPU check: {desc} — torch.cuda.is_available()={ok}")
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
