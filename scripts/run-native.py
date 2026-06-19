#!/usr/bin/env python3
"""Run the turboquant-service natively on the host GPU — cross-platform.

Why this exists
---------------
Docker cannot expose an AMD ROCm GPU to containers on Windows (and ROCm-in-WSL2
is unsupported for most consumer Radeon cards). So on Windows the GPU-bound
turboquant-service has to run natively on the host. This launcher does that on
**Windows, Linux and macOS** from one file — nothing here is hard-coded to
Windows, and on Linux it's just an alternative to the Docker path.

It creates an isolated venv, installs the right wheels for the platform
(PyTorch ROCm, Triton, bitsandbytes), and starts the FastAPI service against
your local GPU. The rest of the stack (Postgres, Redis, Ollama, web, other AI
services) can keep running in Docker — see scripts/install.sh --native-turboquant
and docker-compose.native-ai.yml, which point the Dockerised ai-backend at this
host-native service.

Examples
--------
    # Linux (AMD ROCm) — installs torch from the ROCm wheel index automatically
    python scripts/run-native.py

    # Windows (AMD ROCm) — install torch per AMD's Radeon guide first, then:
    python scripts\\run-native.py
    #   torch index override (if you have one):
    #   python scripts\\run-native.py --torch-index https://...

    # Skip the install step on subsequent runs:
    python scripts/run-native.py --no-install
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MAC = platform.system() == "Darwin"

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVICE_DIR = REPO_ROOT / "services" / "turboquant-service"

# Default PyTorch ROCm wheel index for Linux. Windows ROCm wheels are not on
# download.pytorch.org yet — Windows users install torch via AMD's Radeon guide
# (https://rocm.docs.amd.com/projects/radeon-ryzen/) and we don't overwrite it.
DEFAULT_LINUX_TORCH_INDEX = "https://download.pytorch.org/whl/rocm6.2"


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
        import json

        info = json.loads(out.splitlines()[-1])
        vendor = "AMD/ROCm" if info.get("hip") else ("NVIDIA/CUDA" if info.get("cuda") else "CPU")
        return bool(info.get("ok")), f"torch {info.get('ver')} ({vendor})"
    except Exception as exc:  # torch not installed yet, or probe failed
        return False, f"torch not importable ({exc})"


def install(py: Path, torch_index: str | None) -> None:
    """Install torch (+Triton, +bitsandbytes) and the service deps into the venv."""
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # --- PyTorch -----------------------------------------------------------
    already, desc = torch_has_gpu(py)
    index = torch_index or os.environ.get("TORCH_INDEX_URL")
    if index:
        log(f"Installing torch from explicit index: {index}")
        run([str(py), "-m", "pip", "install", "torch", "--index-url", index])
    elif IS_LINUX:
        log(f"Installing torch from ROCm wheel index: {DEFAULT_LINUX_TORCH_INDEX}")
        run([str(py), "-m", "pip", "install", "torch", "--index-url", DEFAULT_LINUX_TORCH_INDEX])
    elif already:
        log(f"Using pre-installed {desc} (no torch index given) — leaving it alone.")
    else:
        log(
            "No GPU-enabled torch found and no --torch-index given. On Windows, "
            "install PyTorch for ROCm first via AMD's Radeon guide:\n"
            "  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html\n"
            "then re-run with --no-install, or pass --torch-index <url>."
        )

    # --- Triton ------------------------------------------------------------
    # Linux ROCm torch wheels already bundle pytorch-triton-rocm. On Windows the
    # ROCm-capable Triton comes from the woct0rdho fork, published as
    # `triton-windows` on PyPI.
    if IS_WINDOWS:
        try:
            run([str(py), "-m", "pip", "install", "triton-windows"])
        except subprocess.CalledProcessError:
            log("triton-windows install failed — torch.compile will fall back to eager.")
    else:
        log("Triton provided by the torch ROCm wheels (pytorch-triton-rocm).")

    # --- Service deps (minus the GPU-vendor lines) -------------------------
    req = (SERVICE_DIR / "requirements.txt").read_text().splitlines()
    skip = ("torch", "bitsandbytes", "turboquant-gpu")
    filtered = [
        ln for ln in req
        if ln.strip() and not ln.strip().lower().startswith(skip)
    ]
    tmp = SERVICE_DIR / "requirements.native.txt"
    tmp.write_text("\n".join(filtered) + "\n")
    try:
        run([str(py), "-m", "pip", "install", "-r", str(tmp)])
    finally:
        tmp.unlink(missing_ok=True)

    # --- bitsandbytes (best-effort) ---------------------------------------
    # The PyPI wheel is CUDA-only; on ROCm hosts it may lack GPU kernels, in
    # which case app.py degrades to bf16 automatically. We still try so NVIDIA
    # and any ROCm-enabled bnb builds get 4-bit.
    try:
        run([str(py), "-m", "pip", "install", "bitsandbytes"])
    except subprocess.CalledProcessError:
        log("bitsandbytes install failed — model will load in bf16 (no 4-bit).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run turboquant-service natively on the host GPU.")
    ap.add_argument("--port", default=os.environ.get("TURBOQUANT_PORT", "8430"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                    help="auto|cpu|cuda|rocm|mps (default auto)")
    ap.add_argument("--venv", default=str(SERVICE_DIR / ".venv-native"))
    ap.add_argument("--torch-index", default=None,
                    help="Override the pip index URL used to install torch.")
    ap.add_argument("--no-install", action="store_true", help="Skip dependency install.")
    args = ap.parse_args()

    venv_dir = Path(args.venv)
    py = venv_python(venv_dir)

    if not py.exists():
        log(f"Creating venv at {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if not args.no_install:
        install(py, args.torch_index)

    ok, desc = torch_has_gpu(py)
    log(f"GPU check: {desc} — torch.cuda.is_available()={ok}")
    if not ok:
        log("WARNING: no GPU detected by torch; the service will run on CPU.")

    # --- Launch ------------------------------------------------------------
    env = os.environ.copy()
    env.setdefault("DEVICE", args.device)
    env.setdefault("TURBOQUANT_COMPILE", "1")
    env.setdefault("HF_HOME", str(SERVICE_DIR / "models"))
    env.setdefault("TRANSFORMERS_CACHE", str(SERVICE_DIR / "models"))
    (SERVICE_DIR / "models").mkdir(exist_ok=True)

    log(f"Starting turboquant-service on {args.host}:{args.port} (DEVICE={env['DEVICE']})")
    cmd = [str(py), "-m", "uvicorn", "app:app", "--host", args.host, "--port", str(args.port)]
    try:
        run(cmd, cwd=str(SERVICE_DIR), env=env)
    except KeyboardInterrupt:
        log("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
