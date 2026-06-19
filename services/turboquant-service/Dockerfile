FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of the turboquant-service. NVIDIA CUDA uses Dockerfile.gpu and AMD
# ROCm uses Dockerfile.rocm (both wired up by the compose override files).
#
# torch is installed from a backend wheel index chosen at build time; the
# default is the CPU index so the base `docker compose up` pulls NO
# nvidia-*/cuda-* packages. Pinned to the 2.9 series (~=2.9.0) to match the AMD
# Windows ROCm wheels (torch 2.9.1). bitsandbytes (4-bit) and turboquant-gpu
# (NVIDIA cuTile) are GPU-only, so they are stripped here — the service loads
# the model in bf16/fp32 on CPU and degrades gracefully.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch~=2.9.0"
ARG TURBOQUANT_EXTRAS=""

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    # 1. Install torch from the selected backend index (CPU by default).
    uv pip install --system --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} \
    # 2. Install the rest from PyPI, minus the three GPU-vendor lines.
 && grep -ivE '^(torch|bitsandbytes|turboquant-gpu)([[:space:]<>=!~;,]|$)' requirements.txt > requirements.cpu.txt \
 && uv pip install --system -r requirements.cpu.txt \
 && rm -f requirements.cpu.txt \
    # 3. Optional NVIDIA cuTile extras (only when explicitly requested).
 && if [ -n "$TURBOQUANT_EXTRAS" ]; then \
      uv pip install --system \
        --index-strategy unsafe-best-match \
        "turboquant-gpu[$TURBOQUANT_EXTRAS]" \
        --extra-index-url https://pypi.nvidia.com ; \
    fi

COPY app.py compute_backends.py ./

RUN mkdir -p /app/models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

EXPOSE 8430

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8430"]
