FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch backend selection ---------------------------------------------
# The PyTorch build is chosen at build time so a single Dockerfile serves CPU,
# NVIDIA CUDA and AMD ROCm. The default is the CPU wheel index, which pulls NO
# nvidia-*/cuda-* packages — a plain `docker compose up` stays completely
# CUDA-free. torch is pinned to the 2.9 series (~=2.9.0) so every backend matches
# the AMD Windows ROCm wheels (torch 2.9.1). The GPU/ROCm compose overrides set
# TORCH_INDEX_URL to the CUDA or ROCm wheel index instead:
#   CPU   : https://download.pytorch.org/whl/cpu       (default)
#   CUDA  : https://download.pytorch.org/whl/cu128      (docker-compose.gpu.yml)
#   ROCm  : https://download.pytorch.org/whl/rocm6.4    (docker-compose.rocm.yml, Linux containers)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch~=2.9.0"

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    # 1. Install torch from the selected backend index as the ONLY source for it.
    uv pip install --system --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} \
    # 2. Install everything else from PyPI with torch* stripped, so the backend
    #    wheel installed above is never silently replaced by a CUDA PyPI build.
 && grep -ivE '^(torch|torchaudio|torchvision)([[:space:]<>=!~;,]|$)' requirements.txt > requirements.rest.txt \
 && uv pip install --system -r requirements.rest.txt \
 && rm -f requirements.rest.txt

COPY . .

RUN mkdir -p generated uploads

EXPOSE 8423

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8423"]
