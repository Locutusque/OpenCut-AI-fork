FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch backend selection ---------------------------------------------
# Chosen at build time so one Dockerfile serves CPU, NVIDIA CUDA and AMD ROCm.
# Default = CPU wheel index (no nvidia-*/cuda-* packages). pyannote.audio and
# speechbrain depend on torch + torchaudio, so both come from the selected
# backend index, pinned to the 2.9 series (~=2.9.0) to match the AMD Windows
# ROCm wheels (torch 2.9.1). The GPU/ROCm compose overrides set TORCH_INDEX_URL.
#   CPU   : https://download.pytorch.org/whl/cpu       (default)
#   CUDA  : https://download.pytorch.org/whl/cu128      (docker-compose.gpu.yml)
#   ROCm  : https://download.pytorch.org/whl/rocm6.4    (docker-compose.rocm.yml, Linux containers)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch~=2.9.0 torchaudio~=2.9.0"

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    # 1. Install torch + torchaudio from the selected backend index only.
    uv pip install --system --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} \
    # 2. Install everything else from PyPI with torch* stripped so pyannote /
    #    speechbrain don't drag in a CUDA torch/torchaudio from PyPI.
 && grep -ivE '^(torch|torchaudio|torchvision)([[:space:]<>=!~;,]|$)' requirements.txt > requirements.rest.txt \
 && uv pip install --system -r requirements.rest.txt \
 && rm -f requirements.rest.txt

COPY . .

RUN mkdir -p uploads

EXPOSE 8424

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8424"]
