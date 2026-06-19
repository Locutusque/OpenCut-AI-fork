FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 espeak-ng build-essential cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch backend selection ---------------------------------------------
# Chosen at build time so one Dockerfile serves CPU, NVIDIA CUDA and AMD ROCm.
# Default = CPU wheel index (no nvidia-*/cuda-* packages). The GPU/ROCm compose
# overrides set TORCH_INDEX_URL to the CUDA or ROCm wheel index. coqui-tts pulls
# torchaudio, so it must come from the same backend index as torch — both are
# listed in TORCH_PACKAGES and stripped from the PyPI install below.
# coqui-tts 0.24.2 requires torch<2.6, so torch/torchaudio are pinned to 2.5.1
# (available on the cpu, cu124 and rocm6.2 indexes). The CUDA override uses
# cu124 rather than cu128, which only ships torch>=2.7. Exact pins avoid shell
# redirection metacharacters (<, >) in the unquoted ${TORCH_PACKAGES} below.
#   CPU   : https://download.pytorch.org/whl/cpu       (default)
#   CUDA  : https://download.pytorch.org/whl/cu124      (docker-compose.gpu.yml)
#   ROCm  : https://download.pytorch.org/whl/rocm6.2    (docker-compose.rocm.yml, Linux containers)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGES="torch==2.5.1 torchaudio==2.5.1"

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    # 1. Install torch + torchaudio from the selected backend index only.
    uv pip install --system --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} \
    # 2. Install everything else from PyPI with torch* stripped so coqui-tts
    #    doesn't drag in a CUDA torchaudio/torch from PyPI.
 && grep -ivE '^(torch|torchaudio|torchvision)([[:space:]<>=!~;,]|$)' requirements.txt > requirements.rest.txt \
 && uv pip install --system -r requirements.rest.txt \
 && rm -f requirements.rest.txt

COPY . .

RUN mkdir -p generated/voices
ENV COQUI_TOS_AGREED=1

EXPOSE 8422

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8422"]
