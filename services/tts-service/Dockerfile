FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 espeak-ng build-essential cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- PyTorch backend selection ---------------------------------------------
# Chosen at build time so one Dockerfile serves CPU, NVIDIA CUDA and AMD ROCm.
# Default = CPU wheel index (no nvidia-*/cuda-* packages). The GPU/ROCm compose
# overrides set TORCH_INDEX_URL to the CUDA or ROCm wheel index. torch +
# torchaudio come from that index; both are stripped from the PyPI install.
#
# Pinned to the torch 2.9 series (~=2.9.0) so every backend matches the AMD
# Windows ROCm wheels (torch 2.9.1). coqui-tts 0.27.x supports torch>=2.2 and no
# longer bundles torch, so 2.9 works. The `~=` compatible-release form is
# shell-safe (no <,> redirection metacharacters) when expanded unquoted below.
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
    # 2. torchcodec for torch 2.9 audio decoding (coqui-tts >=0.27.4). Best-effort
    #    and --no-deps so it can't pull a CPU torch over the backend build; if a
    #    matching wheel isn't published for this backend the build still succeeds.
 && (uv pip install --system --no-deps "torchcodec>=0.8.0" \
      || echo "torchcodec unavailable for this backend — coqui-tts falls back to soundfile.") \
    # 3. Install everything else from PyPI with torch* stripped so coqui-tts
    #    doesn't drag in a CUDA torchaudio/torch from PyPI.
 && grep -ivE '^(torch|torchaudio|torchvision|torchcodec)([[:space:]<>=!~;,]|$)' requirements.txt > requirements.rest.txt \
 && uv pip install --system -r requirements.rest.txt \
 && rm -f requirements.rest.txt

COPY . .

RUN mkdir -p generated/voices
ENV COQUI_TOS_AGREED=1

EXPOSE 8422

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8422"]
