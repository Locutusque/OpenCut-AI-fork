# Lightweight adapter image (proxy mode — GPU path via a vLLM/TGI sidecar).
# Default ENGINE=auto resolves to the proxy engine when UPSTREAM_URL is set.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY . .

EXPOSE 8430

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8430/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8430"]
