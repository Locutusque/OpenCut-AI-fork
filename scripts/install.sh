#!/usr/bin/env bash
#
# OpenCut AI — one-shot installer + launcher.
#
# Clones the repo (if needed), detects your GPU (NVIDIA / AMD-ROCm / CPU),
# builds and starts the full Docker stack with the right override file, and
# pulls the default LLM model. Re-running it is safe — it just rebuilds and
# brings the stack back up.
#
# Quick start (from anywhere):
#   curl -fsSL https://raw.githubusercontent.com/Ekaanth/OpenCut-AI/main/scripts/install.sh | bash
#
# Or from a checkout:
#   ./scripts/install.sh
#
# Flags:
#   --cpu            Force CPU-only mode (ignore any GPU)
#   --nvidia         Force NVIDIA GPU mode (docker-compose.gpu.yml)
#   --rocm           Force AMD ROCm GPU mode (docker-compose.rocm.yml)
#   --auto           Auto-detect GPU (default)
#   --model <name>   Ollama model to pull (default: llama3.2:1b)
#   --no-pull        Skip pulling the default Ollama model
#   --dir <path>     Where to clone the repo (default: ./OpenCut-AI)
#   --repo <url>     Git URL to clone (default: upstream OpenCut-AI)
#   -h, --help       Show this help

set -euo pipefail

# --- Config / defaults ------------------------------------------------------
REPO_URL="${OPENCUT_REPO_URL:-https://github.com/Ekaanth/OpenCut-AI.git}"
CLONE_DIR="${OPENCUT_DIR:-OpenCut-AI}"
DEFAULT_MODEL="llama3.2:1b"
GPU_MODE="auto"          # auto | cpu | nvidia | rocm
DO_PULL=1
MAX_RETRIES=30
RETRY_INTERVAL=5
OLLAMA_URL="http://localhost:11434"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}==>${NC} $*"; }

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

# --- Parse args -------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)     GPU_MODE="cpu" ;;
        --nvidia)  GPU_MODE="nvidia" ;;
        --rocm)    GPU_MODE="rocm" ;;
        --auto)    GPU_MODE="auto" ;;
        --model)   DEFAULT_MODEL="${2:?--model needs a value}"; shift ;;
        --no-pull) DO_PULL=0 ;;
        --dir)     CLONE_DIR="${2:?--dir needs a value}"; shift ;;
        --repo)    REPO_URL="${2:?--repo needs a value}"; shift ;;
        -h|--help) usage ;;
        *) log_error "Unknown argument: $1"; usage ;;
    esac
    shift
done

# --- Pre-flight: required tooling -------------------------------------------
log_step "Checking prerequisites"
missing=0
for bin in docker git curl; do
    if ! command -v "$bin" &>/dev/null; then
        log_error "'$bin' is not installed or not in PATH."
        missing=1
    fi
done
[ "$missing" -eq 0 ] || { log_error "Install the missing tools above and re-run."; exit 1; }

if ! docker compose version &>/dev/null; then
    log_error "'docker compose' (v2) is not available. Install Docker Compose v2.3+."
    exit 1
fi
if ! docker info &>/dev/null; then
    log_error "Docker daemon is not running (or you lack permission). Start Docker and re-run."
    exit 1
fi
log_info "docker, docker compose, git, curl all present."

# --- Locate or clone the repo ----------------------------------------------
# If we're already inside a checkout (this script lives in <root>/scripts), use
# that. Otherwise clone fresh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../docker-compose.yml" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    log_step "Using existing checkout at $PROJECT_ROOT"
else
    log_step "Cloning $REPO_URL"
    if [ -d "$CLONE_DIR/.git" ]; then
        log_info "Repo already cloned at $CLONE_DIR — pulling latest."
        git -C "$CLONE_DIR" pull --ff-only || log_warn "git pull failed; using existing checkout."
    else
        git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
    fi
    PROJECT_ROOT="$(cd "$CLONE_DIR" && pwd)"
fi
cd "$PROJECT_ROOT"

# --- Environment file -------------------------------------------------------
log_step "Preparing environment"
if [ ! -f apps/web/.env.local ] && [ -f apps/web/.env.example ]; then
    cp apps/web/.env.example apps/web/.env.local
    log_info "Created apps/web/.env.local from .env.example"
fi
# Root .env holds the compose-level knobs (GPU group ids, model, etc.).
touch .env

# --- GPU detection ----------------------------------------------------------
detect_gpu() {
    # Returns: nvidia | rocm | cpu
    if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
        echo "nvidia"; return
    fi
    # AMD: ROCm tools present, or the kernel compute device node exists.
    if command -v rocminfo &>/dev/null || command -v rocm-smi &>/dev/null; then
        echo "rocm"; return
    fi
    if [ -e /dev/kfd ] && [ -d /dev/dri ]; then
        echo "rocm"; return
    fi
    echo "cpu"
}

if [ "$GPU_MODE" = "auto" ]; then
    GPU_MODE="$(detect_gpu)"
    log_step "Auto-detected compute mode: ${GPU_MODE}"
else
    log_step "Compute mode (forced): ${GPU_MODE}"
fi

# --- Assemble compose command -----------------------------------------------
COMPOSE_FILES=(-f docker-compose.yml)
case "$GPU_MODE" in
    nvidia)
        if [ ! -f docker-compose.gpu.yml ]; then
            log_error "docker-compose.gpu.yml missing — cannot start NVIDIA mode."; exit 1
        fi
        COMPOSE_FILES+=(-f docker-compose.gpu.yml)
        log_info "NVIDIA mode: turboquant-service will run on CUDA with cuTile kernels."
        if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
            log_warn "Docker could not access the GPU. Install the NVIDIA Container Toolkit if startup fails."
        fi
        ;;
    rocm)
        if [ ! -f docker-compose.rocm.yml ]; then
            log_error "docker-compose.rocm.yml missing — cannot start ROCm mode."; exit 1
        fi
        COMPOSE_FILES+=(-f docker-compose.rocm.yml)
        # ROCm needs the host's video/render group ids and the GPU device nodes.
        VIDEO_GID="$(getent group video  | cut -d: -f3 2>/dev/null || echo 44)"
        RENDER_GID="$(getent group render | cut -d: -f3 2>/dev/null || echo 993)"
        : "${VIDEO_GID:=44}"; : "${RENDER_GID:=993}"
        set_env() { # key value
            if grep -q "^$1=" .env 2>/dev/null; then
                sed -i "s|^$1=.*|$1=$2|" .env
            else
                echo "$1=$2" >> .env
            fi
        }
        set_env VIDEO_GID "$VIDEO_GID"
        set_env RENDER_GID "$RENDER_GID"
        # Preserve any GFX override the user already set; otherwise leave blank.
        grep -q "^HSA_OVERRIDE_GFX_VERSION=" .env || \
            echo "# Uncomment + set if your card needs it (RDNA2=10.3.0, RDNA3=11.0.0):" >> .env
        grep -q "^HSA_OVERRIDE_GFX_VERSION=" .env || \
            echo "# HSA_OVERRIDE_GFX_VERSION=10.3.0" >> .env
        log_info "ROCm mode: Ollama (ollama:rocm) + turboquant-service on your AMD GPU."
        log_info "Host GPU groups → video=${VIDEO_GID}, render=${RENDER_GID} (written to .env)."
        if [ ! -e /dev/kfd ]; then
            log_warn "/dev/kfd not found — is the amdgpu/ROCm kernel driver loaded on the host?"
        fi
        log_info "If the GPU stays idle, set HSA_OVERRIDE_GFX_VERSION in .env (see comments there)."
        ;;
    cpu)
        log_info "CPU mode: all AI services run on CPU. (Slower, but works anywhere.)"
        ;;
    *)
        log_error "Unknown GPU mode: $GPU_MODE"; exit 1 ;;
esac

# --- Build + start ----------------------------------------------------------
log_step "Building and starting the stack (this can take a while on first run)"
export DOCKER_BUILDKIT=1
# Persist the chosen model for the ai-backend default too.
docker compose "${COMPOSE_FILES[@]}" up -d --build

# --- Wait for Ollama --------------------------------------------------------
log_step "Waiting for Ollama to come online"
retries=0
until curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; do
    retries=$((retries + 1))
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
        log_warn "Ollama not ready after $((MAX_RETRIES * RETRY_INTERVAL))s — LLM features may be delayed."
        break
    fi
    log_warn "Ollama not ready yet (attempt ${retries}/${MAX_RETRIES}); retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done
[ "$retries" -lt "$MAX_RETRIES" ] && log_info "Ollama is ready."

# --- Pull the default model -------------------------------------------------
if [ "$DO_PULL" -eq 1 ]; then
    log_step "Pulling default LLM model: ${DEFAULT_MODEL}"
    if docker compose "${COMPOSE_FILES[@]}" exec -T ollama ollama pull "$DEFAULT_MODEL"; then
        log_info "Model ${DEFAULT_MODEL} is ready."
    else
        log_warn "Could not pull ${DEFAULT_MODEL}. Pull one later with: docker compose exec ollama ollama pull <model>"
    fi
fi

# --- Status -----------------------------------------------------------------
log_step "Service status"
docker compose "${COMPOSE_FILES[@]}" ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" || true

echo ""
log_info "OpenCut AI is up. Compute mode: ${GPU_MODE}"
log_info "Web App:     http://localhost:3100"
log_info "AI Backend:  http://localhost:8420"
log_info "Ollama API:  ${OLLAMA_URL}"
if [ "$GPU_MODE" = "rocm" ]; then
    echo ""
    log_info "Verify the AMD GPU was picked up:"
    log_info "  curl -s http://localhost:8430/health | grep -E 'gpu_vendor|compute_mode|rocm'"
    log_info "  Expect: \"gpu_vendor\":\"amd\", \"compute_mode\":\"rocm\", \"rocm\":true"
elif [ "$GPU_MODE" = "nvidia" ]; then
    echo ""
    log_info "Verify the NVIDIA GPU was picked up:"
    log_info "  curl -s http://localhost:8430/health | grep -E 'gpu_vendor|compute_mode'"
fi
echo ""
log_info "Stop everything with: docker compose ${COMPOSE_FILES[*]} down"
