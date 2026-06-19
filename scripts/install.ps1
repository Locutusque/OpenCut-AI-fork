<#
.SYNOPSIS
    OpenCut AI -- one-shot installer + launcher for Windows.

.DESCRIPTION
    The Windows counterpart of scripts/install.sh. Clones the repo (if needed),
    detects your GPU, starts the Docker stack, and -- because Docker can't pass an
    AMD GPU through on Windows -- runs the GPU-bound AI services (turboquant,
    image, tts, speaker) natively on the host via scripts/run-native.py while the
    rest stays in Docker.

    Compute modes:
      AMD (ROCm)  -> supporting stack in Docker + GPU services native (host GPU)
      NVIDIA      -> full stack in Docker with the GPU override (Docker Desktop
                     + NVIDIA Container Toolkit expose the GPU through WSL2)
      CPU         -> full stack in Docker, no GPU

.PARAMETER Auto    Auto-detect the GPU (default).
.PARAMETER Rocm    Force AMD ROCm mode (native turboquant).
.PARAMETER Nvidia  Force NVIDIA mode (docker-compose.gpu.yml).
.PARAMETER Cpu     Force CPU-only mode.
.PARAMETER Model   Ollama model to pull (default llama3.2:1b).
.PARAMETER NoPull  Skip pulling the default Ollama model.
.PARAMETER NoNativeLaunch
                   Set up native mode but don't auto-launch run-native.py
                   (it prints the command to run instead).
.PARAMETER Dir     Where to clone the repo (default .\OpenCut-AI).
.PARAMETER Repo    Git URL to clone.

.EXAMPLE
    .\scripts\install.ps1
.EXAMPLE
    .\scripts\install.ps1 -Rocm -Model llama3.2:3b
#>
[CmdletBinding()]
param(
    [switch]$Auto,
    [switch]$Rocm,
    [switch]$Nvidia,
    [switch]$Cpu,
    [string]$Model = "llama3.2:1b",
    [switch]$NoPull,
    [switch]$NoNativeLaunch,
    [string]$Dir = "OpenCut-AI",
    [string]$Repo = "https://github.com/Locutusque/OpenCut-AI-fork.git"
)

$ErrorActionPreference = "Stop"
$OllamaUrl = "http://localhost:11434"
$MaxRetries = 30
$RetryInterval = 5

function Log-Info  { param($m) Write-Host "[INFO]  $m" -ForegroundColor Green }
function Log-Warn  { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }
function Log-Step  { param($m) Write-Host "`n==> $m" -ForegroundColor Cyan }

function Have-Cmd { param($name) return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

# --- Resolve compute mode from switches -------------------------------------
$GpuMode = "auto"
if ($Cpu)    { $GpuMode = "cpu" }
if ($Rocm)   { $GpuMode = "rocm" }
if ($Nvidia) { $GpuMode = "nvidia" }
if ($Auto)   { $GpuMode = "auto" }

# --- Pre-flight: required tooling -------------------------------------------
Log-Step "Checking prerequisites"
$missing = $false
foreach ($bin in @("docker", "git")) {
    if (-not (Have-Cmd $bin)) { Log-Error "'$bin' is not installed or not in PATH."; $missing = $true }
}
if ($missing) { Log-Error "Install the missing tools above and re-run."; exit 1 }

# Python is only required for the native GPU path, but resolve it now.
$Python = $null
foreach ($cand in @("python", "py")) {
    if (Have-Cmd $cand) { $Python = $cand; break }
}

try { docker compose version *> $null } catch {
    Log-Error "'docker compose' (v2) is not available. Install Docker Desktop."; exit 1
}
try { docker info *> $null } catch {
    Log-Error "Docker daemon is not running. Start Docker Desktop and re-run."; exit 1
}
Log-Info "docker, docker compose, git present."

# --- Locate or clone the repo -----------------------------------------------
# Files the compute paths depend on. A stale shallow clone from before these
# were added (or an interrupted clone) can be missing them, so we verify their
# presence and refresh/re-clone when they're absent.
$RequiredFiles = @("docker-compose.yml", "docker-compose.native-all.yml", "scripts\run-native.py")
function Test-RepoComplete {
    param($root)
    foreach ($f in $RequiredFiles) {
        if (-not (Test-Path (Join-Path $root $f))) { return $false }
    }
    return $true
}

$ScriptDir = $PSScriptRoot
if (Test-Path (Join-Path $ScriptDir "..\docker-compose.yml")) {
    $ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
    Log-Step "Using existing checkout at $ProjectRoot"
} else {
    Log-Step "Cloning $Repo"

    # Resolve the remote's default branch so we can force-sync onto its tip.
    $DefaultBranch = "main"
    try {
        $symref = git ls-remote --symref $Repo HEAD 2>$null | Select-String 'refs/heads/'
        if ($symref -and ($symref.ToString() -match 'refs/heads/(\S+)\s+HEAD')) { $DefaultBranch = $Matches[1] }
    } catch {}

    if (Test-Path (Join-Path $Dir ".git")) {
        Log-Info "Repo already cloned at $Dir -- refreshing to latest origin/$DefaultBranch."
        # The previous run may have made a shallow (--depth 1) clone. A plain
        # 'git pull --ff-only' on such a checkout can report "Already up to date"
        # and leave the user on stale code that is missing newer files. Instead,
        # fetch the default branch tip and hard-reset onto it unconditionally.
        try {
            git -C $Dir remote set-url origin $Repo
            git -C $Dir fetch --depth 1 origin $DefaultBranch
            if ($LASTEXITCODE -eq 0) {
                git -C $Dir reset --hard FETCH_HEAD
            } else {
                Log-Warn "Could not fetch origin/$DefaultBranch; will verify the existing checkout below."
            }
        } catch { Log-Warn "Refresh of existing checkout failed; will verify its contents below." }
    } else {
        git clone --depth 1 $Repo $Dir
    }

    # If the checkout is still incomplete (e.g. a stale clone that couldn't be
    # refreshed), discard it and clone fresh so required files are present.
    if ((Test-Path $Dir) -and -not (Test-RepoComplete (Resolve-Path $Dir).Path)) {
        Log-Warn "Checkout at $Dir is missing required files -- re-cloning from scratch."
        Remove-Item -Recurse -Force $Dir -ErrorAction SilentlyContinue
        git clone --depth 1 $Repo $Dir
    }

    $ProjectRoot = (Resolve-Path $Dir).Path
}
Set-Location $ProjectRoot

if (-not (Test-RepoComplete $ProjectRoot)) {
    Log-Error "Repository at $ProjectRoot is missing required files (e.g. docker-compose.native-all.yml)."
    Log-Error "Delete '$ProjectRoot' and re-run, or check out a branch that contains the full stack."
    exit 1
}

# --- Environment file -------------------------------------------------------
Log-Step "Preparing environment"
$envLocal = "apps\web\.env.local"
$envExample = "apps\web\.env.example"
if ((-not (Test-Path $envLocal)) -and (Test-Path $envExample)) {
    Copy-Item $envExample $envLocal
    Log-Info "Created $envLocal from .env.example"
}
if (-not (Test-Path ".env")) { New-Item -ItemType File -Path ".env" | Out-Null }

# --- GPU detection ----------------------------------------------------------
function Detect-Gpu {
    if (Have-Cmd "nvidia-smi") {
        try { & nvidia-smi -L *> $null; if ($LASTEXITCODE -eq 0) { return "nvidia" } } catch {}
    }
    try {
        $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        foreach ($g in $gpus) {
            if ($g.Name -match "AMD|Radeon|gfx") { return "rocm" }
            if ($g.Name -match "NVIDIA") { return "nvidia" }
        }
    } catch {}
    return "cpu"
}

if ($GpuMode -eq "auto") {
    $GpuMode = Detect-Gpu
    Log-Step "Auto-detected compute mode: $GpuMode"
} else {
    Log-Step "Compute mode (forced): $GpuMode"
}

# --- Assemble compose command -----------------------------------------------
# On Windows, AMD ROCm means the GPU service runs natively (no Docker GPU
# passthrough for AMD on Windows). NVIDIA works through Docker Desktop's WSL2
# backend, so it uses the standard GPU override.
$composeArgs = @("-f", "docker-compose.yml")
$upExtra = @()
$Native = $false

switch ($GpuMode) {
    "nvidia" {
        if (-not (Test-Path "docker-compose.gpu.yml")) {
            Log-Error "docker-compose.gpu.yml missing -- cannot start NVIDIA mode."; exit 1
        }
        $composeArgs += @("-f", "docker-compose.gpu.yml")
        Log-Info "NVIDIA mode: turboquant-service runs on CUDA in Docker (Desktop + NVIDIA toolkit)."
    }
    "rocm" {
        if (-not (Test-Path "docker-compose.native-all.yml")) {
            Log-Error "docker-compose.native-all.yml missing -- cannot run native ROCm mode."; exit 1
        }
        $Native = $true
        $composeArgs += @("-f", "docker-compose.native-all.yml")
        # Docker has no AMD GPU passthrough on Windows, so every torch-using
        # service runs natively on the host instead. Scale those containers to 0.
        $upExtra = @(
            "--scale", "turboquant-service=0",
            "--scale", "image-service=0",
            "--scale", "tts-service=0",
            "--scale", "speaker-service=0"
        )
        Log-Info "AMD ROCm mode (Windows): supporting stack in Docker, GPU services native on the host."
        if (-not $Python) {
            Log-Warn "Python not found on PATH -- needed for the native GPU services. Install Python 3.10+."
        }
    }
    "cpu" {
        Log-Info "CPU mode: all AI services run on CPU. (Slower, but works anywhere.)"
    }
    default { Log-Error "Unknown GPU mode: $GpuMode"; exit 1 }
}

# --- Build + start ----------------------------------------------------------
Log-Step "Building and starting the stack (this can take a while on first run)"
$env:DOCKER_BUILDKIT = "1"
docker compose @composeArgs up -d --build @upExtra
if ($LASTEXITCODE -ne 0) { Log-Error "docker compose up failed."; exit 1 }

# --- Wait for Ollama --------------------------------------------------------
Log-Step "Waiting for Ollama to come online"
$ready = $false
for ($i = 1; $i -le $MaxRetries; $i++) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$OllamaUrl/api/tags" -TimeoutSec 5 *> $null
        $ready = $true; break
    } catch {
        Log-Warn "Ollama not ready yet (attempt $i/$MaxRetries); retrying in ${RetryInterval}s..."
        Start-Sleep -Seconds $RetryInterval
    }
}
if ($ready) { Log-Info "Ollama is ready." } else { Log-Warn "Ollama not ready yet -- LLM features may be delayed." }

# --- Pull the default model -------------------------------------------------
if (-not $NoPull) {
    Log-Step "Pulling default LLM model: $Model"
    docker compose @composeArgs exec -T ollama ollama pull $Model
    if ($LASTEXITCODE -eq 0) { Log-Info "Model $Model is ready." }
    else { Log-Warn "Could not pull $Model. Pull one later: docker compose exec ollama ollama pull <model>" }
}

# --- Status -----------------------------------------------------------------
Log-Step "Service status"
docker compose @composeArgs ps --format "table {{.Name}}`t{{.Status}}`t{{.Ports}}"

Write-Host ""
Log-Info "OpenCut AI is up. Compute mode: $GpuMode"
Log-Info "Web App:     http://localhost:3100"
Log-Info "AI Backend:  http://localhost:8420"
Log-Info "Ollama API:  $OllamaUrl"

# --- Native GPU services (AMD on Windows) -----------------------------------
if ($Native) {
    Write-Host ""
    Log-Step "GPU services (run natively so they can use your AMD GPU)"
    Log-Info "PyTorch for ROCm is installed automatically on first run: run-native.py"
    Log-Info "downloads the matching AMD Radeon wheels for your Python version. No"
    Log-Info "manual setup needed (Python 3.10-3.13 supported)."
    Log-Info "If AMD republishes under a new path, override ROCM_WINDOWS_REL /"
    Log-Info "ROCM_WINDOWS_TORCH_VER, or set ROCM_WINDOWS_TORCH_WHEELS to exact URLs."
    Log-Info "AMD's guide: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html"
    Log-Info "bitsandbytes (4-bit) uses an AMD/ROCm build auto-selected from your"
    Log-Info "detected GPU (RDNA vs CDNA). Force a build with ROCM_WINDOWS_BNB_TAG"
    Log-Info "from https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases"

    # turboquant + image need only torch; tts + speaker also need torchaudio.
    $nativeServices = @(
        @{ name = "turboquant"; port = 8430 },
        @{ name = "image";      port = 8423 },
        @{ name = "tts";        port = 8422 },
        @{ name = "speaker";    port = 8424 }
    )
    $runScript = Join-Path $ProjectRoot "scripts\run-native.py"

    if ($Python -and -not $NoNativeLaunch) {
        foreach ($svc in $nativeServices) {
            Log-Info "Launching native $($svc.name)-service (port $($svc.port)) in a new window..."
            Start-Process -FilePath "powershell" -ArgumentList @(
                "-NoExit", "-Command", "& '$Python' '$runScript' --service $($svc.name)"
            ) -WorkingDirectory $ProjectRoot
        }
        Log-Info "Native GPU services starting (one window each)."
        Log-Info "The Dockerised ai-backend points at them via host.docker.internal."
    } else {
        $pyName = if ($Python) { $Python } else { "python" }
        Log-Info "Start the GPU services yourself (one terminal each):"
        foreach ($svc in $nativeServices) {
            Log-Info "  $pyName scripts\run-native.py --service $($svc.name)"
        }
    }
}

Write-Host ""
Log-Info "Stop the Docker part with: docker compose $($composeArgs -join ' ') down"
if ($Native) { Log-Info "Stop the GPU service by closing its window (or Ctrl+C in it)." }
