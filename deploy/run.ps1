<#
.SYNOPSIS
    Run the Romanian TTS API + MCP server as a single Docker container (no compose).

.DESCRIPTION
    Runs the image built by build.ps1 directly with `docker run`. Useful for quick local
    testing or when you do not want to use docker compose. Persists the HuggingFace / TTS
    model caches and bind-mounts the output directory so generated audio lands on the host.

.PARAMETER Tag
    Image tag to run. Default: pocket-romanian-tts:latest

.PARAMETER Port
    Host port to publish (container listens on 8000). Default: 8001

.PARAMETER Name
    Container name. Default: romanian-tts

.PARAMETER Detach
    Run detached (default is foreground so you can watch startup logs).

.EXAMPLE
    ./deploy/run.ps1
    ./deploy/run.ps1 -Port 9000 -Detach
#>
[CmdletBinding()]
param(
    [string]$Tag = "pocket-romanian-tts:latest",
    [int]$Port = 8001,
    [string]$Name = "romanian-tts",
    [switch]$Detach
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$outDir = Join-Path $RepoRoot "tts_outputs/romanian_api"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

# Remove a previous container with the same name, if present.
$existing = docker ps -aq -f "name=^$Name$"
if ($existing) {
    Write-Host "Removing existing container '$Name'..." -ForegroundColor Yellow
    docker rm -f $Name | Out-Null
}

$runArgs = @(
    "run",
    "--name", $Name,
    "-p", "${Port}:8000",
    "-e", "ROMANIAN_PUBLIC_BASE_URL=http://localhost:$Port",
    "-e", "ROMANIAN_PRELOAD_PRIMARY=1",
    "-e", "ROMANIAN_PRELOAD_SECONDARY=1",
    "-e", "COQUI_TOS_AGREED=1",
    "-v", "pocket_hf_cache:/root/.cache/huggingface",
    "-v", "pocket_tts_models:/root/.local/share/tts",
    "-v", "${outDir}:/data/outputs"
)
if ($Detach) { $runArgs += "-d" }
$runArgs += $Tag

Write-Host "Running '$Tag' as '$Name' on http://localhost:$Port ..." -ForegroundColor Cyan
& docker @runArgs
if ($LASTEXITCODE -ne 0) { throw "docker run failed with exit code $LASTEXITCODE" }

if ($Detach) {
    Write-Host "Container '$Name' started." -ForegroundColor Green
    Write-Host "  API/Health: http://localhost:$Port/health" -ForegroundColor Green
    Write-Host "  MCP:        http://localhost:$Port/mcp" -ForegroundColor Green
    Write-Host "Logs: docker logs -f $Name" -ForegroundColor Yellow
}
