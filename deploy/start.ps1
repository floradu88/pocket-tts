<#
.SYNOPSIS
    Start the Romanian TTS API + MCP server via docker compose.

.DESCRIPTION
    Brings up the `romanian-tts` service defined in docker-compose.yaml. Creates a .env from
    .env.example on first run if one does not exist. The API and MCP endpoint are published
    on the host port from .env (ROMANIAN_PORT, default 8001).

.PARAMETER Build
    Rebuild the image before starting.

.PARAMETER Foreground
    Run in the foreground (stream logs) instead of detached.

.EXAMPLE
    ./deploy/start.ps1
    ./deploy/start.ps1 -Build
    ./deploy/start.ps1 -Foreground
#>
[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".env")) {
    Write-Host "No .env found; creating one from .env.example." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# Ensure the host output directory exists (bind-mounted into the container).
$outDir = Join-Path $RepoRoot "tts_outputs/romanian_api"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

$composeArgs = @("compose", "up", "romanian-tts")
if ($Build) { $composeArgs += "--build" }
if (-not $Foreground) { $composeArgs += "-d" }

Write-Host "Starting romanian-tts (docker compose)..." -ForegroundColor Cyan
& docker @composeArgs
if ($LASTEXITCODE -ne 0) { throw "docker compose up failed with exit code $LASTEXITCODE" }

if (-not $Foreground) {
    $port = if ($env:ROMANIAN_PORT) { $env:ROMANIAN_PORT } else { "8001" }
    Write-Host "romanian-tts is starting." -ForegroundColor Green
    Write-Host "  API:    http://localhost:$port" -ForegroundColor Green
    Write-Host "  Health: http://localhost:$port/health" -ForegroundColor Green
    Write-Host "  MCP:    http://localhost:$port/mcp" -ForegroundColor Green
    Write-Host "Logs: docker compose logs -f romanian-tts" -ForegroundColor Yellow
    Write-Host "Stop: docker compose down" -ForegroundColor Yellow
}
