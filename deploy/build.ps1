<#
.SYNOPSIS
    Build the Romanian TTS API + MCP Docker image.

.DESCRIPTION
    Builds the image defined by Dockerfile.romanian. Run this once, and again whenever the
    code or dependencies change.

.PARAMETER Tag
    Image tag to build. Default: pocket-romanian-tts:latest

.PARAMETER NoCache
    Build without using the Docker layer cache.

.EXAMPLE
    ./deploy/build.ps1
    ./deploy/build.ps1 -NoCache
#>
[CmdletBinding()]
param(
    [string]$Tag = "pocket-romanian-tts:latest",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Building image '$Tag' from Dockerfile.romanian..." -ForegroundColor Cyan

$buildArgs = @("build", "-f", "Dockerfile.romanian", "-t", $Tag)
if ($NoCache) { $buildArgs += "--no-cache" }
$buildArgs += "."

& docker @buildArgs
if ($LASTEXITCODE -ne 0) { throw "docker build failed with exit code $LASTEXITCODE" }

Write-Host "Built '$Tag'." -ForegroundColor Green
Write-Host "Next: ./deploy/start.ps1  (compose)  or  ./deploy/run.ps1  (single container)" -ForegroundColor Yellow
