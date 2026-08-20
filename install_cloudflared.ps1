[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsDir = Join-Path $projectRoot 'tools'
$destination = Join-Path $toolsDir 'cloudflared.exe'
New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null

Write-Host 'Downloading the official Cloudflare Tunnel client...'
Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $destination
& $destination '--version'
