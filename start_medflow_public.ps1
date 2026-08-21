[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$envFile = Join-Path $projectRoot '.env'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment was not found: $python"
}
if (-not (Test-Path -LiteralPath $envFile)) {
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $secretBytes = New-Object byte[] 48
    $rng.GetBytes($secretBytes)
    $secret = [Convert]::ToBase64String($secretBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    @(
        'APP_ENV=production'
        'DEV_AUTH_BYPASS=false'
        'SECURE_COOKIE=true'
        "SECRET_KEY=$secret"
        'INITIAL_ADMIN_EMAIL='
        'INITIAL_ADMIN_PASSWORD='
        'PDF_UPLOAD_MAX_MB=25'
    ) | Set-Content -LiteralPath $envFile -Encoding utf8
    Write-Host 'Created local .env with session secret.' -ForegroundColor Yellow
}

$env:APP_ENV = 'production'
$env:DEV_AUTH_BYPASS = 'false'
$env:SECURE_COOKIE = 'true'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONPATH = "$projectRoot$([IO.Path]::PathSeparator)$(Join-Path $projectRoot 'medflow20')"

$logDir = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$backendLog = Join-Path $logDir 'medflow-public-backend.log'
$backendErrorLog = Join-Path $logDir 'medflow-public-backend-error.log'
$tunnelLog = Join-Path $logDir 'medflow-public-tunnel.log'

$backend = $null
$tunnel = $null
try {
    $ready = $false
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/health' -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($health.ready -eq $true -and [int]$health.active_chunks -gt 0) {
            $ready = $true
            Write-Host '[Launcher] Reusing active Medflow20 server on http://127.0.0.1:7860' -ForegroundColor Green
        }
    } catch {}

    if (-not $ready) {
        Remove-Item -LiteralPath $backendLog, $backendErrorLog -Force -ErrorAction SilentlyContinue
        Write-Host 'Starting local Medflow20 backend on http://127.0.0.1:7860 ...'
        $backend = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','7860' -WorkingDirectory $projectRoot -RedirectStandardOutput $backendLog -RedirectStandardError $backendErrorLog -PassThru -NoNewWindow

        for ($i = 0; $i -lt 180; $i++) {
            if ($backend.HasExited) { throw "Medflow backend stopped. See $backendLog" }
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/health' -TimeoutSec 2
                if ($health.ready -eq $true -and [int]$health.active_chunks -gt 0) {
                    $ready = $true
                    break
                }
            } catch {}
            Start-Sleep -Seconds 1
        }
        if (-not $ready) { throw "Medflow20 backend initialization timed out." }
    }

    Write-Host "Medflow20: READY | ChromaDB: READY | RAG: READY | Active chunks: $($health.active_chunks)" -ForegroundColor Green

    Write-Host 'Launching persistent public HTTPS tunnel via Serveo ...' -ForegroundColor Cyan
    Remove-Item -LiteralPath $tunnelLog -Force -ErrorAction SilentlyContinue
    $tunnel = Start-Process -FilePath 'ssh' -ArgumentList '-tt','-o','StrictHostKeyChecking=no','-R','80:127.0.0.1:7860','serveo.net' -WorkingDirectory $projectRoot -RedirectStandardOutput $tunnelLog -RedirectStandardError $tunnelLog -PassThru -NoNewWindow

    $publicUrl = $null
    for ($i = 0; $i -lt 60; $i++) {
        if ($tunnel.HasExited) { break }
        $match = Select-String -LiteralPath $tunnelLog -Pattern 'https://[a-z0-9-]+.serveousercontent.com' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) {
            $publicUrl = $match.Matches[0].Value
            break
        }
        Start-Sleep -Seconds 1
    }

    if (-not $publicUrl) {
        throw "Could not obtain persistent public HTTPS URL. See $tunnelLog"
    }

    Write-Host ''
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "  FINAL LIVE DEMO URL: $publicUrl" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host 'This public HTTPS URL remains live and active. Press Ctrl+C to stop.' -ForegroundColor Yellow

    while ($true) {
        if ($tunnel.HasExited) {
            Write-Host 'Tunnel disconnected. Reconnecting ...' -ForegroundColor Yellow
            Remove-Item -LiteralPath $tunnelLog -Force -ErrorAction SilentlyContinue
            $tunnel = Start-Process -FilePath 'ssh' -ArgumentList '-tt','-o','StrictHostKeyChecking=no','-R','80:127.0.0.1:7860','serveo.net' -WorkingDirectory $projectRoot -RedirectStandardOutput $tunnelLog -RedirectStandardError $tunnelLog -PassThru -NoNewWindow
        }
        Start-Sleep -Seconds 5
    }
} finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue }
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue }
    Write-Host 'Medflow Public Mode stopped cleanly.'
}
