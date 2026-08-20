[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$cloudflared = Join-Path $projectRoot 'tools\cloudflared.exe'
$envFile = Join-Path $projectRoot '.env'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment was not found: $python"
}
if (-not (Test-Path -LiteralPath $cloudflared)) {
    throw "Cloudflare Tunnel was not found: $cloudflared. Run install_cloudflared.ps1 once first."
}
if (-not (Test-Path -LiteralPath $envFile)) {
    # First launch creates a local-only secret.  Admin values intentionally stay
    # empty: public users can register normally and no default credential exists.
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
    Write-Host 'Created a local .env with a new random session secret. No default admin account was created.' -ForegroundColor Yellow
}

# The tunnel is the only public-facing process. FastAPI accepts localhost traffic only.
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
$tunnelErrorLog = Join-Path $logDir 'medflow-public-tunnel-error.log'
Remove-Item -LiteralPath $backendLog, $backendErrorLog, $tunnelLog, $tunnelErrorLog -Force -ErrorAction SilentlyContinue

$backend = $null
$tunnel = $null
try {
    Write-Host 'Starting local Medflow20 on http://127.0.0.1:7860 ...'
    $backend = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','7860' -WorkingDirectory $projectRoot -RedirectStandardOutput $backendLog -RedirectStandardError $backendErrorLog -PassThru -NoNewWindow

    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($backend.HasExited) { throw "Medflow stopped during startup. See $backendLog" }
        try {
            $health = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/health' -TimeoutSec 2
            if ($health.ready -eq $true -and $health.collection -eq 'thyroid_section_aware' -and [int]$health.active_chunks -gt 0) {
                $ready = $true
                break
            }
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "Medflow20 did not become ready. See $backendLog" }

    Write-Host "Medflow20: READY | ChromaDB: READY | RAG: READY | Active chunks: $($health.active_chunks)" -ForegroundColor Green

    function Start-QuickTunnel {
        Remove-Item -LiteralPath $tunnelLog, $tunnelErrorLog -Force -ErrorAction SilentlyContinue
        Write-Host 'Starting secure Cloudflare quick tunnel ...'
        $newTunnel = Start-Process -FilePath $cloudflared -ArgumentList 'tunnel','--url','http://127.0.0.1:7860','--protocol','http2','--no-autoupdate' -WorkingDirectory $projectRoot -RedirectStandardOutput $tunnelLog -RedirectStandardError $tunnelErrorLog -PassThru -NoNewWindow
        for ($i = 0; $i -lt 90; $i++) {
            if ($newTunnel.HasExited) { break }
            $match = Select-String -LiteralPath $tunnelLog, $tunnelErrorLog -Pattern 'https://[-a-z0-9]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($match) { return @{ Process = $newTunnel; Url = $match.Matches[0].Value } }
            Start-Sleep -Seconds 1
        }
        if (-not $newTunnel.HasExited) { Stop-Process -Id $newTunnel.Id -ErrorAction SilentlyContinue }
        throw "Cloudflare did not return a public URL. See $tunnelErrorLog"
    }

    $tunnelInfo = Start-QuickTunnel
    $tunnel = $tunnelInfo.Process
    $publicUrl = $tunnelInfo.Url
    Write-Host ''
    Write-Host "Public URL: $publicUrl" -ForegroundColor Cyan
    Write-Host 'The URL will be recreated automatically if the network changes. Press Ctrl+C to stop.' -ForegroundColor Green

    $failedPublicChecks = 0
    $nextCheck = [DateTime]::UtcNow
    while (-not $backend.HasExited) {
        if ($tunnel.HasExited) { $failedPublicChecks = 3 }
        if ([DateTime]::UtcNow -ge $nextCheck) {
            try {
                $publicHealth = Invoke-RestMethod -Uri "$publicUrl/health" -TimeoutSec 10
                if ($publicHealth.ready -eq $true) { $failedPublicChecks = 0 } else { $failedPublicChecks++ }
            } catch { $failedPublicChecks++ }
            $nextCheck = [DateTime]::UtcNow.AddSeconds(20)
        }
        if ($failedPublicChecks -ge 3) {
            Write-Host 'Network/tunnel changed. Recreating the public HTTPS URL ...' -ForegroundColor Yellow
            if (-not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue }
            $tunnelInfo = Start-QuickTunnel
            $tunnel = $tunnelInfo.Process
            $publicUrl = $tunnelInfo.Url
            $failedPublicChecks = 0
            $nextCheck = [DateTime]::UtcNow.AddSeconds(20)
            Write-Host "New Public URL: $publicUrl" -ForegroundColor Cyan
        }
        Start-Sleep -Seconds 1
    }
} finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue }
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -ErrorAction SilentlyContinue }
    Write-Host 'Medflow Public Mode stopped cleanly.'
}
