# ==============================================================================
# Medflow Medical RAG Engine - Single-Port Production Launcher (Port 7860)
# ==============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location -Path $ProjectRoot

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Medflow Medical RAG Assistant (Medflow20 Core Engine)" -ForegroundColor Cyan
Write-Host "  Local Host URL: http://127.0.0.1:7860" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Resolve Python executable
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "[Launcher] Using virtual environment Python: $PythonExe" -ForegroundColor Green
} else {
    $PythonExe = "python"
    Write-Host "[Launcher] Virtual environment not found. Using system Python." -ForegroundColor Yellow
}

# 2. Check if port 7860 is occupied
$Port7860Occupied = $false
try {
    $Connection = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue
    if ($Connection) {
        $Port7860Occupied = $true
    }
} catch {
    # Fallback
}

if ($Port7860Occupied) {
    Write-Host "[Launcher] Port 7860 is currently occupied. Checking system health..." -ForegroundColor Yellow
    try {
        $HealthResp = Invoke-RestMethod -Uri "http://127.0.0.1:7860/health" -TimeoutSec 2 -ErrorAction Stop
        if ($HealthResp.status -eq "healthy" -or $HealthResp.ready -eq $true) {
            Write-Host "[Launcher] Existing healthy Medflow process detected on port 7860!" -ForegroundColor Green
            Write-Host "[Launcher] Opening Medflow Web Application..." -ForegroundColor Cyan
            Start-Process "http://127.0.0.1:7860"
            Exit 0
        }
    } catch {
        Write-Host "[WARNING] Port 7860 is occupied by a non-responsive process." -ForegroundColor Red
        Write-Host "Please close the process using port 7860 and re-run start_medflow.bat." -ForegroundColor Red
        Exit 1
    }
}

# 3. Environment configuration
$env:PYTHONPATH = "$ProjectRoot;$ProjectRoot\medflow20;" + $env:PYTHONPATH
$env:PORT = "7860"
# Local HTTP cannot store Secure cookies. Public Mode uses its dedicated
# launcher, which sets production/HTTPS values before starting FastAPI.
$env:APP_ENV = "local"
$env:SECURE_COOKIE = "false"
$env:DEV_AUTH_BYPASS = "false"

Write-Host "
[1/2] Initializing Medflow20 Core RAG Engine & FastAPI on http://127.0.0.1:7860..." -ForegroundColor Cyan

# Launch FastAPI Uvicorn process
$BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 7860" -WorkingDirectory $ProjectRoot -PassThru -NoNewWindow

# 4. Wait for readiness probe
Write-Host "[2/2] Waiting for Medflow20 vector store & embeddings to load..." -ForegroundColor Cyan
$MaxRetries = 180
$BackendReady = $false

for ($i = 1; $i -le $MaxRetries; $i++) {
    if ($BackendProcess.HasExited) {
        Write-Host "[ERROR] FastAPI backend process exited unexpectedly with code $($BackendProcess.ExitCode)." -ForegroundColor Red
        Exit 1
    }
    
    try {
        $Resp = Invoke-RestMethod -Uri "http://127.0.0.1:7860/health" -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($Resp.status -eq "healthy" -or $Resp.ready -eq $true) {
            $BackendReady = $true
            break
        }
    } catch {
        # Keep waiting
    }
    
    Start-Sleep -Seconds 1
}

if (-not $BackendReady) {
    Write-Host "[ERROR] Medflow20 backend initialization timed out." -ForegroundColor Red
    Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    Exit 1
}

Write-Host "
=================================================================" -ForegroundColor Green
Write-Host "  SUCCESS! Medflow20 Core Engine & Web Interface Ready!" -ForegroundColor Green
Write-Host "  Application URL: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

# Open browser automatically
Start-Process "http://127.0.0.1:7860"

# Keep launcher process active
try {
    $BackendProcess.WaitForExit()
} catch {
    Write-Host "Stopping Medflow..."
}
