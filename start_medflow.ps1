# ==============================================================================
# Medflow Medical RAG Engine - Single-Port Production Launcher (Port 7860)
# ==============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location -Path $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
} else {
    $PythonExe = "python"
}

& "$PythonExe" "$ProjectRoot\run_app.py"
