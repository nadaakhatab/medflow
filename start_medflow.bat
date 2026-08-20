@echo off
title Medflow Medical RAG Engine (Port 7860)
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0run_app.py"
) else (
    python "%~dp0run_app.py"
)
pause
