@echo off
title Medflow Medical RAG Engine (Port 7860)
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_medflow.ps1"
pause
