@echo off
title Medflow Public HTTPS Tunnel Launcher (Port 7860)
cd /d "%~dp0"
echo =================================================================
echo   Medflow Medical RAG - Public Tunnel Launcher
echo =================================================================
powershell -ExecutionPolicy Bypass -File "%~dp0start_medflow_public.ps1"
pause
