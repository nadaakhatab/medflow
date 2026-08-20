@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ================================================
echo MedFlow - Final Agenda Compliance Patch
echo ================================================
echo.
set "TARGET=C:\medflow20"
if not exist "%TARGET%\day4" (
  echo ERROR: %TARGET% was not found or is not the MedFlow project.
  pause
  exit /b 1
)

for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set "D=%%d%%b%%c"
for /f "tokens=1-3 delims=:,. " %%a in ("%time%") do set "T=%%a%%b%%c"
set "BACKUP=%TARGET%\_backup_before_agenda_patch_%D%_%T%"
mkdir "%BACKUP%" >nul 2>&1
mkdir "%BACKUP%\day1" >nul 2>&1
mkdir "%BACKUP%\day2" >nul 2>&1
mkdir "%BACKUP%\day3" >nul 2>&1
mkdir "%BACKUP%\day4" >nul 2>&1

for %%F in (README.md verify_project.py AGENDA_COMPLIANCE.md) do if exist "%TARGET%\%%F" copy /Y "%TARGET%\%%F" "%BACKUP%\" >nul
for %%F in (README.md) do if exist "%TARGET%\day1\%%F" copy /Y "%TARGET%\day1\%%F" "%BACKUP%\day1\" >nul
for %%F in (README.md agenda_retrieval_summary.py) do if exist "%TARGET%\day2\%%F" copy /Y "%TARGET%\day2\%%F" "%BACKUP%\day2\" >nul
for %%F in (README.md) do if exist "%TARGET%\day3\%%F" copy /Y "%TARGET%\day3\%%F" "%BACKUP%\day3\" >nul
for %%F in (README.md risk_classifier.py day4_pipeline.py test_day4_compliance.py) do if exist "%TARGET%\day4\%%F" copy /Y "%TARGET%\day4\%%F" "%BACKUP%\day4\" >nul

echo [1/5] Backup created:
echo %BACKUP%
echo.

echo [2/5] Applying root files...
copy /Y "README.md" "%TARGET%\README.md" >nul
copy /Y "AGENDA_COMPLIANCE.md" "%TARGET%\AGENDA_COMPLIANCE.md" >nul
copy /Y "verify_project.py" "%TARGET%\verify_project.py" >nul

echo [3/5] Applying Day 1-3 documentation and retrieval scorecard...
copy /Y "day1\README.md" "%TARGET%\day1\README.md" >nul
copy /Y "day2\README.md" "%TARGET%\day2\README.md" >nul
copy /Y "day2\agenda_retrieval_summary.py" "%TARGET%\day2\agenda_retrieval_summary.py" >nul
copy /Y "day3\README.md" "%TARGET%\day3\README.md" >nul

echo [4/5] Applying Day 4 agenda alignment...
copy /Y "day4\README.md" "%TARGET%\day4\README.md" >nul
copy /Y "day4\risk_classifier.py" "%TARGET%\day4\risk_classifier.py" >nul
copy /Y "day4\day4_pipeline.py" "%TARGET%\day4\day4_pipeline.py" >nul
copy /Y "day4\test_day4_compliance.py" "%TARGET%\day4\test_day4_compliance.py" >nul

echo [5/5] Done.
echo.
echo No vector database, PDF, .env, or Day 1-3 implementation code was changed.
echo Frozen index settings remain: 1470 chunks, 200 tokens, 0 overlap, Top-K 4.
echo.
echo Next commands:
echo   cd /d C:\medflow20
echo   .venv\Scripts\activate
echo   python -m unittest discover -s day4 -p "test_*.py" -v

echo   python verify_project.py

echo   python day2\agenda_retrieval_summary.py

echo.
pause
