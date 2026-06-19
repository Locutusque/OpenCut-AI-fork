@echo off
setlocal
title OpenCut AI Installer

REM ===========================================================================
REM OpenCut AI - Windows launcher (.cmd)
REM
REM Double-click this file, or run it from cmd / PowerShell / Terminal.
REM
REM Why this exists: install.ps1 is a PowerShell script, and Windows blocks
REM running .ps1 files by default ("running scripts is disabled on this system").
REM This .cmd is a batch file, which is NOT subject to that policy. It simply
REM launches the PowerShell installer with -ExecutionPolicy Bypass for this one
REM process, so nothing about your system's policy is changed.
REM
REM Any arguments you pass are forwarded to install.ps1, e.g.:
REM   install.cmd -Rocm
REM   install.cmd -Nvidia -Model llama3.2:3b
REM   install.cmd -Cpu
REM ===========================================================================

set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%install.ps1"

if exist "%PS1%" goto run

REM Standalone use (e.g. you downloaded only install.cmd): fetch install.ps1
REM from the repository into the temp folder. Override the source with
REM   set OPENCUT_RAW_BASE=https://raw.githubusercontent.com/<owner>/<repo>/<branch>
if "%OPENCUT_RAW_BASE%"=="" set "OPENCUT_RAW_BASE=https://raw.githubusercontent.com/Ekaanth/OpenCut-AI/main"
set "PS1=%TEMP%\opencut-install.ps1"
echo [install] install.ps1 not found next to this file.
echo [install] Downloading it from %OPENCUT_RAW_BASE%/scripts/install.ps1 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing -Uri '%OPENCUT_RAW_BASE%/scripts/install.ps1' -OutFile '%PS1%' } catch { Write-Host ('[install] Download failed: ' + $_.Exception.Message) -ForegroundColor Red; exit 1 }"
if errorlevel 1 goto fail

:run
echo [install] Launching the installer (PowerShell execution policy bypassed for this run only)...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
if errorlevel 1 goto fail

echo.
echo [install] Finished.
goto end

:fail
echo.
echo [install] The installer reported an error - see the messages above.

:end
echo.
echo Press any key to close this window.
pause >nul
endlocal
