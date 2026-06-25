@echo off
setlocal enabledelayedexpansion

REM Change current directory to the script location (optional)
cd /d "%~dp0"

REM Command to create venv
..\..\uv\uv.exe venv --python ..\..\python\3.10.11
if errorlevel 1 (
  echo Failed to create venv. Aborting.
  exit /b 1
)

REM Path to .venv\pyenv.cfg (convert UNIX-style path to Windows)
set "CFG_PATH=.venv\pyvenv.cfg"

if not exist "%CFG_PATH%" (
  echo %CFG_PATH% not found.
  exit /b 1
)

REM Temporary file
set "TMP=%TEMP%\pyenv_tmp_%RANDOM%.tmp"

REM Replace the first line and write to the temporary file
set "REPLACEMENT=home = ../../python/3.10.11"
set /a line=0

> "%TMP%" (
  for /f "usebackq delims=" %%L in ("%CFG_PATH%") do (
    set /a line+=1
    if !line! EQU 1 (
      echo %REPLACEMENT%
    ) else (
      echo(%%L
    )
  )
)

REM Replace the original file with the temporary file
move /y "%TMP%" "%CFG_PATH%" >nul
if errorlevel 1 (
  echo Failed to update file.
  del "%TMP%" 2>nul
  exit /b 1
)

echo Done: replaced the first line in %CFG_PATH%.
pause
