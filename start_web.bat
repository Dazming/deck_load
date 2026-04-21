@echo off
setlocal

set "ROOT=%~dp0"
set "FRONTEND_PORT=4173"
set "FRONTEND_URL=http://localhost:%FRONTEND_PORT%"
set "CONDA_EXE="

for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
  if not defined CONDA_EXE set "CONDA_EXE=%%I"
)
if not defined CONDA_EXE (
  for /f "delims=" %%I in ('where conda 2^>nul') do (
    if not defined CONDA_EXE set "CONDA_EXE=%%I"
  )
)
if not defined CONDA_EXE (
  echo [ERROR] Conda not found in PATH.
  echo Please open Anaconda Prompt or add conda to PATH first.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found in PATH.
  echo Please install Node.js with npm and reopen terminal.
  pause
  exit /b 1
)

echo [INFO] Starting backend (Flask)...
start "Deck Load Backend" cmd /k "cd /d ""%ROOT%"" && ""%CONDA_EXE%"" run -n test --no-capture-output python api_server.py"

echo [INFO] Starting frontend (Vite on port %FRONTEND_PORT%)...
start "Deck Load Frontend" cmd /k "cd /d ""%ROOT%frontend"" && npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%"

echo [INFO] Waiting for dev server startup...
ping 127.0.0.1 -n 6 >nul

echo [INFO] Opening browser at %FRONTEND_URL% ...
start "" "%FRONTEND_URL%"

echo [DONE] Backend: http://localhost:5000
echo [DONE] Frontend: %FRONTEND_URL%
endlocal
