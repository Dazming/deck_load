@echo off
setlocal

set "ROOT=%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda not found in PATH.
  echo Please open Anaconda Prompt or add conda to PATH first.
  pause
  exit /b 1
)

echo [INFO] Starting backend (Flask)...
start "Deck Load Backend" cmd /k "cd /d ""%ROOT%"" && conda run -n test --no-capture-output python api_server.py"

echo [INFO] Starting frontend (Vite)...
start "Deck Load Frontend" cmd /k "cd /d ""%ROOT%frontend"" && npm run dev"

echo [INFO] Waiting for dev server startup...
timeout /t 5 >nul

echo [INFO] Opening browser...
start "" "http://localhost:5173"

echo [DONE] If 5173 is occupied, check frontend terminal for actual port (e.g. 5174/5175).
endlocal
