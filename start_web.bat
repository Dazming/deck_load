@echo off
setlocal

set "ROOT=%~dp0"
set "FRONTEND_PORT="
set "FRONTEND_URL="
set "BACKEND_URL="
set "CONDA_EXE="
set "BACKEND_HEALTH=http://127.0.0.1:5000/api/health"
set "OPEN_BROWSER=1"
set "BACKEND_LAUNCHER=%TEMP%\deck_load_backend.cmd"
set "FRONTEND_LAUNCHER=%TEMP%\deck_load_frontend.cmd"

for /f "delims=" %%I in ('where conda.exe 2^>nul') do (
  if not defined CONDA_EXE set "CONDA_EXE=%%I"
)
if not defined CONDA_EXE (
  for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
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

set "BACKEND_URL=http://127.0.0.1:5000"
call :write_backend_launcher

echo [INFO] Checking existing backend...
call :is_http_ok "%BACKEND_HEALTH%"
if errorlevel 1 (
  echo [INFO] Backend not ready, starting Flask...
  start "Deck Load Backend" cmd /k "%BACKEND_LAUNCHER%"
) else (
  echo [OK] Backend already running at %BACKEND_URL%
)

echo [INFO] Waiting backend health...
call :wait_http_ok "%BACKEND_HEALTH%" 45
if errorlevel 1 (
  echo [ERROR] Backend health check failed: %BACKEND_HEALTH%
  echo [ERROR] Please check "Deck Load Backend" terminal output.
  set "OPEN_BROWSER=0"
) else (
  echo [OK] Backend is ready.
)

echo [INFO] Checking existing frontend...
set "FRONTEND_PORT="
for %%P in (4173 5173 5174 5175) do (
  if not defined FRONTEND_PORT (
    netstat -ano | findstr /r /c:":%%P .*LISTENING" >nul
    if not errorlevel 1 (
      set "FRONTEND_PORT=%%P"
    )
  )
)
if defined FRONTEND_PORT goto :frontend_reuse

call :pick_frontend_port
if not defined FRONTEND_PORT (
  echo [ERROR] No available frontend port found in candidates: 4173, 5173, 5174, 5175.
  pause
  exit /b 1
)
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%"
call :write_frontend_launcher
echo [INFO] Starting frontend (Vite on port %FRONTEND_PORT%)...
start "Deck Load Frontend" cmd /k "%FRONTEND_LAUNCHER%"
goto :frontend_ready_check

:frontend_reuse
set "FRONTEND_URL=http://127.0.0.1:%FRONTEND_PORT%"
echo [OK] Frontend already running at %FRONTEND_URL%

:frontend_ready_check

echo [INFO] Waiting frontend URL...
call :wait_http_ok "%FRONTEND_URL%" 45
if errorlevel 1 (
  echo [ERROR] Frontend URL not reachable: %FRONTEND_URL%
  echo [ERROR] Check "Deck Load Frontend" terminal logs.
  set "OPEN_BROWSER=0"
) else (
  echo [OK] Frontend is ready.
)

if "%OPEN_BROWSER%"=="1" (
  echo [INFO] Opening browser at %FRONTEND_URL% ...
  start "" "%FRONTEND_URL%"
)

echo [DONE] Backend: http://localhost:5000
echo [DONE] Frontend: %FRONTEND_URL%
endlocal
exit /b 0

:pick_frontend_port
for %%P in (4173 5173 5174 5175) do (
  call :is_port_in_use %%P
  if errorlevel 1 (
    set "FRONTEND_PORT=%%P"
    goto :eof
  )
)
goto :eof

:is_port_in_use
set "PORT=%~1"
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if errorlevel 1 (
  rem not in use
  exit /b 1
)
rem in use
exit /b 0

:is_http_ok
set "URL=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%'; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 400){ exit 0 } else { exit 1 } } catch { exit 1 }"
exit /b %errorlevel%

:wait_http_ok
set "URL=%~1"
set "MAX_TRIES=%~2"
set /a TRY=0
:wait_loop
set /a TRY+=1
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r=Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%'; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 400){ exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 exit /b 0
if %TRY% GEQ %MAX_TRIES% exit /b 1
ping 127.0.0.1 -n 2 >nul
goto :wait_loop

:write_backend_launcher
> "%BACKEND_LAUNCHER%" echo @echo off
>> "%BACKEND_LAUNCHER%" echo cd /d "%ROOT%"
>> "%BACKEND_LAUNCHER%" echo "%CONDA_EXE%" run -n test --no-capture-output python api_server.py
goto :eof

:write_frontend_launcher
> "%FRONTEND_LAUNCHER%" echo @echo off
>> "%FRONTEND_LAUNCHER%" echo cd /d "%ROOT%frontend"
>> "%FRONTEND_LAUNCHER%" echo npm run dev -- --host 0.0.0.0 --port %FRONTEND_PORT%
goto :eof
