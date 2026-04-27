@REM @echo off

@REM REM Obtain Root Path
@REM set "BASE_DIR=%~dp0..\"
@REM D:\AI_Data\Computer\Agent-WQB\wqbagentvenv\Scripts\python.exe "%BASE_DIR%wqbagent_v2.py"
@REM pause

@echo off
setlocal enabledelayedexpansion

:: --- NEW: Force Windows terminal to use UTF-8 ---
chcp 65001 >nul

:: 1. Get robust date and time formatting using PowerShell
for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-Date -Format 'yyyyMM'"`) do set "DIR_DATE=%%i"
for /f "usebackq tokens=*" %%i in (`powershell -Command "Get-Date -Format 'yyyyMMdd-HHmmss'"`) do set "FILE_DATE=%%i"

:: 2. Define Directories 
set "PARENT_DIR=%~dp0.."
set "LOG_DIR=%PARENT_DIR%\logs\%DIR_DATE%"

:: 3. Create the log directory if it does not exist
if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
    echo [INFO] Created new log directory: %LOG_DIR%
)

set "OUTPUT_FILE=%LOG_DIR%\wqb_agent-%FILE_DATE%.html"
set "TRANSCRIPT=%LOG_DIR%\wqb_agent-%FILE_DATE%.transcript.txt"

echo ========================================================
echo 🚀 Starting WorldQuant Brain Agent Pipeline...
echo 📂 Output will be saved to: 
echo    %OUTPUT_FILE%
echo ========================================================

:: 4. Move into the parent directory
cd /d "%PARENT_DIR%"

:: 5. ACTIVATE YOUR VIRTUAL ENVIRONMENT
echo [INFO] Activating virtual environment...
call D:\AI_Data\Computer\Agent-WQB\wqbagentvenv\Scripts\activate.bat

:: 6. Force colors on and unbuffered output
set FORCE_COLOR=1
set TERM=xterm-256color
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

:: Run Python, pipe to PowerShell to "Tee" (split) the stream.
:: It will print to the console AND pass it forward to ansi2html.
echo [INFO] Running Agent Pipeline...
:: === TO THIS (just change "powershell" → "pwsh") ===
pwsh -Command "$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8; python -u \"%PARENT_DIR%\wqbagent_v2.py\" | ForEach-Object { $_ | Out-Host; $_ } | ansi2html > \"%OUTPUT_FILE%\""

:: 7. Deactivate the environment when done
call deactivate

echo.
echo ✅ Process Complete! You can view the log in your browser.
pause