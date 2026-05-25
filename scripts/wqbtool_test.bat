@echo off
setlocal enabledelayedexpansion

:: 1. Force UTF-8 to prevent the crashing
chcp 65001 >nul
set PYTHONUTF8=1

:: 2. Define Relative Paths
set "PARENT_DIR=%~dp0.."
set "PYTHON_EXE=%~dp0..\..\wqbagentportablevenv312\python.exe"

cd /d "%PARENT_DIR%"

echo ========================================================
echo 🚀 Starting WorldQuant Brain Agent Pipeline...
echo ========================================================

:: 3. Run the script normally without ANY redirection or pipes
"%PYTHON_EXE%" -u "wqbquant_searchtool_test.py"

echo.
echo ✅ Process Complete! 
pause