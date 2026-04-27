# wqb_agent_run.ps1   ← FINAL FIXED VERSION
$ScriptDir = $PSScriptRoot
$PARENT_DIR = Join-Path $ScriptDir ".."

# === FORCE UTF-8 EVERYWHERE (this fixes the 鉁? garbage) ===
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:FORCE_COLOR = "1"
$env:TERM = "xterm-256color"
$PSStyle.OutputRendering = "Ansi"

$DIR_DATE = Get-Date -Format "yyyyMM"
$LOG_DIR = Join-Path $PARENT_DIR "logs\$DIR_DATE"

if (-not (Test-Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
    Write-Host "[INFO] Created new log directory: $LOG_DIR" -ForegroundColor Yellow
}

$FILE_DATE = Get-Date -Format "yyyyMMdd-HHmmss"
$OUTPUT_FILE = Join-Path $LOG_DIR "wqb_agent-$FILE_DATE.html"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "🚀 Starting WorldQuant Brain Agent Pipeline..." -ForegroundColor Cyan
Write-Host "📂 Output will be saved to:" -ForegroundColor Cyan
Write-Host "   $OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Activate virtual environment
$venvActivate = Join-Path $PARENT_DIR "wqbagentvenv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "[INFO] Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "[ERROR] Venv activate script not found!" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[INFO] Running Agent Pipeline..." -ForegroundColor Green

$ansiLog = $OUTPUT_FILE -replace '\.html$', '.ansi'
$pythonScript = Join-Path $PARENT_DIR "wqbagent_test.py"

# LIVE colored output + perfect raw capture
python -u $pythonScript | ForEach-Object {
    $line = $_ + "`n"
    Write-Host -NoNewline $line          # ← Full colors + correct emojis
    $line | Out-File -FilePath $ansiLog -Encoding utf8 -Append
}

# Convert to colored HTML (light background)
Get-Content -Path $ansiLog -Raw -Encoding utf8 | ansi2html > $OUTPUT_FILE

Remove-Item $ansiLog -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "✅ Process Complete! Colored HTML saved → $OUTPUT_FILE" -ForegroundColor Green
Write-Host "Open the HTML file in your browser." -ForegroundColor Green
pause