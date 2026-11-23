@echo off
setlocal
cd /d "%~dp0"

if "%CTX%"=="" set CTX=16384
if "%NGL%"=="" set NGL=99

set "MODEL=%~dp0Caphlon-2-27B-PTQ1_SB.gguf"
set "MODEL_URL=https://huggingface.co/ssuasd/Caphlon-2/resolve/main/Caphlon-2-27B-PTQ1_SB.gguf"
set "MODEL_BYTES=4856213824"

powershell -NoProfile -Command "if ((Test-Path -LiteralPath $env:MODEL) -and ((Get-Item -LiteralPath $env:MODEL).Length -eq [int64]$env:MODEL_BYTES)) { exit 0 } else { exit 1 }"
if errorlevel 1 (
  echo Downloading Caphlon-2 ^(4.52 GiB^) from Hugging Face...
  echo This only happens once. Download is resumable if interrupted.
  echo.
  curl.exe -L -C - --fail --retry 5 -o "%MODEL%" "%MODEL_URL%"
  powershell -NoProfile -Command "if ((Test-Path -LiteralPath $env:MODEL) -and ((Get-Item -LiteralPath $env:MODEL).Length -eq [int64]$env:MODEL_BYTES)) { exit 0 } else { exit 1 }"
  if errorlevel 1 (
    echo.
    echo ERROR: Download incomplete or failed. Double-click CHAT-TERMINAL.bat again to resume.
    pause
    exit /b 1
  )
)

"%~dp0bin\llama-cli.exe" -m "%MODEL%" -ngl %NGL% -c %CTX% -fa on -ctk q8_0 -ctv q8_0 --jinja
if errorlevel 1 (
  echo.
  echo llama-cli exited with an error.
  pause
  exit /b 1
)
