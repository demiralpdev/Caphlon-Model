@echo off
setlocal
cd /d "%~dp0"

if "%CTX%"=="" set CTX=16384
if "%NGL%"=="" set NGL=99
if "%PORT%"=="" set PORT=8080

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
    echo ERROR: Download incomplete or failed. Double-click START-CAPHLON2.bat again to resume.
    pause
    exit /b 1
  )
)

echo.
echo ============================================================
echo Starting Caphlon-2 on NVIDIA GPU...
echo Chat UI will open automatically at: http://127.0.0.1:%PORT%
echo OpenAI-compatible API endpoint:    http://127.0.0.1:%PORT%/v1
echo Keep this window open while chatting.
echo ============================================================
echo.

start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{$r=Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $env:PORT + '/health') -TimeoutSec 2; if($r.StatusCode -eq 200){Start-Process ('http://127.0.0.1:' + $env:PORT); break}}catch{}; Start-Sleep -Seconds 1}"

"%~dp0bin\llama-server.exe" ^
  -m "%MODEL%" ^
  -ngl %NGL% ^
  -c %CTX% ^
  -np 1 ^
  -b 2048 -ub 512 ^
  -fa on ^
  -ctk q8_0 -ctv q8_0 ^
  --cache-ram 0 ^
  --ctx-checkpoints 32 ^
  -cms 128 ^
  --no-context-shift ^
  --jinja ^
  --reasoning-format deepseek ^
  --logit-bias 13275-1.2 --logit-bias 13624-1.2 --logit-bias 3633-1.2 --logit-bias 11041-1.2 --logit-bias 62490-1.2 --logit-bias 58941-1.2 --logit-bias 62200-1.2 ^
  --logit-bias 36496-1 --logit-bias 64160-1 --logit-bias 53703-1 --logit-bias 56978-1 --logit-bias 67148-1 --logit-bias 65560-1 ^
  --logit-bias 32069-0.8 --logit-bias 46247-0.8 --logit-bias 3384-0.8 --logit-bias 55810-0.8 --logit-bias 17952-0.8 --logit-bias 8027-0.8 --logit-bias 29922-0.8 --logit-bias 52275-0.8 --logit-bias 10278-0.8 --logit-bias 6909-0.8 --logit-bias 20432-0.8 ^
  --host 127.0.0.1 ^
  --port %PORT%

if errorlevel 1 (
  echo.
  echo Server exited with an error.
  echo If out of GPU memory on a 6GB card, edit START-CAPHLON2.bat and set CTX=8192 or NGL=45.
  pause
  exit /b 1
)
