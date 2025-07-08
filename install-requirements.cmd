@echo off
echo Installing AutoCI requirements for Windows...
echo.

REM Upgrade pip first
py -m pip install --upgrade pip

echo Installing core packages...
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing transformers and accelerate...
py -m pip install transformers accelerate bitsandbytes

echo Installing other requirements...
py -m pip install flask flask-cors psutil dataclasses asyncio pathlib
py -m pip install sentencepiece protobuf
py -m pip install rich termcolor colorama
py -m pip install numpy pandas matplotlib seaborn
py -m pip install requests beautifulsoup4
py -m pip install pyyaml toml

echo.
echo Installation complete!
pause