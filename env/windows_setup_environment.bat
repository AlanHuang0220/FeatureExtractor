@echo off
REM 創建 conda 環境
conda env create -f environment.yml

REM activate 此環境
call conda activate alan

REM 使用特定的 index URL 安裝 pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 環境建置完成

pause
