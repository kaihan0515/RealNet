@echo off
REM Train RealNet on Windows (single GPU, gloo backend).
REM Usage: train_realnet_windows.bat [class_name]
REM Default class: bottle. Config: experiments\MVTec-AD\realnet_11g.yaml (batch_size=8 for 11GB GPUs)

set CLASS=%1
if "%CLASS%"=="" set CLASS=bottle

set PYTHON=C:\Users\Tien\anaconda3\envs\MVA_py310_cu121\python.exe

cd /d "%~dp0"
"%PYTHON%" train_realnet.py --dataset MVTec-AD --class_name %CLASS% --config experiments/{}/realnet_11g.yaml
