@echo off
REM RealNet 訓練儀表板：瀏覽器會自動開啟 http://127.0.0.1:8123
cd /d "%~dp0"
C:\Users\Tien\anaconda3\envs\MVA_py310_cu121\python.exe train_dashboard.py
pause
