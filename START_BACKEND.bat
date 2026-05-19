@echo off
REM Backend Server - Adversarial Detection API
REM Double-click to run

cd /d "%~dp0"
echo ============================================================
echo Starting Backend API Server
echo ============================================================
echo Server will run on: http://127.0.0.1:9000
echo Keep this window open while using dashboard
echo ============================================================
python START_BACKEND.py
pause