@echo off
REM Production System - REST API Startup
REM Exposes detection via REST API on port 5001

cd /d %~dp0
python api_production.py
pause
