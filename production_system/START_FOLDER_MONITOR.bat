@echo off
REM Production System - Folder Monitor Startup
REM Watches for images in monitored_folders/input

cd /d %~dp0
python folder_monitor.py
pause
