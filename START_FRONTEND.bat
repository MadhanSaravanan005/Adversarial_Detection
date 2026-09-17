@echo off
REM Frontend Dashboard - Streamlit Launcher
REM Double-click or run from terminal

cd /d "%~dp0"
echo ============================================================
echo Starting Streamlit Dashboard
echo ============================================================
echo Dashboard URL: http://localhost:8501
echo Keep this window open while using the dashboard
echo ============================================================
python -m streamlit run frontend\dashboard.py --server.port=8501
pause
