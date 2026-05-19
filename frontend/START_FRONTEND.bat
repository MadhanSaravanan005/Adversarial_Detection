@echo off
REM Frontend Dashboard - Streamlit
REM Double-click to run

cd /d "%~dp0"
echo ============================================================
echo Starting Streamlit Dashboard
echo ============================================================
echo Dashboard will open in browser at: http://localhost:8501
echo Keep this window open while using dashboard
echo ============================================================
python -m streamlit run dashboard.py --server.port=8501
pause
