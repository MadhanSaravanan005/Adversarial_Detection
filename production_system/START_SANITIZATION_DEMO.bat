@echo off
REM Sanitization Feature Demo - Standalone
REM Shows how to recover adversarial images using defense techniques
REM Double-click to run

cd /d "%~dp0"
echo ============================================================
echo ADVERSARIAL IMAGE SANITIZATION DEMO
echo ============================================================
echo This feature works INDEPENDENTLY:
echo - Detects adversarial images
echo - Tries 6+ defense techniques
echo - Recovers attacked images or blocks them safely
echo ============================================================
echo.
python INTERACTIVE_SANITIZATION_DEMO.py
pause
