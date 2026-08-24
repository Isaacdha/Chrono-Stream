@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Chrono Stream's local environment is not set up yet.
    echo Follow the installation steps in README.md first.
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run chrono_app.py
