@echo off
cd /d "%~dp0"
set "PYTHON=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\python.exe"
"%PYTHON%" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 pause
