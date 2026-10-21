@echo off
cd /d "%~dp0"
"C:\Users\ahmet\anaconda3\python.exe" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 pause
