@echo off
echo ============================================
echo  SmartDoc Search - Starting Backend Server
echo ============================================
echo.

REM Use the full path to the Python interpreter where packages are installed
set PYTHON="C:\Program Files\Python312\python.exe"

echo Checking Python interpreter...
%PYTHON% -c "import flask; print('[OK] Flask', flask.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Flask not found for this Python. Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt
)

echo.
echo Starting Flask server on http://127.0.0.1:5000 ...
echo Press Ctrl+C to stop.
echo.
%PYTHON% server.py
