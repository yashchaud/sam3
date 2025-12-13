@echo off
REM SAM3 API - Server Starter Script for Windows
REM This script activates the virtual environment and starts the API server

echo ==================================
echo SAM3 API - Starting Server
echo ==================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found
    echo Please run setup.bat first to set up the environment
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Check if SAM3 is installed
echo Checking SAM3 installation...
python -c "import sam3" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] SAM3 import issue detected
    echo Attempting to fix numpy compatibility...
    pip install "numpy>=1.26.0,<2.0" --force-reinstall --no-deps >nul 2>&1

    REM Check again
    python -c "import sam3" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] SAM3 is not installed properly
        echo Error details:
        python -c "import sam3"
        exit /b 1
    )
)
echo [OK] SAM3 is ready
echo.

REM Start the server
echo ==================================
echo [OK] Starting SAM3 API Server
echo ==================================
echo.
echo API will be available at: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
