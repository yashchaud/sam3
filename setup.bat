@echo off
REM SAM3 API - Quick Setup Script for Windows
REM This script sets up the SAM3 API server on your local machine

echo ==================================
echo SAM3 API - Quick Setup (Windows)
echo ==================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 is not installed
    echo Please install Python 3.12+ from https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION% found
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded
echo.

REM Install API dependencies
echo Installing API dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)
echo [OK] API dependencies installed
echo.

REM Check for CUDA
echo Checking for CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] CUDA not detected
    echo Installing PyTorch (CPU version)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    echo [WARNING] PyTorch (CPU) installed - inference will be slower
) else (
    echo [OK] CUDA detected
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    echo [OK] PyTorch with CUDA installed
)
echo.

REM Check if SAM3 is installed
echo Checking for SAM3...
python -c "import sam3" >nul 2>&1
if errorlevel 1 (
    echo SAM3 not found. Installing...
    echo.
    echo SAM3 requires access to the model on Hugging Face.
    echo Please ensure you have:
    echo   1. Requested access at: https://huggingface.co/facebook/sam3
    echo   2. Created a Hugging Face token at: https://huggingface.co/settings/tokens
    echo.

    set /p CONTINUE="Have you completed these steps? (y/n): "
    if /i "%CONTINUE%"=="y" (
        REM Install huggingface_hub
        echo Installing huggingface_hub...
        pip install huggingface_hub

        REM Login to Hugging Face
        echo.
        echo Please login to Hugging Face:
        python -m huggingface_hub.commands.huggingface_cli login

        REM Clone and install SAM3
        if not exist "sam3_repo" (
            echo.
            echo Cloning SAM3 repository...
            git clone https://github.com/facebookresearch/sam3.git sam3_repo
        )

        echo Installing SAM3...
        cd sam3_repo
        pip install -e .
        cd ..
        echo [OK] SAM3 installed
    ) else (
        echo [WARNING] Skipping SAM3 installation
        echo You can install it later by running:
        echo   git clone https://github.com/facebookresearch/sam3.git sam3_repo
        echo   cd sam3_repo ^&^& pip install -e . ^&^& cd ..
    )
) else (
    echo [OK] SAM3 already installed
)
echo.

REM Create output directory
if not exist "output" mkdir output

echo.
echo ==================================
echo [OK] Setup Complete!
echo ==================================
echo.
echo To start the server, run:
echo   venv\Scripts\activate.bat
echo   python app.py
echo.
echo Or use the quick start script:
echo   start_server.bat
echo.
echo API will be available at: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo.

pause
