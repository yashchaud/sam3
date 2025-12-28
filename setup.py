"""Setup script for anomaly_detection package."""

from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1.0",
    description="Phase-1 Visual Anomaly Detection using RF-DETR + SAM3",
    author="Your Name",
    python_requires=">=3.11",
    packages=find_packages(exclude=["tests", "examples", "scripts"]),
    install_requires=[
        "numpy>=1.26.0,<2",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "torch>=2.7.0",
        "torchvision>=0.20.0",
        "supervision>=0.21.0",
        "rfdetr>=1.2.0",
        # SAM3 core dependencies
        "timm>=1.0.17",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "typing_extensions",
        "huggingface_hub",
        "decord",
        "einops",
    ],
    extras_require={
        "api": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "python-multipart>=0.0.6",
            "pydantic>=2.0.0",
        ],
        "serverless": [
            "runpod>=1.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
)
