"""Setup script for anomaly detection package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="anomaly-detection",
    version="0.2.0",
    description="Real-time structural anomaly detection with VLM guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/anomaly-detection",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0,<2",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "torch>=2.7.0",
        "torchvision>=0.20.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "detection": ["rfdetr>=1.2.0"],
        "vlm": [
            "transformers>=4.40.0",
            "accelerate>=0.25.0",
        ],
        "api": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "python-multipart>=0.0.6",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "all": [
            "rfdetr>=1.2.0",
            "transformers>=4.40.0",
            "accelerate>=0.25.0",
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "python-multipart>=0.0.6",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "anomaly-detect=examples.run_single_image:main",
            "anomaly-video=examples.run_realtime_video:main",
        ],
    },
)
