"""Configuration management for anomaly detection."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Default class mappings
DEFAULT_STRUCTURE_CLASSES = {
    "beam", "column", "wall", "slab", "pipe",
    "foundation", "joint", "girder", "truss", "deck",
}

DEFAULT_ANOMALY_CLASSES = {
    "crack", "corrosion", "spalling", "deformation", "stain",
    "efflorescence", "exposed_rebar", "delamination", "scaling",
    "popout", "honeycomb", "rust",
}

# Performance constants
MAX_IMAGE_SIZE = 2048
MIN_MASK_AREA = 10
DEFAULT_BATCH_SIZE = 4


def _get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass
class EnvironmentConfig:
    """Configuration loaded from environment variables."""
    # Model paths
    detector_model_path: Path | None = None
    segmenter_model_path: Path | None = None

    # Device
    device: str = "auto"

    # Detection settings
    confidence_threshold: float = 0.3

    # Output
    mask_output_dir: Path | None = None

    # VLM settings
    enable_vlm_judge: bool = False
    vlm_provider: str = "local"
    qwen_model_path: str | None = None
    openrouter_api_key: str | None = None
    openrouter_model: str = "qwen/qwen-2.5-vl-72b-instruct"
    vlm_every_n_frames: int = 10
    vlm_max_generation_frames: int = 60

    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # Video processing
    target_fps: float = 30.0
    frame_buffer_size: int = 120
    enable_tiling: bool = True

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """Load configuration from environment variables."""
        # Try to load .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        detector_path = os.environ.get("ANOMALY_DETECTOR_MODEL")
        segmenter_path = os.environ.get("ANOMALY_SEGMENTER_MODEL")
        mask_dir = os.environ.get("ANOMALY_MASK_OUTPUT_DIR")
        qwen_path = os.environ.get("QWEN_MODEL_PATH")

        return cls(
            # Model paths
            detector_model_path=Path(detector_path) if detector_path else None,
            segmenter_model_path=Path(segmenter_path) if segmenter_path else None,

            # Device
            device=os.environ.get("ANOMALY_DEVICE", "auto"),

            # Detection
            confidence_threshold=_get_float_env("ANOMALY_CONFIDENCE_THRESHOLD", 0.3),

            # Output
            mask_output_dir=Path(mask_dir) if mask_dir else None,

            # VLM
            enable_vlm_judge=_get_bool_env("ENABLE_VLM_JUDGE", False),
            vlm_provider=os.environ.get("VLM_PROVIDER", "local"),
            qwen_model_path=qwen_path,
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "qwen/qwen-2.5-vl-72b-instruct"),
            vlm_every_n_frames=_get_int_env("VLM_EVERY_N_FRAMES", 10),
            vlm_max_generation_frames=_get_int_env("VLM_MAX_GENERATION_FRAMES", 60),

            # Web server
            web_host=os.environ.get("WEB_HOST", "0.0.0.0"),
            web_port=_get_int_env("WEB_PORT", 8000),

            # Video processing
            target_fps=_get_float_env("TARGET_FPS", 30.0),
            frame_buffer_size=_get_int_env("FRAME_BUFFER_SIZE", 120),
            enable_tiling=_get_bool_env("ENABLE_TILING", True),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.detector_model_path and not self.detector_model_path.exists():
            errors.append(f"Detector model not found: {self.detector_model_path}")

        if self.segmenter_model_path and not self.segmenter_model_path.exists():
            errors.append(f"Segmenter model not found: {self.segmenter_model_path}")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"Invalid confidence threshold: {self.confidence_threshold}")

        if self.enable_vlm_judge and self.vlm_provider == "openrouter":
            if not self.openrouter_api_key:
                errors.append("OpenRouter API key required when using openrouter provider")

        return errors

    def print_config(self) -> None:
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("Anomaly Detection Configuration")
        print("=" * 60)
        print(f"  Segmenter Model:  {self.segmenter_model_path or 'Not set'}")
        print(f"  Detector Model:   {self.detector_model_path or 'Pretrained'}")
        print(f"  Device:           {self.device}")
        print(f"  Confidence:       {self.confidence_threshold}")
        print(f"  Enable Tiling:    {self.enable_tiling}")
        print()
        print("VLM Configuration:")
        print(f"  Enabled:          {self.enable_vlm_judge}")
        if self.enable_vlm_judge:
            print(f"  Provider:         {self.vlm_provider}")
            print(f"  Every N Frames:   {self.vlm_every_n_frames}")
            print(f"  Max Wait Frames:  {self.vlm_max_generation_frames}")
        print()
        print("Web Server:")
        print(f"  Host:             {self.web_host}")
        print(f"  Port:             {self.web_port}")
        print("=" * 60 + "\n")


def get_device(preference: str = "auto") -> str:
    """Determine device to use based on preference and availability."""
    if preference == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return preference


# Global config instance (lazy loaded)
_config: EnvironmentConfig | None = None


def get_config() -> EnvironmentConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = EnvironmentConfig.from_env()
    return _config


def reload_config() -> EnvironmentConfig:
    """Reload configuration from environment."""
    global _config
    _config = EnvironmentConfig.from_env()
    return _config
