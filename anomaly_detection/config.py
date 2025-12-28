"""Configuration management for anomaly detection.

Pipeline: VLM (OpenRouter) -> SAM3 Segmentation
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Default anomaly classes that VLM can detect
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
    """Configuration loaded from environment variables.

    Pipeline: VLM (OpenRouter) -> SAM3 Segmentation
    """
    # HuggingFace token (optional)
    hf_token: str | None = None

    # Device
    device: str = "auto"

    # Detection settings
    confidence_threshold: float = 0.3

    # Output
    mask_output_dir: Path | None = None

    # OpenRouter VLM settings (always enabled, primary detection method)
    openrouter_api_key: str | None = None
    openrouter_model: str = "bytedance-seed/seed-1.6-flash"
    vlm_every_n_frames: int = 10
    vlm_max_generation_frames: int = 60

    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # Video processing
    target_fps: float = 30.0
    frame_buffer_size: int = 120

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """Load configuration from environment variables."""
        # Try to load .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        mask_dir = os.environ.get("ANOMALY_MASK_OUTPUT_DIR")

        return cls(
            # HuggingFace token
            hf_token=os.environ.get("HF_TOKEN"),

            # Device
            device=os.environ.get("ANOMALY_DEVICE", "auto"),

            # Detection
            confidence_threshold=_get_float_env("ANOMALY_CONFIDENCE_THRESHOLD", 0.3),

            # Output
            mask_output_dir=Path(mask_dir) if mask_dir else None,

            # OpenRouter VLM (primary detection)
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            openrouter_model=os.environ.get("OPENROUTER_MODEL", "bytedance-seed/seed-1.6-flash"),
            vlm_every_n_frames=_get_int_env("VLM_EVERY_N_FRAMES", 10),
            vlm_max_generation_frames=_get_int_env("VLM_MAX_GENERATION_FRAMES", 60),

            # Web server
            web_host=os.environ.get("WEB_HOST", "0.0.0.0"),
            web_port=_get_int_env("WEB_PORT", 8000),

            # Video processing
            target_fps=_get_float_env("TARGET_FPS", 30.0),
            frame_buffer_size=_get_int_env("FRAME_BUFFER_SIZE", 120),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # SAM3 loads from HuggingFace Hub automatically, no path needed

        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            errors.append(f"Invalid confidence threshold: {self.confidence_threshold}")

        return errors

    def print_config(self) -> None:
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("Anomaly Detection Pipeline (VLM + SAM3)")
        print("=" * 60)
        print(f"  SAM3 Model:       facebook/sam3 (HuggingFace)")
        print(f"  HF Token:         {'***' + self.hf_token[-4:] if self.hf_token else 'Not set'}")
        print(f"  Device:           {self.device}")
        print(f"  Confidence:       {self.confidence_threshold}")
        print()
        print("VLM Configuration (OpenRouter):")
        print(f"  Model:            {self.openrouter_model}")
        print(f"  API Key:          {'***' + self.openrouter_api_key[-4:] if self.openrouter_api_key else 'Not set'}")
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
