"""
Default configuration and environment-based settings.

Provides sensible defaults for production deployment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class EnvironmentConfig:
    """
    Configuration loaded from environment variables.

    Environment variables:
        ANOMALY_DETECTOR_MODEL: Path to RT-DETR model
        ANOMALY_SEGMENTER_MODEL: Path to SAM3 model
        ANOMALY_DEVICE: Compute device (cuda/cpu/auto)
        ANOMALY_CONFIDENCE_THRESHOLD: Detection threshold
        ANOMALY_MASK_OUTPUT_DIR: Directory for mask outputs
    """
    detector_model_path: Path
    segmenter_model_path: Path
    device: str = "auto"
    confidence_threshold: float = 0.3
    mask_output_dir: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """Load configuration from environment variables."""
        detector_path = os.environ.get("ANOMALY_DETECTOR_MODEL")
        segmenter_path = os.environ.get("ANOMALY_SEGMENTER_MODEL")

        if detector_path is None:
            raise ValueError("ANOMALY_DETECTOR_MODEL environment variable required")
        if segmenter_path is None:
            raise ValueError("ANOMALY_SEGMENTER_MODEL environment variable required")

        mask_dir = os.environ.get("ANOMALY_MASK_OUTPUT_DIR")

        return cls(
            detector_model_path=Path(detector_path),
            segmenter_model_path=Path(segmenter_path),
            device=os.environ.get("ANOMALY_DEVICE", "auto"),
            confidence_threshold=float(
                os.environ.get("ANOMALY_CONFIDENCE_THRESHOLD", "0.3")
            ),
            mask_output_dir=Path(mask_dir) if mask_dir else None,
        )


# Default class mappings for structural inspection
DEFAULT_STRUCTURE_CLASSES = (
    "beam",
    "column",
    "wall",
    "slab",
    "pipe",
    "foundation",
    "joint",
    "girder",
    "truss",
    "deck",
)

DEFAULT_ANOMALY_CLASSES = (
    "crack",
    "corrosion",
    "spalling",
    "deformation",
    "stain",
    "efflorescence",
    "exposed_rebar",
    "delamination",
    "scaling",
    "popout",
    "honeycomb",
    "rust",
)


# Performance constants
MAX_IMAGE_SIZE = 2048  # Maximum dimension for processing
MIN_MASK_AREA = 10  # Minimum pixels for valid mask
DEFAULT_BATCH_SIZE = 4  # For batch processing
