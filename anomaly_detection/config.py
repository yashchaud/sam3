"""Configuration management for anomaly detection."""

import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class EnvironmentConfig:
    """Configuration loaded from environment variables."""
    detector_model_path: Path | None = None
    segmenter_model_path: Path | None = None
    device: str = "auto"
    confidence_threshold: float = 0.3
    mask_output_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """Load configuration from environment variables."""
        detector_path = os.environ.get("ANOMALY_DETECTOR_MODEL")
        segmenter_path = os.environ.get("ANOMALY_SEGMENTER_MODEL")
        device = os.environ.get("ANOMALY_DEVICE", "auto")
        confidence = float(os.environ.get("ANOMALY_CONFIDENCE_THRESHOLD", "0.3"))
        mask_dir = os.environ.get("ANOMALY_MASK_OUTPUT_DIR")

        return cls(
            detector_model_path=Path(detector_path) if detector_path else None,
            segmenter_model_path=Path(segmenter_path) if segmenter_path else None,
            device=device,
            confidence_threshold=confidence,
            mask_output_dir=Path(mask_dir) if mask_dir else None,
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

        return errors


def get_device(preference: str = "auto") -> str:
    """Determine device to use based on preference and availability."""
    if preference == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return preference
