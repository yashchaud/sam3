"""RF-DETR object detector for structural inspection."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import numpy as np

from ..models import Detection, DetectionType, BoundingBox
from ..config import DEFAULT_STRUCTURE_CLASSES, DEFAULT_ANOMALY_CLASSES


class RFDETRVariant(Enum):
    """RF-DETR model variants."""
    NANO = "rf_detr_nano"      # 48.4 mAP, 2.32ms
    SMALL = "rf_detr_small"    # 53.0 mAP, 3.52ms
    MEDIUM = "rf_detr_medium"  # 54.7 mAP, 4.52ms (recommended)
    LARGE = "rf_detr_large"    # 56.7 mAP, 9.64ms


@dataclass
class DetectorConfig:
    """Configuration for RF-DETR detector."""
    variant: RFDETRVariant = RFDETRVariant.MEDIUM
    pretrain_weights: Path | None = None
    num_classes: int | None = None
    confidence_threshold: float = 0.3
    device: str = "auto"
    optimize_inference: bool = True

    structure_classes: set[str] = field(default_factory=lambda: DEFAULT_STRUCTURE_CLASSES.copy())
    anomaly_classes: set[str] = field(default_factory=lambda: DEFAULT_ANOMALY_CLASSES.copy())


@dataclass
class DetectorOutput:
    """Output from detector."""
    detections: list[Detection]
    inference_time_ms: float

    @property
    def structures(self) -> list[Detection]:
        return [d for d in self.detections if d.detection_type == DetectionType.STRUCTURE]

    @property
    def anomalies(self) -> list[Detection]:
        return [d for d in self.detections if d.detection_type == DetectionType.ANOMALY]


class RFDETRDetector:
    """RF-DETR detector for structural anomaly detection."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._model = None
        self._device = None
        self._is_loaded = False

    def load(self) -> None:
        """Load the model into memory."""
        if self._is_loaded:
            return

        try:
            from rfdetr import RFDETRBase
            import torch

            # Determine device
            if self.config.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.config.device

            # Load model based on variant
            variant_map = {
                RFDETRVariant.NANO: "rf_detr_nano",
                RFDETRVariant.SMALL: "rf_detr_small",
                RFDETRVariant.MEDIUM: "rf_detr_medium",
                RFDETRVariant.LARGE: "rf_detr_large",
            }

            model_name = variant_map[self.config.variant]

            if self.config.pretrain_weights:
                # Load custom weights
                self._model = RFDETRBase.from_pretrained(
                    str(self.config.pretrain_weights),
                    num_classes=self.config.num_classes,
                )
            else:
                # Load pretrained
                self._model = RFDETRBase.from_pretrained(model_name)

            self._model = self._model.to(self._device)
            self._model.eval()

            # Optimize if requested
            if self.config.optimize_inference and self._device == "cuda":
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                except Exception:
                    pass  # Fallback to non-compiled

            self._is_loaded = True

        except ImportError as e:
            raise ImportError(
                f"RF-DETR not installed. Install with: pip install rfdetr. Error: {e}"
            )

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    def detect(self, image: np.ndarray) -> DetectorOutput:
        """
        Run detection on an image.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            DetectorOutput with all detections
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.perf_counter()

        # Run inference
        results = self._model.predict(image, threshold=self.config.confidence_threshold)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Convert to Detection objects
        detections = []
        for i, (bbox, score, class_id) in enumerate(zip(
            results.xyxy,
            results.confidence,
            results.class_id,
        )):
            class_name = results.names.get(int(class_id), f"class_{class_id}")

            # Determine type
            if class_name.lower() in self.config.structure_classes:
                det_type = DetectionType.STRUCTURE
            elif class_name.lower() in self.config.anomaly_classes:
                det_type = DetectionType.ANOMALY
            else:
                # Default to anomaly (favor detection)
                det_type = DetectionType.ANOMALY

            detection = Detection(
                detection_id=f"det_{i:04d}",
                detection_type=det_type,
                class_name=class_name,
                confidence=float(score),
                bbox=BoundingBox(
                    x_min=int(bbox[0]),
                    y_min=int(bbox[1]),
                    x_max=int(bbox[2]),
                    y_max=int(bbox[3]),
                ),
            )
            detections.append(detection)

        return DetectorOutput(
            detections=detections,
            inference_time_ms=inference_time,
        )

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
