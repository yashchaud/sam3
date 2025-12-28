"""
RF-DETR based object detector for structures and anomalies.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from enum import Enum
import numpy as np

from anomaly_detection.models.data_models import (
    Detection,
    DetectionType,
    BoundingBox,
    generate_id,
)


class RFDETRVariant(Enum):
    """
    Available RF-DETR model variants.

    Attributes:
        NANO: Fastest, lowest accuracy (48.4 mAP, 2.32ms)
        SMALL: Balanced speed/accuracy (53.0 mAP, 3.52ms)
        MEDIUM: Recommended default (54.7 mAP, 4.52ms)
        LARGE: Highest accuracy (56.7 mAP, 9.64ms)
    """
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"   # Recommended
    LARGE = "large"


# Default class mappings for structural inspection
DEFAULT_STRUCTURE_CLASSES: tuple[str, ...] = (
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

DEFAULT_ANOMALY_CLASSES: tuple[str, ...] = (
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


@dataclass
class DetectorConfig:
    """
    Configuration for RF-DETR detector.

    Attributes:
        variant: Model size variant (NANO, SMALL, MEDIUM, LARGE)
        pretrain_weights: Path to custom fine-tuned weights (.pth file)
                         If None, uses COCO pretrained weights
        num_classes: Number of classes for custom models
                    If None, uses default (80 for COCO or inferred from weights)
        confidence_threshold: Minimum confidence to keep detections [0.0, 1.0]
        device: Compute device ('cuda', 'cpu', or 'auto')
        optimize_inference: Apply torch.compile optimization for faster inference
        structure_classes: Class names considered as structural elements
        anomaly_classes: Class names considered as anomalies/defects

    Example:
        # Use pretrained COCO model
        config = DetectorConfig(variant=RFDETRVariant.MEDIUM)

        # Use fine-tuned model
        config = DetectorConfig(
            variant=RFDETRVariant.MEDIUM,
            pretrain_weights=Path("weights/rfdetr_structural.pth"),
            num_classes=22,  # Your custom class count
        )
    """
    variant: RFDETRVariant = RFDETRVariant.MEDIUM
    pretrain_weights: Optional[Path] = None
    num_classes: Optional[int] = None
    confidence_threshold: float = 0.3
    device: str = "auto"
    optimize_inference: bool = True
    structure_classes: tuple[str, ...] = DEFAULT_STRUCTURE_CLASSES
    anomaly_classes: tuple[str, ...] = DEFAULT_ANOMALY_CLASSES

    def __post_init__(self):
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}"
            )
        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {self.num_classes}")


@dataclass
class DetectorOutput:
    """
    Output container for detector results.

    Attributes:
        detections: All detections (structures + anomalies)
        structures: Detections classified as structural elements
        anomalies: Detections classified as anomalies/defects
        inference_time_ms: Time spent on inference in milliseconds
    """
    detections: list[Detection]
    structures: list[Detection]
    anomalies: list[Detection]
    inference_time_ms: float


class RFDETRDetector:
    """
    RF-DETR based detector for structural elements and anomalies.

    This detector uses Roboflow's RF-DETR architecture, which combines
    a DINOv2 vision backbone with a detection transformer head. It provides
    state-of-the-art accuracy with real-time performance.

    Key Features:
        - First real-time model to exceed 60 AP on COCO
        - DINOv2 backbone provides excellent transfer learning
        - Designed for fine-tuning on custom datasets
        - Apache 2.0 license (full commercial use)

    Usage:
        # Basic usage with pretrained COCO weights
        config = DetectorConfig(variant=RFDETRVariant.MEDIUM)
        detector = RFDETRDetector(config)
        detector.load()

        output = detector.detect(image)
        for detection in output.anomalies:
            print(f"Found {detection.class_name} at {detection.bbox}")

        # With custom fine-tuned weights
        config = DetectorConfig(
            variant=RFDETRVariant.MEDIUM,
            pretrain_weights=Path("weights/my_model.pth"),
            num_classes=15,
        )
        detector = RFDETRDetector(config)
        detector.load()

    Note:
        Call load() before detect(). Use unload() to free GPU memory.
    """

    def __init__(self, config: DetectorConfig):
        """
        Initialize the detector.

        Args:
            config: Detector configuration
        """
        self.config = config
        self._model = None
        self._device: Optional[str] = None
        self._class_names: list[str] = []
        self._optimized = False

    def load(self) -> None:
        """
        Load the RF-DETR model into memory.

        This method initializes the model based on the configured variant
        and optionally loads custom weights. Call this before detect().

        Raises:
            RuntimeError: If rfdetr package is not installed
            RuntimeError: If weight file specified but not found
            RuntimeError: If model loading fails
        """
        import torch

        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        # Validate weights path if specified
        if self.config.pretrain_weights is not None:
            if not self.config.pretrain_weights.exists():
                raise RuntimeError(
                    f"Weight file not found: {self.config.pretrain_weights}"
                )

        # Import and instantiate the appropriate model variant
        try:
            model_class = self._get_model_class()
            self._model = self._create_model(model_class)
        except ImportError as e:
            raise RuntimeError(
                "rfdetr package not installed. Install with:\n"
                "  pip install rfdetr\n"
                f"Original error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load RF-DETR model: {e}")

        # Optimize for inference if requested
        if self.config.optimize_inference:
            try:
                self._model.optimize_for_inference()
                self._optimized = True
            except Exception:
                # Optimization may fail on some platforms, continue without it
                self._optimized = False

        # Load class names
        self._load_class_names()

    def _get_model_class(self):
        """Get the appropriate RF-DETR model class for the configured variant."""
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

        variant_to_class = {
            RFDETRVariant.NANO: RFDETRNano,
            RFDETRVariant.SMALL: RFDETRSmall,
            RFDETRVariant.MEDIUM: RFDETRMedium,
            RFDETRVariant.LARGE: RFDETRLarge,
        }
        return variant_to_class[self.config.variant]

    def _create_model(self, model_class):
        """Create and configure the model instance."""
        kwargs = {}

        # Add custom weights if specified
        if self.config.pretrain_weights is not None:
            kwargs["pretrain_weights"] = str(self.config.pretrain_weights)

        # Add custom class count if specified
        if self.config.num_classes is not None:
            kwargs["num_classes"] = self.config.num_classes

        return model_class(**kwargs)

    def _load_class_names(self) -> None:
        """Load class names from model or use defaults."""
        try:
            # Try to get class names from COCO classes
            from rfdetr.util.coco_classes import COCO_CLASSES
            self._class_names = list(COCO_CLASSES)
        except ImportError:
            # Fallback to combined structure + anomaly classes
            self._class_names = list(
                self.config.structure_classes + self.config.anomaly_classes
            )

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._model is not None

    def detect(self, image: np.ndarray) -> DetectorOutput:
        """
        Run object detection on a single image.

        Args:
            image: Input image as numpy array with shape (H, W, C) in RGB format.
                   Accepts uint8 [0-255] or float32 [0-1] values.

        Returns:
            DetectorOutput containing:
                - detections: All detected objects
                - structures: Objects classified as structural elements
                - anomalies: Objects classified as anomalies/defects
                - inference_time_ms: Processing time

        Raises:
            RuntimeError: If model not loaded (call load() first)
            ValueError: If image format is invalid

        Example:
            output = detector.detect(image)
            print(f"Found {len(output.anomalies)} anomalies")
            for anomaly in output.anomalies:
                print(f"  {anomaly.class_name}: {anomaly.confidence:.1%}")
        """
        import time

        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Validate image
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array (H, W, C), got shape {image.shape}")
        if image.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3 or 4 channels, got {image.shape[2]}")

        # Run inference
        start_time = time.perf_counter()
        detections = self._run_inference(image)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Split detections by type
        structures = [d for d in detections if d.detection_type == DetectionType.STRUCTURE]
        anomalies = [d for d in detections if d.detection_type == DetectionType.ANOMALY]

        return DetectorOutput(
            detections=detections,
            structures=structures,
            anomalies=anomalies,
            inference_time_ms=inference_time_ms,
        )

    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        """
        Execute model inference and parse results.

        Args:
            image: Preprocessed image array

        Returns:
            List of Detection objects
        """
        # RF-DETR accepts numpy arrays directly (RGB format)
        # The predict() method returns a supervision.Detections object
        sv_detections = self._model.predict(
            image,
            threshold=self.config.confidence_threshold,
        )

        # Parse supervision detections into our Detection format
        detections = []

        if sv_detections is None or len(sv_detections) == 0:
            return detections

        for i in range(len(sv_detections)):
            # Extract detection data
            xyxy = sv_detections.xyxy[i]
            confidence = float(sv_detections.confidence[i])
            class_id = int(sv_detections.class_id[i])

            # Get class name
            class_name = self._get_class_name(class_id)
            if class_name is None:
                class_name = f"class_{class_id}"

            # Classify as structure or anomaly
            detection_type = self._classify_detection(class_name)

            # Create bounding box
            bbox = BoundingBox(
                x_min=float(xyxy[0]),
                y_min=float(xyxy[1]),
                x_max=float(xyxy[2]),
                y_max=float(xyxy[3]),
            )

            # Create detection object
            detection = Detection(
                detection_id=generate_id("det"),
                detection_type=detection_type,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
            )
            detections.append(detection)

        return detections

    def _get_class_name(self, class_id: int) -> Optional[str]:
        """
        Get class name from class ID.

        Args:
            class_id: Integer class identifier

        Returns:
            Class name string, or None if ID is out of range
        """
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return None

    def _classify_detection(self, class_name: str) -> DetectionType:
        """
        Classify a detection as structure or anomaly based on class name.

        Args:
            class_name: Name of the detected class

        Returns:
            DetectionType.STRUCTURE or DetectionType.ANOMALY

        Note:
            Unknown classes default to ANOMALY to favor recall
            (better to flag potential issues than miss them).
        """
        class_lower = class_name.lower()

        # Check if it's a structure
        for structure_class in self.config.structure_classes:
            if structure_class.lower() == class_lower:
                return DetectionType.STRUCTURE

        # Check if it's a known anomaly
        for anomaly_class in self.config.anomaly_classes:
            if anomaly_class.lower() == class_lower:
                return DetectionType.ANOMALY

        # Default to anomaly for unknown classes (favor recall)
        return DetectionType.ANOMALY

    def unload(self) -> None:
        """
        Release model from memory.

        Frees GPU memory and clears internal state. Call this when
        the detector is no longer needed or before loading a new model.
        """
        self._model = None
        self._device = None
        self._class_names = []
        self._optimized = False

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __repr__(self) -> str:
        """Return string representation of detector state."""
        status = "loaded" if self.is_loaded() else "not loaded"
        opt = ", optimized" if self._optimized else ""
        return (
            f"RFDETRDetector(variant={self.config.variant.value}, "
            f"status={status}{opt})"
        )
