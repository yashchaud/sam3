"""
Image loading and preprocessing utilities.

Provides consistent image handling across the pipeline.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np


def load_image(
    path: Union[str, Path],
    mode: str = "rgb",
) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        path: Path to image file
        mode: Color mode - 'rgb', 'bgr', or 'gray'

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if mode == "gray":
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    if mode == "rgb" and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    mode: str = "rgb",
) -> None:
    """
    Save an image to disk.

    Args:
        image: Image array to save
        path: Output file path
        mode: Color mode of input - 'rgb' or 'bgr'
    """
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "rgb" and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image)


def ensure_rgb(image: np.ndarray, source_mode: str = "bgr") -> np.ndarray:
    """
    Convert image to RGB format.

    Args:
        image: Input image
        source_mode: Current color mode ('bgr' or 'rgb')

    Returns:
        Image in RGB format
    """
    import cv2

    if image.ndim == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if source_mode == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_for_inference(
    image: np.ndarray,
    max_size: int = 1024,
    min_size: int = 512,
) -> Tuple[np.ndarray, float]:
    """
    Resize image for model inference while maintaining aspect ratio.

    Args:
        image: Input image (H, W, C)
        max_size: Maximum dimension size
        min_size: Minimum dimension size

    Returns:
        Tuple of (resized image, scale factor)
    """
    import cv2

    h, w = image.shape[:2]
    max_dim = max(h, w)
    min_dim = min(h, w)

    # Determine scale factor
    if max_dim > max_size:
        scale = max_size / max_dim
    elif min_dim < min_size:
        scale = min_size / min_dim
    else:
        return image, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def validate_image(image: np.ndarray) -> None:
    """
    Validate image array format.

    Args:
        image: Image to validate

    Raises:
        ValueError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image)}")

    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got {image.shape[2]}")

    if image.size == 0:
        raise ValueError("Empty image array")
