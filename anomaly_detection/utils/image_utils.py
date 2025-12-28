"""Image I/O and manipulation utilities."""

from pathlib import Path
import numpy as np
import cv2


def load_image(path: str | Path, mode: str = "rgb") -> np.ndarray:
    """
    Load image from disk.

    Args:
        path: Path to image file
        mode: 'rgb', 'bgr', or 'gray'

    Returns:
        Image as numpy array
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    if mode == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif mode == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # bgr
        return image


def save_image(image: np.ndarray, path: str | Path, mode: str = "rgb") -> None:
    """
    Save image to disk.

    Args:
        image: Image array
        path: Output path
        mode: Input mode ('rgb' or 'bgr')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "rgb" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image)


def ensure_rgb(image: np.ndarray, source_mode: str = "bgr") -> np.ndarray:
    """
    Convert image to RGB format.

    Args:
        image: Input image
        source_mode: Current mode ('bgr' or 'gray')

    Returns:
        RGB image
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif source_mode == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_for_inference(
    image: np.ndarray,
    max_size: int = 1024,
    min_size: int = 512,
) -> tuple[np.ndarray, float]:
    """
    Resize image for inference while maintaining aspect ratio.

    Args:
        image: Input image
        max_size: Maximum dimension
        min_size: Minimum dimension

    Returns:
        (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    min_dim = min(h, w)

    # Calculate scale
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


def validate_image(image: np.ndarray) -> list[str]:
    """
    Validate image array format.

    Args:
        image: Image to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not isinstance(image, np.ndarray):
        errors.append("Image must be numpy array")
        return errors

    if len(image.shape) not in [2, 3]:
        errors.append(f"Invalid dimensions: {image.shape}")

    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        errors.append(f"Invalid channels: {image.shape[2]}")

    if image.size == 0:
        errors.append("Image is empty")

    return errors


def draw_detections(
    image: np.ndarray,
    detections: list,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.

    Args:
        image: RGB image
        detections: List of Detection objects
        color: Box color (RGB)
        thickness: Line thickness

    Returns:
        Image with boxes drawn
    """
    output = image.copy()

    for det in detections:
        bbox = det.bbox
        cv2.rectangle(
            output,
            (bbox.x_min, bbox.y_min),
            (bbox.x_max, bbox.y_max),
            color,
            thickness,
        )

        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(
            output,
            (bbox.x_min, bbox.y_min - text_h - 4),
            (bbox.x_min + text_w, bbox.y_min),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (bbox.x_min, bbox.y_min - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return output


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay mask on image with transparency.

    Args:
        image: RGB image
        mask: Binary mask
        color: Overlay color (RGB)
        alpha: Transparency (0-1)

    Returns:
        Image with mask overlay
    """
    output = image.copy()

    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color

    # Blend
    output = cv2.addWeighted(output, 1, overlay, alpha, 0)

    return output
