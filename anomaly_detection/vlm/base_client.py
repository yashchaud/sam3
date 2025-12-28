"""Base client interface for VLM providers."""

from abc import ABC, abstractmethod
import numpy as np

from .models import VLMPrediction, VLMResponse, VLMConfig


class BaseVLMClient(ABC):
    """Abstract base class for VLM clients."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass

    @abstractmethod
    async def predict(
        self,
        image: np.ndarray,
        grid_image: np.ndarray,
        grid_description: str,
        frame_id: str,
    ) -> VLMResponse:
        """
        Run prediction on an image with grid overlay.

        Args:
            image: Original image (RGB, HWC)
            grid_image: Image with grid overlay drawn
            grid_description: Text description of grid cells
            frame_id: Identifier for the frame

        Returns:
            VLMResponse with predictions
        """
        pass

    def get_system_prompt(self) -> str:
        """Get system prompt for structural anomaly detection."""
        return """You are an expert structural inspection AI analyzing images for defects and anomalies.

Your task is to identify potential structural issues including:
- Cracks (hairline, structural, pattern cracking)
- Corrosion and rust
- Spalling (concrete surface deterioration)
- Deformation (bowing, bulging, misalignment)
- Staining (water damage, efflorescence)
- Exposed rebar
- Delamination
- Honeycomb (concrete voids)
- Scaling and popouts

The image has a grid overlay with labeled cells (e.g., A1, A2, B1, B2).
Identify any defects and specify their location using grid cell references.

Respond ONLY with a JSON object in this exact format:
{
  "predictions": [
    {
      "cell": "A1",
      "defect_type": "crack",
      "confidence": 0.85,
      "description": "Hairline crack running diagonally"
    }
  ]
}

If no defects are found, respond with:
{"predictions": []}

Rules:
- Only report genuine structural concerns, not normal wear or shadows
- Use exact cell labels from the grid
- Confidence should be between 0.0 and 1.0
- Keep descriptions brief but specific
- Maximum 10 predictions per image"""

    def parse_response(self, response_text: str, frame_id: str) -> list[VLMPrediction]:
        """Parse VLM response text into predictions."""
        import json
        import re
        from .models import PredictionType

        predictions = []

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return predictions

            data = json.loads(json_match.group())

            if "predictions" not in data:
                return predictions

            for pred in data["predictions"]:
                cell_label = pred.get("cell", "").strip().upper()
                defect_type = pred.get("defect_type", "unknown")
                confidence = float(pred.get("confidence", 0.5))

                if not cell_label:
                    continue

                # Clamp confidence
                confidence = max(0.0, min(1.0, confidence))

                if confidence < self.config.min_confidence:
                    continue

                predictions.append(VLMPrediction(
                    prediction_type=PredictionType.POINT,  # Will be resolved to actual coords later
                    confidence=confidence,
                    defect_type=defect_type,
                    grid_cell=None,  # Will be set by caller
                ))

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        return predictions[:self.config.max_predictions_per_frame]
