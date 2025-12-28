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
        return """You are an expert structural inspection AI. Analyze the image for ANY visible defects.

DEFECT TYPES TO DETECT:
- Crack: Any line, fracture, or break pattern (hairline, structural, pattern cracking, crazing)
- Corrosion: Rust, oxidation, metal degradation
- Spalling: Surface deterioration, flaking, chipping
- Deformation: Bowing, bulging, warping
- Stain: Discoloration, water marks, efflorescence
- Exposed_Rebar: Visible reinforcement steel
- Delamination: Layer separation
- Honeycomb: Voids, air pockets in concrete
- Scaling: Surface peeling
- Rust: Metal oxidation

GRID SYSTEM:
The image has a labeled grid (A1, A2, A3, B1, B2, B3, C1, C2, C3 for 3x3).
Row letters go top to bottom (A=top, B=middle, C=bottom).
Column numbers go left to right (1=left, 2=center, 3=right).

RESPOND WITH JSON ONLY:
{
  "predictions": [
    {"cell": "A1", "defect_type": "crack", "confidence": 0.9, "description": "visible crack"}
  ]
}

IMPORTANT:
- Report ALL visible defects, even minor ones
- If you see ANY crack-like patterns, report them as "crack"
- Use confidence 0.5-1.0 based on visibility
- Report defects in EVERY cell where they appear
- For images showing crack patterns/textures, report cracks in all affected cells"""

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
