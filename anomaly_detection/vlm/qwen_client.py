"""Local Qwen3 VL client for anomaly detection."""

import asyncio
import time
import numpy as np
from pathlib import Path
import json
import re
from io import BytesIO
import base64

from .base_client import BaseVLMClient
from .models import VLMConfig, VLMPrediction, VLMResponse, VLMProvider, PredictionType


class QwenVLClient(BaseVLMClient):
    """Client for local Qwen3 VL 8B model."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._model = None
        self._processor = None
        self._device = config.qwen_device

    def load(self) -> None:
        """Load Qwen VL model into memory."""
        if self._is_loaded:
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch

            model_path = self.config.qwen_model_path or "Qwen/Qwen2.5-VL-7B-Instruct"

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map=self._device if self._device != "cpu" else None,
                trust_remote_code=True,
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            self._model.eval()
            self._is_loaded = True

        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers. Install with: pip install transformers>=4.40.0. Error: {e}"
            )

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    def _image_to_pil(self, image: np.ndarray):
        """Convert numpy array to PIL Image."""
        from PIL import Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    async def predict(
        self,
        image: np.ndarray,
        grid_image: np.ndarray,
        grid_description: str,
        frame_id: str,
    ) -> VLMResponse:
        """Run prediction using local Qwen VL model."""
        if not self.is_loaded():
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=0,
                is_valid=False,
                error_message="Model not loaded",
                provider=VLMProvider.LOCAL_QWEN,
            )

        start_time = time.perf_counter()

        try:
            # Run inference in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                self._run_inference,
                grid_image,
                grid_description,
            )

            generation_time_ms = (time.perf_counter() - start_time) * 1000

            # Parse predictions
            predictions = self._parse_predictions(response_text, grid_description)

            return VLMResponse(
                frame_id=frame_id,
                predictions=predictions,
                generation_time_ms=generation_time_ms,
                is_valid=True,
                provider=VLMProvider.LOCAL_QWEN,
            )

        except Exception as e:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=generation_time_ms,
                is_valid=False,
                error_message=str(e),
                provider=VLMProvider.LOCAL_QWEN,
            )

    def _run_inference(self, grid_image: np.ndarray, grid_description: str) -> str:
        """Run synchronous inference."""
        import torch

        pil_image = self._image_to_pil(grid_image)

        # Build conversation
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": f"{grid_description}\n\nAnalyze this image for structural defects."},
                ],
            },
        ]

        # Process inputs
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        if self._device == "cuda":
            inputs = inputs.to("cuda")

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response

    def _parse_predictions(self, response_text: str, grid_description: str) -> list[VLMPrediction]:
        """Parse model response into predictions with grid coordinates."""
        predictions = []

        try:
            # Extract JSON from response
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

                confidence = max(0.0, min(1.0, confidence))

                if confidence < self.config.min_confidence:
                    continue

                # Parse cell label to get grid position
                col_row = self._parse_cell_label(cell_label)
                if col_row is None:
                    continue

                predictions.append(VLMPrediction(
                    prediction_type=PredictionType.POINT,
                    confidence=confidence,
                    defect_type=defect_type,
                    grid_cell=col_row,
                ))

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        return predictions[:self.config.max_predictions_per_frame]

    def _parse_cell_label(self, label: str) -> tuple[int, int] | None:
        """Parse cell label like 'A1' to (col, row)."""
        if len(label) < 2:
            return None

        row_char = label[0]
        col_str = label[1:]

        if not row_char.isalpha() or not col_str.isdigit():
            return None

        row = ord(row_char) - ord('A')
        col = int(col_str) - 1

        grid_config = self.config.grid_config
        if 0 <= row < grid_config.rows and 0 <= col < grid_config.cols:
            return (col, row)

        return None
