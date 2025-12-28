"""OpenRouter client for VLM inference."""

import asyncio
import time
import numpy as np
import json
import re
import base64
from io import BytesIO
import aiohttp

from .base_client import BaseVLMClient
from .models import VLMConfig, VLMPrediction, VLMResponse, VLMProvider, PredictionType


class OpenRouterClient(BaseVLMClient):
    """Client for OpenRouter API (cloud VLM inference)."""

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None

    def load(self) -> None:
        """Initialize HTTP session."""
        if self._is_loaded:
            return

        if not self.config.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")

        self._is_loaded = True

    def unload(self) -> None:
        """Close HTTP session."""
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded and self.config.openrouter_api_key is not None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds + 5)
            )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 encoded image."""
        from PIL import Image

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)

        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def predict(
        self,
        image: np.ndarray,
        grid_image: np.ndarray,
        grid_description: str,
        frame_id: str,
    ) -> VLMResponse:
        """Run prediction using OpenRouter API."""
        if not self.is_loaded():
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=0,
                is_valid=False,
                error_message="Client not initialized (missing API key)",
                provider=VLMProvider.OPENROUTER,
            )

        start_time = time.perf_counter()

        try:
            session = await self._ensure_session()

            # Convert image to base64
            image_b64 = self._image_to_base64(grid_image)

            # Build request
            headers = {
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://anomaly-detection.local",
                "X-Title": "Anomaly Detection Pipeline",
            }

            payload = {
                "model": self.config.openrouter_model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": f"{grid_description}\n\nAnalyze this image for structural defects.",
                            },
                        ],
                    },
                ],
                "max_tokens": 512,
                "temperature": 0,
            }

            async with session.post(
                f"{self.config.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            ) as response:
                generation_time_ms = (time.perf_counter() - start_time) * 1000

                if response.status != 200:
                    error_text = await response.text()
                    return VLMResponse(
                        frame_id=frame_id,
                        predictions=[],
                        generation_time_ms=generation_time_ms,
                        is_valid=False,
                        error_message=f"API error {response.status}: {error_text[:200]}",
                        provider=VLMProvider.OPENROUTER,
                    )

                data = await response.json()

            # Extract response text
            response_text = ""
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                response_text = message.get("content", "")

            # Parse predictions
            predictions = self._parse_predictions(response_text)

            return VLMResponse(
                frame_id=frame_id,
                predictions=predictions,
                generation_time_ms=generation_time_ms,
                is_valid=True,
                provider=VLMProvider.OPENROUTER,
            )

        except asyncio.TimeoutError:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=generation_time_ms,
                is_valid=False,
                error_message="Request timed out",
                provider=VLMProvider.OPENROUTER,
            )

        except aiohttp.ClientError as e:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=generation_time_ms,
                is_valid=False,
                error_message=f"HTTP error: {str(e)}",
                provider=VLMProvider.OPENROUTER,
            )

        except Exception as e:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            return VLMResponse(
                frame_id=frame_id,
                predictions=[],
                generation_time_ms=generation_time_ms,
                is_valid=False,
                error_message=str(e),
                provider=VLMProvider.OPENROUTER,
            )

    def _parse_predictions(self, response_text: str) -> list[VLMPrediction]:
        """Parse model response into predictions."""
        predictions = []

        try:
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
