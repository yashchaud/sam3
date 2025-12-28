"""OpenRouter client for VLM inference."""

import asyncio
import time
import logging
import numpy as np
import json
import re
import base64
from io import BytesIO
import aiohttp

from .base_client import BaseVLMClient
from .models import VLMConfig, VLMPrediction, VLMResponse, VLMProvider, PredictionType

logger = logging.getLogger(__name__)


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
        logger.info(f"[OpenRouter] Starting API call for frame {frame_id}")

        try:
            session = await self._ensure_session()

            # Convert image to base64
            image_b64 = self._image_to_base64(grid_image)
            logger.info(f"[OpenRouter] Image encoded, size: {len(image_b64)} bytes")

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

            logger.info(f"[OpenRouter] Sending request to {self.config.openrouter_base_url}/chat/completions")
            logger.info(f"[OpenRouter] Model: {self.config.openrouter_model}")

            async with session.post(
                f"{self.config.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            ) as response:
                generation_time_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"[OpenRouter] Response status: {response.status}, latency: {generation_time_ms:.0f}ms")

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[OpenRouter] API error: {error_text[:500]}")
                    return VLMResponse(
                        frame_id=frame_id,
                        predictions=[],
                        generation_time_ms=generation_time_ms,
                        is_valid=False,
                        error_message=f"API error {response.status}: {error_text[:200]}",
                        provider=VLMProvider.OPENROUTER,
                    )

                # Read response body inside the context manager
                response_body = await response.text()
                logger.info(f"[OpenRouter] Response body length: {len(response_body)} chars")
                logger.info(f"[OpenRouter] Response body preview: {response_body[:500]}")

                try:
                    data = json.loads(response_body)
                except json.JSONDecodeError as e:
                    logger.error(f"[OpenRouter] Failed to parse response JSON: {e}")
                    logger.error(f"[OpenRouter] Raw body: {response_body[:1000]}")
                    return VLMResponse(
                        frame_id=frame_id,
                        predictions=[],
                        generation_time_ms=generation_time_ms,
                        is_valid=False,
                        error_message=f"Invalid JSON response: {str(e)}",
                        provider=VLMProvider.OPENROUTER,
                    )

            # Extract response text from API response
            response_text = ""
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                response_text = message.get("content", "")
                logger.info(f"[OpenRouter] Extracted content from choices: {len(response_text)} chars")
            else:
                logger.warning(f"[OpenRouter] No 'choices' in response. Keys: {list(data.keys())}")

            # Log raw VLM response
            logger.info(f"[OpenRouter] Model: {self.config.openrouter_model}")
            logger.info(f"[OpenRouter] Raw response: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")

            # Parse predictions
            predictions = self._parse_predictions(response_text)
            logger.info(f"[OpenRouter] Parsed {len(predictions)} predictions from response")

            return VLMResponse(
                frame_id=frame_id,
                predictions=predictions,
                generation_time_ms=generation_time_ms,
                is_valid=True,
                provider=VLMProvider.OPENROUTER,
            )

        except asyncio.TimeoutError as e:
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"[OpenRouter] TIMEOUT ERROR: {e}")
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
            logger.error(f"[OpenRouter] CLIENT ERROR: {type(e).__name__}: {e}")
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
            logger.error(f"[OpenRouter] UNEXPECTED ERROR: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[OpenRouter] Traceback: {traceback.format_exc()}")
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

        logger.info(f"[OpenRouter] Parsing response text ({len(response_text)} chars)")

        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning(f"[OpenRouter] No JSON found in response: {response_text[:200]}")
                return predictions

            json_str = json_match.group()
            logger.info(f"[OpenRouter] Found JSON: {json_str[:300]}")

            data = json.loads(json_str)

            if "predictions" not in data:
                logger.warning(f"[OpenRouter] No 'predictions' key in JSON. Keys: {list(data.keys())}")
                return predictions

            logger.info(f"[OpenRouter] Found {len(data['predictions'])} predictions in JSON")

            for i, pred in enumerate(data["predictions"]):
                cell_label = pred.get("cell", "").strip().upper()
                defect_type = pred.get("defect_type", "unknown")
                confidence = float(pred.get("confidence", 0.5))

                logger.info(f"[OpenRouter] Prediction {i}: cell={cell_label}, type={defect_type}, conf={confidence}")

                if not cell_label:
                    logger.warning(f"[OpenRouter] Skipping prediction {i}: empty cell label")
                    continue

                confidence = max(0.0, min(1.0, confidence))

                if confidence < self.config.min_confidence:
                    logger.warning(f"[OpenRouter] Skipping prediction {i}: confidence {confidence} < min {self.config.min_confidence}")
                    continue

                col_row = self._parse_cell_label(cell_label)
                if col_row is None:
                    logger.warning(f"[OpenRouter] Skipping prediction {i}: invalid cell label '{cell_label}'")
                    continue

                logger.info(f"[OpenRouter] Adding prediction: {defect_type} at {cell_label} -> col={col_row[0]}, row={col_row[1]}")
                predictions.append(VLMPrediction(
                    prediction_type=PredictionType.POINT,
                    confidence=confidence,
                    defect_type=defect_type,
                    grid_cell=col_row,
                ))

        except json.JSONDecodeError as e:
            logger.error(f"[OpenRouter] JSON parse error: {e}")
            logger.error(f"[OpenRouter] Response was: {response_text[:500]}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"[OpenRouter] Parse error: {e}")

        logger.info(f"[OpenRouter] Final parsed predictions: {len(predictions)}")
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
