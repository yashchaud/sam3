"""Debug output system for pipeline analysis and optimization.

Saves intermediate results from each processing stage:
1. Original frames
2. SAM3 candidate masks (per defect type)
3. VLM grid overlays
4. VLM judgments (JSON)
5. Final approved detections
6. Timing statistics per stage
"""

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import cv2
from dataclasses import asdict, is_dataclass


class DebugOutputManager:
    """Manages debug output for pipeline analysis."""

    def __init__(self, output_dir: str | Path, enabled: bool = True):
        """
        Initialize debug output manager.

        Args:
            output_dir: Root directory for debug outputs
            enabled: Whether debug output is enabled
        """
        self.enabled = enabled
        if not enabled:
            return

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each stage
        self.dirs = {
            "frames": self.output_dir / "01_frames",
            "sam3_candidates": self.output_dir / "02_sam3_candidates",
            "vlm_grids": self.output_dir / "03_vlm_grids",
            "vlm_responses": self.output_dir / "04_vlm_responses",
            "approved_detections": self.output_dir / "05_approved_detections",
            "timing": self.output_dir / "06_timing",
            "summary": self.output_dir / "00_summary",
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize timing log
        self.timing_log = []

        print(f"[Debug Output] Enabled - Output directory: {self.output_dir}")
        print(f"[Debug Output] Subdirectories created:")
        for name, path in self.dirs.items():
            print(f"  - {name}: {path.name}")

    def save_original_frame(self, frame_id: str, frame_index: int, image: np.ndarray) -> None:
        """Save original input frame."""
        if not self.enabled:
            return

        filename = self.dirs["frames"] / f"frame_{frame_index:06d}_{frame_id}.jpg"
        cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def save_sam3_candidate(
        self,
        frame_id: str,
        frame_index: int,
        defect_type: str,
        mask: np.ndarray,
        image: np.ndarray,
        confidence: float,
    ) -> None:
        """Save SAM3 candidate mask with overlay."""
        if not self.enabled:
            return

        # Create frame-specific directory
        frame_dir = self.dirs["sam3_candidates"] / f"frame_{frame_index:06d}"
        frame_dir.mkdir(exist_ok=True)

        # Save mask as binary image
        mask_filename = frame_dir / f"{defect_type}_mask.png"
        cv2.imwrite(str(mask_filename), mask * 255)

        # Save overlay visualization
        overlay = image.copy()
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = [255, 0, 0]  # Red for mask
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

        # Add text
        cv2.putText(
            overlay,
            f"{defect_type} (conf: {confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        overlay_filename = frame_dir / f"{defect_type}_overlay.jpg"
        cv2.imwrite(str(overlay_filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def save_sam3_summary(
        self,
        frame_id: str,
        frame_index: int,
        candidates_count: int,
        defect_types: list[str],
        processing_time_ms: float,
    ) -> None:
        """Save SAM3 processing summary."""
        if not self.enabled:
            return

        summary = {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "timestamp": time.time(),
            "sam3_candidates_count": candidates_count,
            "defect_types_found": defect_types,
            "processing_time_ms": processing_time_ms,
        }

        filename = self.dirs["sam3_candidates"] / f"frame_{frame_index:06d}_summary.json"
        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)

    def save_vlm_grid(self, frame_id: str, frame_index: int, grid_image: np.ndarray) -> None:
        """Save VLM grid overlay image."""
        if not self.enabled:
            return

        filename = self.dirs["vlm_grids"] / f"frame_{frame_index:06d}_{frame_id}_grid.jpg"
        cv2.imwrite(str(filename), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))

    def save_vlm_response(
        self,
        frame_id: str,
        frame_index: int,
        predictions: list,
        latency_ms: float,
        request_frame_index: int,
    ) -> None:
        """Save VLM judgment response."""
        if not self.enabled:
            return

        # Convert predictions to serializable format
        predictions_data = []
        for pred in predictions:
            pred_dict = {
                "defect_type": pred.defect_type,
                "confidence": pred.confidence,
                "prediction_type": pred.prediction_type.value if hasattr(pred.prediction_type, 'value') else str(pred.prediction_type),
            }
            if pred.point:
                pred_dict["point"] = pred.point
            if pred.box:
                pred_dict["box"] = pred.box
            if pred.grid_cell:
                pred_dict["grid_cell"] = pred.grid_cell
            predictions_data.append(pred_dict)

        response_data = {
            "frame_id": frame_id,
            "response_frame_index": frame_index,
            "request_frame_index": request_frame_index,
            "frame_delay": frame_index - request_frame_index,
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "predictions_count": len(predictions),
            "predictions": predictions_data,
        }

        filename = self.dirs["vlm_responses"] / f"response_frame_{frame_index:06d}_{frame_id}.json"
        with open(filename, "w") as f:
            json.dump(response_data, f, indent=2)

    def save_approved_detections(
        self,
        frame_id: str,
        frame_index: int,
        image: np.ndarray,
        approved_anomalies: list,
        sam_candidates_count: int,
    ) -> None:
        """Save final approved detections with visualization."""
        if not self.enabled:
            return

        # Create visualization
        output = image.copy()

        # Draw all approved masks
        for i, anomaly in enumerate(approved_anomalies):
            if anomaly.mask and anomaly.mask.data is not None:
                # Generate color for this detection
                color = self._get_color(i)
                colored_mask = np.zeros_like(output)
                colored_mask[anomaly.mask.data > 0] = color
                output = cv2.addWeighted(output, 0.7, colored_mask, 0.3, 0)

                # Draw bounding box
                if anomaly.bbox:
                    x1, y1, x2, y2 = map(int, [anomaly.bbox.x_min, anomaly.bbox.y_min, anomaly.bbox.x_max, anomaly.bbox.y_max])
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f"{anomaly.defect_type} ({anomaly.detection_confidence:.2f})"
                    cv2.putText(
                        output,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # Add summary text
        summary_text = f"Approved: {len(approved_anomalies)}/{sam_candidates_count} SAM3 candidates"
        cv2.putText(
            output,
            summary_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        filename = self.dirs["approved_detections"] / f"frame_{frame_index:06d}_{frame_id}_approved.jpg"
        cv2.imwrite(str(filename), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

        # Save JSON metadata
        metadata = {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "sam_candidates_count": sam_candidates_count,
            "approved_count": len(approved_anomalies),
            "approval_rate": len(approved_anomalies) / sam_candidates_count if sam_candidates_count > 0 else 0,
            "detections": [
                {
                    "defect_type": a.defect_type,
                    "vlm_confidence": a.detection_confidence,
                    "sam3_confidence": a.segmentation_confidence,
                    "bbox": [a.bbox.x_min, a.bbox.y_min, a.bbox.x_max, a.bbox.y_max] if a.bbox else None,
                }
                for a in approved_anomalies
            ],
        }

        json_filename = self.dirs["approved_detections"] / f"frame_{frame_index:06d}_{frame_id}_metadata.json"
        with open(json_filename, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_frame_timing(
        self,
        frame_id: str,
        frame_index: int,
        sam3_time_ms: float,
        vlm_time_ms: float,
        total_time_ms: float,
        sam_candidates_count: int,
        vlm_approved_count: int,
    ) -> None:
        """Save timing information for a frame."""
        if not self.enabled:
            return

        timing_entry = {
            "frame_id": frame_id,
            "frame_index": frame_index,
            "timestamp": time.time(),
            "timing": {
                "sam3_ms": sam3_time_ms,
                "vlm_ms": vlm_time_ms,
                "total_ms": total_time_ms,
            },
            "counts": {
                "sam3_candidates": sam_candidates_count,
                "vlm_approved": vlm_approved_count,
            },
            "efficiency": {
                "sam3_ms_per_candidate": sam3_time_ms / sam_candidates_count if sam_candidates_count > 0 else 0,
                "approval_rate": vlm_approved_count / sam_candidates_count if sam_candidates_count > 0 else 0,
            },
        }

        self.timing_log.append(timing_entry)

        # Save individual frame timing
        filename = self.dirs["timing"] / f"frame_{frame_index:06d}_timing.json"
        with open(filename, "w") as f:
            json.dump(timing_entry, f, indent=2)

    def save_summary(self) -> None:
        """Save overall summary and analysis."""
        if not self.enabled or not self.timing_log:
            return

        # Calculate statistics
        total_frames = len(self.timing_log)
        total_sam3_time = sum(t["timing"]["sam3_ms"] for t in self.timing_log)
        total_vlm_time = sum(t["timing"]["vlm_ms"] for t in self.timing_log)
        total_time = sum(t["timing"]["total_ms"] for t in self.timing_log)

        total_candidates = sum(t["counts"]["sam3_candidates"] for t in self.timing_log)
        total_approved = sum(t["counts"]["vlm_approved"] for t in self.timing_log)

        summary = {
            "session_info": {
                "total_frames": total_frames,
                "output_directory": str(self.output_dir),
                "timestamp": time.time(),
            },
            "timing_totals": {
                "sam3_total_ms": total_sam3_time,
                "vlm_total_ms": total_vlm_time,
                "total_processing_ms": total_time,
            },
            "timing_averages": {
                "sam3_avg_ms": total_sam3_time / total_frames if total_frames > 0 else 0,
                "vlm_avg_ms": total_vlm_time / total_frames if total_frames > 0 else 0,
                "total_avg_ms": total_time / total_frames if total_frames > 0 else 0,
                "fps": 1000 / (total_time / total_frames) if total_time > 0 else 0,
            },
            "detection_stats": {
                "total_sam3_candidates": total_candidates,
                "total_vlm_approved": total_approved,
                "overall_approval_rate": total_approved / total_candidates if total_candidates > 0 else 0,
                "avg_candidates_per_frame": total_candidates / total_frames if total_frames > 0 else 0,
                "avg_approved_per_frame": total_approved / total_frames if total_frames > 0 else 0,
            },
            "bottleneck_analysis": {
                "sam3_percentage": (total_sam3_time / total_time * 100) if total_time > 0 else 0,
                "vlm_percentage": (total_vlm_time / total_time * 100) if total_time > 0 else 0,
                "recommendation": self._get_bottleneck_recommendation(total_sam3_time, total_vlm_time),
            },
            "per_frame_data": self.timing_log,
        }

        # Save summary
        filename = self.dirs["summary"] / "session_summary.json"
        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary to console
        print("\n" + "=" * 80)
        print("DEBUG OUTPUT SUMMARY")
        print("=" * 80)
        print(f"Total frames processed: {total_frames}")
        print(f"Output directory: {self.output_dir}")
        print()
        print("TIMING ANALYSIS:")
        print(f"  SAM3:  {total_sam3_time:.1f}ms total, {summary['timing_averages']['sam3_avg_ms']:.1f}ms avg ({summary['bottleneck_analysis']['sam3_percentage']:.1f}%)")
        print(f"  VLM:   {total_vlm_time:.1f}ms total, {summary['timing_averages']['vlm_avg_ms']:.1f}ms avg ({summary['bottleneck_analysis']['vlm_percentage']:.1f}%)")
        print(f"  Total: {total_time:.1f}ms total, {summary['timing_averages']['total_avg_ms']:.1f}ms avg")
        print(f"  FPS:   {summary['timing_averages']['fps']:.2f}")
        print()
        print("DETECTION STATS:")
        print(f"  SAM3 candidates:  {total_candidates} total, {summary['detection_stats']['avg_candidates_per_frame']:.1f} avg/frame")
        print(f"  VLM approved:     {total_approved} total, {summary['detection_stats']['avg_approved_per_frame']:.1f} avg/frame")
        print(f"  Approval rate:    {summary['detection_stats']['overall_approval_rate']:.1%}")
        print()
        print("BOTTLENECK:")
        print(f"  {summary['bottleneck_analysis']['recommendation']}")
        print("=" * 80)

        return summary

    def _get_bottleneck_recommendation(self, sam3_time: float, vlm_time: float) -> str:
        """Analyze bottleneck and provide recommendation."""
        total = sam3_time + vlm_time
        if total == 0:
            return "No processing time recorded"

        sam3_pct = sam3_time / total * 100
        vlm_pct = vlm_time / total * 100

        if sam3_pct > 70:
            return f"SAM3 is the bottleneck ({sam3_pct:.1f}%). Consider: reducing defect classes, using smaller model, or GPU optimization."
        elif vlm_pct > 70:
            return f"VLM is the bottleneck ({vlm_pct:.1f}%). Consider: increasing VLM_EVERY_N_FRAMES, using faster model, or batching."
        else:
            return f"Balanced pipeline (SAM3: {sam3_pct:.1f}%, VLM: {vlm_pct:.1f}%)"

    def _get_color(self, index: int) -> tuple[int, int, int]:
        """Get color for detection index."""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
        return colors[index % len(colors)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_summary()
