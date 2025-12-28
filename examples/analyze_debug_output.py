"""
Analyze debug output to identify performance bottlenecks and optimization opportunities.

This script parses the debug output from a session and generates analysis reports.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze debug output")
    parser.add_argument("debug_dir", type=str, help="Debug output directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-frame analysis")
    return parser.parse_args()


def load_session_summary(debug_dir: Path) -> dict:
    """Load the session summary JSON."""
    summary_file = debug_dir / "00_summary" / "session_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Session summary not found: {summary_file}")

    with open(summary_file) as f:
        return json.load(f)


def load_timing_files(debug_dir: Path) -> list[dict]:
    """Load all timing JSON files."""
    timing_dir = debug_dir / "06_timing"
    if not timing_dir.exists():
        return []

    timing_files = sorted(timing_dir.glob("frame_*_timing.json"))
    timings = []

    for file in timing_files:
        with open(file) as f:
            timings.append(json.load(f))

    return timings


def load_vlm_responses(debug_dir: Path) -> list[dict]:
    """Load all VLM response JSON files."""
    vlm_dir = debug_dir / "04_vlm_responses"
    if not vlm_dir.exists():
        return []

    vlm_files = sorted(vlm_dir.glob("response_frame_*.json"))
    responses = []

    for file in vlm_files:
        with open(file) as f:
            responses.append(json.load(f))

    return responses


def analyze_timing_trends(timings: list[dict]) -> dict:
    """Analyze timing trends across frames."""
    if not timings:
        return {}

    sam3_times = [t["timing"]["sam3_ms"] for t in timings]
    vlm_times = [t["timing"]["vlm_ms"] for t in timings]
    total_times = [t["timing"]["total_ms"] for t in timings]

    return {
        "sam3": {
            "min": min(sam3_times),
            "max": max(sam3_times),
            "mean": statistics.mean(sam3_times),
            "median": statistics.median(sam3_times),
            "stdev": statistics.stdev(sam3_times) if len(sam3_times) > 1 else 0,
        },
        "vlm": {
            "min": min(vlm_times),
            "max": max(vlm_times),
            "mean": statistics.mean(vlm_times),
            "median": statistics.median(vlm_times),
            "stdev": statistics.stdev(vlm_times) if len(vlm_times) > 1 else 0,
        },
        "total": {
            "min": min(total_times),
            "max": max(total_times),
            "mean": statistics.mean(total_times),
            "median": statistics.median(total_times),
            "stdev": statistics.stdev(total_times) if len(total_times) > 1 else 0,
        },
    }


def analyze_defect_distribution(debug_dir: Path) -> dict:
    """Analyze distribution of detected defect types."""
    sam3_dir = debug_dir / "02_sam3_candidates"
    if not sam3_dir.exists():
        return {}

    defect_counts = defaultdict(int)
    summary_files = sam3_dir.glob("frame_*_summary.json")

    for file in summary_files:
        with open(file) as f:
            data = json.load(f)
            for defect_type in data.get("defect_types_found", []):
                defect_counts[defect_type] += 1

    return dict(sorted(defect_counts.items(), key=lambda x: x[1], reverse=True))


def analyze_vlm_latency(responses: list[dict]) -> dict:
    """Analyze VLM latency and frame delays."""
    if not responses:
        return {}

    latencies = [r["latency_ms"] for r in responses]
    frame_delays = [r["frame_delay"] for r in responses]

    return {
        "latency": {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
        },
        "frame_delay": {
            "min": min(frame_delays),
            "max": max(frame_delays),
            "mean": statistics.mean(frame_delays),
            "median": statistics.median(frame_delays),
            "max_observed": max(frame_delays),
        },
    }


def generate_recommendations(summary: dict, timing_trends: dict, vlm_analysis: dict) -> list[str]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []

    # Bottleneck analysis
    bottleneck = summary.get("bottleneck_analysis", {})
    sam3_pct = bottleneck.get("sam3_percentage", 0)
    vlm_pct = bottleneck.get("vlm_percentage", 0)

    if sam3_pct > 70:
        recommendations.append(
            f"🔴 SAM3 BOTTLENECK ({sam3_pct:.1f}%): "
            "Consider reducing defect classes or using GPU acceleration"
        )
    elif vlm_pct > 70:
        recommendations.append(
            f"🔴 VLM BOTTLENECK ({vlm_pct:.1f}%): "
            "Consider increasing VLM_EVERY_N_FRAMES or using a faster model"
        )
    else:
        recommendations.append(
            f"🟢 BALANCED PIPELINE: SAM3 {sam3_pct:.1f}%, VLM {vlm_pct:.1f}%"
        )

    # FPS analysis
    fps = summary.get("timing_averages", {}).get("fps", 0)
    if fps < 1.0:
        recommendations.append(
            f"⚠️  LOW FPS ({fps:.2f}): Pipeline is too slow for real-time processing"
        )
    elif fps < 5.0:
        recommendations.append(
            f"⚠️  MODERATE FPS ({fps:.2f}): May struggle with high-resolution video"
        )
    else:
        recommendations.append(f"✓ GOOD FPS ({fps:.2f}): Suitable for real-time use")

    # Approval rate analysis
    detection_stats = summary.get("detection_stats", {})
    approval_rate = detection_stats.get("overall_approval_rate", 0)

    if approval_rate < 0.3:
        recommendations.append(
            f"⚠️  LOW APPROVAL RATE ({approval_rate:.1%}): "
            "SAM3 generating too many false positives - consider adjusting defect classes"
        )
    elif approval_rate > 0.9:
        recommendations.append(
            f"ℹ️  HIGH APPROVAL RATE ({approval_rate:.1%}): "
            "Could reduce VLM frequency since SAM3 quality is already high"
        )

    # VLM latency analysis
    if vlm_analysis:
        max_delay = vlm_analysis.get("frame_delay", {}).get("max_observed", 0)
        if max_delay > 60:
            recommendations.append(
                f"🔴 HIGH VLM DELAY ({max_delay} frames): "
                "VLM too slow, predictions arrive very late"
            )
        elif max_delay > 30:
            recommendations.append(
                f"⚠️  MODERATE VLM DELAY ({max_delay} frames): "
                "Consider using faster VLM model"
            )

    return recommendations


def print_report(summary: dict, timing_trends: dict, defect_dist: dict, vlm_analysis: dict, detailed: bool):
    """Print formatted analysis report."""
    print("=" * 80)
    print("DEBUG OUTPUT ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Session info
    session = summary.get("session_info", {})
    print("SESSION INFO:")
    print(f"  Total frames: {session.get('total_frames', 0)}")
    print(f"  Output directory: {session.get('output_directory', 'N/A')}")
    print()

    # Timing summary
    print("TIMING SUMMARY:")
    timing_avg = summary.get("timing_averages", {})
    print(f"  Average SAM3:  {timing_avg.get('sam3_avg_ms', 0):.1f}ms")
    print(f"  Average VLM:   {timing_avg.get('vlm_avg_ms', 0):.1f}ms")
    print(f"  Average Total: {timing_avg.get('total_avg_ms', 0):.1f}ms")
    print(f"  FPS:           {timing_avg.get('fps', 0):.2f}")
    print()

    # Detection stats
    print("DETECTION STATS:")
    detection = summary.get("detection_stats", {})
    print(f"  SAM3 candidates:  {detection.get('total_sam3_candidates', 0)} "
          f"({detection.get('avg_candidates_per_frame', 0):.1f} avg/frame)")
    print(f"  VLM approved:     {detection.get('total_vlm_approved', 0)} "
          f"({detection.get('avg_approved_per_frame', 0):.1f} avg/frame)")
    print(f"  Approval rate:    {detection.get('overall_approval_rate', 0):.1%}")
    print()

    # Timing trends
    if timing_trends:
        print("TIMING TRENDS:")
        for stage in ["sam3", "vlm", "total"]:
            stats = timing_trends.get(stage, {})
            print(f"  {stage.upper()}:")
            print(f"    Min:    {stats.get('min', 0):.1f}ms")
            print(f"    Max:    {stats.get('max', 0):.1f}ms")
            print(f"    Mean:   {stats.get('mean', 0):.1f}ms")
            print(f"    Median: {stats.get('median', 0):.1f}ms")
            print(f"    StdDev: {stats.get('stdev', 0):.1f}ms")
        print()

    # Defect distribution
    if defect_dist:
        print("DEFECT TYPE DISTRIBUTION:")
        total_detections = sum(defect_dist.values())
        for defect_type, count in list(defect_dist.items())[:10]:  # Top 10
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"  {defect_type:20s}: {count:4d} ({percentage:5.1f}%)")
        print()

    # VLM analysis
    if vlm_analysis:
        print("VLM LATENCY ANALYSIS:")
        latency = vlm_analysis.get("latency", {})
        print(f"  Min:    {latency.get('min', 0):.1f}ms")
        print(f"  Max:    {latency.get('max', 0):.1f}ms")
        print(f"  Mean:   {latency.get('mean', 0):.1f}ms")
        print(f"  Median: {latency.get('median', 0):.1f}ms")

        delay = vlm_analysis.get("frame_delay", {})
        print(f"\n  Frame Delay:")
        print(f"    Min:    {delay.get('min', 0)} frames")
        print(f"    Max:    {delay.get('max_observed', 0)} frames")
        print(f"    Mean:   {delay.get('mean', 0):.1f} frames")
        print(f"    Median: {delay.get('median', 0):.1f} frames")
        print()

    # Recommendations
    recommendations = generate_recommendations(summary, timing_trends, vlm_analysis)
    print("RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  {rec}")
    print()

    print("=" * 80)


def main():
    args = parse_args()
    debug_dir = Path(args.debug_dir)

    if not debug_dir.exists():
        print(f"Error: Debug directory not found: {debug_dir}")
        return

    print(f"Analyzing: {debug_dir}")
    print()

    # Load data
    summary = load_session_summary(debug_dir)
    timings = load_timing_files(debug_dir)
    vlm_responses = load_vlm_responses(debug_dir)

    # Run analysis
    timing_trends = analyze_timing_trends(timings)
    defect_dist = analyze_defect_distribution(debug_dir)
    vlm_analysis = analyze_vlm_latency(vlm_responses)

    # Print report
    print_report(summary, timing_trends, defect_dist, vlm_analysis, args.detailed)

    # Detailed frame-by-frame if requested
    if args.detailed and timings:
        print("DETAILED FRAME ANALYSIS:")
        print("-" * 80)
        for timing in timings:
            frame_idx = timing["frame_index"]
            sam3_ms = timing["timing"]["sam3_ms"]
            vlm_ms = timing["timing"]["vlm_ms"]
            total_ms = timing["timing"]["total_ms"]
            candidates = timing["counts"]["sam3_candidates"]
            approved = timing["counts"]["vlm_approved"]

            print(f"Frame {frame_idx:4d}: "
                  f"SAM3={sam3_ms:6.1f}ms, VLM={vlm_ms:6.1f}ms, Total={total_ms:6.1f}ms, "
                  f"Candidates={candidates:2d}, Approved={approved:2d}")


if __name__ == "__main__":
    main()
