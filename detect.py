"""Thin orchestrator: runs pose tracking, ball detection, and fault correlation.

Usage:
    python detect.py <video_path> --calibration <config.json> [--pose-model <path>] [--ball-model <path>]

Outputs:
    output/{video_name}_faults.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from pickleball.annotate import annotate_frame
from pickleball.ball import (
    LIVE,
    UNKNOWN,
    BallStateMachine,
    classify_bounce_side,
    compute_vertical_velocity,
    detect_bounces,
    interpolate_positions,
)
from pickleball.constants import (
    BALL_UNKNOWN_GAP_FRAMES,
    CONSECUTIVE_FRAMES_MIN,
    scale_frame_threshold,
)
from pickleball.fault import correlate_fault
from pickleball.pose import (
    check_player_in_kitchen,
    compute_homography,
    get_ankle_confidence,
    transform_to_court,
)


def load_calibration(config_path: str) -> dict:
    """Load and validate calibration config."""
    with open(config_path) as f:
        config = json.load(f)

    required = ["pixel_corners", "homography_matrix", "net_x_pixel", "input_resolution"]
    for key in required:
        if key not in config:
            print(f"Error: calibration missing '{key}'")
            sys.exit(1)

    return config


def validate_resolution(config: dict, video_w: int, video_h: int) -> None:
    """Check video resolution matches calibration."""
    cal_w, cal_h = config["input_resolution"]
    if cal_w != video_w or cal_h != video_h:
        print(f"Error: resolution mismatch.")
        print(f"  Calibration: {cal_w}x{cal_h}")
        print(f"  Video: {video_w}x{video_h}")
        print("Re-run calibrate.py with a frame from this video.")
        sys.exit(1)


def run_detection(
    video_path: str,
    config_path: str,
    pose_model_path: str = "models/yolov8m-pose.pt",
    ball_model_path: str | None = None,
    debug_video_path: str | None = None,
) -> str:
    """Run the full detection pipeline.

    Returns:
        Path to output fault JSON.
    """
    # Load calibration
    config = load_calibration(config_path)
    H = np.array(config["homography_matrix"], dtype=np.float64)
    net_x_pixel = config["net_x_pixel"]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    validate_resolution(config, video_w, video_h)

    # Scale frame thresholds for actual fps
    min_consecutive = scale_frame_threshold(CONSECUTIVE_FRAMES_MIN, fps)
    unknown_gap = scale_frame_threshold(BALL_UNKNOWN_GAP_FRAMES, fps)

    # Load models
    pose_model = YOLO(pose_model_path)

    ball_model = None
    if ball_model_path:
        ball_model = YOLO(ball_model_path)

    # ── Pass 1: Collect pose and ball detections ─────────────────────────────
    # Per-frame pose data: {track_id: keypoints}
    pose_data = []
    # Per-frame ball detection: {x, y} or None
    ball_detections: list[dict | None] = []

    print(f"Processing {total_frames} frames at {fps:.1f} fps...")

    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        ret, frame = cap.read()
        if not ret:
            break

        # Pose tracking
        pose_results = pose_model.track(frame, persist=True, verbose=False)
        frame_poses = {}

        if pose_results and pose_results[0].boxes is not None:
            boxes = pose_results[0].boxes
            keypoints_all = pose_results[0].keypoints

            if keypoints_all is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                kps = keypoints_all.data.cpu().numpy()  # (N, num_kp, 3)

                for i, track_id in enumerate(ids):
                    frame_poses[int(track_id)] = kps[i]

        pose_data.append(frame_poses)

        # Ball detection
        ball_det = None
        if ball_model is not None:
            ball_results = ball_model(frame, verbose=False)
            if ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
                # Take highest confidence detection
                boxes = ball_results[0].boxes
                best_idx = boxes.conf.argmax()
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                conf = float(boxes.conf[best_idx])

                # Transform to court coords
                court_x, court_y = transform_to_court((float(cx), float(cy)), H)

                ball_det = {
                    "x": float(cx),
                    "y": float(cy),
                    "court_x": court_x,
                    "court_y": court_y,
                    "conf": conf,
                }

        ball_detections.append(ball_det)

    cap.release()

    # ── Ball post-processing ─────────────────────────────────────────────────
    ball_detections = interpolate_positions(ball_detections)
    velocities = compute_vertical_velocity(ball_detections)
    bounce_frames = detect_bounces(ball_detections, velocities)

    # Build bounce events list for annotation (frame index + side)
    bounce_events = []
    for bf in bounce_frames:
        side = classify_bounce_side(bf, ball_detections)
        bounce_events.append({"frame": bf, "side": side})

    # Check ball detection rate for fallback mode
    detected_count = sum(1 for d in ball_detections if d is not None and not d.get("interpolated"))
    detection_rate = detected_count / max(total_frames, 1)
    fallback_mode = ball_model is None or detection_rate < 0.4

    if fallback_mode and ball_model is not None:
        print(f"Warning: ball detection rate {detection_rate:.1%} < 40%. Using fallback mode.")
        print("All kitchen entries will be REVIEW_NEEDED.")

    # Build ball state machine
    ball_sm = BallStateMachine(unknown_gap_frames=unknown_gap)

    # Pre-compute ball states per frame
    ball_states_per_frame = []
    bounce_set = set(bounce_frames)

    for frame_idx in range(len(ball_detections)):
        det = ball_detections[frame_idx]
        detected = det is not None

        ball_side = None
        if detected and "court_y" in det:
            ball_side = "left" if det["court_y"] < 13.5 else "right"

        ball_sm.update_detection(detected, ball_side=ball_side)

        if frame_idx in bounce_set:
            side = classify_bounce_side(frame_idx, ball_detections)
            if side:
                ball_sm.update_bounce(side)

        ball_states_per_frame.append({
            "left": ball_sm.get_state("left"),
            "right": ball_sm.get_state("right"),
        })

    # ── Pass 2: Fault correlation ────────────────────────────────────────────
    # Track consecutive frames per player per side
    # Key: (track_id, side) -> consecutive frame count
    consecutive_tracker: dict[tuple[int, str], int] = {}
    faults = []
    fault_id = 0
    fault_frame_set: set[int] = set()  # frame indices with at least one fault

    for frame_idx in range(len(pose_data)):
        frame_poses = pose_data[frame_idx]
        frame_states = ball_states_per_frame[frame_idx] if frame_idx < len(ball_states_per_frame) else {"left": UNKNOWN, "right": UNKNOWN}

        # Track which (track_id, zone) pairs are active this frame
        active_this_frame = set()

        for track_id, keypoints in frame_poses.items():
            hits = check_player_in_kitchen(keypoints, H)
            ankle_conf = get_ankle_confidence(keypoints)

            for hit in hits:
                zone = hit["zone"]
                key = (track_id, zone)
                active_this_frame.add(key)

                consecutive_tracker[key] = consecutive_tracker.get(key, 0) + 1
                consec = consecutive_tracker[key]

                # Determine ball state and confidence
                if fallback_mode:
                    ball_state = UNKNOWN
                    ball_conf = 0.0
                else:
                    ball_state = frame_states.get(zone, UNKNOWN)
                    # Ball confidence from detection, or 1.0 if state came from rules
                    ball_det = ball_detections[frame_idx] if frame_idx < len(ball_detections) else None
                    ball_conf = ball_det["conf"] if ball_det is not None and "conf" in ball_det else 0.0
                    if ball_state in (LIVE, "BOUNCED") and ball_conf == 0.0:
                        ball_conf = 1.0  # state inferred from rules

                result = correlate_fault(
                    zone_hit=hit,
                    ball_state=ball_state,
                    consecutive_frames=consec,
                    ankle_conf=ankle_conf,
                    ball_conf=ball_conf,
                    min_consecutive=min_consecutive,
                )

                if result is not None:
                    fault_id += 1
                    fault_entry = {
                        "fault_id": fault_id,
                        "frame_number": frame_idx,
                        "timestamp_seconds": round(frame_idx / fps, 3),
                        "player_track_id": track_id,
                        **result,
                        "review_decision": None,
                    }
                    faults.append(fault_entry)
                    fault_frame_set.add(frame_idx)

        # Reset consecutive count for players who left the zone
        for key in list(consecutive_tracker.keys()):
            if key not in active_this_frame:
                del consecutive_tracker[key]

    # ── Output ───────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/{video_name}_faults.json"

    output = {
        "schema_version": 2,
        "video_path": os.path.abspath(video_path),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "calibration_file": os.path.abspath(config_path),
        "total_frames": total_frames,
        "fps": fps,
        "ball_detection_rate": round(detection_rate, 4),
        "fallback_mode": fallback_mode,
        "faults": faults,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"  Total faults: {len(faults)}")
    auto = sum(1 for f in faults if f["confidence_tier"] == "AUTO_FAULT")
    review = sum(1 for f in faults if f["confidence_tier"] == "REVIEW_NEEDED")
    print(f"  AUTO_FAULT: {auto}")
    print(f"  REVIEW_NEEDED: {review}")
    if fallback_mode:
        print(f"  (fallback mode: ball detection rate {detection_rate:.1%})")

    # ── Pass 3: Debug video (optional) ───────────────────────────────────────
    if debug_video_path:
        _write_debug_video(
            video_path=video_path,
            debug_video_path=debug_video_path,
            total_frames=total_frames,
            fps=fps,
            video_w=video_w,
            video_h=video_h,
            config=config,
            H=H,
            pose_data=pose_data,
            ball_detections=ball_detections,
            ball_states_per_frame=ball_states_per_frame,
            bounce_events=bounce_events,
            fault_frame_set=fault_frame_set,
            fallback_mode=fallback_mode,
        )

    return output_path


def _write_debug_video(
    video_path: str,
    debug_video_path: str,
    total_frames: int,
    fps: float,
    video_w: int,
    video_h: int,
    config: dict,
    H: np.ndarray,
    pose_data: list[dict],
    ball_detections: list,
    ball_states_per_frame: list[dict],
    bounce_events: list[dict],
    fault_frame_set: set[int],
    fallback_mode: bool,
) -> None:
    """Write annotated debug video. Called after all detection passes complete."""
    os.makedirs(os.path.dirname(debug_video_path) if os.path.dirname(debug_video_path) else ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (video_w, video_h))

    if not writer.isOpened():
        print(f"Warning: could not open VideoWriter for {debug_video_path}")
        cap.release()
        return

    ball_trail: list[tuple[int, int]] = []

    print(f"\nWriting debug video to {debug_video_path}...")
    for frame_idx in tqdm(range(total_frames), desc="Annotating"):
        ret, frame = cap.read()
        if not ret:
            break

        # Update ball trail
        det = ball_detections[frame_idx] if frame_idx < len(ball_detections) else None
        if det is not None:
            ball_trail.append((int(det["x"]), int(det["y"])))
            if len(ball_trail) > 10:
                ball_trail = ball_trail[-10:]

        ball_states = (
            ball_states_per_frame[frame_idx]
            if frame_idx < len(ball_states_per_frame)
            else {"left": UNKNOWN, "right": UNKNOWN}
        )

        annotated = annotate_frame(
            frame=frame,
            frame_idx=frame_idx,
            config=config,
            pose_data_frame=pose_data[frame_idx] if frame_idx < len(pose_data) else {},
            ball_trail=list(ball_trail),
            ball_states=ball_states,
            bounce_events=bounce_events,
            fault_frame_set=fault_frame_set,
            H=H,
            fallback_mode=fallback_mode,
        )
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Debug video saved to {debug_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect kitchen faults in pickleball video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--calibration", required=True, help="Path to calibration JSON")
    parser.add_argument("--pose-model", default="models/yolov8m-pose.pt", help="Path to pose model")
    parser.add_argument("--ball-model", default=None, help="Path to ball detection model")
    parser.add_argument("--debug-video", default=None, metavar="PATH",
                        help="Write annotated debug video to PATH (e.g. output/debug.mp4)")
    args = parser.parse_args()

    run_detection(args.video, args.calibration, args.pose_model, args.ball_model, args.debug_video)


if __name__ == "__main__":
    main()
