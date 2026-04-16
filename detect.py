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

import pickleball._cuda_preload  # noqa: F401  — must precede onnxruntime/rtmlib imports

import cv2
import numpy as np
import torch
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
    CONSECUTIVE_GAP_TOLERANCE,
    scale_frame_threshold,
)
from pickleball.fault import correlate_fault
from pickleball.pose import (
    check_player_in_kitchen,
    get_pose_confidence,
    is_wholebody_model,
)
from pickleball.pose_backend import make_backend


def load_calibration(config_path: str) -> dict:
    """Load and validate calibration config."""
    with open(config_path) as f:
        config = json.load(f)

    required = ["kitchen_polygon", "net_x_pixel", "input_resolution"]
    for key in required:
        if key not in config:
            print(f"Error: calibration missing '{key}'. "
                  "Re-run calibrate.py — the schema has changed (4-corner kitchen polygon).")
            sys.exit(1)

    if len(config["kitchen_polygon"]) < 3:
        print("Error: kitchen_polygon must have at least 3 points.")
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
    pose_model_path: str = "models/yolov8x-pose-p6.pt",
    ball_model_path: str | None = None,
    debug_video_path: str | None = None,
    imgsz: int = 640,
    ball_stride: int = 2,
    half: bool | None = None,
    pose_backend: str = "wholebody",
    detector_model_path: str = "models/yolov8s.pt",
    wholebody_mode: str = "balanced",
) -> str:
    """Run the full detection pipeline.

    Returns:
        Path to output fault JSON.
    """
    # Load calibration
    config = load_calibration(config_path)
    polygon = np.array(config["kitchen_polygon"], dtype=np.int32)
    net_x_pixel = float(config["net_x_pixel"])

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
    gap_tolerance = scale_frame_threshold(CONSECUTIVE_GAP_TOLERANCE, fps)

    # fp16 inference only benefits CUDA. On CPU/MPS ultralytics falls back to fp32.
    if half is None:
        half = torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # All ultralytics weights live under models/. Ensure the dir exists so
    # ultralytics auto-download can write there for missing weights.
    os.makedirs("models", exist_ok=True)

    # Load pose backend
    backend = make_backend(
        pose_backend,
        pose_model_path=pose_model_path,
        detector_model_path=detector_model_path,
        wholebody_mode=wholebody_mode,
        device=device,
    )

    ball_model = None
    if ball_model_path:
        ball_model = YOLO(ball_model_path)

    infer_kwargs = {"imgsz": imgsz, "half": half, "verbose": False}

    # ── Pass 1: Collect pose and ball detections ─────────────────────────────
    pose_data = []
    ball_detections: list[dict | None] = []

    _model_type_warned = False

    print(f"Processing {total_frames} frames at {fps:.1f} fps...")

    for frame_idx in tqdm(range(total_frames), desc="Detecting"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_poses = backend.track(frame, imgsz=imgsz, half=half)

        if not _model_type_warned and frame_poses:
            _model_type_warned = True
            first_kps = next(iter(frame_poses.values()))
            if not is_wholebody_model(first_kps):
                print(
                    "Warning: COCO 17-keypoint model detected. "
                    "Ankle fallback active. "
                    "For accurate shoe-tip detection, run with --pose-backend wholebody."
                )

        pose_data.append(frame_poses)

        # Ball detection (skip every Nth frame; interpolation fills the gaps)
        ball_det = None
        if ball_model is not None and (frame_idx % ball_stride == 0):
            ball_results = ball_model(frame, **infer_kwargs)
            if ball_results and ball_results[0].boxes is not None and len(ball_results[0].boxes) > 0:
                boxes = ball_results[0].boxes
                best_idx = boxes.conf.argmax()
                x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                conf = float(boxes.conf[best_idx])

                ball_det = {
                    "x": float(cx),
                    "y": float(cy),
                    "conf": conf,
                }

        ball_detections.append(ball_det)

    cap.release()

    # ── Ball post-processing ─────────────────────────────────────────────────
    ball_detections = interpolate_positions(ball_detections)
    velocities = compute_vertical_velocity(ball_detections)
    bounce_frames = detect_bounces(ball_detections, velocities)

    bounce_events = []
    for bf in bounce_frames:
        side = classify_bounce_side(bf, ball_detections, net_x_pixel)
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

    ball_states_per_frame = []
    bounce_set = set(bounce_frames)

    for frame_idx in range(len(ball_detections)):
        det = ball_detections[frame_idx]
        detected = det is not None

        ball_side = None
        if detected:
            ball_side = "left" if det["x"] < net_x_pixel else "right"

        ball_sm.update_detection(detected, ball_side=ball_side)

        if frame_idx in bounce_set:
            side = classify_bounce_side(frame_idx, ball_detections, net_x_pixel)
            if side:
                ball_sm.update_bounce(side)

        ball_states_per_frame.append({
            "left": ball_sm.get_state("left"),
            "right": ball_sm.get_state("right"),
        })

    # ── Pass 2: Fault correlation ────────────────────────────────────────────
    # Consecutive-frame streak per track_id, with a small gap tolerance so that
    # single-frame tracker flickers don't reset a legitimate streak.
    consecutive_tracker: dict[int, int] = {}
    gap_seen: dict[int, int] = {}
    faults = []
    fault_id = 0
    fault_frame_set: set[int] = set()

    for frame_idx in range(len(pose_data)):
        frame_poses = pose_data[frame_idx]
        frame_states = ball_states_per_frame[frame_idx] if frame_idx < len(ball_states_per_frame) else {"left": UNKNOWN, "right": UNKNOWN}

        active_this_frame: set[int] = set()

        for track_id, keypoints in frame_poses.items():
            hits = check_player_in_kitchen(keypoints, polygon, net_x_pixel=net_x_pixel)
            if not hits:
                continue

            pose_conf = get_pose_confidence(keypoints)
            active_this_frame.add(track_id)
            gap_seen[track_id] = 0
            consecutive_tracker[track_id] = consecutive_tracker.get(track_id, 0) + 1
            consec = consecutive_tracker[track_id]

            # Pick the ball state for the side of the net the foot is on.
            # Default to left if foot_side is missing.
            foot_sides = {h.get("foot_side") for h in hits if h.get("foot_side")}
            foot_side = next(iter(foot_sides), "left")

            if fallback_mode:
                ball_state = UNKNOWN
                ball_conf = 1.0
            else:
                ball_state = frame_states.get(foot_side, UNKNOWN)
                ball_det = ball_detections[frame_idx] if frame_idx < len(ball_detections) else None
                ball_conf = ball_det["conf"] if ball_det is not None and "conf" in ball_det else 0.0
                if ball_state in (LIVE, "BOUNCED") and ball_conf == 0.0:
                    ball_conf = 1.0
                elif ball_state == UNKNOWN:
                    ball_conf = 1.0

            # Emit one fault per hit this frame (one per foot keypoint that's inside).
            for hit in hits:
                result = correlate_fault(
                    zone_hit=hit,
                    ball_state=ball_state,
                    consecutive_frames=consec,
                    ankle_conf=pose_conf,
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

        # Decay inactive keys; delete only after tolerance exceeded.
        for key in list(consecutive_tracker.keys()):
            if key in active_this_frame:
                continue
            gap_seen[key] = gap_seen.get(key, 0) + 1
            if gap_seen[key] > gap_tolerance:
                del consecutive_tracker[key]
                del gap_seen[key]

    # ── Output ───────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"output/{video_name}_faults.json"

    output = {
        "schema_version": 3,
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
            polygon=polygon,
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
    polygon: np.ndarray,
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
            polygon=polygon,
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
    parser.add_argument("--pose-model", default="models/yolov8x-pose-p6.pt",
                        help="Path to ultralytics pose model. Used when --pose-backend=ultralytics. "
                             "COCO 17-kp models use ankle keypoints (WholeBody preferred for accuracy).")
    parser.add_argument("--pose-backend", choices=["ultralytics", "wholebody"], default="wholebody",
                        help="'wholebody' (default) = ultralytics person tracker + rtmlib RTMPose WholeBody "
                             "(133 kp, true shoe-tip keypoints). "
                             "'ultralytics' = single-model YOLO pose (COCO 17 kp, ankle-based zone check).")
    parser.add_argument("--detector-model", default="models/yolov8s.pt",
                        help="Person detector for --pose-backend=wholebody (auto-downloads via ultralytics into models/). "
                             "Default models/yolov8s.pt is plenty for 2-4 large subjects on a static court; "
                             "use models/yolov8x.pt only if you see missed detections.")
    parser.add_argument("--wholebody-mode", choices=["performance", "balanced", "lightweight"],
                        default="balanced",
                        help="rtmlib pose tier. Default 'balanced' (DWPose-L @ 192x256). "
                             "'performance' = DWPose-L @ 288x384 (slower, marginal accuracy gain). "
                             "'lightweight' = RTMPose-M (fastest, for mobile or low-end CPU).")
    parser.add_argument("--ball-model", default=None, help="Path to ball detection model")
    parser.add_argument("--debug-video", default=None, metavar="PATH",
                        help="Write annotated debug video to PATH (e.g. output/debug.mp4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Detector inference image size (long edge). Lower = faster, less accurate. "
                             "Affects ultralytics detector + ball model only; rtmlib pose has its own input size. Default 640.")
    parser.add_argument("--ball-stride", type=int, default=2,
                        help="Run ball detection every Nth frame; interpolate between. Default 2.")
    parser.add_argument("--half", dest="half", action="store_true", default=None,
                        help="Force fp16 inference. Default: auto (on when CUDA available).")
    parser.add_argument("--no-half", dest="half", action="store_false",
                        help="Force fp32 inference.")
    args = parser.parse_args()

    run_detection(
        args.video, args.calibration, args.pose_model, args.ball_model, args.debug_video,
        imgsz=args.imgsz, ball_stride=args.ball_stride, half=args.half,
        pose_backend=args.pose_backend, detector_model_path=args.detector_model,
        wholebody_mode=args.wholebody_mode,
    )


if __name__ == "__main__":
    main()
