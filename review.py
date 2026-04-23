"""Fault review tool: navigate faults, accept/reject, update JSON.

Usage:
    python review.py <fault_json> [--video <video_path>]

Keyboard controls:
    N - next fault
    P - previous fault
    A - accept fault
    R - reject fault
    Q - quit and save
"""

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np


def load_faults(json_path: str) -> tuple[dict, float]:
    """Load fault JSON and return (data, mtime).

    Returns:
        Tuple of (fault data dict, file modification time).
    """
    mtime = os.path.getmtime(json_path)
    with open(json_path) as f:
        data = json.load(f)
    return data, mtime


def save_faults(json_path: str, data: dict, original_mtime: float) -> bool:
    """Save fault JSON with mtime guard.

    Checks that the file hasn't been modified since we read it.
    Writes atomically via temp file + rename.

    Args:
        json_path: path to fault JSON.
        data: fault data to save.
        original_mtime: mtime when we first read the file.

    Returns:
        True if saved successfully, False if mtime mismatch.
    """
    current_mtime = os.path.getmtime(json_path)
    if abs(current_mtime - original_mtime) > 0.001:
        print(f"Warning: {json_path} was modified by another process!")
        print("Your changes were NOT saved. Reload and review again.")
        return False

    # Atomic write: write to temp, then rename
    dir_name = os.path.dirname(os.path.abspath(json_path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, json_path)
        return True
    except Exception:
        os.unlink(tmp_path)
        raise


def draw_fault_overlay(
    frame: np.ndarray,
    fault: dict,
    fault_idx: int,
    total_faults: int,
) -> np.ndarray:
    """Draw fault information overlay on frame."""
    display = frame.copy()

    # Draw crosshair at fault pixel location
    px, py = fault["pixel"]
    px, py = int(px), int(py)
    color = (0, 0, 255) if fault["confidence_tier"] == "AUTO_FAULT" else (0, 165, 255)

    cv2.circle(display, (px, py), 15, color, 2)
    cv2.line(display, (px - 20, py), (px + 20, py), color, 2)
    cv2.line(display, (px, py - 20), (px, py + 20), color, 2)

    # Info panel background
    panel_h = 200
    cv2.rectangle(display, (0, 0), (450, panel_h), (0, 0, 0), -1)
    cv2.rectangle(display, (0, 0), (450, panel_h), color, 2)

    # Info text
    y_offset = 25
    lines = [
        f"Fault {fault_idx + 1}/{total_faults}",
        f"Frame: {fault['frame_number']} ({fault['timestamp_seconds']:.2f}s)",
        f"Player: #{fault['player_track_id']} | Side: {fault['keypoint_side']}",
        f"Tier: {fault['confidence_tier']}",
        f"Confidence: {fault['composite_confidence']:.3f}",
        f"Ball: {fault['ball_state']} | Consec: {fault['consecutive_frames_in_zone']}",
        f"Decision: {fault.get('review_decision', 'PENDING')}",
        "[N]ext [P]rev [A]ccept [R]eject [Q]uit",
    ]

    for line in lines:
        cv2.putText(display, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 22

    return display


def run_review(json_path: str, video_path: str | None = None) -> None:
    """Run the interactive review tool."""
    data, mtime = load_faults(json_path)
    faults = data.get("faults", [])

    if not faults:
        print("No faults to review.")
        return

    # Determine video path
    if video_path is None:
        video_path = data.get("video_path")
    if video_path is None or not os.path.exists(video_path):
        print(f"Error: video not found at {video_path}")
        print("Specify video path with --video")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_idx = 0
    changed = False

    window_name = "Fault Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show_fault(idx: int) -> None:
        fault = faults[idx]
        frame_num = fault["frame_number"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: cannot read frame {frame_num}")
            return

        display = draw_fault_overlay(frame, fault, idx, len(faults))
        cv2.imshow(window_name, display)

    show_fault(current_idx)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            if current_idx < len(faults) - 1:
                current_idx += 1
                show_fault(current_idx)
        elif key == ord("p"):
            if current_idx > 0:
                current_idx -= 1
                show_fault(current_idx)
        elif key == ord("a"):
            faults[current_idx]["review_decision"] = "ACCEPTED"
            changed = True
            print(f"Fault {current_idx + 1}: ACCEPTED")
            show_fault(current_idx)
        elif key == ord("r"):
            faults[current_idx]["review_decision"] = "REJECTED"
            changed = True
            print(f"Fault {current_idx + 1}: REJECTED")
            show_fault(current_idx)

    cap.release()
    cv2.destroyAllWindows()

    # Save if changed
    if changed:
        data["faults"] = faults
        if save_faults(json_path, data, mtime):
            accepted = sum(1 for f in faults if f.get("review_decision") == "ACCEPTED")
            rejected = sum(1 for f in faults if f.get("review_decision") == "REJECTED")
            pending = sum(1 for f in faults if f.get("review_decision") is None)
            print(f"\nSaved to {json_path}")
            print(f"  Accepted: {accepted}")
            print(f"  Rejected: {rejected}")
            print(f"  Pending: {pending}")
    else:
        print("No changes made.")


def main():
    parser = argparse.ArgumentParser(description="Review kitchen fault detections")
    parser.add_argument("faults", help="Path to fault JSON file")
    parser.add_argument("--video", default=None, help="Path to video file (overrides JSON)")
    args = parser.parse_args()

    run_review(args.faults, args.video)


if __name__ == "__main__":
    main()
