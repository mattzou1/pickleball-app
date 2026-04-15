"""Court calibration tool: click 8 corners + 2 net points, compute homography.

Usage:
    python calibrate.py <video_path>

Opens video with frame picker. Use WASD to navigate to a frame where both kitchen
zones are clearly visible, then press Enter to confirm. User clicks 8 court corners
in order:
    1-4: Left kitchen (BL, BR, TR, TL)
    5-8: Right kitchen (BL, BR, TR, TL)
    9-10: Net left and right top corners

Saves calibration JSON to configs/{video_name}.json.
Draws validation overlay showing kitchen zones projected back onto frame.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np

from pickleball.constants import WORLD_CORNERS
from pickleball.pose import compute_homography


CORNER_LABELS = [
    "Left kitchen: baseline left corner",
    "Left kitchen: baseline right corner",
    "Left kitchen: kitchen line right corner",
    "Left kitchen: kitchen line left corner",
    "Right kitchen: kitchen line left corner",
    "Right kitchen: kitchen line right corner",
    "Right kitchen: baseline right corner",
    "Right kitchen: baseline left corner",
    "Net left top corner",
    "Net right top corner",
]

CORNER_COLORS = [
    (0, 255, 0),    # green for left kitchen
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 0, 0),    # blue for right kitchen
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 0),
    (0, 255, 255),  # yellow for net
    (0, 255, 255),
]


class CalibrationState:
    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.done = False

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 10:
            return

        self.points.append((x, y))
        idx = len(self.points) - 1

        # Draw point and label
        color = CORNER_COLORS[idx]
        cv2.circle(self.display, (x, y), 5, color, -1)
        cv2.putText(
            self.display,
            f"{idx + 1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Draw lines connecting corners
        if idx > 0 and idx < 4:
            cv2.line(self.display, self.points[idx - 1], (x, y), (0, 255, 0), 1)
        if idx == 3:
            cv2.line(self.display, (x, y), self.points[0], (0, 255, 0), 1)
        if idx > 4 and idx < 8:
            cv2.line(self.display, self.points[idx - 1], (x, y), (255, 0, 0), 1)
        if idx == 7:
            cv2.line(self.display, (x, y), self.points[4], (255, 0, 0), 1)

        if len(self.points) >= 10:
            self.done = True


def draw_validation_overlay(frame: np.ndarray, H: np.ndarray, pixel_corners: list) -> np.ndarray:
    """Draw kitchen zones and net line projected back onto frame for visual validation."""
    overlay = frame.copy()

    # Draw left kitchen zone (green, semi-transparent)
    left_pts = np.array(pixel_corners[:4], dtype=np.int32)
    cv2.fillPoly(overlay, [left_pts], (0, 255, 0))

    # Draw right kitchen zone (blue, semi-transparent)
    right_pts = np.array(pixel_corners[4:8], dtype=np.int32)
    cv2.fillPoly(overlay, [right_pts], (255, 0, 0))

    # Blend
    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw corner points
    for i, pt in enumerate(pixel_corners[:8]):
        color = CORNER_COLORS[i]
        cv2.circle(result, (int(pt[0]), int(pt[1])), 8, color, -1)
        cv2.putText(result, str(i + 1), (int(pt[0]) + 12, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw net line between the two net corner points
    if len(pixel_corners) >= 10:
        net_left = pixel_corners[8]
        net_right = pixel_corners[9]
        pt_left = (int(net_left[0]), int(net_left[1]))
        pt_right = (int(net_right[0]), int(net_right[1]))
        cv2.line(result, pt_left, pt_right, (0, 255, 255), 3)
        cv2.circle(result, pt_left, 8, (0, 255, 255), -1)
        cv2.circle(result, pt_right, 8, (0, 255, 255), -1)
        cv2.putText(result, "NET L", (pt_left[0] + 10, pt_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(result, "NET R", (pt_right[0] + 10, pt_right[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(result, "Validation overlay - press any key to save, 'r' to redo",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def pick_calibration_frame(cap: cv2.VideoCapture, total_frames: int) -> np.ndarray:
    """Navigate through video frames to pick one where the kitchen zones are visible.

    Keys: d=+1 frame, a=-1 frame, s=+30 frames, w=-30 frames, Enter/Space=confirm, q=quit.

    Note: seeking is approximate for compressed video (H.264/H.265) — display snaps
    to the nearest keyframe. Use large steps (s/w) to navigate efficiently.

    Args:
        cap: open VideoCapture. Caller is responsible for releasing it.
        total_frames: total frame count (0 if unknown).

    Returns:
        The confirmed frame as a numpy array.
    """
    frame_idx = 0
    last_good_frame = None
    window_name = "Select frame — a/d: ±1  w/s: ±30  Enter: confirm  q: quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            last_good_frame = frame.copy()
            display = frame.copy()
            total_str = f"/{total_frames - 1}" if total_frames > 0 else ""
            info = f"Frame {frame_idx}{total_str}  |  a/d: +-1  w/s: +-30  Enter: confirm  q: quit"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Couldn't decode this position — show error, keep navigating
            display = last_good_frame.copy() if last_good_frame is not None else None
            if display is None:
                display = np.zeros((200, 600, 3), dtype=np.uint8)
            err = f"Error reading frame {frame_idx} — try another position"
            cv2.putText(display, err, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('d'):
            frame_idx = min(frame_idx + 1, total_frames - 1) if total_frames > 0 else frame_idx + 1
        elif key == ord('a'):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('s'):
            frame_idx = min(frame_idx + 30, total_frames - 1) if total_frames > 0 else frame_idx + 30
        elif key == ord('w'):
            frame_idx = max(frame_idx - 30, 0)
        elif key in (13, 32):  # Enter or Space
            if ret and last_good_frame is not None:
                cv2.destroyWindow(window_name)
                return last_good_frame
        elif key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            print("Calibration cancelled.")
            sys.exit(0)


def run_calibration(video_path: str) -> str:
    """Run the full calibration workflow.

    Returns:
        Path to saved calibration JSON.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nSelect a frame for calibration.")
    print("Use a frame where both kitchen zones are clearly visible.")
    frame = pick_calibration_frame(cap, total_frames)
    cap.release()

    h, w = frame.shape[:2]

    while True:
        state = CalibrationState(frame)
        window_name = "Calibration - click 8 corners + net point"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, state.on_click)

        print("\nClick 8 court corners + 2 net corners in order:")
        for i, label in enumerate(CORNER_LABELS):
            print(f"  {i + 1}. {label}")
        print("\nPress 'q' to quit, 'u' to undo last point.")

        while not state.done:
            # Show current instruction
            remaining = 10 - len(state.points)
            if remaining > 0:
                label = CORNER_LABELS[len(state.points)]
                info = f"Click point {len(state.points) + 1}/10: {label}"
            else:
                info = "All points placed!"

            display = state.display.copy()
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                print("Calibration cancelled.")
                sys.exit(0)
            elif key == ord("u") and state.points:
                state.points.pop()
                state.display = state.frame.copy()
                for i, pt in enumerate(state.points):
                    color = CORNER_COLORS[i]
                    cv2.circle(state.display, pt, 5, color, -1)
                    cv2.putText(state.display, f"{i + 1}", (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        pixel_corners_8 = [list(pt) for pt in state.points[:8]]
        net_left = list(state.points[8])
        net_right = list(state.points[9])
        net_x_pixel = (net_left[0] + net_right[0]) / 2

        # Compute homography
        try:
            H = compute_homography(pixel_corners_8)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again.")
            cv2.destroyAllWindows()
            continue

        # Show validation overlay (pass all 10 points)
        validation = draw_validation_overlay(frame, H, [list(pt) for pt in state.points])
        cv2.imshow(window_name, validation)
        key = cv2.waitKey(0) & 0xFF

        cv2.destroyAllWindows()

        if key == ord("r"):
            print("Redoing calibration...")
            continue

        # Save
        os.makedirs("configs", exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"configs/{video_name}.json"

        config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pixel_corners": pixel_corners_8,
            "homography_matrix": H.tolist(),
            "net_left_pixel": net_left,
            "net_right_pixel": net_right,
            "net_x_pixel": net_x_pixel,
            "court_surface_y_pixel": max(pt[1] for pt in pixel_corners_8),
            "input_resolution": [w, h],
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nCalibration saved to {output_path}")
        print(f"  Resolution: {w}x{h}")
        print(f"  Net left: {net_left}  Net right: {net_right}")
        print(f"  Net x-pixel (midpoint): {net_x_pixel:.1f}")
        print(f"  Court surface y-pixel: {config['court_surface_y_pixel']}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate court for kitchen fault detection")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    run_calibration(args.video)


if __name__ == "__main__":
    main()
