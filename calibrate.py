"""Court calibration tool: click 8 corners + 1 net point, compute homography.

Usage:
    python calibrate.py <video_path> --court-id <id> --camera-id <id>

Opens first frame of video. User clicks 8 court corners in order:
    1-4: Near kitchen (BL, BR, TR, TL)
    5-8: Far kitchen (BL, BR, TR, TL)
    9:   Net center point

Saves calibration JSON to configs/{court_id}_{camera_id}.json.
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
    "Near BL (baseline left)",
    "Near BR (baseline right)",
    "Near TR (kitchen line right)",
    "Near TL (kitchen line left)",
    "Far BL (kitchen line left)",
    "Far BR (kitchen line right)",
    "Far TR (baseline right)",
    "Far TL (baseline left)",
    "Net center point",
]

CORNER_COLORS = [
    (0, 255, 0),    # green for near kitchen
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 0, 0),    # blue for far kitchen
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 0),
    (0, 255, 255),  # yellow for net
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
        if len(self.points) >= 9:
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

        if len(self.points) >= 9:
            self.done = True


def draw_validation_overlay(frame: np.ndarray, H: np.ndarray, pixel_corners: list) -> np.ndarray:
    """Draw kitchen zones projected back onto frame for visual validation."""
    overlay = frame.copy()

    # Draw near kitchen zone (green, semi-transparent)
    near_pts = np.array(pixel_corners[:4], dtype=np.int32)
    cv2.fillPoly(overlay, [near_pts], (0, 255, 0))

    # Draw far kitchen zone (blue, semi-transparent)
    far_pts = np.array(pixel_corners[4:8], dtype=np.int32)
    cv2.fillPoly(overlay, [far_pts], (255, 0, 0))

    # Blend
    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Draw corner points
    for i, pt in enumerate(pixel_corners[:8]):
        color = CORNER_COLORS[i]
        cv2.circle(result, (int(pt[0]), int(pt[1])), 8, color, -1)
        cv2.putText(result, str(i + 1), (int(pt[0]) + 12, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw net point
    if len(pixel_corners) > 8:
        net_pt = pixel_corners[8]
        cv2.circle(result, (int(net_pt[0]), int(net_pt[1])), 8, (0, 255, 255), -1)
        cv2.putText(result, "NET", (int(net_pt[0]) + 12, int(net_pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(result, "Validation overlay - press any key to save, 'r' to redo",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result


def run_calibration(video_path: str) -> str:
    """Run the full calibration workflow.

    Returns:
        Path to saved calibration JSON.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read first frame")
        sys.exit(1)

    h, w = frame.shape[:2]

    while True:
        state = CalibrationState(frame)
        window_name = "Calibration - click 8 corners + net point"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, state.on_click)

        print("\nClick the 8 court corners in order:")
        for i, label in enumerate(CORNER_LABELS):
            print(f"  {i + 1}. {label}")
        print("\nPress 'q' to quit, 'u' to undo last point.")

        while not state.done:
            # Show current instruction
            remaining = 9 - len(state.points)
            if remaining > 0:
                label = CORNER_LABELS[len(state.points)]
                info = f"Click point {len(state.points) + 1}/9: {label}"
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
        net_point = list(state.points[8])

        # Compute homography
        try:
            H = compute_homography(pixel_corners_8)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please try again.")
            cv2.destroyAllWindows()
            continue

        # Show validation overlay
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
            "net_point_pixel": net_point,
            "net_x_pixel": net_point[0],
            "court_surface_y_pixel": max(pt[1] for pt in pixel_corners_8),
            "input_resolution": [w, h],
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nCalibration saved to {output_path}")
        print(f"  Resolution: {w}x{h}")
        print(f"  Net x-pixel: {net_point[0]}")
        print(f"  Court surface y-pixel: {config['court_surface_y_pixel']}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate court for kitchen fault detection")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    run_calibration(args.video)


if __name__ == "__main__":
    main()
