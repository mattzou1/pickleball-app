"""Court calibration tool: click 4 kitchen-polygon corners + net point.

Usage:
    python calibrate.py <video_path>

Opens video with frame picker. Use WASD to navigate to a frame where the
combined kitchen region (both sides of the net) is clearly visible, then press
Enter to confirm. User clicks 4 corners enclosing both kitchens as a single
polygon, then 1 net point.

Saves calibration JSON to configs/{video_name}.json.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np


CORNER_LABELS = [
    "Kitchen polygon: corner 1 (e.g. top-left)",
    "Kitchen polygon: corner 2 (e.g. top-right)",
    "Kitchen polygon: corner 3 (e.g. bottom-right)",
    "Kitchen polygon: corner 4 (e.g. bottom-left)",
    "Net left top corner",
    "Net right top corner",
]

NUM_POINTS = len(CORNER_LABELS)  # 6

COLOR_KITCHEN = (0, 255, 0)    # green for kitchen polygon corners
COLOR_NET = (0, 255, 255)      # yellow for net points
COLORS = [COLOR_KITCHEN] * 4 + [COLOR_NET] * 2

# Reject duplicate clicks / degenerate polygons.
MIN_POINT_DISTANCE_PX = 10


class CalibrationState:
    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.points: list[tuple[int, int]] = []
        self.done = False

    def _redraw(self) -> None:
        self.display = self.frame.copy()
        for i, pt in enumerate(self.points):
            color = COLORS[i]
            cv2.circle(self.display, pt, 5, color, -1)
            cv2.putText(self.display, f"{i + 1}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Connect polygon edges as they're drawn
        for i in range(1, min(len(self.points), 4)):
            cv2.line(self.display, self.points[i - 1], self.points[i], COLOR_KITCHEN, 1)
        if len(self.points) >= 4:
            cv2.line(self.display, self.points[3], self.points[0], COLOR_KITCHEN, 1)

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= NUM_POINTS:
            return
        self.points.append((x, y))
        self._redraw()
        if len(self.points) >= NUM_POINTS:
            self.done = True

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw()


def validate_polygon(polygon: list[list[int]]) -> str | None:
    """Return an error message if the polygon is invalid, else None."""
    # Duplicate-point check
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            dx = polygon[i][0] - polygon[j][0]
            dy = polygon[i][1] - polygon[j][1]
            if (dx * dx + dy * dy) < (MIN_POINT_DISTANCE_PX ** 2):
                return (f"Points {i + 1} and {j + 1} are too close "
                        f"(< {MIN_POINT_DISTANCE_PX}px). Re-click distinct corners.")
    # Convexity check
    pts = np.array(polygon, dtype=np.int32)
    if not cv2.isContourConvex(pts):
        return "Polygon is not convex (points likely out of order). Use a consistent winding order."
    return None


def draw_validation_overlay(
    frame: np.ndarray,
    polygon: list[list[int]],
    net_left: list[int],
    net_right: list[int],
) -> np.ndarray:
    """Draw the kitchen polygon and net line projected onto the frame for review."""
    pts = np.array(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], COLOR_KITCHEN)
    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    cv2.polylines(result, [pts], True, COLOR_KITCHEN, 2)

    for i, pt in enumerate(polygon):
        cv2.circle(result, (int(pt[0]), int(pt[1])), 8, COLOR_KITCHEN, -1)
        cv2.putText(result, str(i + 1), (int(pt[0]) + 12, int(pt[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_KITCHEN, 2)

    pt_left = (int(net_left[0]), int(net_left[1]))
    pt_right = (int(net_right[0]), int(net_right[1]))
    cv2.line(result, pt_left, pt_right, COLOR_NET, 3)
    cv2.circle(result, pt_left, 8, COLOR_NET, -1)
    cv2.circle(result, pt_right, 8, COLOR_NET, -1)
    cv2.putText(result, "NET L", (pt_left[0] + 10, pt_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 2)
    cv2.putText(result, "NET R", (pt_right[0] + 10, pt_right[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 2)

    cv2.putText(result, "Validation overlay - press any key to save, 'r' to redo",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return result


def pick_calibration_frame(cap: cv2.VideoCapture, total_frames: int) -> np.ndarray:
    """Navigate through video frames to pick one where the kitchen is visible.

    Keys: d=+1 frame, a=-1 frame, s=+30 frames, w=-30 frames, Enter/Space=confirm, q=quit.
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
    print("Use a frame where the combined kitchen region is clearly visible.")
    frame = pick_calibration_frame(cap, total_frames)
    cap.release()

    h, w = frame.shape[:2]

    while True:
        state = CalibrationState(frame)
        window_name = "Calibration - click 4 kitchen corners + 2 net points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, state.on_click)

        print("\nClick 6 points in order:")
        for i, label in enumerate(CORNER_LABELS):
            print(f"  {i + 1}. {label}")
        print("\nKitchen corners should enclose BOTH kitchens as one convex quad.")
        print("Press 'q' to quit, 'u' to undo last point.")

        while not state.done:
            remaining = NUM_POINTS - len(state.points)
            if remaining > 0:
                label = CORNER_LABELS[len(state.points)]
                info = f"Click point {len(state.points) + 1}/{NUM_POINTS}: {label}"
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
            elif key == ord("u"):
                state.undo()

        kitchen_polygon = [list(pt) for pt in state.points[:4]]
        net_left = list(state.points[4])
        net_right = list(state.points[5])
        net_x_pixel = (net_left[0] + net_right[0]) / 2

        # Sanity checks
        err = validate_polygon(kitchen_polygon)
        if err:
            print(f"Error: {err}")
            print("Please re-click.")
            cv2.destroyAllWindows()
            continue

        # Show validation overlay
        validation = draw_validation_overlay(frame, kitchen_polygon, net_left, net_right)
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
            "schema_version": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kitchen_polygon": kitchen_polygon,
            "net_left_pixel": net_left,
            "net_right_pixel": net_right,
            "net_x_pixel": net_x_pixel,
            "input_resolution": [w, h],
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nCalibration saved to {output_path}")
        print(f"  Resolution: {w}x{h}")
        print(f"  Kitchen polygon: {kitchen_polygon}")
        print(f"  Net left: {net_left}  Net right: {net_right}")
        print(f"  Net x-pixel (midpoint): {net_x_pixel:.1f}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate court for kitchen fault detection")
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    run_calibration(args.video)


if __name__ == "__main__":
    main()
