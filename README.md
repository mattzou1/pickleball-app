# Pickleball Kitchen Fault Detector

A Python CLI pipeline that processes recorded MP4s of pickleball games and detects kitchen (non-volley zone) foot faults using computer vision.

Camera mounts at net center looking down the court. The pipeline detects when a player's foot enters the kitchen zone while the ball is live (not yet bounced on their side).

## How it works

1. **Calibrate** -- click 8 court corners + 2 net corners on the first frame. Computes a homography that maps pixels to court coordinates (feet).
2. **Detect** -- runs pose tracking (YOLOv8-pose or WholeBody model) and optional ball detection on every frame. Checks foot keypoints against kitchen zone boundaries in court coordinates. Correlates with ball state to determine faults.
3. **Debug** (optional) -- generate an annotated video showing foot positions, ball side, and bounce events for tuning and verification.
4. **Review** -- opens an interactive viewer to step through detected faults and accept/reject each one.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need a YOLOv8 pose model. Download one:

```bash
# Standard COCO 17-keypoint (works out of the box)
# The model auto-downloads on first run if not present
# Or manually: place yolov8m-pose.pt in models/
```

For better accuracy, use a WholeBody model (133 keypoints with toe/heel detection). The code auto-detects the model type and uses foot keypoints when available, falling back to ankle keypoints + buffer otherwise.

## Usage

### 1. Calibrate

```bash
python calibrate.py game.mp4
```

Click 10 points in order:
1. Near baseline left
2. Near baseline right
3. Near kitchen line right
4. Near kitchen line left
5. Far kitchen line left
6. Far kitchen line right
7. Far baseline right
8. Far baseline left
9. Net left top corner (left post)
10. Net right top corner (right post)

Press `u` to undo the last point, `q` to quit. After placing all 10 points, a validation overlay shows the kitchen zones and net line. Press any key to save, or `r` to redo.

Saves to `configs/{video_name}.json`.

### 2. Detect faults

```bash
python detect.py game.mp4 --calibration configs/game.json

# With ball detection model (optional but recommended)
python detect.py game.mp4 --calibration configs/game.json --ball-model ball_detect.pt

# With a different pose model
python detect.py game.mp4 --calibration configs/game.json --pose-model yolov8l-pose.pt

# With debug video output
python detect.py game.mp4 --calibration configs/game.json --debug-video output/game_debug.mp4
```

Without `--ball-model`, all kitchen entries are logged as REVIEW_NEEDED (fallback mode). Add a ball model for automatic fault vs. non-fault classification.

Outputs to `output/{video_name}_faults.json`.

### 3. Debug video (optional)

The `--debug-video` flag writes a full annotated MP4 alongside the fault JSON. Useful for verifying that foot keypoints, ball tracking, and bounce detection are working correctly before reviewing faults.

Each frame shows:
- Kitchen zones highlighted (green = near, blue = far)
- Net line
- Foot keypoint dots: green when outside the kitchen, red when inside
- Player track IDs
- Ball position with a 10-frame trail
- Status bar: `Near: LIVE | Far: BOUNCED`
- Bounce flash text when a bounce is detected (shows for 15 frames)
- Red border on frames where a fault was triggered

### 4. Review faults

```bash
python review.py output/game_faults.json

# Override video path if moved
python review.py output/game_faults.json --video /path/to/game.mp4
```

Keyboard controls:
- `N` -- next fault
- `P` -- previous fault
- `A` -- accept (mark as real fault)
- `R` -- reject (false positive)
- `Q` -- quit and save

Review decisions are written back to the same JSON file (atomic writes with mtime guard).

## Project structure

```
pickleball-app/
  calibrate.py          # Court calibration CLI
  detect.py             # Detection pipeline orchestrator
  review.py             # Interactive fault review tool
  requirements.txt      # Python dependencies
  pickleball/           # Core library
    __init__.py
    constants.py        # All thresholds, zone bounds, keypoint indices
    pose.py             # Pose tracking, homography, zone check
    ball.py             # Ball interpolation, bounce detection, state machine
    fault.py            # Confidence scoring, tier classification
    annotate.py         # Debug video annotation helpers
  tests/                # Unit tests (pytest)
    test_pose.py
    test_ball.py
    test_fault.py
    test_constants.py
    test_annotate.py
  configs/              # Calibration JSONs (generated)
  models/               # Model weights (not tracked in git)
  output/               # Detection results (generated)
```

## How fault detection works

**Zone check:** Each frame, foot keypoints (or ankle keypoints with a 0.5ft buffer) are transformed through the homography to court coordinates. If any keypoint lands inside a kitchen zone (0-7ft from baseline on either side), including on the line, it's a zone hit.

**Ball state machine:** Tracks per-side state (LIVE, BOUNCED, UNKNOWN). A bounce on the near side sets near=BOUNCED, far=LIVE. Ball crossing the net resets the destination side to LIVE. No ball detection for >10 frames sets both to UNKNOWN.

**Fault rule:** A kitchen zone entry is a fault only when the ball is LIVE on that side. If BOUNCED, it's legal (player is allowed in the kitchen after the ball bounces). If UNKNOWN, it's flagged for review.

**Confidence scoring:**
```
composite = ankle_confidence * min(consecutive_frames / 10, 1.0) * ball_confidence
```

Tiers:
- AUTO_FAULT: confidence >= 0.5
- REVIEW_NEEDED: confidence >= 0.15
- FILTERED: confidence < 0.15 (not shown)

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

All tests use synthetic data (no video files or model weights needed).
