# Pickleball Kitchen Fault Detector

A Python CLI pipeline that processes recorded MP4s of pickleball games and detects kitchen (non-volley zone) foot faults using computer vision.

Camera mounts at net center looking down the court. The pipeline detects when a player's foot enters the kitchen zone while the ball is live (not yet bounced on their side).

## How it works

1. **Calibrate** -- click 4 corners enclosing the combined kitchen region + 2 net corners on the first frame. Saves the kitchen polygon directly in pixel space (no homography).
2. **Detect** -- runs pose tracking (ultralytics YOLOv8-pose, or the WholeBody backend with rtmlib RTMPose for true shoe-tip keypoints) and optional ball detection on every frame. Tests foot keypoints against the kitchen polygon in pixel space (`cv2.pointPolygonTest` with a small outward tolerance). Correlates with ball state to determine faults.
3. **Debug** (optional) -- generate an annotated video showing foot positions, ball side, and bounce events for tuning and verification.
4. **Review** -- opens an interactive viewer to step through detected faults and accept/reject each one.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pose models

Two backends are supported:

**1. WholeBody (default)** — two-stage: ultralytics YOLO for person detection + tracking, then [rtmlib](https://github.com/Tau-J/rtmlib) RTMPose for 133 COCO-WholeBody keypoints including big toe, small toe, and heel per foot. Zone check uses real foot keypoints directly. Weights auto-download to `~/.cache/rtmlib/` on first run (~500 MB).

**2. Ultralytics** — single-model YOLOv8-pose, COCO 17 keypoints. Ankle keypoints + 0.75ft buffer approximate toe position. Model auto-downloads on first run. Pass `--pose-backend ultralytics` to use it.

### GPU setup (Linux + NVIDIA)

`requirements.txt` installs `onnxruntime-gpu` and the CUDA 12 runtime wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.). `pickleball/_cuda_preload.py` is imported at the top of `detect.py` and ctypes-loads those `.so` files so rtmlib's onnxruntime binds to `CUDAExecutionProvider` without any `LD_LIBRARY_PATH` tweaking. This works even when torch ships its own (different-version) CUDA libs, since the two install into separate directories. No GPU available? Swap `onnxruntime-gpu` for `onnxruntime` in `requirements.txt` and drop the `nvidia-*-cu12` lines.

## Usage

### 1. Calibrate

```bash
python calibrate.py game.mp4
```

Pick a frame where the combined kitchen region (both sides of the net) is clearly visible (`a`/`d` step 1 frame, `w`/`s` step 30, Enter to confirm), then click 6 points in order:

1. Kitchen polygon corner 1 (e.g. top-left)
2. Kitchen polygon corner 2 (e.g. top-right)
3. Kitchen polygon corner 3 (e.g. bottom-right)
4. Kitchen polygon corner 4 (e.g. bottom-left)
5. Net left top corner
6. Net right top corner

The 4 kitchen corners enclose BOTH kitchens as a single convex quad. Press `u` to undo the last point, `q` to quit. After placing all 6 points, a validation overlay shows the kitchen polygon and net line. Press any key to save, or `r` to redo.

Saves to `configs/{video_name}.json`.

### 2. Detect faults

```bash
python detect.py game.mp4 --calibration configs/game.json

# With ball detection model (optional but recommended)
python detect.py game.mp4 --calibration configs/game.json --ball-model ball_detect.pt

# With a different ultralytics pose model (only applies to --pose-backend ultralytics)
python detect.py game.mp4 --calibration configs/game.json \
    --pose-backend ultralytics --pose-model yolov8l-pose.pt

# With debug video output
python detect.py game.mp4 --calibration configs/game.json --debug-video output/game_debug.mp4
```

Without `--ball-model`, all kitchen entries are logged as REVIEW_NEEDED (fallback mode). Add a ball model for automatic fault vs. non-fault classification.

**Speed knobs:**
- `--imgsz 640` (default) — detector long-edge inference size. Affects ultralytics detector + ball model only; rtmlib pose has its own input size.
- `--ball-stride 2` (default) — run ball detection every Nth frame; interpolation fills gaps.
- `--half` / `--no-half` — force fp16/fp32. Default: fp16 when CUDA is available.

**WholeBody knobs:**
- `--pose-backend {wholebody,ultralytics}` — default `wholebody`.
- `--detector-model models/yolov8s.pt` (default) — person detector for WholeBody. Bump to `models/yolov8x.pt` only if you see missed detections. All ultralytics weights live in `models/`; `detect.py` creates the directory on startup so auto-download writes there.
- `--wholebody-mode {balanced,performance,lightweight}` — default `balanced` (DWPose-L @ 192x256). `performance` adds resolution for marginal accuracy gain; `lightweight` is fastest, for mobile or low-end CPU.

Outputs to `output/{video_name}_faults.json`.

### 3. Debug video (optional)

The `--debug-video` flag writes a full annotated MP4 alongside the fault JSON. Useful for verifying that foot keypoints, ball tracking, and bounce detection are working correctly before reviewing faults.

Each frame shows:
- Combined kitchen polygon overlay (green)
- Net line
- Foot keypoint dots: green when outside the kitchen, red when inside (the zone test uses an outward tolerance, so a dot just outside the drawn polygon line can still render red)
- Player track IDs
- Ball position with a 10-frame trail
- Status bar: `Left: LIVE | Right: BOUNCED` (or `FALLBACK MODE` if no ball model)
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
    constants.py        # All thresholds, zone tolerance, keypoint indices
    pose.py             # Pixel-space zone check, keypoint extraction
    pose_backend.py     # Pluggable pose backends (ultralytics / wholebody)
    ball.py             # Ball interpolation, bounce detection, state machine
    fault.py            # Confidence scoring, tier classification
    annotate.py         # Debug video annotation helpers
    _cuda_preload.py    # Preloads CUDA 12 libs for onnxruntime-gpu
  tests/                # Unit tests (pytest)
    test_pose.py
    test_pose_backend.py
    test_ball.py
    test_fault.py
    test_constants.py
    test_annotate.py
  configs/              # Calibration JSONs (generated)
  models/               # Model weights (not tracked in git)
  output/               # Detection results (generated)
```

## How fault detection works

**Zone check:** Each frame, foot keypoints (WholeBody backend: big toe, small toe, heel per foot) or ankle keypoints (Ultralytics backend) are tested against the kitchen polygon in pixel space via `cv2.pointPolygonTest`. A small outward tolerance (`KITCHEN_BOUNDARY_TOLERANCE_PX` in `constants.py`) compensates for pose-keypoint noise, so a toe on or just outside the line still registers as a zone hit. The side of the net a hit is attributed to is derived from its pixel x vs `net_x_pixel`.

**Ball state machine:** Tracks per-side state (LIVE, BOUNCED, UNKNOWN) in pixel space. A bounce on one side sets that side=BOUNCED, the other=LIVE. Ball crossing the net (pixel x crosses `net_x_pixel`) resets the destination side to LIVE. No ball detection for >10 frames (fps-scaled) sets both to UNKNOWN.

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
