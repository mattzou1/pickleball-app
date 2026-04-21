# Pickleball Kitchen Fault Detector

## Project overview

Python CLI pipeline for detecting kitchen (non-volley zone) foot faults in recorded pickleball games. Camera at net center, processes MP4 files through calibration, pose tracking, ball detection, and fault correlation. This will eventually be the backend for an IOS app. 

## Architecture

Three CLI scripts orchestrate a core library:

- `calibrate.py` -- 4 kitchen polygon corners + 2 net top corners (6 clicks total), outputs calibration JSON with a pixel-space kitchen polygon and net endpoints. No homography.
- `detect.py` -- thin orchestrator that runs pose tracking, ball detection, and fault correlation
- `review.py` -- interactive OpenCV viewer for accept/reject decisions on detected faults

Core logic lives in `pickleball/`:
- `constants.py` -- single source of truth for all thresholds, zone tolerance, keypoint indices
- `pose.py` -- pixel-space zone check (`cv2.pointPolygonTest` on the kitchen polygon with an outward pixel tolerance), foot/ankle keypoint extraction
- `pose_backend.py` -- pluggable pose backends: `UltralyticsBackend` (single-model YOLO pose, COCO 17) and `WholeBodyBackend` (ultralytics detect+track + rtmlib RTMPose, COCO-WholeBody 133 kp)
- `ball.py` -- interpolation, bounce detection (in court coords), per-side state machine
- `fault.py` -- confidence scoring, tier classification, fault correlation
- `_cuda_preload.py` -- ctypes-preloads the `nvidia-*-cu12` wheels' `.so` files at import time so `onnxruntime-gpu` finds CUDA 12 runtime libs even when torch ships CUDA 13. Imported as the first line of `detect.py`; no-op if the wheels aren't present.

## Key design decisions

- **Pixel-space kitchen polygon** (no homography). Calibration saves 4 clicked corners enclosing both kitchens as a single convex quad. Zone check is `cv2.pointPolygonTest` on that polygon with an outward tolerance (`KITCHEN_BOUNDARY_TOLERANCE_PX` in `constants.py`, default 5 px) so a toe on or just outside the kitchen line still registers as a fault. The polygon stored in the JSON is untouched — the tolerance is a detection-time setting, so tuning doesn't force a re-calibration.
- **Two pose backends**. `WholeBodyBackend` (default) does two-stage inference: ultralytics YOLO detector+tracker for person boxes + track IDs, then rtmlib RTMPose (COCO-WholeBody, 133 kp) for keypoints. Zone check uses real foot keypoints (17-22 = big toe, small toe, heel per foot). `UltralyticsBackend` is a single-model YOLO pose; typically COCO 17, so zone check falls back to ankle keypoints (15-16). Both backends run the same pixel-polygon test — the outward tolerance covers keypoint noise uniformly. Auto-detection at `pose.py:is_wholebody_model` switches between foot and ankle keypoints based on keypoint count.
- **Player selection** (WholeBody backend). After person detection, `pose_backend._keep_largest` keeps only the `MAX_PLAYERS_KEPT` (4) largest bboxes — with a net-center camera, bbox pixel area is a reliable "closest to camera" proxy that survives occluded/cropped feet. This replaced an earlier homography-based court-bounds gate that silently dropped foreground players whose torso sat mid-court while a foot reached into the kitchen. Do not re-introduce a homography- or kitchen-proximity-based gate.
- **Split confidence thresholds**: FOOT_KP_CONF_THRESHOLD=0.2 for zone check (foot keypoints are noisy), ANKLE_CONF_THRESHOLD=0.5 for confidence scoring.
- **Ball state machine** is per-side (left/right). Side is derived from ball pixel x vs `net_x_pixel`; net crossing is detected when that side flips between frames. No homography.
- **Bounce detection in pixel space**. Vertical velocity is pixel-y/frame with a 3-frame moving-average smoother; a bounce is a sign reversal. Side of the bounce is classified by the ball's pixel x vs `net_x_pixel`.
- **Fallback mode**: if no ball model or detection rate < 40%, all kitchen entries go to REVIEW_NEEDED. Invariant: in fallback (and any UNKNOWN ball state) `detect.py` sets `ball_conf=1.0` before calling `correlate_fault`. The confidence formula multiplies by `ball_conf`, so a zeroed ball_conf would tier everything as FILTERED and silently drop every entry. Preserve this when touching `detect.py`'s ball-state block.
- **FPS scaling**: all frame-count thresholds scale by actual_fps / 30.0.
- **Atomic JSON writes** with mtime guard in review.py to prevent data loss from concurrent edits.
- **All ultralytics weights live in `models/`**. CLI defaults (`--pose-model`, `--detector-model`) and `pose_backend.py` defaults all point inside `models/`. `detect.py` ensures the directory exists at startup so ultralytics' auto-download writes there. rtmlib weights are separate — they cache to `~/.cache/rtmlib/` by design (managed by rtmlib, not us).

## Commands

```bash
# Run tests
source .venv/bin/activate
python -m pytest tests/ -v

# Calibrate (6 clicks: 4 kitchen corners + 2 net top corners)
python calibrate.py <video>

# Detect (WholeBody backend, default — shoe-tip accurate)
python detect.py <video> --calibration <config.json> [--ball-model <path>]

# Detect (legacy ultralytics single-model backend)
python detect.py <video> --calibration <config.json> --pose-backend ultralytics [--pose-model models/yolov8x-pose-p6.pt] [--ball-model <path>]

# Speed knobs: --imgsz 640, --ball-stride 2, --half / --no-half (default: auto, fp16 on CUDA)
# WholeBody-only knobs: --detector-model models/yolov8s.pt (default), --wholebody-mode {balanced,performance,lightweight} (default balanced)
# Ultralytics-only knobs: --pose-model models/yolov8x-pose-p6.pt (default)

# Review
python review.py <faults.json> [--video <path>]
```

## Code conventions

- All model weights must be stored in `models/`. This applies to ultralytics weights, ball detection models, and any other model files — never reference or download models to another directory.
- All tunable constants live in `pickleball/constants.py`. Tuning is a one-file change.
- Everything is pixel-space — no court coordinates, no homography. The kitchen polygon, net position, ball tracking, and bounce detection all operate on pixel coordinates from the input frame.
- Calibration JSON (schema_version 3) fields: `kitchen_polygon` (4 pixel-space `[x, y]` points, convex quad enclosing both kitchens), `net_left_pixel`, `net_right_pixel`, `net_x_pixel` (midpoint, used for ball-side classification), `input_resolution` (`[w, h]`, checked against the video at detect time).
- Fault JSON is the single source of truth across all pipeline stages. `review_decision` is null until review.py updates it.
- Tests use synthetic data only (no model weights or video fixtures needed). Run with `pytest tests/ -v`.

## Dependencies

- ultralytics (YOLOv8 pose tracking + ball detection, person detector for WholeBody backend)
- rtmlib + onnxruntime-gpu (default on Linux; swap to `onnxruntime` for CPU-only)
- `nvidia-*-cu12` runtime wheels on Linux (cublas, cudnn, cuda-runtime, cuda-nvrtc, curand, cufft) — preloaded by `_cuda_preload.py` so `onnxruntime-gpu` binds to `CUDAExecutionProvider` alongside a CUDA-13 torch
- opencv-python (polygon hit test, video I/O, calibration UI)
- numpy
- torch (device detection for fp16 auto-toggle, ultralytics GPU inference)
- tqdm (progress bars)
- pytest (testing)
