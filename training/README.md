# Ball Detection Training Pipeline

Train a pickleball-specific detection model to replace the missing ball model
in `detect.py`. Without a ball model, every kitchen entry tiers to
`REVIEW_NEEDED` in fallback mode — training a model unlocks `AUTO_FAULT` tiering.

## Prerequisites

- Roboflow account (free tier, https://roboflow.com).
- `.venv` activated: `source .venv/bin/activate`.
- Videos in `inputs/`. Reserve one video (e.g. `inputs/video3.mp4`) for
  held-out evaluation — do **not** extract frames from it.

## Pipeline

### 1. Extract frames

```bash
python training/extract_frames.py \
    --videos inputs/video1.mp4 inputs/video2.mp4 inputs/video4.mp4 \
    --out training/dataset/frames \
    --stride 15
```

Every 15th frame (~2 fps @ 30 fps source) with perceptual-hash dedup.
Expect ~1k frames for 10 min of footage across 3 videos.

### 2. Pre-label with generic sports-ball proposer

```bash
python training/prelabel.py \
    --frames training/dataset/frames \
    --model models/yolov8x.pt \
    --out training/dataset/prelabels \
    --conf 0.1
```

Runs `yolov8x.pt` (already in `models/`) on every frame, keeps COCO class 32
(`sports ball`) detections, and writes YOLO-format `.txt` files alongside
the jpgs. Low confidence (0.1) maximizes recall — faster to delete false
boxes in Roboflow than to draw missing ones.

### 3. Upload to Roboflow

1. Create a new project → **Object Detection** → single class `pickleball`.
2. Zip the prelabels folder:
   ```bash
   cd training/dataset && zip -r prelabels.zip prelabels && cd -
   ```
3. Upload `prelabels.zip` via Roboflow web UI ("Upload → with annotations,
   YOLOv8 format"). Roboflow will auto-detect the txt labels.
4. Correct boxes: delete false positives, add missed balls, tighten boxes.
5. Auto-split: **Generate → Train/Valid/Test** split 70/20/10.
6. Augmentation preset (recommended): motion blur, brightness ±25%,
   mosaic, 90° rotation off.
7. **Generate version** → **Export dataset** → **YOLOv8** format →
   **download zip**.
8. Unzip into `training/dataset/roboflow_export/` so `data.yaml` is at
   `training/dataset/roboflow_export/data.yaml`.

### 4. Train

```bash
python training/train.py \
    --data training/dataset/roboflow_export/data.yaml \
    --model yolov8s.pt \
    --epochs 100
```

Best weights save to `training/runs/train/weights/best.pt`.
Early stopping (patience 30) will stop sooner if val plateaus.

### 5. Evaluate on held-out video

```bash
python training/evaluate.py \
    --weights training/runs/train/weights/best.pt \
    --video inputs/video3.mp4
```

Pass criterion: detection rate ≥ 40% (clears `detect.py`'s fallback gate).
Stretch goal: ≥ 70% with mean conf ≥ 0.4.

### 6. Promote and integration test

```bash
cp training/runs/train/weights/best.pt models/ball.pt
python detect.py inputs/video3.mp4 \
    --calibration configs/video3.json \
    --ball-model models/ball.pt
```

Check `output/video3_faults.json`: `ball_detection_rate` should match the
evaluate step, `fallback_mode` should be `false`, and `AUTO_FAULT` entries
should appear.

## Adding more footage later

1. Drop new MP4s in `inputs/`.
2. Re-run steps 1–2 on only the new videos.
3. In Roboflow, upload the new prelabels → correct → bump dataset version.
4. Re-export and re-train (step 4 onwards). No code changes needed.

## Directory layout

```
training/
├── extract_frames.py      # step 1
├── prelabel.py            # step 2
├── train.py               # step 4
├── evaluate.py            # step 5
├── dataset/               # (gitignored) frames + labels
│   ├── frames/
│   ├── prelabels/
│   └── roboflow_export/   # unzipped Roboflow export
└── runs/                  # (gitignored) ultralytics outputs
    └── train/weights/best.pt
```
