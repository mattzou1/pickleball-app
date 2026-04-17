"""Evaluate a trained ball model on a held-out video, using the same detection
loop shape as detect.py so the reported metric matches runtime behavior.

The key metric is detection rate: fraction of processed frames with >=1 box.
detect.py falls back to REVIEW_NEEDED for every kitchen entry when this rate
is below 40%, so that's the minimum pass bar.

Usage:
    python training/evaluate.py --weights training/runs/train/weights/best.pt \
        --video inputs/video3.mp4
"""

import argparse
import statistics
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="Trained .pt weights")
    parser.add_argument("--video", required=True, help="Held-out video (e.g. inputs/video3.mp4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Match detect.py default 640.")
    parser.add_argument("--ball-stride", type=int, default=2,
                        help="Match detect.py default 2 (every Nth frame).")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Inference conf threshold. Default 0.25 (ultralytics default).")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    half = torch.cuda.is_available()
    model = YOLO(args.weights)

    processed = 0
    detected = 0
    confs: list[float] = []

    for frame_idx in tqdm(range(total), desc="Evaluating"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.ball_stride != 0:
            continue
        processed += 1

        results = model(frame, imgsz=args.imgsz, half=half, conf=args.conf, verbose=False)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            detected += 1
            confs.append(float(r.boxes.conf.max()))

    cap.release()

    rate = detected / max(processed, 1)
    mean_conf = statistics.mean(confs) if confs else 0.0
    median_conf = statistics.median(confs) if confs else 0.0

    print(f"\nVideo: {video_path}  ({total} frames @ {fps:.1f} fps)")
    print(f"Weights: {args.weights}")
    print(f"Processed frames: {processed} (stride {args.ball_stride})")
    print(f"Frames with >=1 detection: {detected}")
    print(f"Detection rate: {rate:.1%}")
    print(f"Mean conf (on detected frames): {mean_conf:.3f}")
    print(f"Median conf: {median_conf:.3f}")

    if rate < 0.40:
        print("\nFAIL: below 40% — detect.py would fall back to REVIEW_NEEDED for all entries.")
    elif rate < 0.70:
        print("\nPASS (minimum): clears the 40% fallback threshold. Stretch goal is 70%+.")
    else:
        print("\nPASS (stretch): detection rate >=70% — good model.")


if __name__ == "__main__":
    main()
