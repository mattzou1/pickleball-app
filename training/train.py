"""Train a pickleball detection model on a Roboflow-exported YOLOv8 dataset.

Usage:
    python training/train.py --data training/dataset/roboflow_export/data.yaml \
        --model yolov8s.pt --epochs 100

Augmentation is tuned for small, fast-moving balls under variable lighting:
strong mosaic + scale jitter for small-object robustness, moderate hsv_v for
indoor/outdoor lighting variance, fliplr for left/right symmetry.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Roboflow-exported data.yaml")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base weights. Default yolov8s.pt (auto-downloads to cwd).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size. -1 = auto (ultralytics picks based on VRAM). Default -1.")
    parser.add_argument("--project", default="training/runs",
                        help="Output parent dir for runs. Default training/runs.")
    parser.add_argument("--name", default="train", help="Run name. Default train.")
    parser.add_argument("--device", default=None,
                        help="Device string, e.g. '0' or 'cpu'. Default: auto (CUDA if available).")
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=device,
        # Pickleball-tuned augmentation
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,
        degrees=5.0,
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        # Save every 10 epochs so long runs are recoverable
        save_period=10,
        # Early stopping: patience epochs without val improvement
        patience=30,
    )

    print(f"\nBest weights: {args.project}/{args.name}/weights/best.pt")
    print("Evaluate on held-out video with:")
    print(f"  python training/evaluate.py --weights {args.project}/{args.name}/weights/best.pt "
          "--video inputs/video3.mp4")


if __name__ == "__main__":
    main()
