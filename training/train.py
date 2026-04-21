"""Train a pickleball detection model on a Roboflow-exported YOLOv8 dataset.

Usage:
    python training/train.py --data training/dataset/roboflow_export/data.yaml \
        --model yolov8s.pt --epochs 100

Augmentation is tuned for small, fast-moving balls under variable lighting:
strong mosaic + scale jitter for small-object robustness, moderate hsv_v for
indoor/outdoor lighting variance, fliplr for left/right symmetry.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Must precede any CUDA import — preloads nvidia-*-cu12 .so files so torch
# finds CUDA 12 runtime libs even when torch ships CUDA 13.
sys.path.insert(0, str(Path(__file__).parent.parent))
import pickleball._cuda_preload  # noqa: F401, E402

import logging

import torch
from ultralytics import YOLO, settings as yolo_settings

# Suppress ultralytics' per-batch tqdm/logging noise.
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Roboflow-exported data.yaml")
    parser.add_argument("--model", default="models/yolov8s.pt",
                        help="Base weights. Default models/yolov8s.pt (auto-downloads into models/).")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max epochs. Early stopping will cut this short if val plateaus. Default 200.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping: stop after this many epochs with no val mAP50 improvement. Default 30.")
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

    # Redirect ultralytics auto-downloads into models/.
    Path("models").mkdir(exist_ok=True)
    yolo_settings.update({"weights_dir": str(Path("models").resolve())})

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    model = YOLO(args.model)

    def _print_map50(trainer):
        map50 = trainer.metrics["metrics/mAP50(B)"]
        print(f"Epoch {trainer.epoch + 1}/{trainer.epochs}  mAP50: {map50:.4f}", flush=True)

    model.add_callback("on_fit_epoch_end", _print_map50)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(Path(args.project).resolve()),
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
        # Early stopping: stop after patience epochs with no val mAP50 improvement
        patience=args.patience,
        verbose=False,
    )

    # Resolve best.pt — try trainer.best first, then search the project tree.
    best_src = Path(model.trainer.best)
    print(f"\nTrainer reports best.pt at: {best_src}")

    if not best_src.exists():
        # Ultralytics sometimes restructures the project path (e.g. inserts detect/).
        # Search the resolved project dir for any best.pt as a fallback.
        project_dir = Path(args.project).resolve()
        candidates = sorted(project_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit(
                f"ERROR: best.pt not found at {best_src} or anywhere under {project_dir}.\n"
                f"Training may have failed. Check the run directory manually."
            )
        best_src = candidates[0]
        print(f"  Not found — using most recently modified: {best_src}")

    dest = Path("models") / "ball.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(best_src, dest)
    print(f"Best weights copied to {dest}")
    print("Evaluate on held-out video with:")
    print(f"  python training/evaluate.py --weights {dest} --video inputs/video3.mp4")


if __name__ == "__main__":
    main()
