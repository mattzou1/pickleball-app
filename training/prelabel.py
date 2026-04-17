"""Pre-label extracted frames with a generic YOLO sports-ball proposer.

Uses the existing models/yolov8x.pt (COCO-trained) and keeps class 32
('sports ball') detections, rewritten as class 0 ('pickleball') in YOLO
txt format. Upload the output directory to Roboflow as an annotated dataset,
then correct/prune in the Roboflow UI.

Usage:
    python training/prelabel.py --frames training/dataset/frames \
        --model models/yolov8x.pt \
        --out training/dataset/prelabels --conf 0.1
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

COCO_SPORTS_BALL = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", required=True, help="Directory of input jpgs")
    parser.add_argument("--model", default="models/yolov8x.pt",
                        help="COCO-trained YOLO model. Default models/yolov8x.pt (already in repo).")
    parser.add_argument("--out", required=True, help="Output directory (jpg + txt pairs)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Low confidence, high recall — easier to delete false boxes in "
                             "Roboflow than to draw missing ones. Default 0.1.")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference image size. Higher helps catch small pickleballs. Default 1280.")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(frames_dir.glob("*.jpg"))
    if not jpgs:
        print(f"No jpgs in {frames_dir}")
        return

    model = YOLO(args.model)

    frames_with_ball = 0
    total_boxes = 0

    for jpg in tqdm(jpgs, desc="Pre-labeling"):
        results = model(str(jpg), conf=args.conf, imgsz=args.imgsz, verbose=False,
                        classes=[COCO_SPORTS_BALL])
        r = results[0]
        h, w = r.orig_shape

        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                # Rewrite class 32 → 0 (single-class pickleball dataset).
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        shutil.copy2(jpg, out_dir / jpg.name)
        # Always write a txt (empty = no ball), so Roboflow registers the frame as annotated.
        (out_dir / f"{jpg.stem}.txt").write_text("\n".join(lines))

        if lines:
            frames_with_ball += 1
            total_boxes += len(lines)

    print(f"\n{len(jpgs)} frames processed.")
    print(f"  {frames_with_ball} frames have >=1 proposed box ({100*frames_with_ball/len(jpgs):.1f}%)")
    print(f"  {total_boxes} total proposed boxes (avg {total_boxes/max(frames_with_ball,1):.2f}/frame)")
    print(f"Output: {out_dir}")
    print("\nNext: zip and upload to Roboflow. See training/README.md.")


if __name__ == "__main__":
    main()
