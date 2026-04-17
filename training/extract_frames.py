"""Extract frames from pickleball videos for ball-detection training.

Usage:
    python training/extract_frames.py --videos inputs/video1.mp4 inputs/video2.mp4 \
        --out training/dataset/frames --stride 15

Samples every Nth frame. Optional perceptual-hash dedup drops near-duplicate
frames (useful when the ball is off-screen during timeouts / between points).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm


def phash(gray: "cv2.Mat", hash_size: int = 8) -> int:
    """Tiny difference-hash. Fast, no extra deps."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_from_video(
    video_path: Path,
    out_dir: Path,
    stride: int,
    dedup_threshold: int,
    jpg_quality: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  skip: cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video_path.stem
    written = 0
    last_hash: int | None = None

    for frame_idx in tqdm(range(total), desc=stem, leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            continue

        if dedup_threshold > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = phash(gray)
            if last_hash is not None and hamming(h, last_hash) <= dedup_threshold:
                continue
            last_hash = h

        out_path = out_dir / f"{stem}_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        written += 1

    cap.release()
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", nargs="+", required=True, help="Video files to extract from")
    parser.add_argument("--out", required=True, help="Output directory for jpgs")
    parser.add_argument("--stride", type=int, default=15,
                        help="Sample every Nth frame. Default 15 (~2 fps @ 30 fps source).")
    parser.add_argument("--dedup-threshold", type=int, default=4,
                        help="Hamming distance below which consecutive sampled frames are "
                             "considered duplicates and skipped. 0 disables dedup. Default 4.")
    parser.add_argument("--jpg-quality", type=int, default=92, help="JPEG quality 1-100. Default 92.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for v in args.videos:
        vp = Path(v)
        if not vp.exists():
            print(f"Warning: {vp} does not exist, skipping.")
            continue
        n = extract_from_video(vp, out_dir, args.stride, args.dedup_threshold, args.jpg_quality)
        print(f"{vp.name}: {n} frames")
        total_written += n

    print(f"\nWrote {total_written} frames to {out_dir}")
    if total_written == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
