"""Pluggable pose backends.

Two implementations, both returning ``{track_id: (num_kp, 3) ndarray}`` per frame
where the last column is per-keypoint confidence:

- ``UltralyticsBackend``: single-model YOLO pose + BYTETrack. COCO 17-keypoint
  (or WholeBody if a compatible custom weights file is supplied).
- ``WholeBodyBackend``: two-stage. Ultralytics person detector+tracker for
  bboxes and IDs, then rtmlib RTMPose (COCO-WholeBody, 133 kp) for keypoints.
  Indices 17-22 are foot keypoints, matching ``constants.WHOLEBODY_FOOT_INDICES``.
"""

from __future__ import annotations

import numpy as np

from pickleball.constants import MAX_PLAYERS_KEPT


class PoseBackend:
    """Abstract pose backend."""

    def track(self, frame, **kwargs) -> dict[int, np.ndarray]:
        raise NotImplementedError


class UltralyticsBackend(PoseBackend):
    """Single-model ultralytics pose + tracking (current default path)."""

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def track(self, frame, *, imgsz: int = 960, half: bool = False) -> dict[int, np.ndarray]:
        results = self.model.track(frame, persist=True, imgsz=imgsz, half=half, verbose=False)
        out: dict[int, np.ndarray] = {}
        if not results or results[0].boxes is None:
            return out
        boxes = results[0].boxes
        kps_all = results[0].keypoints
        if kps_all is None or boxes.id is None:
            return out
        ids = boxes.id.cpu().numpy().astype(int)
        kps = kps_all.data.cpu().numpy()  # (N, num_kp, 3)
        for i, tid in enumerate(ids):
            out[int(tid)] = kps[i]
        return out


class WholeBodyBackend(PoseBackend):
    """Ultralytics detector+tracker + rtmlib RTMPose (COCO-WholeBody, 133 kp).

    The ultralytics model here is a *person detector* (e.g. models/yolov8s.pt),
    not a pose model. We use it only for bounding boxes and track IDs.
    """

    # Pose ONNX URLs and input sizes per rtmlib mode. Mirrors rtmlib.Wholebody.MODE
    # (rtmlib/tools/solution/wholebody.py) but skips the YOLOX detector entry —
    # we use ultralytics for detection so loading YOLOX wastes startup + GPU memory.
    POSE_MODE = {
        "performance": {
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
            "input_size": (288, 384),
        },
        "balanced": {
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
            "input_size": (192, 256),
        },
        "lightweight": {
            "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
            "input_size": (192, 256),
        },
    }

    def __init__(
        self,
        detector_model_path: str = "models/yolov8s.pt",
        mode: str = "balanced",
        device: str = "cpu",
    ):
        from ultralytics import YOLO
        from rtmlib import RTMPose

        self.detector = YOLO(detector_model_path)
        cfg = self.POSE_MODE[mode]
        self.pose_model = RTMPose(
            cfg["url"],
            model_input_size=cfg["input_size"],
            backend="onnxruntime",
            device=device,
        )

    def track(self, frame, *, imgsz: int = 960, half: bool = False) -> dict[int, np.ndarray]:
        # COCO person class id is 0.
        results = self.detector.track(
            frame, persist=True, classes=[0], imgsz=imgsz, half=half, verbose=False
        )
        out: dict[int, np.ndarray] = {}
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return out
        boxes = results[0].boxes
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        if len(xyxy) == 0:
            return out

        # Keep the MAX_PLAYERS_KEPT largest bboxes — biggest bbox = closest to
        # the (net-center) camera. Pure image-space signal, so it survives
        # occluded/cropped feet that would break any homography-based gate.
        xyxy, ids = _keep_largest(xyxy, ids, MAX_PLAYERS_KEPT)

        keypoints, scores = self.pose_model(frame, xyxy)  # (N, 133, 2), (N, 133)
        kps = np.concatenate([keypoints, scores[..., None]], axis=-1)  # (N, 133, 3)
        for i, tid in enumerate(ids):
            out[int(tid)] = kps[i]
        return out


def _keep_largest(
    xyxy: np.ndarray,
    ids: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the n detections with the largest bbox area.

    Bbox pixel area is a reliable "closest to the camera" proxy for a net-center
    camera: on-court players are physically closest and appear largest, while
    adjacent-court players and spectators appear smaller. No homography
    dependency, so it doesn't misfire when feet are occluded or cropped.
    """
    if len(xyxy) <= n:
        return xyxy, ids
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    top = np.argsort(-area)[:n]
    return xyxy[top], ids[top]


def make_backend(
    backend: str,
    pose_model_path: str,
    detector_model_path: str = "yolov8x.pt",
    wholebody_mode: str = "performance",
    device: str = "cpu",
) -> PoseBackend:
    if backend == "ultralytics":
        return UltralyticsBackend(pose_model_path)
    if backend == "wholebody":
        return WholeBodyBackend(detector_model_path, mode=wholebody_mode, device=device)
    raise ValueError(f"Unknown pose backend: {backend!r}")
