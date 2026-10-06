"""Module 1 - Input & Preprocessing (OpenCV).

Accepts a static image (path / URL / array / PIL image), a video file or a live
camera, and standardises every frame into a model-ready form:

    acquire frame -> quality check -> (low-light enhancement / denoise)
    -> resize or crop -> RGB uint8 image + normalised 1x3xHxW float tensor

The RGB image is what the SmolVLM processor consumes (it applies the same
normalisation internally); the normalised tensor is kept for Phase II
(Grad-CAM, Insertion-Deletion), which operates on pixel tensors directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import PreprocessConfig
from .schemas import PreprocessedFrame

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


# ----------------------------------------------------------------------------- loading
def load_image_bgr(source) -> tuple[np.ndarray, str]:
    """Load an image from a path, URL, numpy array (RGB or BGR) or PIL image as BGR uint8."""
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return img, "array"

    try:
        from PIL import Image
        if isinstance(source, Image.Image):
            return cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR), "pil"
    except ImportError:  # pragma: no cover
        pass

    src = str(source)
    if src.startswith(("http://", "https://")):
        import requests
        resp = requests.get(src, timeout=20)
        resp.raise_for_status()
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image from URL: {src}")
        return img, src

    path = Path(src).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)          # applies EXIF orientation
    if img is None:                                         # e.g. GIF / HEIC -> fall back to PIL
        from PIL import Image, ImageOps
        with Image.open(path) as pil:
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img, str(path)


# ----------------------------------------------------------------------------- quality
def assess_quality(bgr: np.ndarray, cfg: PreprocessConfig) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Measure blur at a fixed resolution so the score is comparable across image sizes.
    h, w = gray.shape
    s = 640.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_score": round(blur, 2),
        "is_blurry": blur < cfg.blur_threshold,
        "is_dark": brightness < cfg.low_light_threshold,
        "is_low_contrast": contrast < 25.0,
    }


def enhance_low_light(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB plus mild gamma lift - brightens without colour shift."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 0.8
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, table)


def resize_frame(bgr: np.ndarray, cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    target = int(cfg.target_longest_side)
    if cfg.resize_mode == "none" or target <= 0:
        return bgr, 1.0
    if cfg.resize_mode == "center_crop":
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = bgr[y0:y0 + side, x0:x0 + side]
        scale = target / side
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(crop, (target, target), interpolation=interp), scale
    if cfg.resize_mode != "fit":
        raise ValueError(f"Unknown resize_mode: {cfg.resize_mode}")
    scale = target / max(h, w)
    if scale > 1 and not cfg.upscale_small:
        return bgr, 1.0
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp), scale


def to_normalized_tensor(rgb: np.ndarray, mean, std):
    import torch
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0)


# ----------------------------------------------------------------------------- preprocessor
class Preprocessor:
    def __init__(self, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()

    def __call__(self, source, frame_index: int | None = None,
                 timestamp_s: float | None = None) -> PreprocessedFrame:
        bgr, name = load_image_bgr(source)
        return self.process_bgr(bgr, name, frame_index, timestamp_s)

    def process_bgr(self, bgr: np.ndarray, source: str = "array", frame_index: int | None = None,
                    timestamp_s: float | None = None) -> PreprocessedFrame:
        if bgr is None or bgr.size == 0:
            raise ValueError("Empty frame")
        cfg = self.cfg
        orig_h, orig_w = bgr.shape[:2]
        quality = assess_quality(bgr, cfg)
        applied = []
        if cfg.enhance_low_light and (quality["is_dark"] or quality["is_low_contrast"]):
            bgr = enhance_low_light(bgr)
            applied.append("clahe_low_light")
        if cfg.denoise:
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
            applied.append("denoise")
        bgr, scale = resize_frame(bgr, cfg)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = to_normalized_tensor(rgb, cfg.normalize_mean, cfg.normalize_std)
        quality["enhancements"] = applied
        if quality["is_blurry"]:
            log.warning("Input frame looks blurry (Laplacian variance %.1f) - predictions may degrade.",
                        quality["blur_score"])
        return PreprocessedFrame(
            image=rgb, tensor=tensor, original_size=(orig_w, orig_h),
            processed_size=(rgb.shape[1], rgb.shape[0]), scale=round(scale, 4), quality=quality,
            source=source, frame_index=frame_index, timestamp_s=timestamp_s,
        )


def preprocess_input(source, cfg: PreprocessConfig | None = None) -> PreprocessedFrame:
    """Functional form matching the report pseudocode."""
    return Preprocessor(cfg)(source)


# ----------------------------------------------------------------------------- streams
@dataclass
class RawFrame:
    bgr: np.ndarray
    index: int
    timestamp_s: float


def _hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(bgr, (160, 120)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


class FrameStream:
    """Iterate over frames of a video file or live camera.

    ``sampled()`` yields only frames worth analysing: at most one every
    ``sample_interval_s`` seconds, not blurry, and visibly different from the
    last analysed frame (scene-change detection).
    """

    def __init__(self, source: int | str, cfg: PreprocessConfig | None = None):
        self.cfg = cfg or PreprocessConfig()
        self.is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        self.source = int(source) if self.is_camera else str(source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._t0 = time.time()

    def __iter__(self) -> Iterator[RawFrame]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break
            ts = (time.time() - self._t0) if self.is_camera else idx / self.fps
            yield RawFrame(frame, idx, ts)
            idx += 1

    def sampled(self) -> Iterator[RawFrame]:
        interval = self.cfg.sample_interval_s
        last_check, last_yield, last_hist = -1e9, -1e9, None
        for raw in self:
            if raw.timestamp_s - last_check < interval:
                continue
            q = assess_quality(raw.bgr, self.cfg)
            if self.cfg.skip_blurry_frames and q["is_blurry"]:
                continue
            last_check = raw.timestamp_s
            hist = _hist(raw.bgr)
            # Same scene as last analysed frame: skip, but still refresh every 3 intervals
            # because gestures (e.g. a handshake starting) barely change the colour histogram.
            if last_hist is not None and raw.timestamp_s - last_yield < 3 * interval:
                dist = cv2.compareHist(last_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if dist < self.cfg.scene_change_threshold:
                    continue
            last_yield, last_hist = raw.timestamp_s, hist
            yield raw

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS
