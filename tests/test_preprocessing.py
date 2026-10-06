import cv2
import numpy as np
import pytest

from svcr.config import PreprocessConfig
from svcr.preprocessing import FrameStream, Preprocessor, assess_quality, load_image_bgr


def _pattern(h=600, w=900, value=None):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    if value is not None:
        img = np.clip(img // 8 + value, 0, 255).astype(np.uint8)
    return img


def test_fit_resize_keeps_aspect_and_normalises():
    frame = Preprocessor(PreprocessConfig(target_longest_side=384))(_pattern())
    assert frame.processed_size == (384, 256)
    assert frame.original_size == (900, 600)
    assert frame.image.dtype == np.uint8 and frame.image.shape == (256, 384, 3)
    t = frame.tensor
    assert tuple(t.shape) == (1, 3, 256, 384)
    assert float(t.min()) >= -1.0001 and float(t.max()) <= 1.0001


def test_center_crop_is_square():
    frame = Preprocessor(PreprocessConfig(target_longest_side=384, resize_mode="center_crop"))(_pattern())
    assert frame.processed_size == (384, 384)


def test_small_images_not_upscaled_by_default():
    frame = Preprocessor(PreprocessConfig(target_longest_side=1152))(_pattern(100, 150))
    assert frame.processed_size == (150, 100)


def test_dark_image_is_enhanced():
    dark = _pattern(value=5)
    cfg = PreprocessConfig(target_longest_side=384)
    assert assess_quality(dark, cfg)["is_dark"]
    frame = Preprocessor(cfg)(dark)
    assert "clahe_low_light" in frame.quality["enhancements"]
    assert frame.image.mean() > cv2.cvtColor(dark, cv2.COLOR_BGR2RGB).mean()


def test_blur_detection():
    cfg = PreprocessConfig()
    sharp = _pattern()
    blurry = cv2.GaussianBlur(sharp, (31, 31), 15)
    assert not assess_quality(sharp, cfg)["is_blurry"]
    assert assess_quality(blurry, cfg)["is_blurry"]


def test_load_from_path_and_missing(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), _pattern(50, 80))
    img, name = load_image_bgr(p)
    assert img.shape == (50, 80, 3) and name == str(p)
    with pytest.raises(FileNotFoundError):
        load_image_bgr(tmp_path / "nope.jpg")


def test_frame_stream_sampling(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(60):                               # 6 s of video, scene changes at 3 s
        frame = _pattern(120, 160) if i < 30 else _pattern(120, 160, value=200)
        writer.write(frame)
    writer.release()
    cfg = PreprocessConfig(sample_interval_s=1.0, skip_blurry_frames=False)
    with FrameStream(path, cfg) as stream:
        sampled = list(stream.sampled())
    assert 2 <= len(sampled) <= 6
    assert sampled[0].index == 0
