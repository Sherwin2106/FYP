import cv2
import numpy as np

from svcr.video import select_key_moments, summarize_moments


def _clip(path, scenes, fps=10, blur_every_second=True):
    rng = np.random.default_rng(0)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for color, secs in scenes:
        base = (rng.integers(0, 60, (240, 320, 3)) + np.array(color)).clip(0, 255).astype(np.uint8)
        for f in range(secs * fps):
            frame = base.copy()
            cv2.circle(frame, (40 + (f * 7) % 240, 120), 20, (255, 255, 255), -1)
            if blur_every_second and f % fps in (0, 1):
                frame = cv2.GaussianBlur(frame, (21, 21), 8)
            w.write(frame)
    w.release()


def test_scenes_detected_and_spread(tmp_path):
    path = tmp_path / "three_scenes.avi"
    _clip(path, [((40, 40, 200), 4), ((200, 60, 40), 10), ((40, 180, 60), 20)])
    moments, info = select_key_moments(str(path), max_moments=7)
    assert info.scene_segments == 3                    # blurred frames are not scene cuts
    assert len(moments) == 7                           # long scenes split to cover the clip
    starts = sorted({m.segment[0] for m in moments})
    assert 4.0 in starts and 14.0 in starts            # real cuts are respected
    assert moments[-1].time_s > 25                     # the end of the clip is represented
    assert min(m.sharpness for m in moments) > 1000    # never a blurred frame


def test_cap_and_static_clip(tmp_path):
    path = tmp_path / "three_scenes.avi"
    _clip(path, [((40, 40, 200), 4), ((200, 60, 40), 10), ((40, 180, 60), 20)])
    moments, _ = select_key_moments(str(path), max_moments=2)
    assert len(moments) == 2

    static = tmp_path / "static.avi"
    _clip(static, [((90, 90, 90), 10)], blur_every_second=False)
    moments, _ = select_key_moments(str(static), max_moments=7)
    assert 1 <= len(moments) <= 2                      # no 7 near-identical moments


def test_summary_weighted_majority():
    def r(act, conf):
        task = {"label_id": act, "label": act.title(), "confidence": conf}
        return {"prediction": {"activity": task, "relationship": task, "intention": task}}
    s = summarize_moments([r("greeting", 0.9), r("greeting", 0.8), r("talking", 0.95)])
    assert s["activity"]["label_id"] == "greeting"
    assert s["activity"]["moments"] == 2
