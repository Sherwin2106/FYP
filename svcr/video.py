"""Key-moment selection for uploaded videos.

Analysing a frame takes ~30-45 s on a laptop GPU, so a video is reduced to a handful of
representative moments (default: at most 7). Selection works in two passes:

1. **Scan** the whole clip at a few frames per second, recording for each sampled frame
   its sharpness (variance of the Laplacian) and an HSV colour histogram of a small,
   smoothed copy. Only these statistics are kept, never the frames themselves, so long
   videos don't fill memory.
2. **Choose**:
   * split the clip into scene segments wherever the picture differs enough from the
     start of the current segment (Bhattacharyya distance between histograms); a change
     must persist into the next sample, so a single blurred or flashing frame is no cut;
   * if there are more segments than moments allowed, merge the shortest segment into
     its shorter neighbour until the count fits;
   * if there are fewer, split the longest segment in half while it is longer than
     ``min_split_s`` - one long scene (e.g. two people talking for a minute) still changes
     over time even when its colours don't; a short static clip stays at 1-2 moments;
   * from each segment take the sharpest frame from its central part (frames at the
     edges of a segment tend to be transition or motion-blurred frames).

A second pass over the video then reads just those frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SCAN_SAMPLES = 900          # cap on frames scored in pass 1 (sets the scan rate for long videos)


@dataclass
class KeyMoment:
    index: int
    frame_index: int
    time_s: float
    segment: tuple[float, float]     # (start_s, end_s) of the scene segment it represents
    sharpness: float
    bgr: np.ndarray = field(repr=False)


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    frames_scanned: int
    scene_segments: int


def _hist(bgr: np.ndarray) -> np.ndarray:
    # Small, smoothed frame + coarse bins: insensitive to blur, noise and small motion,
    # sensitive to an actual change of scene.
    small = cv2.GaussianBlur(cv2.resize(bgr, (64, 48), interpolation=cv2.INTER_AREA), (5, 5), 0)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def _sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = 640.0 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_key_moments(path: str, max_moments: int = 7, scan_fps: float = 4.0,
                       change_threshold: float = 0.35, min_split_s: float = 6.0,
                       progress=None) -> tuple[list[KeyMoment], VideoInfo]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("The video could not be opened. Please upload an MP4, MOV, AVI or WEBM file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = total / fps if total else 0.0
    rate = scan_fps if not duration else min(scan_fps, MAX_SCAN_SAMPLES / max(duration, 1e-6))
    step = max(1, round(fps / rate))

    # ---------------------------------------------------------------- pass 1: score frames
    times, idxs, sharp, hists = [], [], [], []
    i = 0
    while True:
        if i % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(i / fps)
            idxs.append(i)
            sharp.append(_sharpness(frame))
            hists.append(_hist(frame))
            if progress and len(times) % 20 == 0 and total:
                progress(min(0.95, i / total))
        elif not cap.grab():
            break
        i += 1
    cap.release()
    if not times:
        raise ValueError("No frames could be read from the video.")
    duration = max(duration, times[-1] + 1.0 / fps)
    n = len(times)

    def dist(a, b):
        return cv2.compareHist(hists[a], hists[b], cv2.HISTCMP_BHATTACHARYYA)

    # ---------------------------------------------------------------- scene segments
    segments: list[list[int]] = [[0, 0]]                 # inclusive sample-index ranges
    for k in range(1, n):
        start = segments[-1][0]
        persists = k + 1 >= n or dist(start, k + 1) > change_threshold
        if dist(start, k) > change_threshold and persists:
            segments.append([k, k])
        else:
            segments[-1][1] = k
    n_scenes = len(segments)

    def seg_len(s):                                       # seconds covered by a segment
        end_t = times[s[1] + 1] if s[1] + 1 < n else duration
        return end_t - times[s[0]]

    # too many: merge the shortest into its shorter neighbour
    while len(segments) > max_moments:
        j = min(range(len(segments)), key=lambda q: seg_len(segments[q]))
        if j == 0:
            nb = 1
        elif j == len(segments) - 1:
            nb = j - 1
        else:
            nb = j - 1 if seg_len(segments[j - 1]) <= seg_len(segments[j + 1]) else j + 1
        a, b = sorted((j, nb))
        segments[a] = [segments[a][0], segments[b][1]]
        del segments[b]

    # too few: split long segments so a long single scene is still covered over time
    while len(segments) < max_moments:
        j = max(range(len(segments)), key=lambda q: seg_len(segments[q]))
        s = segments[j]
        if seg_len(s) < min_split_s or s[1] - s[0] < 3:
            break
        mid = (s[0] + s[1]) // 2
        segments[j:j + 1] = [[s[0], mid], [mid + 1, s[1]]]

    # ---------------------------------------------------------------- sharpest frame per segment
    picks = []
    for s in segments:
        lo, hi = s
        trim = (hi - lo) // 10                            # avoid transition frames at the edges
        cand = range(lo + trim, hi - trim + 1) if hi - lo >= 4 else range(lo, hi + 1)
        best = max(cand, key=lambda k: sharp[k])
        end_t = times[s[1] + 1] if s[1] + 1 < n else duration
        picks.append((idxs[best], times[best], (times[lo], end_t), sharp[best]))

    # ---------------------------------------------------------------- pass 2: read the chosen frames
    wanted = {p[0] for p in picks}
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    i, last = 0, max(wanted)
    while i <= last:
        if i in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            frames[i] = frame
        elif not cap.grab():
            break
        i += 1
    cap.release()

    moments = [KeyMoment(index=k, frame_index=fi, time_s=round(t, 2),
                         segment=(round(a, 2), round(b, 2)), sharpness=round(sh, 1), bgr=frames[fi])
               for k, (fi, t, (a, b), sh) in enumerate(p for p in picks if p[0] in frames)]
    info = VideoInfo(duration_s=round(duration, 2), fps=round(fps, 2), width=width, height=height,
                     frames_scanned=n, scene_segments=n_scenes)
    return moments, info


def summarize_moments(results: list[dict]) -> dict:
    """Overall activity / relationship / intention across moments, weighted by confidence."""
    summary = {}
    for task in ("activity", "relationship", "intention"):
        votes: dict[str, dict] = {}
        for r in results:
            t = r["prediction"][task]
            v = votes.setdefault(t["label_id"], {"label": t["label"], "weight": 0.0, "count": 0})
            v["weight"] += t["confidence"]
            v["count"] += 1
        if not votes:
            continue
        best_id, best = max(votes.items(), key=lambda kv: (kv[1]["weight"], kv[1]["count"]))
        summary[task] = {"label_id": best_id, "label": best["label"], "moments": best["count"],
                         "share": round(best["count"] / len(results), 3),
                         "mean_confidence": round(best["weight"] / best["count"], 4)}
    return summary
