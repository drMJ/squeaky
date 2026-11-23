"""Tracks the robot (a white paper box on wheels) in the tablet's rear camera.

Detection by hue and saturation: the camera's auto white balance is set by the warm wood floor, so the white paper
box shows up blue-tinted (hue ~100-107, saturation ~75-150 on OpenCV's 0-179 / 0-255 scales), while the floor and the
glare on it are nearly unsaturated (saturation < 25). Each frame is downscaled and converted to HSV; pixels inside
hue_range and saturation_range and not darker than the frame's median value + value_offset (drops shadows) are kept,
cleaned up with morphology, and the blobs are scored by size, by distance to the last position while tracking, and
down-weighted when cut by the frame edge (window reflections). Clicking the robot in the camera window recalibrates
the hue/saturation ranges from the clicked pixels and re-seeds the track.
Keys in the window: m = show mask, r = reset track, q = quit.
"""
import logging
import math
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("squeaky.vision")

WINDOW = "squeaky - robot tracking"


@dataclass
class Track:
    found: bool = False
    x: float = 0.0           # centre, normalized 0..1 (left -> right)
    y: float = 0.0           # centre, normalized 0..1 (top -> bottom)
    bbox: tuple = None       # (x, y, w, h) in frame pixels
    vx: float = 0.0          # velocity, normalized units per second
    vy: float = 0.0
    timestamp: float = 0.0

    def describe(self):
        if not self.found:
            return "robot not visible"
        horiz = "left" if self.x < 0.33 else "right" if self.x > 0.67 else "centre"
        depth = "far" if self.y < 0.4 else "near" if self.y > 0.7 else "middle"
        text = f"robot at x={self.x:.2f} y={self.y:.2f} ({horiz}, {depth})"
        heading = self.heading()
        return text + (f", moving {heading}" if heading else ", still")

    def heading(self, min_speed=0.03):
        if math.hypot(self.vx, self.vy) < min_speed:
            return None
        if abs(self.vx) > abs(self.vy):
            return "right" if self.vx > 0 else "left"
        return "away" if self.vy < 0 else "closer"


class RobotDetector:
    def __init__(self, process_width=640, hue_range=(90, 130), saturation_range=(50, 255), value_offset=-10,
                 min_area=0.002, max_area=0.3,
                 min_solidity=0.45):
        self.process_width = process_width
        self.hue_range = tuple(hue_range)
        self.saturation_range = tuple(saturation_range)
        self.value_offset = value_offset
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity

    def _hsv(self, frame):
        h, w = frame.shape[:2]
        scale = self.process_width / w if w > self.process_width else 1.0
        small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale != 1.0 else frame
        return cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV), scale

    def calibrate(self, frame, x, y, radius=12):
        """Sets the hue/saturation ranges from the pixels around frame pixel (x, y), which must be on the robot."""
        hsv, scale = self._hsv(frame)
        px, py, r = round(x * scale), round(y * scale), max(2, round(radius * scale))
        patch = hsv[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1].reshape(-1, 3).astype(int)
        if patch.size == 0:
            return False
        hue = np.percentile(patch[:, 0], [10, 90])
        sat = np.percentile(patch[:, 1], [10, 90])
        self.hue_range = (max(0, int(hue[0]) - 10), min(179, int(hue[1]) + 10))
        self.saturation_range = (max(0, int(sat[0]) - 25), min(255, int(sat[1]) + 60))
        log.info(f"vision: calibrated hue={self.hue_range} saturation={self.saturation_range}")
        return True

    def candidates(self, frame):
        """Returns ([(cx, cy, (x, y, w, h), area_fraction, touches_border)] in frame pixels, mask, scale)."""
        hsv, scale = self._hsv(frame)
        hue, sat, val = cv2.split(hsv)
        (h0, h1), (s0, s1) = self.hue_range, self.saturation_range
        in_hue = (hue >= h0) & (hue <= h1) if h0 <= h1 else (hue >= h0) | (hue <= h1)  # h0 > h1 wraps red
        v_min = float(np.median(val)) + self.value_offset
        mask = (in_hue & (sat >= s0) & (sat <= s1) & (val >= v_min)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mh, mw = mask.shape
        total = mh * mw
        found = []
        for c in contours:
            area = cv2.contourArea(c)
            frac = area / total
            if not self.min_area <= frac <= self.max_area:
                continue
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull <= 0 or area / hull < self.min_solidity:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            border = x <= 1 or y <= 1 or x + bw >= mw - 1 or y + bh >= mh - 1
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            found.append((cx / scale, cy / scale, (round(x / scale), round(y / scale), round(bw / scale),
                                                     round(bh / scale)), frac, border))
        return found, mask, scale


class RobotTracker:
    """Chooses the robot among the detector's candidates and smooths its position over time."""

    def __init__(self, detector=None, smoothing=0.5, lost_after=0.7, max_jump=0.35, border_penalty=0.3):
        self.detector = detector or RobotDetector()
        self.smoothing = smoothing      # EMA weight of the previous position
        self.lost_after = lost_after    # seconds without a detection before the track is lost
        self.max_jump = max_jump        # max movement between frames while tracking (fraction of the diagonal)
        self.border_penalty = border_penalty  # score factor for blobs cut by the frame edge (often reflections)
        self.track = Track()
        self.mask = None
        self._seed = None
        self._last_seen = 0.0

    def seed(self, x, y, frame=None):
        """Prefer the blob nearest to frame pixel (x, y) in the next frame; with a frame, also recalibrate the
        detector's colour ranges from the pixels there."""
        if frame is not None:
            self.detector.calibrate(frame, x, y)
        self._seed = (x, y)

    def reset(self):
        self.track = Track()
        self._seed = None

    def update(self, frame, now=None):
        now = time.monotonic() if now is None else now
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)
        candidates, self.mask, _ = self.detector.candidates(frame)
        prev = self.track
        best, best_score = None, 0.0
        for cand in candidates:
            cx, cy, _, frac, border = cand
            score = frac * (self.border_penalty if border else 1.0)
            if self._seed:
                score = 1.0 / (1.0 + math.hypot(cx - self._seed[0], cy - self._seed[1]))
            elif prev.found:
                dist = math.hypot(cx - prev.x * w, cy - prev.y * h) / diag
                if dist > self.max_jump:
                    continue
                score *= math.exp(-(dist / 0.1) ** 2)
            if score > best_score:
                best, best_score = cand, score
        self._seed = None

        if best is None:
            if prev.found and now - self._last_seen > self.lost_after:
                self.track = Track(timestamp=now)
            return self.track

        cx, cy, bbox, _, _ = best
        x, y = cx / w, cy / h
        if prev.found:
            a = self.smoothing
            dt = max(1e-3, now - prev.timestamp)
            sx, sy = a * prev.x + (1 - a) * x, a * prev.y + (1 - a) * y
            vx = a * prev.vx + (1 - a) * (sx - prev.x) / dt
            vy = a * prev.vy + (1 - a) * (sy - prev.y) / dt
            self.track = Track(True, sx, sy, bbox, vx, vy, now)
        else:
            self.track = Track(True, x, y, bbox, 0.0, 0.0, now)
        self._last_seen = now
        return self.track


class CameraTracker:
    """Background thread: grabs frames from the camera, tracks the robot, logs changes and shows an annotated view."""

    def __init__(self, index=1, resolution=(1280, 720), tracker=None, show=True, log_interval=1.0, min_move=0.03,
                 on_quit=None):
        self.index = index
        self.resolution = tuple(resolution) if resolution else None
        self.tracker = tracker or RobotTracker()
        self.show = show
        self.log_interval = log_interval
        self.min_move = min_move
        self.on_quit = on_quit
        self.show_mask = False
        self._lock = threading.Lock()
        self._track = Track()
        self._frame = None
        self._trail = []
        self._running = False
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._last_log = (0.0, None)

    @property
    def track(self):
        with self._lock:
            return self._track

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=3)

    def _open(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if self.resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"vision: camera {self.index} opened at {w}x{h}")
        return cap

    def _run(self):
        cap = None
        window_ready = False
        while self._running:
            if cap is None:
                cap = self._open()
                if cap is None:
                    log.warning(f"vision: cannot open camera {self.index}; retrying in 2s")
                    time.sleep(2)
                    continue
            ok, frame = cap.read()
            if not ok:
                log.warning("vision: camera read failed; reopening")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            self._frame = frame
            track = self.tracker.update(frame)
            with self._lock:
                self._track = track
            self._log(track)
            if self.show:
                if not window_ready:
                    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(WINDOW, self._on_mouse)
                    window_ready = True
                cv2.imshow(WINDOW, self._annotate(frame, track))
                self._on_key(cv2.waitKey(1) & 0xFF)
        if cap is not None:
            cap.release()
        if window_ready:
            cv2.destroyWindow(WINDOW)

    def _log(self, track):
        last_time, last = self._last_log
        now = time.monotonic()
        if last is None or track.found != last.found:
            log.info(f"vision: {track.describe()}" if track.found else "vision: robot lost")
        elif track.found and now - last_time >= self.log_interval and \
                math.hypot(track.x - last.x, track.y - last.y) >= self.min_move:
            log.info(f"vision: {track.describe()}")
        else:
            return
        self._last_log = (now, track)

    def _annotate(self, frame, track):
        view = frame.copy()
        if self.show_mask and self.tracker.mask is not None:
            mask = cv2.resize(self.tracker.mask, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_NEAREST)
            view[mask > 0] = (0.5 * view[mask > 0] + (0, 127, 0)).astype(np.uint8)
        h, w = view.shape[:2]
        if track.found:
            self._trail.append((int(track.x * w), int(track.y * h)))
            self._trail = self._trail[-40:]
            x, y, bw, bh = track.bbox
            cv2.rectangle(view, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            centre = (int(track.x * w), int(track.y * h))
            cv2.circle(view, centre, 5, (0, 0, 255), -1)
            end = (int(centre[0] + track.vx * w * 0.5), int(centre[1] + track.vy * h * 0.5))
            cv2.arrowedLine(view, centre, end, (0, 255, 255), 2)
        else:
            self._trail.clear()
        for a, b in zip(self._trail, self._trail[1:]):
            cv2.line(view, a, b, (255, 0, 255), 2)
        cv2.putText(view, track.describe(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return view

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            log.info(f"vision: robot clicked at ({x}, {y}); recalibrating colour and re-seeding the track")
            self.tracker.seed(x, y, self._frame)

    def _on_key(self, key):
        if key == ord("m"):
            self.show_mask = not self.show_mask
        elif key == ord("r"):
            log.info("vision: track reset")
            self.tracker.reset()
        elif key in (ord("q"), 27) and self.on_quit:
            self.on_quit()


def list_cameras(max_index=5):
    """Probe camera indices; on Surface tablets the rear camera is usually index 1 (front is 0)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if cap.isOpened():
            ok, frame = cap.read()
            found.append((i, frame.shape[1], frame.shape[0]) if ok else (i, 0, 0))
        cap.release()
    return found
