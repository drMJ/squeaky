import glob
import os
import unittest

import cv2

from squeaky.vision import RobotDetector, RobotTracker

IMAGES = os.path.join(os.path.dirname(__file__), "images")

# approximate robot centre (pixels) in each sample photo (1365x768)
EXPECTED = {
    "WIN_20261122_15_56_34_Pro.jpg": (670, 560),
    "WIN_20261122_15_56_43_Pro.jpg": (570, 390),
    "WIN_20261122_15_56_53_Pro.jpg": (640, 340),
    "WIN_20261122_15_56_59_Pro.jpg": (640, 390),
    # live rear-camera frame (1280x720): the glossy floor in the lower left is brighter than the robot
    "live_rear_camera.jpg": (360, 450),
}
GLOSSY_FLOOR = (181, 584)  # in live_rear_camera.jpg


class VisionTest(unittest.TestCase):
    def test_detects_robot_in_sample_photos(self):
        paths = sorted(glob.glob(os.path.join(IMAGES, "*.jpg")))
        self.assertEqual(len(paths), len(EXPECTED))
        for path in paths:
            frame = cv2.imread(path)
            h, w = frame.shape[:2]
            track = RobotTracker().update(frame)
            with self.subTest(os.path.basename(path)):
                self.assertTrue(track.found)
                ex, ey = EXPECTED[os.path.basename(path)]
                x, y, bw, bh = track.bbox
                self.assertTrue(x <= ex <= x + bw and y <= ey <= y + bh, (track.bbox, (ex, ey)))
                self.assertLess(bw * bh, 0.08 * w * h)

    def test_ignores_glossy_floor(self):
        frame = cv2.imread(os.path.join(IMAGES, "live_rear_camera.jpg"))
        candidates, _, _ = RobotDetector().candidates(frame)
        self.assertEqual(len(candidates), 1)
        x, y, bw, bh = candidates[0][2]
        gx, gy = GLOSSY_FLOOR
        self.assertFalse(x <= gx <= x + bw and y <= gy <= y + bh)

    def test_click_calibration(self):
        frame = cv2.imread(os.path.join(IMAGES, "live_rear_camera.jpg"))
        detector = RobotDetector(hue_range=(45, 65), saturation_range=(100, 255))  # green: nothing in the scene
        tracker = RobotTracker(detector)
        self.assertFalse(tracker.update(frame).found)
        tracker.seed(*EXPECTED["live_rear_camera.jpg"], frame)
        h0, h1 = detector.hue_range
        self.assertTrue(h0 <= 105 <= h1, detector.hue_range)
        x, y, bw, bh = tracker.update(frame).bbox
        ex, ey = EXPECTED["live_rear_camera.jpg"]
        self.assertTrue(x <= ex <= x + bw and y <= ey <= y + bh)

    def test_tracks_motion_between_frames(self):
        frames = [cv2.imread(os.path.join(IMAGES, name)) for name in
                  ("WIN_20261122_15_56_53_Pro.jpg", "WIN_20261122_15_56_34_Pro.jpg")]
        tracker = RobotTracker(smoothing=0.0, max_jump=0.5)
        first = tracker.update(frames[0], now=0.0)
        second = tracker.update(frames[1], now=1.0)
        self.assertTrue(first.found and second.found)
        self.assertGreater(second.y, first.y)  # the robot came closer to the camera
        self.assertEqual(second.heading(), "closer")

    def test_lost_after_timeout(self):
        frame = cv2.imread(os.path.join(IMAGES, "WIN_20261122_15_56_43_Pro.jpg"))
        tracker = RobotTracker(lost_after=0.5)
        self.assertTrue(tracker.update(frame, now=0.0).found)
        blank = frame.copy()
        blank[:] = 120
        self.assertTrue(tracker.update(blank, now=0.3).found)
        self.assertFalse(tracker.update(blank, now=0.8).found)


if __name__ == "__main__":
    unittest.main()
