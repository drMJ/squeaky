"""squeaky: voice-controlled littleBits R2-D2 with rear-camera tracking.

Usage: python -m squeaky [--config config.yaml] [--camera N] [--no-agent] [--no-vision] [--no-robot] [--list-cameras]

Console input while running:
  speak <sound> | move <direction> [seconds] | stop    send a command straight to the robot
  status                                             robot link and tracking state
  anything else                                      sent to the agent as if spoken
  quit                                               exit (also Ctrl+C, or q in the camera window)
"""
import argparse
import logging
import os
import sys
import threading

import yaml

from .r2d2_protocol import SOUND_NAMES
from .robot import DIRECTIONS, R2D2Robot

log = logging.getLogger("squeaky")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT_DIR, "config.yaml")
FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"


def setup_logging(level="INFO", file=None):
    handlers = [logging.StreamHandler(sys.stderr)]
    if file:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    squeaky = logging.getLogger("squeaky")
    squeaky.handlers[:] = handlers
    squeaky.setLevel(level)
    squeaky.propagate = False
    try:
        from groc.eventlog import configure
        # groc logs the agent side (connection, barge-in, tool calls) with the same format
        configure(console_level=level, file=file, file_level=level)
    except ImportError:
        pass


def load_config(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class App:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.quit = threading.Event()
        self.robot = None
        self.agent = None
        self.vision = None
        self._move_start = None

    def run(self):
        robot_cfg = self.cfg.get("robot") or {}
        agent_cfg = self.cfg.get("agent") or {}
        camera_cfg = self.cfg.get("camera") or {}
        vision_cfg = self.cfg.get("vision") or {}

        self.robot = R2D2Robot(**robot_cfg) if not self.args.no_robot else _OfflineRobot()
        self.robot.motion_listeners.append(self._on_motion)
        self.robot.start()

        if not self.args.no_vision:
            from .vision import CameraTracker, RobotDetector, RobotTracker
            index = self.args.camera if self.args.camera is not None else camera_cfg.get("index", 1)
            detector = RobotDetector(**(vision_cfg.get("detector") or {}))
            tracker = RobotTracker(detector, **(vision_cfg.get("tracker") or {}))
            self.vision = CameraTracker(index=index, resolution=camera_cfg.get("resolution", (1280, 720)),
                                        tracker=tracker, show=camera_cfg.get("show", True),
                                        log_interval=float(vision_cfg.get("log_interval", 1.0)),
                                        on_quit=self.quit.set)
            self.vision.start()

        if not self.args.no_agent:
            from .agent import RobotAgent
            self.agent = RobotAgent(self.robot, realtime=agent_cfg.get("realtime", "gemini"),
                                    prompt=agent_cfg.get("prompt"))
            self.agent.start()
            threading.Thread(target=self._audio_loop, name="audio", daemon=True).start()

        threading.Thread(target=self._console_loop, name="console", daemon=True).start()
        log.info(f"ready. type: speak <sound> | move <{'|'.join(DIRECTIONS)}> [s] | stop | status | quit"
                 + (" | anything else is sent to the agent" if self.agent else ""))
        try:
            while not self.quit.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        log.info("shutting down")
        if self.agent:
            self.agent.close()
        if self.vision:
            self.vision.stop()
        if self.robot:
            self.robot.close()

    def _audio_loop(self):
        try:
            self.agent.run_audio()
        except Exception as e:
            log.error(f"audio loop stopped: {type(e).__name__}: {e}")
            self.quit.set()

    def _console_loop(self):
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            word = text.split()[0].lower()
            if word in ("quit", "exit"):
                self.quit.set()
                return
            if word == "status":
                log.info(f"status: robot {'connected' if self.robot.connected else 'NOT connected'}"
                         + (f"; vision: {self.vision.track.describe()}" if self.vision else ""))
            elif word == "sounds":
                log.info(f"sounds: {', '.join(SOUND_NAMES)}")
            elif word in ("speak", "move", "stop") and _is_command(text):
                log.info(f"console -> robot: {text} -> {self.robot.execute(text)}")
            elif self.agent:
                self.agent.send_text(text)
            elif word in ("speak", "move", "stop"):
                log.info(f"console -> robot: {text} -> {self.robot.execute(text)}")
            else:
                log.warning(f"unknown input {text!r} (agent disabled)")

    def _on_motion(self, event, direction, duration):
        """Relates each move to what the camera saw, e.g. to check the drive direction (robot.invert_drive)."""
        if not self.vision:
            return
        track = self.vision.track
        if event == "start":
            self._move_start = track if track.found else None
            return
        start, self._move_start = self._move_start, None
        if not (start and track.found):
            log.info(f"vision: move {direction} done; robot not tracked")
            return
        dx, dy = track.x - start.x, track.y - start.y
        log.info(f"vision: move {direction} {duration:.1f}s -> robot moved dx={dx:+.2f} dy={dy:+.2f} "
                 f"(now {track.describe()})")


def _is_command(text):
    from .robot import CommandError, parse_command
    try:
        parse_command(text)
        return True
    except CommandError:
        return False


class _OfflineRobot:
    """Stand-in for --no-robot: logs commands instead of sending them."""
    connected = False

    def __init__(self):
        self.motion_listeners = []
        self.log = logging.getLogger("squeaky.robot")

    def start(self):
        self.log.info("robot: offline mode (--no-robot), commands are only logged")

    def close(self):
        pass

    def command(self, verb, args=None):
        self.log.info(f"robot (offline): {verb} {args or ''}")
        return "ok (robot offline)"

    def execute(self, text):
        from .robot import CommandError, parse_command
        try:
            return self.command(*parse_command(text))
        except CommandError as e:
            return f"error: {e}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="squeaky", description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file (default: config.yaml)")
    parser.add_argument("--camera", type=int, help="camera index (overrides camera.index)")
    parser.add_argument("--no-agent", action="store_true", help="no voice agent; console commands only")
    parser.add_argument("--no-vision", action="store_true", help="no camera tracking")
    parser.add_argument("--no-robot", action="store_true", help="no BLE; robot commands are only logged")
    parser.add_argument("--list-cameras", action="store_true", help="list camera indices and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log_cfg = cfg.get("log") or {}
    setup_logging(log_cfg.get("level", "INFO"), log_cfg.get("file"))

    if args.list_cameras:
        from .vision import list_cameras
        for index, w, h in list_cameras():
            log.info(f"camera {index}: {w}x{h}")
        return
    App(cfg, args).run()


if __name__ == "__main__":
    main()
