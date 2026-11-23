"""Thread-safe controller for the littleBits R2-D2 over BLE (bleak).

A private asyncio loop thread owns the BLE client. A watchdog thread checks the link every check_interval seconds,
reconnects with backoff when it drops, and sends a keepalive stop frame while idle (the droid auto-sleeps and the
Windows BLE stack does not always report dropped links until a write fails).

Public methods (speak / move / stop / execute) never block on BLE: they queue work on the loop and return a short
status string, which is what the agent sees as the tool result.
"""
import asyncio
import logging
import threading
import time

from . import r2d2_protocol as proto

log = logging.getLogger("squeaky.robot")

DIRECTIONS = ("forward", "backward", "left", "right")
_DIRECTION_ALIASES = {"back": "backward", "backwards": "backward", "reverse": "backward", "ahead": "forward",
                      "forwards": "forward"}
MAX_QUEUED_MOVES = 5


class CommandError(ValueError):
    pass


def parse_command(text):
    """'speak beep' / 'move left [seconds]' / 'stop' -> (verb, args dict). Raises CommandError."""
    words = str(text).strip().lower().split()
    if not words:
        raise CommandError("empty command")
    verb, rest = words[0], words[1:]
    if verb == "stop" and not rest:
        return "stop", {}
    if verb == "speak" and rest:
        return "speak", {"sound": " ".join(rest)}
    if verb == "move" and rest:
        args = {"direction": rest[0]}
        if len(rest) == 2:
            try:
                args["duration"] = float(rest[1])
            except ValueError:
                raise CommandError(f"bad duration {rest[1]!r}") from None
        elif len(rest) > 2:
            raise CommandError(f"usage: move <{'|'.join(DIRECTIONS)}> [seconds]")
        return "move", args
    raise CommandError(f"unknown command {text!r}; use: speak <sound> | move <direction> [seconds] | stop")


def normalize_direction(direction):
    d = str(direction).strip().lower()
    d = _DIRECTION_ALIASES.get(d, d)
    if d not in DIRECTIONS:
        raise CommandError(f"unknown direction {direction!r}; use one of {', '.join(DIRECTIONS)}")
    return d


class R2D2Robot:
    def __init__(self, name=proto.DEVICE_NAME, address=None, speed=0.6, turn_speed=0.5, move_duration=1.0,
                 max_move_duration=5.0, invert_drive=False, check_interval=2.0, keepalive=20.0, write_gap=0.04,
                 scan_timeout=10.0, max_backoff=15.0, client_factory=None, scanner=None):
        if client_factory is None or scanner is None:
            from bleak import BleakClient, BleakScanner
            client_factory = client_factory or BleakClient
            scanner = scanner or BleakScanner
        self.name = name
        self.address = address
        self.speed = speed
        self.turn_speed = turn_speed
        self.move_duration = move_duration
        self.max_move_duration = max_move_duration
        self.invert_drive = invert_drive
        self.check_interval = check_interval
        self.keepalive = keepalive
        self.write_gap = write_gap
        self.scan_timeout = scan_timeout
        self.max_backoff = max_backoff
        self._client_factory = client_factory
        self._scanner = scanner
        self.motion_listeners = []  # callables (event, direction, duration); event is "start" or "end"

        self._client = None
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, name="ble-loop", daemon=True)
        self._watchdog = threading.Thread(target=self._watch, name="ble-watchdog", daemon=True)
        self._closing = threading.Event()
        self._wake = threading.Event()
        self._write_lock = None
        self._moves = None
        self._move_task = None
        self._motion_task = None
        self._last_write = 0.0
        self._moving = False
        self.reconnects = 0

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        self._loop_thread.start()
        asyncio.run_coroutine_threadsafe(self._init_loop(), self._loop).result(timeout=5)
        self._watchdog.start()
        log.info(f"robot: looking for '{self.address or self.name}' (watchdog every {self.check_interval:.0f}s)")

    def close(self):
        self._closing.set()
        self._wake.set()
        if self._loop_thread.is_alive():
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=5)
            except Exception as e:
                log.warning(f"robot: shutdown error: {e}")
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=2)
        if self._watchdog.is_alive():
            self._watchdog.join(timeout=2)
        log.info("robot: closed")

    @property
    def connected(self):
        client = self._client
        return bool(client and client.is_connected)

    def wait_connected(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.connected:
            if self._closing.is_set() or (deadline and time.monotonic() > deadline):
                return False
            time.sleep(0.1)
        return True

    # ---------------------------------------------------------------- public commands
    def execute(self, text):
        """Run a text command ('speak beep', 'move left', 'stop'); returns a status string."""
        try:
            verb, args = parse_command(text)
        except CommandError as e:
            log.warning(f"robot: {e}")
            return f"error: {e}"
        return self.command(verb, args)

    def command(self, verb, args=None):
        args = args or {}
        try:
            if verb == "speak":
                return self.speak(args.get("sound", ""))
            if verb == "move":
                return self.move(args.get("direction", ""), args.get("duration"))
            if verb == "stop":
                return self.stop()
            raise CommandError(f"unknown command {verb!r}")
        except CommandError as e:
            log.warning(f"robot: {e}")
            return f"error: {e}"

    def speak(self, sound):
        name, frame = proto.find_sound(sound)
        if frame is None:
            raise CommandError(f"unknown sound {sound!r}; available: {', '.join(proto.SOUND_NAMES)}")
        if not self._ready("speak " + name):
            return "error: robot is not connected (reconnecting)"
        log.info(f"robot: speak {name}")
        self._submit(self._write(frame), f"speak {name}")
        return f"playing sound {name}"

    def move(self, direction, duration=None):
        direction = normalize_direction(direction)
        duration = self.move_duration if duration is None else float(duration)
        duration = max(0.1, min(self.max_move_duration, duration))
        if not self._ready(f"move {direction}"):
            return "error: robot is not connected (reconnecting)"
        try:
            self._call(self._moves.put_nowait, (direction, duration))
        except asyncio.QueueFull:
            log.warning(f"robot: move {direction} dropped, {MAX_QUEUED_MOVES} moves already queued")
            return "error: too many moves queued"
        log.info(f"robot: move {direction} {duration:.1f}s queued")
        return f"moving {direction} for {duration:.1f}s"

    def stop(self):
        log.info("robot: stop")
        if not self.connected:
            self._call(self._drain_moves)
            log.warning("robot: stop while not connected; a stop is sent on reconnect")
            return "robot is not connected; it will be stopped on reconnect"
        self._submit(self._stop(), "stop")
        return "stopped"

    # ---------------------------------------------------------------- loop-side implementation
    async def _init_loop(self):
        self._write_lock = asyncio.Lock()
        self._moves = asyncio.Queue(maxsize=MAX_QUEUED_MOVES)
        self._motion_task = asyncio.ensure_future(self._motion_loop())

    async def _shutdown(self):
        if self._motion_task:
            self._motion_task.cancel()
        if self.connected:
            try:
                await self._stop()
            except Exception:
                pass
        client, self._client = self._client, None
        if client:
            try:
                await client.disconnect()
            except Exception:
                pass

    async def _connect(self):
        target = self.address
        if not target:
            target = await self._scanner.find_device_by_name(self.name, timeout=self.scan_timeout)
            if target is None:
                raise ConnectionError(f"'{self.name}' not found (power-cycle the droid: it only advertises ~30s)")
        client = self._client_factory(target, disconnected_callback=self._on_disconnect)
        await client.connect()
        self._client = client
        address = getattr(target, "address", target)
        # safety: never resume a motion that was running when the link dropped
        await self._write(proto.STOP_FRAME)
        await self._write(proto.STRAIGHT_FRAME)
        return address

    def _on_disconnect(self, client):
        if client is not self._client:
            return
        self._client = None
        log.warning("robot: BLE link dropped")
        self._drain_moves()
        if self._move_task:
            self._move_task.cancel()
        self._wake.set()

    async def _write(self, frame):
        async with self._write_lock:
            wait = self.write_gap - (time.monotonic() - self._last_write)
            if wait > 0:
                await asyncio.sleep(wait)  # the hub drops the link if writes come faster than ~20-50ms
            client = self._client
            if not client or not client.is_connected:
                raise ConnectionError("robot is not connected")
            try:
                await client.write_gatt_char(proto.CHAR_UUID, bytes.fromhex(frame), response=True)
                log.debug(f"robot: TX {frame}")
            except Exception as e:
                log.warning(f"robot: write failed ({type(e).__name__}: {e}); dropping link")
                await self._drop(client)
                raise ConnectionError(f"write failed: {e}") from e
            finally:
                self._last_write = time.monotonic()

    async def _drop(self, client):
        if client is self._client:
            self._client = None
        self._drain_moves()
        self._wake.set()
        try:
            await client.disconnect()
        except Exception:
            pass

    async def _stop(self):
        self._drain_moves()
        if self._move_task:
            self._move_task.cancel()
        await self._write(proto.STOP_FRAME)
        await self._write(proto.STRAIGHT_FRAME)

    def _drain_moves(self):
        while self._moves and not self._moves.empty():
            self._moves.get_nowait()

    async def _motion_loop(self):
        while True:
            direction, duration = await self._moves.get()
            self._move_task = asyncio.ensure_future(self._do_move(direction, duration))
            try:
                await self._move_task
            except asyncio.CancelledError:
                if not self._move_task.cancelled():
                    raise  # the motion loop itself was cancelled
                log.info(f"robot: move {direction} interrupted")
            except ConnectionError as e:
                log.warning(f"robot: move {direction} failed: {e}")
            finally:
                self._move_task = None

    async def _do_move(self, direction, duration):
        steer = {"left": -1.0, "right": 1.0}.get(direction, 0.0)
        speed = {"forward": self.speed, "backward": -self.speed}.get(direction, self.turn_speed)
        self._moving = True
        self._notify("start", direction, duration)
        log.info(f"robot: moving {direction} for {duration:.1f}s")
        try:
            await self._write(proto.turn_frame(steer))
            await self._write(proto.drive_frame(speed, self.invert_drive))
            await asyncio.sleep(duration)
            await self._write(proto.STOP_FRAME)
            if steer:
                await self._write(proto.STRAIGHT_FRAME)
            log.info(f"robot: move {direction} done")
        finally:
            self._moving = False
            self._notify("end", direction, duration)

    def _notify(self, event, direction, duration):
        for listener in list(self.motion_listeners):
            try:
                listener(event, direction, duration)
            except Exception as e:
                log.warning(f"robot: motion listener failed: {e}")

    # ---------------------------------------------------------------- helpers
    def _ready(self, what):
        if self.connected:
            return True
        log.warning(f"robot: '{what}' ignored, robot is not connected (watchdog is reconnecting)")
        self._wake.set()
        return False

    def _call(self, fn, *args):
        """Run a non-async loop-side function on the loop thread and return its result."""
        async def run():
            return fn(*args)
        return asyncio.run_coroutine_threadsafe(run(), self._loop).result(timeout=5)

    def _submit(self, coro, what):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def done(f):
            if not f.cancelled() and f.exception():
                log.warning(f"robot: {what} failed: {f.exception()}")
        future.add_done_callback(done)
        return future

    # ---------------------------------------------------------------- watchdog thread
    def _watch(self):
        backoff = 1.0
        while not self._closing.is_set():
            if not self.connected:
                self.reconnects += 1
                log.info(f"robot: connecting to '{self.address or self.name}' (attempt {self.reconnects})")
                future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
                try:
                    address = future.result(timeout=self.scan_timeout + 20)
                    log.info(f"robot: connected to {address}")
                    backoff = 1.0
                except Exception as e:
                    future.cancel()
                    log.warning(f"robot: connect failed ({type(e).__name__}: {e}); retrying in {backoff:.0f}s")
                    self._closing.wait(backoff)
                    backoff = min(self.max_backoff, backoff * 2)
                    continue
            elif self.keepalive and not self._moving and time.monotonic() - self._last_write > self.keepalive:
                # keeps the droid awake and detects links that died silently (the write fails)
                future = asyncio.run_coroutine_threadsafe(self._write(proto.STOP_FRAME), self._loop)
                try:
                    future.result(timeout=10)
                    log.debug("robot: keepalive ok")
                except Exception as e:
                    log.warning(f"robot: keepalive failed: {e}")
                    continue
            self._wake.wait(self.check_interval)
            self._wake.clear()
