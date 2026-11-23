# squeaky

Voice-controlled littleBits R2-D2 (a white paper-box droid) with rear-camera tracking, for a Surface tablet console.

- **Robot** ([squeaky/robot.py](squeaky/robot.py)): BLE control of the droid's `w32 ControlHub` with
  [bleak](https://github.com/hbldh/bleak), using the frames from `C:\code\R2D2\R2D2-mcp.py`. Writes are spaced 40 ms
  apart (the hub drops the link when flooded). A background watchdog thread checks the link every 2 s, reconnects
  with backoff when it drops, sends a keepalive while idle and always sends stop + straight on (re)connect.
- **Agent** ([squeaky/agent.py](squeaky/agent.py)): a groc (`C:\code\groc`) realtime voice model, configured for
  speech in, commands out. It cannot talk (its audio is never played); it only calls `speak <sound>`,
  `move <forward|backward|left|right> [seconds]` and `stop`. A watchdog replaces the realtime session if the server
  closes it.
- **Vision** ([squeaky/vision.py](squeaky/vision.py)): tracks the robot in the rear camera by hue/saturation (the
  white box reads blue-tinted to the camera, the wood floor and its glare are unsaturated), shows an annotated window
  and logs position and heading. Each move is also logged with the displacement the camera saw.

Everything is logged to the console: what the agent heard, the commands it issued, BLE link events, and tracking.

## Run

```powershell
C:\venv\squeaky\Scripts\python.exe -m squeaky                # robot + voice agent + camera
C:\venv\squeaky\Scripts\python.exe -m squeaky --list-cameras # find the rear camera index (camera.index)
C:\venv\squeaky\Scripts\python.exe -m squeaky --no-agent     # console commands only
C:\venv\squeaky\Scripts\python.exe -m squeaky --no-robot     # no BLE, robot commands are only logged
```

Console input: `speak <sound>`, `move <direction> [seconds]`, `stop`, `status`, `sounds`, `quit`; any other text is
sent to the agent as if spoken. Camera window: click the robot to recalibrate its colour and re-seed the track,
`m` shows the detection mask, `r` resets the track, `q` quits.

Settings (BLE name/address, speeds, `invert_drive` if forward and backward are swapped, camera index, colour
ranges, realtime model) are in [config.yaml](config.yaml).

## Setup

```powershell
C:\venv\squeaky\Scripts\python.exe -m pip install -r requirements.txt
C:\venv\squeaky\Scripts\python.exe -m pip install --no-deps -e C:\code\groc
```

`win_aec` is not on PyPI; it is built from `C:\code\win-aec` (or copied from a venv with the same Python version).

**Smart App Control:** the tablet has Windows Smart App Control on, which only loads unsigned DLLs that have cloud
reputation. The newest `winrt-runtime` 4.x (bleak's Windows Bluetooth bindings) is blocked
(`ImportError: DLL load failed while importing _winrt: An Application Control policy has blocked this file`), so
[requirements.txt](requirements.txt) pins the winrt packages to 3.2.1, which load. Smart App Control has no per-file
exceptions; the alternative is turning it off (Windows Security > App & browser control > Smart App Control).

**Connecting:** the droid advertises for only ~30 s after power-on and not at all while another device (e.g. the
`C:\code\R2D2\R2D2.html` page) is connected. If the log keeps showing `'w32 ControlHub' not found`, disconnect other
controllers and power-cycle the droid; the watchdog picks it up on its next attempt.

## Tests

```powershell
C:\venv\squeaky\Scripts\python.exe -m unittest discover -s tests
```
