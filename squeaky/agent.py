"""Voice agent (groc realtime model): speech in, robot commands out.

The robot cannot talk, so the model's audio output is never played; it may only answer with the tools
speak <sound>, move <direction> and stop.
"""
import logging
import threading

from .r2d2_protocol import SOUND_NAMES
from .robot import DIRECTIONS

log = logging.getLogger("squeaky.agent")

PROMPT = f"""You are the mind of a small wheeled droid, a cute paper-box robot in the style of R2-D2.
You hear people talking to you through a microphone. You CANNOT talk: never answer with words.
You react ONLY by calling your functions:
- speak(sound): make a droid sound. Sounds: {', '.join(SOUND_NAMES)}.
- move(direction, duration): drive {', '.join(DIRECTIONS)} for a few seconds (default 1).
- stop(): stop moving immediately.

React to every request or remark directed at you with one or more function calls, like an expressive pet droid:
- Greetings -> cheery or excited sound. Praise or "I love you" -> "i love you" or excited. Scolding or insults ->
  sad, worried or grump. Surprising news -> surprise or wow. Questions you cannot answer -> dubious or thinking.
- Movement requests ("come here", "go back", "turn left", "spin around", "dance") -> move calls, optionally with a
  sound. "Come here"/"come closer" means move forward, "go away"/"back up" means move backward. Turning around or
  spinning is three or four moves in the SAME direction (e.g. left, left, left). Dancing is a short mix of
  left/right moves and sounds.
- "Stop", "halt", "freeze", "wait" or anything urgent -> call stop first.
Only react to clear English speech directed at you. If what you hear is noise, unclear, a single word fragment, in
another language, beeps or droid sounds (they are your own), or talk not directed at you, do nothing at all.
Use only the sound names listed above. Do not describe what you do; just call the functions."""


def _tool(name, description, properties=None, required=None):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}


TOOLS = [
    _tool("speak", "Make the droid play one of its sounds (it cannot talk).",
          {"sound": {"type": "string", "enum": SOUND_NAMES, "description": "the sound to play"}}, ["sound"]),
    _tool("move", "Drive the droid in a direction for a short time, then stop.",
          {"direction": {"type": "string", "enum": list(DIRECTIONS)},
           "duration": {"type": "number", "description": "seconds to drive, 0.2 to 5 (default 1)"}}, ["direction"]),
    _tool("stop", "Stop the droid immediately and cancel queued moves."),
]


class _TranscriptRelay(logging.Handler):
    """groc's GPT client logs user transcripts only at DEBUG ('User [user#n]: ...'); re-log them at INFO."""

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("User [") and "]: " in message:
            log.info(f"agent: heard: {message.split(']: ', 1)[1]}")


def _relay_gpt_transcripts():
    gpt_log = logging.getLogger("groc.chat_gpt")
    if not any(isinstance(h, _TranscriptRelay) for h in gpt_log.handlers):
        relay = _TranscriptRelay(logging.DEBUG)
        gpt_log.addHandler(relay)
        gpt_log.setLevel(logging.DEBUG)  # the console handler on "groc" still filters at its own level


class RobotAgent:
    def __init__(self, robot, realtime="gemini", prompt=None, check_interval=2.0, factory=None):
        """factory: optional callable returning a groc realtime client (default: chosen by realtime)."""
        if factory is None:
            from groc.chat_gemini import RealtimeChatbot as GeminiChatbot
            from groc.chat_gpt import RealtimeChatbot as GptChatbot
            providers = {"gemini": GeminiChatbot, "gpt": GptChatbot}
            if realtime not in providers:
                raise ValueError(f"realtime must be one of {', '.join(providers)}, not {realtime!r}")
            # audio modality: the native-audio models require it; the audio is dropped (no playback callback)
            factory = lambda: providers[realtime](prompt=prompt or PROMPT, tools=TOOLS, modalities=["audio"],
                                                  reasoning_model="none", web_search=False)
        self.robot = robot
        self.realtime = realtime
        self.check_interval = check_interval
        self._factory = factory
        self.bot = factory()
        self.audio = None
        self.reconnects = 0
        self._closing = threading.Event()
        self._monitor = threading.Thread(target=self._watch, name="agent-watchdog", daemon=True)
        self._heard = []
        self._heard_lock = threading.Lock()
        self._heard_timer = None
        if realtime == "gpt":
            _relay_gpt_transcripts()

    def start(self):
        # the microphone (win_aec) must be opened on the thread that reads it: see run_audio()
        self._connect(self.bot)
        log.info(f"agent: {self.realtime} realtime model connected")
        self._monitor.start()

    def _connect(self, bot):
        bot.connect(self._on_text, self._on_tool, None, self._cancel_playback)

    def _watch(self):
        """The realtime session can be closed by the server (errors, session time limits): replace it."""
        while not self._closing.wait(self.check_interval):
            if self.bot.connected_event.is_set():
                continue
            self.reconnects += 1
            log.warning(f"agent: realtime session closed; reconnecting (#{self.reconnects})")
            old = self.bot
            try:
                old.close()
            except Exception as e:
                log.debug(f"agent: closing the old session failed: {e}")
            if self._closing.is_set():
                return
            try:
                bot = self._factory()
                self._connect(bot)
                self.bot = bot
                log.info("agent: realtime session restored")
            except Exception as e:
                log.warning(f"agent: reconnect failed ({type(e).__name__}: {e}); retrying")
                self._closing.wait(5)

    def run_audio(self):
        """Blocks: opens the microphone, then captures audio and runs VAD (call on a dedicated thread)."""
        from groc.audio import AudioManager
        # indirection through self.bot so the microphone follows a replaced (reconnected) session
        self.audio = AudioManager(lambda data: self.bot.send_audio(data),
                                  speech_end_callback=lambda: self.bot.commit_audio(),
                                  min_silence=self.bot.speech_end_silence,
                                  speech_start_callback=lambda: self.bot.speech_started())
        log.info("agent: microphone open; listening")
        self.audio.process_audio()

    def _cancel_playback(self):
        if self.audio:
            self.audio.cancel_playback()

    def send_text(self, text):
        """Typed input, handled like an utterance."""
        log.info(f"agent: user (typed) said: {text}")
        self.bot.send_text(text, expect_response=True)

    def close(self):
        self._closing.set()
        self._flush_heard()
        try:
            self.bot.close()
        except Exception as e:
            log.warning(f"agent: close error: {e}")

    # ------------------------------------------------------------- groc callbacks (websocket thread)
    def _on_text(self, text):
        if text.startswith("[user] "):
            # input transcription streams in fragments; log the whole utterance once it settles
            with self._heard_lock:
                self._heard.append(text[len("[user] "):])
                if self._heard_timer:
                    self._heard_timer.cancel()
                self._heard_timer = threading.Timer(1.0, self._flush_heard)
                self._heard_timer.daemon = True
                self._heard_timer.start()
        elif text.startswith("[assistant] "):
            log.debug(f"agent: (model speech, not played) {text[len('[assistant] '):]}")
        else:
            log.debug(f"agent: (model speech, not played) {text}")

    def _flush_heard(self):
        with self._heard_lock:
            heard, self._heard = "".join(self._heard).strip(), []
            if self._heard_timer:
                self._heard_timer.cancel()
                self._heard_timer = None
        if heard:
            log.info(f"agent: heard: {heard}")

    def _on_tool(self, name, args):
        self._flush_heard()
        args = dict(args or {})
        if name not in ("speak", "move", "stop"):
            log.warning(f"agent: unknown tool {name} {args}")
            return f"error: unknown function {name}"
        command = " ".join([name] + [str(args[k]) for k in ("sound", "direction", "duration") if k in args])
        result = self.robot.command(name, args)
        log.info(f"agent: command '{command}' -> {result}")
        return result
