import threading
import time
import unittest

from squeaky.agent import TOOLS, RobotAgent


class FakeBot:
    instances = []
    speech_end_silence = 0.5

    def __init__(self):
        self.connected_event = threading.Event()
        self.closed = False
        FakeBot.instances.append(self)

    def connect(self, text_cb, tool_cb, audio_play=None, audio_cancel=None):
        self.tool_cb = tool_cb
        self.connected_event.set()

    def close(self):
        self.closed = True


class FakeRobot:
    def __init__(self):
        self.calls = []

    def command(self, verb, args=None):
        self.calls.append((verb, args))
        return "ok"


class AgentTest(unittest.TestCase):
    def setUp(self):
        FakeBot.instances = []
        self.robot = FakeRobot()
        self.agent = RobotAgent(self.robot, factory=FakeBot, check_interval=0.05)
        self.agent.start()

    def tearDown(self):
        self.agent.close()

    def test_tools_are_routed_to_the_robot(self):
        self.assertEqual({t["name"] for t in TOOLS}, {"speak", "move", "stop"})
        self.assertEqual(self.agent._on_tool("move", {"direction": "left", "duration": 2}), "ok")
        self.assertTrue(self.agent._on_tool("dance", {}).startswith("error"))
        self.assertEqual(self.robot.calls, [("move", {"direction": "left", "duration": 2})])

    def test_reconnects_when_the_session_closes(self):
        first = self.agent.bot
        first.connected_event.clear()  # server closed the websocket
        deadline = time.monotonic() + 2
        while self.agent.bot is first and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNot(self.agent.bot, first)
        self.assertTrue(first.closed)
        self.assertTrue(self.agent.bot.connected_event.is_set())


if __name__ == "__main__":
    unittest.main()
