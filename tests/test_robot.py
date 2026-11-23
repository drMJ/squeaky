import time
import unittest

from squeaky import r2d2_protocol as proto
from squeaky.robot import CommandError, R2D2Robot, parse_command


class FakeClient:
    instances = []

    def __init__(self, target, disconnected_callback=None):
        self.target = target
        self.disconnected_callback = disconnected_callback
        self.is_connected = False
        self.writes = []
        FakeClient.instances.append(self)

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, uuid, data, response=True):
        assert uuid == proto.CHAR_UUID
        self.writes.append(data.hex().upper())

    def drop(self, robot):
        """Simulate the droid dropping the link (bleak calls the callback on its loop)."""
        self.is_connected = False
        robot._loop.call_soon_threadsafe(self.disconnected_callback, self)


class FakeScanner:
    @staticmethod
    async def find_device_by_name(name, timeout=10.0):
        return "AA:BB:CC:DD:EE:FF"


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class RobotTest(unittest.TestCase):
    def setUp(self):
        FakeClient.instances = []
        self.robot = R2D2Robot(client_factory=FakeClient, scanner=FakeScanner, check_interval=0.05, keepalive=0,
                               write_gap=0.0, move_duration=0.2)
        self.robot.start()
        self.assertTrue(self.robot.wait_connected(3))

    def tearDown(self):
        self.robot.close()

    @property
    def client(self):
        return FakeClient.instances[-1]

    def test_connect_sends_safety_stop(self):
        self.assertEqual(self.client.writes[:2], [proto.STOP_FRAME, proto.STRAIGHT_FRAME])

    def test_move_drives_then_stops(self):
        self.client.writes.clear()
        self.assertIn("moving left", self.robot.execute("move left 0.2"))
        self.assertTrue(wait_for(lambda: len(self.client.writes) >= 4))
        self.assertEqual(self.client.writes, [proto.turn_frame(-1), proto.drive_frame(0.5), proto.STOP_FRAME,
                                              proto.STRAIGHT_FRAME])
        self.assertEqual(proto.turn_frame(-1), proto.TURN_FRAMES[0])

    def test_stop_interrupts_move_and_clears_queue(self):
        self.client.writes.clear()
        self.robot.move("forward", 5)
        self.robot.move("backward", 5)
        self.assertTrue(wait_for(lambda: proto.drive_frame(0.6) in self.client.writes))
        self.robot.stop()
        self.assertTrue(wait_for(lambda: self.client.writes[-2:] == [proto.STOP_FRAME, proto.STRAIGHT_FRAME]))
        time.sleep(0.2)
        self.assertNotIn(proto.drive_frame(-0.6), self.client.writes)

    def test_watchdog_reconnects_after_drop(self):
        first = self.client
        first.drop(self.robot)
        self.assertTrue(wait_for(lambda: len(FakeClient.instances) == 2 and self.robot.connected))
        self.assertEqual(self.client.writes[:2], [proto.STOP_FRAME, proto.STRAIGHT_FRAME])
        self.assertIn("playing sound", self.robot.speak("Wow!"))
        self.assertTrue(wait_for(lambda: proto.SOUNDS["wow"] in self.client.writes))
        self.assertEqual(self.robot.speak("chatter"), "playing sound chattering")

    def test_commands_while_disconnected_are_rejected(self):
        robot = R2D2Robot(client_factory=FakeClient, scanner=FakeScanner)  # not started: never connects
        self.assertIn("not connected", robot.execute("speak beep"))
        self.assertIn("not connected", robot.execute("move forward"))


class ParseTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_command("speak i love you"), ("speak", {"sound": "i love you"}))
        self.assertEqual(parse_command("Move Left 2"), ("move", {"direction": "left", "duration": 2.0}))
        self.assertEqual(parse_command("stop"), ("stop", {}))
        for bad in ("", "jump", "move", "move left fast", "stop now"):
            with self.assertRaises(CommandError):
                parse_command(bad)

    def test_unknown_sound_and_direction(self):
        robot = R2D2Robot(client_factory=FakeClient, scanner=FakeScanner)
        self.assertTrue(robot.command("speak", {"sound": "laugh"}).startswith("error: unknown sound"))
        self.assertTrue(robot.command("move", {"direction": "up"}).startswith("error: unknown direction"))


if __name__ == "__main__":
    unittest.main()
