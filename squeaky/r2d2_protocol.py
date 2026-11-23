"""BLE protocol of the littleBits R2-D2 (w32 ControlHub).

UUIDs and command frames are taken from C:\\code\\R2D2\\R2D2-mcp.py (reverse engineered by meetar/littlebits-r2d2-controls).
"""
import re

DEVICE_NAME = "w32 ControlHub"
SERVICE_UUID = "d9d9e9e0-aa4e-4797-8151-cb41cedaf2ad"
CHAR_UUID = "d9d9e9e1-aa4e-4797-8151-cb41cedaf2ad"

# full reverse (0) -> stop (31) -> full forward (62)
DRIVE_FRAMES = [
    "140202FF659D", "140202FA3538", "140202F6F4B4", "140202F2B430", "140202ED57EE",
    "140202EA2709", "140202E6E685", "140202E3B620", "140202DF41FF", "140202DA115A",
    "140202D6D0D6", "140202D29052", "140202CE43EF", "140202CA036B", "140202C5F284",
    "140202C28263", "140202BE3D78", "140202BA7DFC", "140202B6BC70", "140202B2FCF4",
    "140202AB7FEC", "140202AA6FCD", "140202A6AE41", "140202A2EEC5", "140202A1DEA6",
    "1402029E191A", "1402029969FD", "140202969812", "14020292D896", "1402028D3B48",
    "1402028A4BAF", "140202897BCC",
    "1402027DD457", "1402027994D3", "14020276653C", "1402027225B8", "1402026EF605",
    "1402026DC666", "1402026896C3", "1402026327A8", "1402026107EA", "1402025CE014",
    "14020259B0B1", "14020255713D", "140202502198", "1402024B82C2", "14020249A280",
    "14020245630C", "140202412388", "1402023BFC55", "14020239DC17", "140202337D5D",
    "140202304D3E", "1402022BEE64", "14020229CE26", "140202214F2E", "140202205F0F",
    "1402021BD837", "14020218E854", "1402021429D8", "1402020F8A82", "1402020CBAE1",
    "140202070B8A",
]
DRIVE_STOP_INDEX = 31

# hard left (0) -> straight (17) -> hard right (32)
TURN_FRAMES = [
    "140201FC00AD", "140201EA725A", "140201E49394", "140201DF14AC", "140201D9746A",
    "140201D1F562", "140201CC36FE", "140201C697B4", "140201C0F772", "140201B918CC",
    "140201B4C961", "140201AE7A1A", "140201A7EB33", "140201A2BB96", "1402019C6C0B",
    "14020196CD41", "14020190AD87",
    "1402018A1EFC",
    "14020184FF32", "1402017EB167", "14020178D1A1", "1402017270EB", "1402016C8314",
    "14020166225E", "140201604298", "1402015AD581", "14020154344F", "1402014F9715",
    "14020149F7D3", "1402014246B8", "1402013CD9E1", "1402013548C8", "140201002E3E",
]
TURN_STRAIGHT_INDEX = 17

STOP_FRAME = DRIVE_FRAMES[DRIVE_STOP_INDEX]
STRAIGHT_FRAME = TURN_FRAMES[TURN_STRAIGHT_INDEX]

_RAW_SOUNDS = {
    "grump": "1E011B42AA", "scold": "1E011A528B", "chitter": "1E011962E8", "chattering": "1E011872C9",
    "i love you": "1E01178326", "bleep": "1E01169307", "beep": "1E0115A364", "whistle": "1E0114B345",
    "descending": "1E0113C3A2", "excited": "1E0112D383", "cheery": "1E0111E3E0", "sad": "1E0110F3C1",
    "scream!!": "1E010F101F", "startup": "1E010E003E", "surprise!!": "1E010C207C", "story": "1E010A40BA",
    "wow!": "1E010860F8", "thbt": "1E01068136", "worried": "1E0104A174", "dubious": "1E0102C1B2",
    "board startup!!": "1E0101F1D1", "thinking": "1E0100E1F0",
}


def normalize_sound(name):
    """'Scream!!' -> 'scream', 'I-love-you' -> 'i love you'."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split())


SOUNDS = {normalize_sound(name): frame for name, frame in _RAW_SOUNDS.items()}
SOUND_NAMES = sorted(SOUNDS)


def find_sound(name):
    """Sound frame for a name, tolerating case, punctuation and unique prefixes ('Chatter' -> 'chattering')."""
    key = normalize_sound(name)
    if key in SOUNDS:
        return key, SOUNDS[key]
    matches = [n for n in SOUND_NAMES if n.startswith(key)] if key else []
    if len(matches) == 1:
        return matches[0], SOUNDS[matches[0]]
    return key, None


def drive_frame(speed, invert=False):
    """speed -1 (full reverse) .. 0 (stop) .. +1 (full forward)."""
    speed = max(-1.0, min(1.0, float(speed)))
    if invert:
        speed = -speed
    index = round(DRIVE_STOP_INDEX + speed * DRIVE_STOP_INDEX)
    return DRIVE_FRAMES[max(0, min(len(DRIVE_FRAMES) - 1, index))]


def turn_frame(steer):
    """steer -1 (hard left) .. 0 (straight) .. +1 (hard right)."""
    steer = max(-1.0, min(1.0, float(steer)))
    span = TURN_STRAIGHT_INDEX if steer < 0 else len(TURN_FRAMES) - 1 - TURN_STRAIGHT_INDEX
    return TURN_FRAMES[round(TURN_STRAIGHT_INDEX + steer * span)]
