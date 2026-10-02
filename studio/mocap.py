"""Motion-capture clips (CMU Graphics Lab database, free for all uses) retargeted onto the stickman skeleton by tools/build_motion_lib.py.

A clip is an (n, 11) array of angles at 30 fps in the engine's convention: torso, head, a1, a2, b1, b2, l1, l2, m1, m2, lift.
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KEYS = ["torso", "head", "a1", "a2", "b1", "b2", "l1", "l2", "m1", "m2", "lift"]
_LIB = None
BENDERS = {"duck", "dodge", "fall", "get_up", "pick_up", "sit_down", "scared", "stumble", "cry", "push", "pull", "kick", "punch", "boxing", "sword_1", "sword_2", "run", "surprised"}
LEG = 0.44                                      # leg length in actor units (thigh + shin)


def _load():
    global _LIB
    if _LIB is None:
        _LIB = {}
        f = HERE / "motion_data.npz"
        if f.exists():
            z = np.load(f)
            meta = json.loads((HERE / "motion_index.json").read_text())
            for k in z.files:
                a = z[k].astype(np.float32)
                for c in (0, 2, 4, 6, 8):                                   # absolute angles: unwrap so interpolation never spins the long way round
                    a[:, c] = np.degrees(np.unwrap(np.radians(a[:, c])))
                for c in (1, 3, 5, 7, 9):                                   # relative joint angles: wrap to +-180 so interpolation never spins through zero
                    a[:, c] = (a[:, c] + 180.0) % 360.0 - 180.0
                a[:, 3] = np.clip(np.abs(a[:, 3]), 0.0, 150.0)              # elbows only fold one way: a projected capture bending into the screen reads as a backwards, broken arm
                a[:, 5] = np.clip(np.abs(a[:, 5]), 0.0, 150.0)
                a[:, 7] = np.clip(a[:, 7], -150.0, 3.0)                     # knees only fold backwards
                a[:, 9] = np.clip(a[:, 9], -150.0, 3.0)
                _LIB[k] = (a, meta[k])
    return _LIB


def has(name):
    return name in _load()


def info(name):
    return _load()[name][1]


def duration(name):
    a, m = _load()[name]
    return len(a) / m["fps"]


GESTURE = {"talk_1", "talk_2", "talk_3", "talk_4", "talk_5", "talk_6", "explain_a", "explain_b", "quarrel_a", "quarrel_b", "threat_a", "threat_b", "sad", "cry", "happy", "point", "think", "look_around", "shrug", "wave"}


def sample(name, tt, loop=None, boost=None):
    """Pose dict for clip `name` at clip-time tt seconds. Loops when the clip is a loop/cycle, otherwise holds the last frame."""
    a, m = _load()[name]
    n = len(a)
    f = max(0.0, tt) * m["fps"]
    if loop is None:
        loop = m["loops"] or m["kind"] == "cycle"
    if loop:
        f = f % n
        i0 = int(f); i1 = (i0 + 1) % n
    else:
        f = min(f, n - 1.001)
        i0 = int(f); i1 = min(i0 + 1, n - 1)
    u = f - int(f)
    row = a[i0] * (1 - u) + a[i1] * u
    if name not in BENDERS:                                                  # nobody leans through the person they are talking to
        row = row.copy()
        row[0] = min(row[0], 24.0)
        row[1] = max(-30.0, min(row[1], 30.0))
    if boost is None:
        boost = 1.55 if name in GESTURE else 1.15
    if boost != 1.0 and m["kind"] != "cycle":                                # cartoon exaggeration of the upper body around the clip's mean pose; legs stay planted
        mean = a.mean(axis=0)
        row = row.copy()
        row[:6] = mean[:6] + (row[:6] - mean[:6]) * boost
    if name in GESTURE:
        row = row.copy()                                                     # stand tall: capture actors relax into bent knees, a sitting hip or a forward lean that reads as hunched on a cartoon
        mean = a.mean(axis=0)
        row[6] = 4 + (row[6] - 4) * 0.35; row[8] = -4 + (row[8] + 4) * 0.35
        row[7] *= 0.3; row[9] *= 0.3
        row[0] -= 0.75 * max(mean[0] - 2.0, 0.0)
        row[1] -= 0.5 * mean[1]
    if name in ("sword_1", "sword_2", "punch", "boxing", "kick", "dodge", "stumble", "surprised"):
        row = row.copy()                                                     # fighters keep a ready stance, not a deep squat
        row[7] *= 0.55; row[9] *= 0.55
        row[6] = 4 + (row[6] - 4) * 0.7; row[8] = -4 + (row[8] + 4) * 0.7
    return dict(zip(KEYS, [float(v) for v in row]))
