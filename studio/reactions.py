"""Hand-keyed reactions with real animation timing: a fast move into the pose that overshoots, a hold with a little life in it, and a soft settle back.

These replace the motion-capture crouches (they squat and bend through the other character). Each reaction is a target pose reached through an
underdamped spring (so it snaps in and overshoots like a cartoon take), held, then relaxed with a gentler spring.
"""
import math

from . import stick

# name: (target pose spec, spring omega, damping, hold seconds or None = stay, relax omega)
SPEC = {
    "flinch":  (dict(torso=-16, head=-10, a1=118, a2=-62, b1=-96, b2=-70, l1=10, l2=-16, m1=-12, m2=-14), 26, 0.34, 0.9, 9),
    "cower":   (dict(torso=14, head=18, a1=128, a2=-92, b1=-110, b2=-100, l1=24, l2=-34, m1=-4, m2=-30), 15, 0.45, 2.2, 7),
    "duck":    (dict(torso=26, head=14, a1=60, a2=70, b1=-40, b2=70, l1=22, l2=-40, m1=6, m2=-38), 20, 0.4, 0.9, 8),
    "slump":   (dict(torso=24, head=26, a1=-5, a2=6, b1=-20, b2=6, l1=18, l2=-28, m1=-8, m2=-18), 8, 0.7, None, 6),
    "stumble": (dict(torso=24, head=-6, a1=70, a2=30, b1=-60, b2=30, l1=-26, l2=-10, m1=24, m2=-30), 22, 0.38, 0.35, 10),
    "cheer_up": (dict(torso=-6, head=-10, a1=156, a2=-8, b1=-156, b2=-8), 18, 0.4, 1.4, 8),
    "recoil":  (dict(torso=-10, head=-6, a1=84, a2=-30, b1=-70, b2=-30, l1=6, l2=-8, m1=-8, m2=-8), 24, 0.36, 0.6, 9),
}


def _spring(t, wn, z):
    """Step response of an underdamped spring: 0 at t=0, overshoots 1, settles at 1."""
    if t <= 0:
        return 0.0
    wd = wn * math.sqrt(1 - z * z)
    return 1.0 - math.exp(-z * wn * t) * (math.cos(wd * t) + z * wn / wd * math.sin(wd * t))


def sample(name, tt, seed=0.0):
    tgt, wn, z, hold, wn_out = SPEC[name]
    base = stick.pose_at("stand", tt, seed)
    if hold is None or tt <= hold:
        w = _spring(tt, wn, z)
    else:
        w = _spring(hold, wn, z) * (1.0 - _spring(tt - hold, wn_out, 0.8))
    p = dict(base)
    for k, v in tgt.items():
        p[k] = base[k] + (v - base[k]) * w
    if name in ("cower", "flinch") and w > 0.6:                      # shaking: the pose is never perfectly still
        p["head"] += 2.2 * math.sin(tt * 55 + seed)
        p["a1"] += 2.0 * math.sin(tt * 48 + seed * 2)
    return p


def look(tt, seed=0.0):
    """Glancing around: the head and eyes scan, the body hardly moves."""
    p = stick.pose_at("stand", tt, seed)
    p["head"] += 24 * math.sin(tt * 1.5 + seed) * (0.6 + 0.4 * math.sin(tt * 0.5))
    p["torso"] += 3 * math.sin(tt * 0.9 + seed)
    return p
