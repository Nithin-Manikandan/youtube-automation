"""Procedural sound effects timed to what happens on screen. No sound files, no licences."""
import numpy as np

from pipeline.audio import _bp, _lp, whoosh as _whoosh
from pipeline.tts import SR

rng0 = np.random.default_rng(42)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _norm(x, peak=1.0):
    m = np.abs(x).max()
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def footstep(seed=0):
    r = np.random.default_rng(seed)
    t = _t(0.14)
    f = 70 + r.random() * 30
    thud = np.sin(2 * np.pi * f * t) * np.exp(-t * 45)
    scuff = _lp(r.standard_normal(len(t)), 1800) * np.exp(-t * 60) * 0.5
    return _norm(thud + scuff, 0.8)


def clash(seed=0):
    r = np.random.default_rng(seed)
    t = _t(0.9)
    y = np.zeros_like(t)
    for f, a, d in ((1850, 1, 7), (2790, .8, 9), (3920, .6, 12), (5180, .4, 15), (6700, .25, 20)):
        f *= 1 + r.uniform(-0.03, 0.03)
        y += a * np.sin(2 * np.pi * f * t + r.random() * 6) * np.exp(-t * d)
    y += _bp(r.standard_normal(len(t)), 2500, 9000) * np.exp(-t * 80) * 1.4
    y += np.sin(2 * np.pi * 120 * t) * np.exp(-t * 40) * 0.6
    return _norm(y, 0.9)


def sword_whoosh(seed=0):
    w = _whoosh(0.32, seed)
    return _norm(w, 0.6)


def thunder():
    r = np.random.default_rng(9)
    t = _t(2.6)
    env = np.minimum(t / 0.25, 1) * np.exp(-t * 1.3)
    y = _lp(r.standard_normal(len(t)), 220) * env + np.sin(2 * np.pi * 42 * t) * env * 0.6
    crack = _bp(r.standard_normal(len(t)), 1200, 6000) * np.exp(-t * 18) * 0.5
    return _norm(y + crack, 0.9)


def rain_bed(seconds):
    r = np.random.default_rng(5)
    n = int(seconds * SR)
    x = _bp(r.standard_normal(n), 1800, 9000)
    x *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.4 * np.arange(n) / SR)
    return _norm(x, 0.5)


def cheer(seconds=1.8, seed=0):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for lo, hi in ((300, 900), (900, 2200), (2200, 3800)):
        x += _bp(r.standard_normal(n), lo, hi) * (0.6 + 0.4 * np.sin(2 * np.pi * r.uniform(4, 8) * t + r.random() * 6))
    env = np.minimum(t / 0.2, 1) * np.minimum((seconds - t) / 0.5, 1)
    return _norm(x * env, 0.55)


def clink():
    t = _t(0.35)
    y = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((3100, 1, 16), (4700, .6, 22), (6200, .3, 30)))
    return _norm(y, 0.6)


def thump():
    t = _t(0.9)
    y = np.sin(2 * np.pi * (70 * np.exp(-t * 5) + 30) * t) * np.exp(-t * 5)
    return _norm(y, 1.0)


def pen():
    r = np.random.default_rng(2)
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    x = _bp(r.standard_normal(n), 900, 4500) * np.sin(np.pi * t / 0.6) ** 1.5
    return _norm(x, 0.35)


def pop():
    t = _t(0.12)
    y = np.sin(2 * np.pi * (500 + 900 * t / 0.12) * t) * np.exp(-t * 28)
    return _norm(y, 0.35)


SOUNDS = {"clash": clash, "whoosh": sword_whoosh, "thunder": lambda seed=0: thunder(), "clink": lambda seed=0: clink(),
          "thump": lambda seed=0: thump(), "pen": lambda seed=0: pen(), "pop": lambda seed=0: pop(), "foot": footstep}
GAIN = {"clash": .8, "whoosh": .55, "thunder": .7, "clink": .5, "thump": .8, "pen": .5, "pop": .5, "foot": .42}


def scene_events(sc, t0):
    """Sound events for one scene (scene dict from recipes.build_stage / map / card), times are absolute."""
    ev, dur = [], sc["duration"]
    ev.append((t0 - 0.05, "whoosh", 0.4))  # soft transition into the scene
    for tx in sc.get("text", []):
        ev.append((t0 + tx["t"], "pop", 1.0))
    for ar in sc.get("arrows", []):
        ev.append((t0 + ar["t0"], "pen", 1.0))
    for a in sc.get("actors", []):
        ks = a["keys"]
        for k0, k1 in zip(ks, ks[1:]):
            moving = abs(k1["x"] - k0["x"]) > 0.02 and (k0["pose"] in ("walk", "run") or k1["pose"] in ("walk", "run"))
            if moving:
                run = "run" in (k0["pose"], k1["pose"])
                rate = 5.2 if run else 3.2
                n = int((k1["t"] - k0["t"]) * rate)
                for i in range(n):
                    ev.append((t0 + k0["t"] + i / rate, "foot", .7 if run else .45))
        for k in ks:
            if k["pose"] == "swing":
                n = int((dur - k["t"]) * 1.4)
                for i in range(n):
                    ev.append((t0 + k["t"] + i / 1.4, "whoosh", .7))
        if any(k["pose"] == "cheer" for k in ks):
            ev.append((t0 + 0.3, "cheer", 1.0))
        if a.get("crown_fall") is not None:
            tf = a["crown_fall"]
            ev.append((t0 + tf + .55, "clink", 1.0))
            ev.append((t0 + tf + .85, "clink", .5))
    for fx in sc.get("fx", []):
        if fx["type"] == "sparks":
            ev.append((t0 + fx["t"], "clash", 1.0))
            ev.append((t0 + fx["t"] + .32, "clash", .7))
        elif fx["type"] == "flash":
            ev.append((t0 + fx["t"] + .25, "thunder", 1.0))
        elif fx["type"] == "rain":
            ev.append((t0 + fx["t0"], "RAIN", (fx["t1"] - fx["t0"])))
    for sh in sc.get("shake", []):
        ev.append((t0 + sh["t"], "thump", 1.0))
    return ev


def render_sfx(events, total):
    n = int(total * SR)
    out = np.zeros(n, dtype=np.float32)
    cache = {}
    for i, (t, kind, arg) in enumerate(events):
        if kind == "RAIN":
            seg = rain_bed(arg) * 0.22
        elif kind == "cheer":
            seg = cheer(seed=i) * 0.5
        else:
            seg = SOUNDS[kind](i % 5) if kind in ("foot", "clash", "whoosh") else cache.setdefault(kind, SOUNDS[kind]())
            seg = seg * GAIN[kind] * arg
        i0 = int(max(t, 0) * SR)
        seg = seg[:max(0, n - i0)]
        out[i0:i0 + len(seg)] += seg
    return out
