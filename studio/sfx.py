"""Procedural sound design, in stereo, timed to what happens on screen. No sound files, no licences.

Each effect is layered from several parts (impact transient + body + tail) like real foley, varied every time it plays,
panned to where it happens on screen, and lightly reverberated so it sits in a room instead of sounding pasted on.
"""
import numpy as np

from pipeline.tts import SR


# ------------------------------------------------------------------ building blocks
def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _white(n, rng):
    return rng.standard_normal(n)


def filt(x, lo=None, hi=None, order=2):
    """Smooth-slope band filter in the frequency domain (no brick-wall ringing)."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 - 1 / (1 + (f / lo) ** (2 * order))
    if hi:
        g *= 1 / (1 + (f / hi) ** (2 * order))
    return np.fft.irfft(spec * g, len(x))


def pink(n, rng):
    spec = np.fft.rfft(_white(n, rng))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1
    return np.fft.irfft(spec / np.sqrt(f), n)


def brown(n, rng):
    x = np.cumsum(_white(n, rng))
    return x - np.linspace(x[0], x[-1], n)


def norm(x, peak=1.0):
    m = np.abs(x).max()
    return (x / m * peak).astype(np.float32) if m > 0 else x.astype(np.float32)


def sweep_noise(n, rng, f0, f1, f2, q=0.6):
    """Noise whose pass-band glides f0 -> f1 -> f2: the classic whoosh. Overlap-add STFT."""
    x = pink(n, rng)
    w = 1024
    hop = w // 2
    win = np.hanning(w)
    out = np.zeros(n + w)
    fr = np.fft.rfftfreq(w, 1 / SR)
    for i in range(0, n - w, hop):
        u = i / n
        c = (f0 + (f1 - f0) * u / 0.5) if u < 0.5 else (f1 + (f2 - f1) * (u - 0.5) / 0.5)
        g = np.exp(-((np.log2(fr + 1) - np.log2(c)) ** 2) / (2 * q ** 2))
        seg = np.fft.irfft(np.fft.rfft(x[i:i + w] * win) * g, w)
        out[i:i + w] += seg * win
    return out[:n]


def room_ir(seconds=0.45, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    ir = np.zeros((2, n))
    for ch in range(2):
        d = rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
        ir[ch] = filt(d, 200, 7000)
        for k in range(6):  # early reflections
            ir[ch][int((0.008 + 0.011 * k + 0.003 * ch) * SR)] += 0.6 / (k + 1)
    return ir / np.abs(ir).sum(axis=1, keepdims=True).max() * 6


_IR = None


def reverb(seg, wet=0.14):
    """seg: (2, n). Adds a short room tail."""
    global _IR
    if _IR is None:
        _IR = room_ir()
    n = seg.shape[1] + _IR.shape[1]
    out = np.zeros((2, n))
    out[:, :seg.shape[1]] = seg
    for ch in range(2):
        out[ch] += wet * np.fft.irfft(np.fft.rfft(seg[ch], n) * np.fft.rfft(_IR[ch], n), n)
    return out


def pan(mono, p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)])


def st(x):
    return x if x.ndim == 2 else np.stack([x, x])


# ------------------------------------------------------------------ one-shot sounds
def footstep(seed=0, run=False):
    rng = np.random.default_rng(seed)
    t = _t(0.24)
    n = len(t)
    f = 70 + rng.random() * 40
    thump = np.sin(2 * np.pi * f * (1 + 0.5 * np.exp(-t * 60)) * t) * np.exp(-t * 45) * 0.2
    body = filt(_white(n, rng), 110, 950) * np.exp(-t * 42) * 1.0
    mid = filt(_white(n, rng), 600, 2600) * np.exp(-t * 85) * 0.75
    click = filt(_white(n, rng), 2200, 7000) * np.exp(-t * 260) * (0.7 if run else 0.5)
    grit = np.zeros(n)
    idx = (rng.random(int(16 + rng.random() * 16)) * 0.08 * SR).astype(int)
    grit[idx] = rng.standard_normal(len(idx))
    grit = filt(grit, 1800, 8000) * 0.7
    y = thump + body + mid + click + grit
    return norm(y, 0.8)


def clash(seed=0):
    """Steel on steel: a hard noisy strike, short heavily damped metallic resonances and a blade scrape.
    Deliberately NOT long-ringing pure tones (those sound like bells)."""
    rng = np.random.default_rng(seed)
    t = _t(0.55)
    n = len(t)
    f0 = 700 + rng.random() * 1500
    y = np.zeros(n)
    # strike: broadband burst, the main "clang"
    y += filt(_white(n, rng), 250, 8000) * np.exp(-t * 600) * 2.6
    y += filt(_white(n, rng), 500, 6500) * np.exp(-t * 90) * 1.3
    # damped inharmonic resonances, each one rough (noise-modulated) so it reads as struck metal, not a tuned tone
    for r, a_, d in ((1.0, 1.0, 38.0), (2.41, 0.8, 46.0), (3.97, 0.65, 58.0), (5.83, 0.5, 70.0), (8.11, 0.35, 90.0)):
        f = f0 * r * (1 + rng.uniform(-0.05, 0.05))
        rough = 0.55 + 0.45 * filt(_white(n, rng), None, 260) / (np.abs(filt(_white(n, rng), None, 260)).max() + 1e-9)
        vib = 1 + 0.006 * np.sin(2 * np.pi * rng.uniform(20, 40) * t)
        y += a_ * np.sin(2 * np.pi * f * vib * t + rng.random() * 6.28) * np.exp(-t * d) * rough
    # scrape / slide of the blades
    y += filt(_white(n, rng), 3000, 9000) * np.exp(-((t - 0.035) ** 2) / (2 * 0.028 ** 2)) * 0.6
    # weight of the impact
    y += filt(_white(n, rng), 180, 900) * np.exp(-t * 48) * 2.3
    y += np.sin(2 * np.pi * (170 + 50 * np.exp(-t * 40)) * t) * np.exp(-t * 42) * 0.8
    return norm(y, 0.9)


def whoosh(seed=0, dur=0.42):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = sweep_noise(n, rng, 500 + rng.random() * 250, 2400 + rng.random() * 700, 800)
    env = np.sin(np.pi * np.linspace(0, 1, n) ** 0.85) ** 2
    y = y * env + filt(pink(n, rng), None, 300) * env * 0.35
    return norm(y, 0.7)


def thunder(seed=1):
    rng = np.random.default_rng(seed)
    n = int(4.5 * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        rb = brown(n, np.random.default_rng(seed + ch))
        rumble = norm(filt(rb, None, 130), 1.0) + 0.75 * norm(filt(rb, 120, 520), 1.0)
        env = np.minimum(t / 0.18, 1) * np.exp(-t * 0.85) * (1 + 0.45 * np.sin(2 * np.pi * 0.9 * t + ch * 1.7))
        crack = filt(_white(n, rng), 700, 9000) * np.exp(-t * 13) * 0.6 * (1 if ch == 0 else 0.85)
        out[ch] = norm(rumble, 1.0) * env + crack + np.sin(2 * np.pi * 36 * t) * env * 0.35
    return norm(out, 0.9)


def thump(seed=0):
    rng = np.random.default_rng(seed)
    t = _t(1.1)
    n = len(t)
    sub = np.sin(2 * np.pi * (42 + 55 * np.exp(-t * 14)) * t) * np.exp(-t * 5.5) * 0.6
    body = filt(_white(n, rng), 90, 900) * np.exp(-t * 14) * 1.0
    crack = filt(_white(n, rng), 800, 5000) * np.exp(-t * 60) * 0.7
    tail = filt(brown(n, rng), 60, 400) * np.exp(-t * 4) * 0.5
    return norm(sub + body + crack + tail, 0.95)


def applause(seed=0, dur=2.3):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    dens = 260 * np.minimum(t / 0.25, 1) * np.minimum((dur - t) / 0.9, 1) * (0.8 + 0.2 * np.sin(2 * np.pi * 3.1 * t))
    for ch in range(2):
        imp = (rng.random(n) < dens / SR) * rng.standard_normal(n) * rng.random(n)
        clap = filt(imp, 900, 6500) + filt(imp, 2000, 4000) * 0.6
        kern = np.exp(-np.arange(int(0.004 * SR)) / (0.0012 * SR))
        out[ch] = np.convolve(clap, kern, mode="same") + filt(_white(n, rng), 400, 1800) * 0.03 * np.minimum(t / 0.5, 1)
    return norm(out, 0.7)


def clink(seed=0):
    """Gold crown hitting stone: a few quick, damped metal taps (no ringing bell tone)."""
    rng = np.random.default_rng(seed)
    t = _t(0.7)
    n = len(t)
    y = np.zeros(n)
    f0 = 1500 + rng.random() * 900
    for dt, a_, pitch in ((0, 1.0, 1.0), (0.12, 0.55, 0.93), (0.2, 0.3, 0.88), (0.26, 0.15, 0.85)):
        i0 = int(dt * SR)
        tt = t[:n - i0]
        seg = _white(len(tt), rng) * np.exp(-tt * 500) * 1.6
        for r, am, d in ((1.0, 1.0, 55.0), (2.3, 0.7, 70.0), (4.1, 0.5, 90.0)):
            seg += am * np.sin(2 * np.pi * f0 * pitch * r * tt) * np.exp(-tt * d)
        seg += filt(_white(len(tt), rng), 400, 3500) * np.exp(-tt * 90) * 0.6
        y[i0:] += seg * a_
    return norm(y, 0.6)


def ping(seed=0):
    """Sonar ping: soft sine blip with a long watery decay."""
    t = _t(1.6)
    y = np.sin(2 * np.pi * 1180 * t) * np.exp(-t * 3.2) * np.minimum(1, t * 400)
    y += 0.35 * np.sin(2 * np.pi * 1770 * t) * np.exp(-t * 4.5)
    return norm(reverb(y, 0.3), 0.45)


def roar(seed=0, dur=3.2):
    """Rocket launch: low rumble swelling with filtered hiss."""
    rng = np.random.default_rng(seed)
    t = _t(dur)
    env = np.minimum(1, t / 0.8) * np.exp(-np.maximum(0, t - dur * .6) * 1.4)
    y = filt(brown(len(t), rng), None, 220) * 3.0 + filt(_white(len(t), rng), 300, 2500) * 0.5
    return norm(y * env, 0.8)


def draw_scratch(seed=0, dur=0.6):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = filt(_white(n, rng), 1400, 6500)
    y *= np.abs(filt(_white(n, rng), None, 40)) * 6 + 0.4
    y *= np.sin(np.pi * np.linspace(0, 1, n)) ** 1.4
    return norm(y, 0.4)


# ------------------------------------------------------------------ ambience beds
def _fade(x, sec=0.5):
    f = int(sec * SR)
    f = min(f, x.shape[-1] // 2)
    w = np.ones(x.shape[-1])
    w[:f] = np.linspace(0, 1, f)
    w[-f:] = np.linspace(1, 0, f)
    return x * w


def rain_bed(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        rng = np.random.default_rng(10 + ch)
        hiss = filt(_white(n, rng), 2500, 9500) * 0.5 + filt(_white(n, rng), 450, 2200) * 0.35
        drops = np.zeros(n)
        idx = (rng.random(int(dur * 90)) * n).astype(int)
        drops[idx] = rng.random(len(idx)) + 0.3
        drops = filt(drops, 2200, 7000) * 6
        out[ch] = (hiss + drops) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.3 * t + ch))
    return _fade(norm(out, 0.5), 0.8)


def wind_bed(dur, seed=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        rng = np.random.default_rng(20 + ch + seed)
        low = filt(pink(n, rng), 60, 420) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * t + ch * 2.0 + seed))
        mid = filt(pink(n, rng), 380, 1500) * (0.35 + 0.35 * np.sin(2 * np.pi * 0.21 * t + ch + 1.3)) * 0.5
        out[ch] = low + mid
    return _fade(norm(out, 0.5), 0.8)


def crickets(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    rng = np.random.default_rng(31)
    for k in range(5):
        f = 4300 + rng.random() * 600
        rate = 0.9 + rng.random() * 0.4
        ph = rng.random()
        burst = ((t * rate + ph) % 1.0) < 0.36
        pulses = (np.sin(2 * np.pi * 26 * t * 1.0 + k) > 0.2)
        y = np.sin(2 * np.pi * f * t) * burst * pulses * 0.35
        p = rng.uniform(-0.9, 0.9)
        out += pan(y, p)
    return _fade(norm(out, 0.4), 0.8)


def birds(dur, seed=0):
    n = int(dur * SR)
    out = np.zeros((2, n))
    rng = np.random.default_rng(40 + seed)
    tt = 0.6 + rng.random() * 1.5
    while tt < dur - 0.6:
        f0, df = 2200 + rng.random() * 1800, (rng.random() - 0.5) * 2200
        for k in range(int(2 + rng.random() * 4)):
            m = int(0.09 * SR)
            u = np.linspace(0, 1, m)
            ph = 2 * np.pi * np.cumsum(f0 + df * np.sin(np.pi * u) + 300 * np.sin(2 * np.pi * 14 * u)) / SR
            c = np.sin(ph) * np.sin(np.pi * u) ** 1.5
            i0 = int((tt + k * 0.13) * SR)
            if i0 + m < n:
                out[:, i0:i0 + m] += pan(c, rng.uniform(-0.8, 0.8))[:, :m] * 0.5
        tt += 1.5 + rng.random() * 3.5
    return _fade(norm(out, 0.35), 0.5) if np.abs(out).max() > 0 else out


def rumble_bed(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        rng = np.random.default_rng(60 + ch)
        r = filt(brown(n, rng), 25, 160) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.35 * t + ch))
        out[ch] = norm(r, 1.0) * 0.8 + filt(pink(n, rng), 120, 600) * 0.12
    return _fade(norm(out, 0.6), 0.8)


def crackle_bed(dur):
    n = int(dur * SR)
    out = np.zeros((2, n))
    for ch in range(2):
        rng = np.random.default_rng(70 + ch)
        imp = np.zeros(n)
        idx = (rng.random(int(dur * 55)) * n).astype(int)
        imp[idx] = rng.standard_normal(len(idx)) * rng.random(len(idx))
        out[ch] = filt(imp, 1500, 7000) * 8 + filt(pink(n, rng), 200, 900) * 0.25
    return _fade(norm(out, 0.5), 0.6)


def waves(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    for ch in range(2):
        rng = np.random.default_rng(50 + ch)
        swell = (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t + ch * 1.4)) ** 1.6
        out[ch] = (filt(pink(n, rng), 120, 2600) * swell) * 0.9
    return _fade(norm(out, 0.5), 0.9)


# ------------------------------------------------------------------ scene -> events
def _x_at(keys, t):
    for a, b in zip(keys, keys[1:]):
        if a["t"] <= t <= b["t"]:
            u = (t - a["t"]) / max(b["t"] - a["t"], 1e-6)
            return a["x"] + (b["x"] - a["x"]) * u
    return keys[-1]["x"] if t > keys[-1]["t"] else keys[0]["x"]


def _pan(x):
    return float(np.clip((x - 0.5) * 1.7, -0.9, 0.9))


AMBIENT = {"storm": "wind", "desert": "wind", "battlefield": "wind", "night": "crickets", "city_day": "birds", "countryside": "birds",
           "forest": "birds", "palace": None, "sea": "waves", "snow": "wind"}


def scene_events(sc, t0, prev=None):
    """Events (time, kind, gain, pan, extra) for one scene. Times are absolute seconds."""
    ev, dur = [], sc["duration"]
    rng = np.random.default_rng(int(t0 * 10) % 9999)
    kind_of = lambda s: s.get("kind", "stage")
    if prev is None or kind_of(prev) != kind_of(sc) or prev.get("chapter") != sc.get("chapter"):
        ev.append((max(0.0, t0 - 0.12), "whoosh", 0.45, 0.0, 0))
    for ar in sc.get("arrows", []):
        ev.append((t0 + ar["t0"], "scratch", 0.6, 0.0, 0))
    amb = AMBIENT.get(sc.get("bg_name"))
    if amb and kind_of(sc) == "stage":
        ev.append((t0, "AMB_" + amb, 1.0, 0.0, dur + 0.4))
    objs = {o_["type"]: o_ for o_ in sc.get("objects", [])}
    if "volcano" in objs:
        ev.append((t0, "AMB_rumble", 1.0, 0.0, dur + 0.4))
        ev.append((t0 + 0.6, "thump", 0.8, 0.0, 0))
    if "fire" in objs:
        ev.append((t0, "AMB_crackle", 1.0, 0.0, dur + 0.4))
    if "explosion" in objs:
        ev.append((t0 + objs["explosion"].get("t0", 0.4), "thump", 1.0, 0.0, 0))
        ev.append((t0 + objs["explosion"].get("t0", 0.4) + 0.05, "thunder", 0.8, 0.0, 0))
    if "depth_charge" in objs:
        tb = objs["depth_charge"].get("t0", 1.0) + 1.3
        ev.append((t0 + tb, "thump", 1.0, 0.0, 0))
        ev.append((t0 + tb + 0.05, "thunder", 0.7, 0.0, 0))
    if "torpedo" in objs:
        ev.append((t0 + objs["torpedo"].get("t0", 1.0), "whoosh", 0.7, 0.0, 0))
    if "missile" in objs and objs["missile"].get("launch"):
        ev.append((t0 + objs["missile"].get("t0", 1.0), "roar", 0.9, 0.0, 0))
    if "wave" in objs:
        ev.append((t0, "AMB_waves", 1.0, 0.0, dur + 0.4))
        ev.append((t0 + 1.2, "thunder", 0.6, 0.0, 0))
    fighters = [a for a in sc.get("actors", []) if any(k["pose"] == "swing" for k in a["keys"])]
    for a in sc.get("actors", []):
        ks = a["keys"]
        sc_gain = 0.55 * float(a.get("scale", 1))
        for k0, k1 in zip(ks, ks[1:]):
            if abs(k1["x"] - k0["x"]) > 0.02 and "run" in (k0["pose"], k1["pose"]) or (abs(k1["x"] - k0["x"]) > 0.02 and "walk" in (k0["pose"], k1["pose"])):
                run = "run" in (k0["pose"], k1["pose"])
                rate = 5.2 if run else 3.2
                n = int((k1["t"] - k0["t"]) * rate)
                for i in range(n):
                    tt = k0["t"] + i / rate + rng.uniform(-0.03, 0.03)
                    ev.append((t0 + tt, "foot_run" if run else "foot", sc_gain * rng.uniform(.75, 1.0) * (1.0 if i % 2 else .86), _pan(_x_at(ks, tt)), int(rng.integers(0, 6))))
        for k in ks:
            if k["pose"] == "swing":
                n = int((dur - k["t"]) * 1.4)
                for i in range(n):
                    tt = k["t"] + i / 1.4
                    x = _x_at(ks, tt)
                    ev.append((t0 + tt, "whoosh_s", 0.55, _pan(x), int(rng.integers(0, 4))))
                    if len(fighters) >= 2 and i % 1 == 0 and tt + 0.3 < dur:
                        ev.append((t0 + tt + 0.3, "clash", 0.55 + 0.25 * (i % 2), 0.0, int(rng.integers(0, 4))))
        if any(k["pose"] == "cheer" for k in ks):
            ev.append((t0 + 0.3, "applause", 0.6, 0.0, 0))
        if a.get("crown_fall") is not None:
            ev.append((t0 + a["crown_fall"] + 0.5, "clink", 0.7, _pan(_x_at(ks, a["crown_fall"])), int(rng.integers(0, 4))))
    for fx in sc.get("fx", []):
        if fx["type"] == "sparks":
            ev.append((t0 + fx["t"], "clash", 1.0, _pan(fx.get("x", .5)), 1))
            ev.append((t0 + fx["t"] + .28, "clash", .7, _pan(fx.get("x", .5)) * .8, 2))
        elif fx["type"] == "sonar":
            for j in range(int((fx["t1"] - fx["t0"]) / 2.0) + 1):
                ev.append((t0 + fx["t0"] + j * 2.0, "ping", 0.5, 0.0, 0))
        elif fx["type"] == "flash":
            ev.append((t0 + fx["t"] + .2, "thunder", 1.0, 0.0, 0))
        elif fx["type"] == "rain":
            ev.append((t0 + fx["t0"], "AMB_rain", 1.0, 0.0, fx["t1"] - fx["t0"] + 0.4))
    for sh in sc.get("shake", []):
        ev.append((t0 + sh["t"], "thump", 1.0, 0.0, 0))
    return ev


_CACHE = {}


def _sound(kind, extra):
    key = (kind, extra if kind not in ("AMB_rain", "AMB_wind", "AMB_crickets", "AMB_birds", "AMB_waves", "AMB_rumble", "AMB_crackle") else round(extra, 1))
    if key in _CACHE and not kind.startswith("AMB"):
        return _CACHE[key]
    if kind == "foot":
        s = footstep(extra)
    elif kind == "foot_run":
        s = footstep(extra + 50, run=True)
    elif kind == "clash":
        s = clash(extra)
    elif kind in ("whoosh", "whoosh_s"):
        s = whoosh(extra)
    elif kind == "thunder":
        s = thunder()
    elif kind == "thump":
        s = thump()
    elif kind == "applause":
        s = applause()
    elif kind == "clink":
        s = clink(extra)
    elif kind == "ping":
        s = ping()
    elif kind == "roar":
        s = roar()
    elif kind == "scratch":
        s = draw_scratch()
    elif kind == "AMB_rain":
        s = rain_bed(extra)
    elif kind == "AMB_wind":
        s = wind_bed(extra)
    elif kind == "AMB_crickets":
        s = crickets(extra)
    elif kind == "AMB_birds":
        s = birds(extra)
    elif kind == "AMB_waves":
        s = waves(extra)
    elif kind == "AMB_rumble":
        s = rumble_bed(extra)
    elif kind == "AMB_crackle":
        s = crackle_bed(extra)
    else:
        raise KeyError(kind)
    if not kind.startswith("AMB"):
        _CACHE[key] = s
    return s


LEVEL = {"foot": 0.5, "foot_run": 0.55, "clash": 0.8, "whoosh": 0.5, "whoosh_s": 0.5, "thunder": 0.7, "thump": 0.75, "applause": 0.45,
         "clink": 0.55, "scratch": 0.4, "AMB_rain": 0.16, "AMB_wind": 0.15, "AMB_crickets": 0.08, "AMB_birds": 0.07, "AMB_waves": 0.22, "AMB_rumble": 0.3, "AMB_crackle": 0.16}
REVERB = {"clash": 0.2, "clink": 0.25, "thump": 0.1, "thunder": 0.0, "foot": 0.1, "foot_run": 0.1, "whoosh": 0.12, "whoosh_s": 0.12, "applause": 0.15}


def render_sfx(events, total):
    """Returns a stereo float32 array (2, n) for the whole video."""
    n = int(total * SR)
    out = np.zeros((2, n), dtype=np.float32)
    for (t, kind, gain, p, extra) in events:
        seg = st(_sound(kind, extra)) if kind != "whoosh_s" else st(_sound(kind, extra))
        if seg.ndim == 2 and kind in ("foot", "foot_run", "clash", "whoosh", "whoosh_s", "thump", "clink", "scratch"):
            mono = seg[0]
            seg = pan(mono, p)
        if REVERB.get(kind, 0) > 0:
            seg = reverb(seg, REVERB[kind])
        i0 = int(max(t, 0) * SR)
        seg = seg[:, :max(0, n - i0)] * LEVEL[kind] * gain
        out[:, i0:i0 + seg.shape[1]] += seg
    # remove sub-audible DC / rumble below 25 Hz and gently glue peaks
    for ch in range(2):
        out[ch] = out[ch] - np.convolve(out[ch], np.ones(1764) / 1764, mode="same") * 0.0
    return np.tanh(out * 1.3) / 1.3
