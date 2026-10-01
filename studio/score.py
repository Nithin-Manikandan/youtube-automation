"""Adaptive documentary score: six moods that share one key (A minor / C major) and one tempo grid, so changing
mood is a smooth crossfade of harmonically compatible layers, never a jarring cut.

calm · tense · epic · sad · triumph · mystery.   Everything is synthesised; there is no music licence.
"""
import numpy as np

from pipeline.tts import SR

BPM = 96
BEAT = 60.0 / BPM
BAR = BEAT * 4
CH_LEN = BAR * 4                      # one chord per 4 bars (10 s); the progression is Am - F - C - G
CHORDS = [(110.0, 130.81, 164.81), (87.31, 110.0, 130.81), (130.81, 164.81, 196.0), (98.0, 123.47, 146.83)]
FADE = 3.6                            # half-width of a mood crossfade, seconds (7.2 s total)
MOODS = ("calm", "tense", "epic", "sad", "triumph", "mystery")
LEVEL = {"calm": 0.75, "tense": 0.8, "epic": 1.0, "sad": 0.7, "triumph": 0.9, "mystery": 0.65}


def _lp(x, cutoff):
    sp = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(sp / (1 + (f / cutoff) ** 4), len(x))


def _hp(x, cutoff):
    sp = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(sp * (1 - 1 / (1 + (f / cutoff) ** 4)), len(x))


def _chord_windows(a, b):
    k0 = int(a // CH_LEN) - 1
    k1 = int(b // CH_LEN) + 1
    for k in range(k0, k1 + 1):
        yield k, CHORDS[k % 4]


def _chord_env(t, k, ramp=2.5):
    s, e = k * CH_LEN, (k + 1) * CH_LEN
    return np.clip((t - (s - ramp)) / (2 * ramp), 0, 1) * np.clip(((e + ramp) - t) / (2 * ramp), 0, 1)


def _win(a, n, k, ramp=2.5):
    """Sample range of this segment that chord window k can touch (its envelope is zero outside)."""
    s, e = k * CH_LEN - ramp, (k + 1) * CH_LEN + ramp
    i0 = max(0, int(np.floor((s - a) * SR)))
    i1 = min(n, int(np.ceil((e - a) * SR)) + 1)
    return i0, i1


def _pad(t, a, b, amps, det=0.004, rng=None):
    """Sustained chord layers. amps = {octave_multiplier: amplitude}."""
    out = np.zeros(len(t))
    for k, chord in _chord_windows(a, b):
        i0, i1 = _win(a, len(t), k)
        if i1 <= i0:
            continue
        tt = t[i0:i1]
        env = _chord_env(tt, k)
        for f in chord:
            for mult, amp in amps.items():
                for d in (-det, 0.0, det):
                    out[i0:i1] += env * amp * np.sin(2 * np.pi * f * mult * (1 + d) * tt + (k * 7 + f) % 6.28)
    return out


def _saw(t, f, harmonics=9):
    y = np.zeros(len(t))
    for h in range(1, harmonics + 1):
        if f * h < 5000:
            y += np.sin(2 * np.pi * f * h * t) / h
    return y


def _note(out, a, te, f, dur, amp, kind="pluck"):
    i0 = int((te - a) * SR)
    m = int(dur * SR)
    if i0 >= len(out) or i0 + m <= 0:
        return
    j0, j1 = max(i0, 0), min(i0 + m, len(out))
    tt = (np.arange(j0, j1) - i0) / SR
    if kind == "pluck":
        y = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 6) + 0.12 * np.sin(2 * np.pi * 3 * f * tt) * np.exp(-tt * 10))
        y *= np.minimum(tt / 0.006, 1) * np.exp(-tt * (3.2 / max(dur, 0.2) * 1.4))
    elif kind == "soft":
        y = np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(2 * np.pi * 2 * f * tt)
        y *= np.minimum(tt / 0.04, 1) * np.exp(-tt * (2.4 / max(dur, 0.3)))
    elif kind == "kick":
        y = np.sin(2 * np.pi * (48 + 90 * np.exp(-tt * 28)) * tt) * np.exp(-tt * 9)
    elif kind == "tom":
        y = np.sin(2 * np.pi * f * (1 + 0.25 * np.exp(-tt * 18)) * tt) * np.exp(-tt * 7)
    elif kind == "hat":
        y = np.random.default_rng(int(te * 1000) % 99991).standard_normal(len(tt)) * np.exp(-tt * 60)
    else:
        y = np.sin(2 * np.pi * f * tt)
    out[j0:j1] += y * amp


def _beats(a, b, step):
    n0 = int(np.ceil(a / step))
    n1 = int(b / step)
    return range(n0, n1 + 1)


def _chord_at(te):
    return CHORDS[int(te // CH_LEN) % 4]


# ------------------------------------------------------------------ the six moods
def calm(t, a, b):
    y = _pad(t, a, b, {1: 0.07, 2: 0.17, 4: 0.09})
    for n in _beats(a, b, BEAT * 2):
        te = n * BEAT * 2
        ch = _chord_at(te)
        _note(y, a, te, ch[(n // 2) % 3] * 4, 2.2, 0.07, "soft")
    return y


def tense(t, a, b):
    y = _pad(t, a, b, {0.5: 0.06, 1: 0.05, 2: 0.07}, det=0.01)
    y += 0.10 * np.sin(2 * np.pi * 55.0 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t))
    y += 0.05 * np.sin(2 * np.pi * 58.27 * t)                                       # a slow, uneasy beat against the drone
    trem = 0.6 + 0.4 * np.sin(2 * np.pi * 5.5 * t)
    for k, ch in _chord_windows(a, b):
        i0, i1 = _win(a, len(t), k)
        if i1 > i0:
            y[i0:i1] += _chord_env(t[i0:i1], k) * trem[i0:i1] * 0.07 * _lp(_saw(t[i0:i1], ch[0] * 2, 7), 1400)
    for n in _beats(a, b, BEAT * 2):                                                  # heartbeat
        te = n * BEAT * 2
        _note(y, a, te, 50, 0.3, 0.30, "kick")
        _note(y, a, te + 0.24, 50, 0.3, 0.18, "kick")
    return y


def epic(t, a, b):
    y = _pad(t, a, b, {0.5: 0.08, 1: 0.13, 2: 0.10}, det=0.006)
    for k, ch in _chord_windows(a, b):
        i0, i1 = _win(a, len(t), k)
        if i1 <= i0:
            continue
        tt = t[i0:i1]
        env = _chord_env(tt, k)
        swell = 0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * tt)
        for f in (ch[0], ch[0] * 1.5, ch[0] * 2):                                     # power chord "brass"
            y[i0:i1] += env * 0.05 * _lp(_saw(tt, f, 10), 2200) * swell
    for n in _beats(a, b, BEAT):
        te = n * BEAT
        beat_in_bar = n % 4
        bar = n // 4
        if beat_in_bar in (0, 2):
            _note(y, a, te, 50, 0.4, 0.55, "kick")
        if beat_in_bar in (1, 3) and bar % 4 != 3:
            _note(y, a, te, 92, 0.45, 0.28, "tom")
        if bar % 4 == 3 and beat_in_bar == 3:                                           # tom fill into the next phrase
            for q in range(6):
                _note(y, a, te + q * BEAT / 6, 120 - q * 9, 0.3, 0.25 + 0.03 * q, "tom")
    for n in _beats(a, b, BEAT / 2):
        _note(y, a, n * BEAT / 2, 0, 0.1, 0.05, "hat")
    return y


def sad(t, a, b):
    y = _pad(t, a, b, {1: 0.05, 2: 0.10, 4: 0.04})
    for n in _beats(a, b, BEAT * 1.0):
        te = n * BEAT
        ch = _chord_at(te)
        seq = (0, 1, 2, 1)
        _note(y, a, te, ch[seq[n % 4]] * 4, 1.6, 0.11, "soft")
    return y


def triumph(t, a, b):
    y = _pad(t, a, b, {1: 0.05, 2: 0.13, 4: 0.12, 8: 0.03})
    for k, ch in _chord_windows(a, b):
        i0, i1 = _win(a, len(t), k)
        if i1 > i0:
            tt = t[i0:i1]
            y[i0:i1] += _chord_env(tt, k) * 0.05 * _lp(_saw(tt, ch[0] * 4, 8), 3800) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * tt))
    for n in _beats(a, b, BEAT / 2):
        te = n * BEAT / 2
        ch = _chord_at(te)
        seq = (0, 1, 2, 1, 0, 2, 1, 2)
        _note(y, a, te, ch[seq[n % 8]] * 4, 0.6, 0.085, "pluck")
    for n in _beats(a, b, BEAT * 2):
        _note(y, a, n * BEAT * 2, 50, 0.4, 0.22, "kick")
    return y


def mystery(t, a, b):
    y = _pad(t, a, b, {1: 0.05, 2: 0.10, 3: 0.05, 4: 0.06}, det=0.008)
    y += 0.03 * np.sin(2 * np.pi * 123.47 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.06 * t))   # a suspended 2nd (B) floating over the chord
    penta = (440.0, 523.25, 587.33, 659.25, 783.99)
    rng = np.random.default_rng(int(a) % 977)
    n0 = int(a // 3.1) - 1
    for n in range(n0, int(b // 3.1) + 2):
        te = n * 3.1 + (n * 7919 % 100) / 100.0
        _note(y, a, te, penta[(n * 3 + n // 2) % 5], 2.4, 0.05, "soft")
    return y


GEN = {"calm": calm, "tense": tense, "epic": epic, "sad": sad, "triumph": triumph, "mystery": mystery}


def render_mood(mood, a, b):
    n = int((b - a) * SR)
    t = a + np.arange(n) / SR
    y = GEN[mood](t, a, b)
    y = _hp(_lp(y, 4200), 65)
    rms = np.sqrt((y ** 2).mean()) + 1e-9
    return (y / rms * 0.12 * LEVEL[mood]).astype(np.float32)


def build(runs, total):
    """runs: [(start_s, end_s, mood)], contiguous, covering the video. Returns a mono float32 score.
    Moods crossfade over 2*FADE seconds centred on each boundary."""
    n = int(total * SR)
    out = np.zeros(n, dtype=np.float32)
    for i, (s, e, mood) in enumerate(runs):
        a = 0.0 if i == 0 else max(0.0, s - FADE)
        b = total if i == len(runs) - 1 else min(total, e + FADE)
        seg = render_mood(mood, a, b)
        t = a + np.arange(len(seg)) / SR
        w = np.ones(len(seg), dtype=np.float32)
        if i > 0:
            w *= np.sin(np.clip((t - (s - FADE)) / (2 * FADE), 0, 1) * np.pi / 2) ** 2
        else:
            w *= np.clip(t / 2.0, 0, 1)
        if i < len(runs) - 1:
            w *= np.cos(np.clip((t - (e - FADE)) / (2 * FADE), 0, 1) * np.pi / 2) ** 2
        else:
            w *= np.clip((total - t) / 3.0, 0, 1)
        i0 = int(a * SR)
        out[i0:i0 + len(seg)] += (seg * w)[:max(0, n - i0)]
    return out
