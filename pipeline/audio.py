"""Procedural dark-ambient bed, cut whooshes and hit sounds, then the final mix.

Everything is generated from maths, so there is no music licence to worry about.
"""
import numpy as np

from .tts import SR


def _lp(x, cutoff):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1 / (1 + (freqs / cutoff) ** 4)
    return np.fft.irfft(spec, len(x))


def _bp(x, lo, hi):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= ((freqs > lo) & (freqs < hi)).astype(np.float32)
    return np.fft.irfft(spec, len(x))


def drone(seconds, seed=7):
    rnd = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f, a in ((55.0, .5), (82.41, .3), (110.9, .22), (164.8, .10), (73.4, .18)):
        det = f * (1 + rnd.uniform(-0.004, 0.004))
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * rnd.uniform(0.03, 0.11) * t + rnd.uniform(0, 6))
        out += a * lfo * np.sin(2 * np.pi * det * t + 0.4 * np.sin(2 * np.pi * 0.07 * t))
    wind = _bp(rnd.standard_normal(n), 120, 900)
    wind *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.09 * t)
    out += 0.25 * wind / (np.abs(wind).max() + 1e-9)
    # cold high shimmer that tremolos in and out
    out += 0.03 * np.sin(2 * np.pi * 1760 * t) * np.clip(np.sin(2 * np.pi * 0.05 * t + 1.3), 0, 1) ** 2
    # slow heartbeat
    beat = np.zeros(n)
    for s in np.arange(0.5, seconds, 1.15):
        i = int(s * SR)
        for off, g in ((0, 1.0), (int(0.22 * SR), 0.6)):
            j = i + off
            m = min(int(0.25 * SR), n - j)
            if m > 0:
                tt = np.arange(m) / SR
                beat[j:j + m] += g * np.sin(2 * np.pi * 48 * tt) * np.exp(-tt * 14)
    out += 0.5 * beat
    fade = np.clip(t / 1.5, 0, 1) * np.clip((seconds - t) / 1.5, 0, 1)
    out *= fade
    return (out / np.abs(out).max()).astype(np.float32)


def whoosh(dur=0.45, seed=0):
    rnd = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = _bp(rnd.standard_normal(n), 250, 5000)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2
    return (noise * env / (np.abs(noise).max() + 1e-9) * 0.5).astype(np.float32)


def boom(dur=1.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = np.sin(2 * np.pi * (60 * np.exp(-t * 1.5) + 28) * t) * np.exp(-t * 3.2)
    hiss = _lp(np.random.default_rng(3).standard_normal(n), 1500) * np.exp(-t * 6) * 0.4
    y = body + hiss
    return (y / np.abs(y).max()).astype(np.float32)


def reverb(x, wet=0.12, tail=0.5):
    n = int(tail * SR)
    ir = np.random.default_rng(1).standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir[:200] = 0
    size = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    y *= np.abs(x).max() / (np.abs(y).max() + 1e-9)
    return (x + wet * y).astype(np.float32)


def _envelope(x, win=0.12):
    k = int(win * SR)
    e = np.convolve(np.abs(x), np.ones(k) / k, mode="same")
    return np.clip(e / (np.percentile(e, 95) + 1e-9), 0, 1)


def mix(voices, starts, total, cfg):
    """voices: list of float32 arrays, starts: seconds each begins, total: video seconds."""
    n = int(total * SR)
    voice = np.zeros(n, dtype=np.float32)
    for v, s in zip(voices, starts):
        i = int(s * SR)
        seg = v[:max(0, n - i)]
        voice[i:i + len(seg)] += seg
    if np.abs(voice).max() > 0:
        voice = reverb(voice)
        voice = np.tanh(1.6 * voice / np.abs(voice).max()) * 0.9   # gentle compression / level

    music = drone(total) * cfg["music"].get("volume", 0.22)
    duck = 1 - 0.65 * _envelope(voice)
    music = music * duck

    sfx = np.zeros(n, dtype=np.float32)
    hit = boom() * 0.55
    sfx[:min(len(hit), n)] += hit[:n]
    for k, s in enumerate(starts[1:]):
        w = whoosh(seed=k)
        i = max(0, int((s - 0.18) * SR))
        seg = w[:max(0, n - i)]
        sfx[i:i + len(seg)] += seg * 0.35

    y = voice + music + sfx
    peak = np.abs(y).max()
    return (y / peak * 0.89).astype(np.float32) if peak > 0 else y
