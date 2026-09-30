"""Calm documentary music bed, generated: evolving pads on a slow minor progression, soft plucks. No licences."""
import numpy as np

from pipeline.tts import SR

CHORDS = [(110.0, 130.81, 164.81), (87.31, 110.0, 130.81), (130.81, 164.81, 196.0), (98.0, 123.47, 146.83)]  # Am F C G
CH_LEN = 12.0


def _lp(x, cutoff):
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(spec / (1 + (f / cutoff) ** 4), len(x))


def loop(seed=3):
    rng = np.random.default_rng(seed)
    total = CH_LEN * len(CHORDS)
    n = int(total * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for ci, chord in enumerate(CHORDS):
        c0 = ci * CH_LEN
        env = np.clip((t - c0) / 4.0, 0, 1) * np.clip((c0 + CH_LEN + 4.0 - t) / 4.0, 0, 1)
        env = np.where((t >= c0 - 4.0) & (t <= c0 + CH_LEN + 4.0), np.clip((t - (c0 - 4.0)) / 4.0, 0, 1) * np.clip((c0 + CH_LEN + 4.0 - t) / 4.0, 0, 1), 0)
        for f in chord:
            for det in (-0.004, 0.0, 0.004):
                ff = f * (1 + det)
                out += env * np.sin(2 * np.pi * ff * t + rng.random() * 6) * 0.22
            out += env * np.sin(2 * np.pi * f * 2 * t) * 0.05
    # wrap-around: blend the tail into the head so the loop is seamless
    for k, note in enumerate(np.concatenate([CHORDS[i] for i in (0, 3, 2, 1)])):
        t0 = 1.0 + k * 3.7
        m = min(int(2.4 * SR), n - int(t0 * SR))
        if m > 0 and t0 < total - 3:
            tt = np.arange(m) / SR
            out[int(t0 * SR):int(t0 * SR) + m] += 0.10 * np.sin(2 * np.pi * note * 4 * tt) * np.exp(-tt * 2.6)
    out = _lp(out, 1800)
    return (out / np.abs(out).max()).astype(np.float32)


def bed(seconds):
    lp = loop()
    n = int(seconds * SR)
    reps = -(-n // len(lp))
    x = np.tile(lp, reps)[:n].copy()
    f = int(2.0 * SR)
    x[:f] *= np.linspace(0, 1, f)
    x[-f:] *= np.linspace(1, 0, f)
    return x
