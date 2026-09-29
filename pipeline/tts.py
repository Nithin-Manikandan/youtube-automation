"""Narration with word-level timestamps (drives the animated captions)."""
import asyncio
import subprocess
import time

import numpy as np

from . import ffmpeg

SR = 44100


def _decode(mp3_bytes):
    p = subprocess.run([ffmpeg.exe(), "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "pipe:1"], input=mp3_bytes, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


async def _edge(text, voice, rate, pitch):
    import edge_tts
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    audio, words = bytearray(), []
    async for ch in comm.stream():
        if ch["type"] == "audio":
            audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            start = ch["offset"] / 1e7
            words.append([ch["text"], start, start + ch["duration"] / 1e7])
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(audio), words


def narrate(text, cfg, retries=4):
    """Returns (float32 mono samples @44.1k, [[word, start_s, end_s], ...])."""
    v = cfg["voice"]
    last = None
    for i in range(retries):
        try:
            mp3, words = asyncio.run(_edge(text, v["name"], v.get("rate", "+0%"), v.get("pitch", "+0Hz")))
            samples = _decode(mp3)
            if not words:  # some voices return no boundaries: spread words evenly
                toks = text.split()
                d = len(samples) / SR
                words = [[t, d * k / len(toks), d * (k + 1) / len(toks)] for k, t in enumerate(toks)]
            return samples, words
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Narration failed: {last}")


def silent(text, wps=2.7):
    """Offline engine-test stand-in: silence with evenly estimated word timings."""
    toks = text.split()
    d = max(1.0, len(toks) / wps)
    words = [[t, d * k / len(toks), d * (k + 1) / len(toks)] for k, t in enumerate(toks)]
    return np.zeros(int(d * SR), dtype=np.float32), words
