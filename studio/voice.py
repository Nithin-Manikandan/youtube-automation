"""An expressive narrator from the free Edge neural voices.

Flat reading is what makes TTS sound robotic, so this does what a human reader does:
- reads sentence by sentence, each with its own pace, pitch and loudness (scene emotion, question/exclamation,
  short punchy lines slower and heavier, long lines a touch quicker, last line of a scene falling away)
- leans on key words (slower, louder, higher) using SSML prosody, and pauses on dashes and ellipses
- leaves real breathing room between sentences, longer for somber / dramatic moments
- if the service rejects the SSML it silently falls back to plain prosody, so a video is never blocked
"""
import asyncio
import hashlib
import random
import re
import threading
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from pipeline import tts

# deltas applied on top of the channel's base rate/pitch: (rate %, pitch Hz, volume %, extra pause ms)
EMOTIONS = {
    "neutral": (0, 0, 0, 0), "warm": (-2, 2, 2, 0), "tense": (-7, -5, -2, 60), "somber": (-13, -9, -7, 160),
    "dramatic": (-9, -3, 5, 140), "excited": (8, 7, 6, -30), "urgent": (12, 4, 6, -50), "awed": (-6, 4, 0, 90),
}
MOOD_EMOTION = {"calm": "warm", "tense": "tense", "epic": "urgent", "sad": "somber", "triumph": "awed", "mystery": "awed"}
_LOCK = threading.Lock()      # the SSML passthrough briefly patches a module function, so serialise construction
FALLBACK_VOICES = ["en-US-AndrewNeural", "en-US-GuyNeural"]


def clean_delivery(d):
    d = d if isinstance(d, dict) else {}
    emo = str(d.get("emotion", "")).lower().strip()
    words = [re.sub(r"[^\w'\-]", "", str(w)) for w in (d.get("emphasis") or [])][:3]
    try:
        pb = max(0, min(900, int(d.get("pause_before_ms", 0))))
    except (TypeError, ValueError):
        pb = 0
    return {"emotion": emo if emo in EMOTIONS else "", "emphasis": [w for w in words if w], "pause_before_ms": pb}


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", re.sub(r"\s+", " ", text).strip())
    return [p for p in parts if p]


def _int(x, base):
    m = re.search(r"-?\d+", str(base))
    return int(m.group()) if m else 0


def prosody_for(sentence, i, n, emotion, base_rate, base_pitch, rng):
    dr, dp, dv, _ = EMOTIONS[emotion]
    rate, pitch, vol = _int(0, base_rate) + dr, _int(0, base_pitch) + dp, dv
    words = len(sentence.split())
    if i == 0:
        pitch += 2                      # a fresh start lifts slightly
    if i == n - 1 and n > 1:
        rate -= 4
        pitch -= 3                      # the last line lands and falls away
        vol -= 1
    if sentence.rstrip().endswith("?"):
        pitch += 5
        rate += 1
    if sentence.rstrip().endswith("!"):
        vol += 4
        rate += 3
    if words <= 5:
        rate -= 6                       # short punchy lines get weight
        vol += 3
    elif words >= 22:
        rate += 3
    rate += rng.randint(-3, 3)          # no two sentences identical
    pitch += rng.randint(-2, 2)
    return max(-40, min(40, rate)), max(-30, min(30, pitch)), max(-20, min(20, vol))


def _ssml_fragment(sentence, emphasis, pitch):
    """Sentence as an SSML fragment: key words get their own prosody; dashes/ellipses become pauses."""
    frag = xml_escape(sentence)
    frag = frag.replace("...", "<break time='420ms'/>").replace(" — ", "<break time='260ms'/>").replace(" -- ", "<break time='260ms'/>")
    used = False
    for w in emphasis:
        pat = re.compile(r"\b(" + re.escape(xml_escape(w)) + r")\b", re.I)
        if pat.search(frag):
            frag = pat.sub(lambda m: f"<prosody pitch='+8Hz' rate='-14%' volume='+10%'>{m.group(1)}</prosody>", frag, count=1)
            used = True
    return frag, used


async def _edge(text, voice, rate, pitch, vol, raw):
    import edge_tts
    import edge_tts.communicate as ec
    with _LOCK:
        orig = ec.escape
        if raw:
            ec.escape = lambda s, *a, **k: s      # we have already escaped; let our SSML tags through
        try:
            comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{vol:+d}%", boundary="WordBoundary")
        finally:
            ec.escape = orig
    audio, words = bytearray(), []
    async for ch in comm.stream():
        if ch["type"] == "audio":
            audio += ch["data"]
        elif ch["type"] == "WordBoundary":
            s = ch["offset"] / 1e7
            words.append([ch["text"], s, s + ch["duration"] / 1e7])
    if not audio:
        raise RuntimeError("no audio")
    return bytes(audio), words


def _synth(text, voices, rate, pitch, vol, raw=False):
    last = None
    for v in voices:
        try:
            mp3, words = asyncio.run(_edge(text, v, rate, pitch, vol, raw))
            return tts._decode(mp3), words
        except Exception as e:
            last = e
    raise RuntimeError(f"voice failed: {last}")


def speak(text, cfg, delivery=None, mood=None, synth=None):
    """Returns (float32 mono samples @44.1k, [[word, start_s, end_s], ...]). `synth` is injectable for tests."""
    synth = synth or _synth
    v = cfg["voice"]
    d = clean_delivery(delivery)
    emotion = d["emotion"] or MOOD_EMOTION.get(mood or "", "neutral")
    voices = [v["name"]] + [x for x in FALLBACK_VOICES if x != v["name"]]
    base_rate, base_pitch = v.get("rate", "+0%"), v.get("pitch", "+0Hz")
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    sents = split_sentences(text) or [text]
    chunks, words_out, cursor = [], [], 0.0
    if d["pause_before_ms"]:
        chunks.append(np.zeros(int(d["pause_before_ms"] / 1000 * tts.SR), dtype=np.float32))
        cursor += d["pause_before_ms"] / 1000
    for i, s in enumerate(sents):
        rate, pitch, vol = prosody_for(s, i, len(sents), emotion, base_rate, base_pitch, rng)
        frag, used = _ssml_fragment(s, d["emphasis"], pitch)
        has_tags = "<" in frag
        try:
            if has_tags:
                samples, w = synth(frag, voices, rate, pitch, vol, raw=True)
            else:
                samples, w = synth(s, voices, rate, pitch, vol, raw=False)
        except Exception:
            samples, w = synth(s, voices, rate, pitch, vol, raw=False)        # SSML rejected: plain prosody still gives expression
        if not w:
            toks = s.split()
            dur = len(samples) / tts.SR
            w = [[t, dur * k / len(toks), dur * (k + 1) / len(toks)] for k, t in enumerate(toks)]
        # tighten the dead air edge-tts leaves at the ends of each clip
        samples = _trim(samples)
        chunks.append(samples)
        words_out += [[x[0], cursor + x[1], cursor + x[2]] for x in w]
        cursor += len(samples) / tts.SR
        if i < len(sents) - 1:
            pause = 170 + EMOTIONS[emotion][3] + (110 if s.rstrip().endswith("?") else 0) + (120 if len(s.split()) <= 5 else 0) + rng.randint(-25, 40)
            pause = max(90, pause)
            chunks.append(np.zeros(int(pause / 1000 * tts.SR), dtype=np.float32))
            cursor += pause / 1000
    return np.concatenate(chunks).astype(np.float32), words_out


def _trim(x, thresh=0.004, keep=0.05):
    idx = np.where(np.abs(x) > thresh)[0]
    if not len(idx):
        return x
    a = max(0, idx[0] - int(keep * tts.SR))
    b = min(len(x), idx[-1] + int(keep * tts.SR))
    return x[a:b]
