"""Puppet scenes: turns a scene dict (actors, keys, narration word timings) into frames using the layered puppet rig over a wide painted background."""
import math
import re

import cv2
import numpy as np
from PIL import Image

from . import puppet

W, H = 1280, 720

# the director's pose names (mocap clips, reactions, legacy poses) -> hand-designed puppet poses
POSEMAP = [
    (r"point", "point"), (r"shrug|surprised|explain_b", "shrug"), (r"flinch|surprise|rx:recoil", "shock"), (r"cower|scared|duck|dodge", "cower"),
    (r"happy|cheer|wave|rx:cheer", "cheer"), (r"think|look_around", "think"), (r"facepalm|cry|sad|slump|rx:slump|sit", "facepalm"),
    (r"quarrel|threat|demand|push|pull|punch|kick|sword|fight", "point"), (r"explain|talk|proud|armscross|rx:look|mc:sneak|stand|idle", None),
]
FACE_MOOD = {"worried": "worried", "shock": "shock", "smile": "smile", "sad": "sad", "angry": "angry", "neutral": "neutral"}
HATS = {"crown": "crown", "hardhat": "hardhat", "helmet": "helmet", "navycap": "cap", "hat": "cap"}


def _viseme(ch):
    ch = ch.lower()
    if ch in "mbp":
        return "A"
    if ch in "fv":
        return "G"
    if ch in "l":
        return "H"
    if ch in "ao":
        return "D" if ch == "a" else "E"
    if ch == "u" or ch == "w":
        return "F"
    if ch in "ei":
        return "C"
    return "B"


def mouth_cues(words, who=None):
    """words: [(text, t0, t1)] relative to the scene start. Letter shapes spread across each word; a held rest between words."""
    cues = []
    for w, a, b in words:
        letters = re.sub(r"[^a-z]", "", w.lower())
        if not letters or b <= a:
            continue
        seq = []
        for ch in letters:
            v = _viseme(ch)
            if not seq or seq[-1] != v:
                seq.append(v)
        seq = seq[:max(2, int((b - a) / 0.07))] if len(seq) > int((b - a) / 0.07) > 1 else seq
        step = (b - a) / len(seq)
        for i, v in enumerate(seq):
            cues.append((a + i * step, a + (i + 1) * step, v))
    return cues


def _timeline(keys, dur):
    tl, last, flip = [], None, 0
    for k in keys:
        nm = str(k["pose"])
        target = None
        for pat, name in POSEMAP:
            if re.search(pat, nm):
                target = name
                break
        if target is None:                                                  # talking/explaining poses alternate between two gestures so the hands keep moving
            flip += 1
            target = "explain" if flip % 2 else "explain2"
        if target != last or nm.startswith(("mc:talk", "mc:explain")):
            tl.append((float(k["t"]), target))
            last = target
    if not tl:
        tl = [(0.0, "idle")]
    out, prev_t = [], -9
    for t, n in tl:                                                         # no two key poses closer than 0.45 s
        if t - prev_t >= 0.45 or not out:
            out.append((t, n)); prev_t = t
    return [(0.0, "idle")] + out if out[0][0] > 0.2 else out


class PuppetScene:
    def __init__(self, scene, bg_path, words, dur, seed=0):
        self.dur, self.seed = dur, seed
        self.bg = np.asarray(Image.open(bg_path).convert("RGB").resize((int(W * 1.16), int(H * 1.16)), Image.LANCZOS))
        self.actors = []
        specs = scene.get("actors", [])
        if len(specs) >= 2:                                                  # keep people from standing inside each other: spread the first two apart
            xs = sorted(range(len(specs)), key=lambda i: specs[i]["keys"][0]["x"])
            for rank, i in enumerate(xs[:2]):
                for k in specs[i]["keys"]:
                    k = dict(k)
            self._xs = {xs[0]: 0.30, xs[1]: 0.70}
        else:
            self._xs = {}
        talkers = [i for i, a in enumerate(specs) if a.get("talks")] or ([0] if specs else [])
        # two talkers take turns by sentence; one talker gets all the words
        chunks, cur = [], []
        for i, (w, a, b) in enumerate(words):
            if cur and a - cur[-1][2] > 0.32:
                chunks.append(cur); cur = []
            cur.append((w, a, b))
        if cur:
            chunks.append(cur)
        for ai, sp in enumerate(specs):
            keys = sp["keys"]
            mine = [wd for ci, ch in enumerate(chunks) for wd in ch if talkers and talkers[ci % len(talkers)] == ai]
            moods, last = [], None
            for k in keys:
                m = FACE_MOOD.get(k.get("face", "neutral"), "neutral")
                if m != last:
                    moods.append((float(k["t"]), m)); last = m
            props = sp.get("props", [])
            hat = next((HATS[p] for p in props if p in HATS), None)
            tun = tuple(int(v) for v in (sp.get("tunic") or sp.get("color") or (150, 100, 60))[:3])
            look = puppet.Look(tunic=tun, hair=tuple(sp.get("hair", (96, 62, 40))), hat=hat, seed=ai * 5 + seed)
            xf = float(keys[0]["x"])
            xf = min(0.84, max(0.16, xf))
            xf = self._xs.get(ai, xf)
            facing = keys[0].get("facing", 1)
            a_ = puppet.Actor(look, _timeline(keys, dur), mouth_cues(mine), x_frac=xf, scale=0.80 if len(specs) < 3 else 0.7, facing=facing, seed=ai * 3 + seed, mood_track=moods or None)
            self.actors.append(a_)

    def frame(self, t):
        u = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
        e = u * u * (3 - 2 * u)
        bh, bw = self.bg.shape[:2]
        zb = 1.0 + 0.06 * e
        ww, hh = bw / 1.16 / zb, bh / 1.16 / zb
        x0 = max(0, min(bw - ww, bw / 2 - ww / 2 + (e - .5) * bw * 0.03))
        y0 = max(0, min(bh - hh, bh / 2 - hh / 2))
        M = np.float32([[W / ww, 0, -x0 * W / ww], [0, H / hh, -y0 * H / hh]])
        out = cv2.warpAffine(self.bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        for a in sorted(self.actors, key=lambda a: a.xf):
            tile, x0_, y0_ = a.render(t, self.dur, W, H)
            ta = np.asarray(tile).astype(np.float32)
            th, tw = ta.shape[:2]
            xa, ya = max(0, x0_), max(0, y0_)
            xb, yb = min(W, x0_ + tw), min(H, y0_ + th)
            if xb <= xa or yb <= ya:
                continue
            sub = ta[ya - y0_:yb - y0_, xa - x0_:xb - x0_]
            al = sub[..., 3:4] / 255.0
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - al) + sub[..., :3] * al
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        out *= np.clip(1 - 0.16 * (((xx - W / 2) / (W * .62)) ** 2 + ((yy - H / 2) / (H * .62)) ** 2), 0.75, 1)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
