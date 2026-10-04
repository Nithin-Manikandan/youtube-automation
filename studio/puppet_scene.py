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


TALK_CYCLE = ["explain", "open", "explain2", "count", "lean", "explain", "smug", "open", "point", "explain2"]


def _timeline(keys, dur):
    """Director pose names -> puppet poses. Talking beats walk through a varied cycle of gestures so nobody keeps repeating one move, and 'point' is saved for real pointing."""
    tl, last, n = [], None, 0
    for k in keys:
        nm = str(k["pose"])
        target = None
        for pat, name in POSEMAP:
            if re.search(pat, nm):
                target = name
                break
        if target == "point" and not re.search(r"point", nm):               # fight/push/demand clips are not pointing
            target = None
        if target is None:
            target = TALK_CYCLE[(n + int(k["t"] * 3)) % len(TALK_CYCLE)]
            n += 1
        if target != last or nm.startswith(("mc:talk", "mc:explain")):
            tl.append((float(k["t"]), target)); last = target
    if not tl:
        tl = [(0.0, "idle")]
    out, prev_t = [], -9
    for t, nme in tl:
        if t - prev_t >= 0.9 or not out:                                    # hold each gesture at least a second
            out.append((t, nme)); prev_t = t
    # keep him gesturing: if a gap is long, drop a fresh gesture in
    full, c = [], 1
    for i, (t, nme) in enumerate(out):
        full.append((t, nme))
        nxt = out[i + 1][0] if i + 1 < len(out) else dur
        gap = nxt - t
        while gap > 2.6:
            t += 2.2; gap -= 2.2
            full.append((t, TALK_CYCLE[(len(full) + c) % len(TALK_CYCLE)])); c += 1
    return [(0.0, "idle")] + full if full[0][0] > 0.2 else full


def _unify(a):
    """One palette for every background: tame the saturation extremes and wash everything with the same warm paper tone so scenes feel like one show."""
    x = a.astype(np.float32)
    g = x.mean(axis=2, keepdims=True)
    x = g + (x - g) * 0.82                                                  # calmer colour
    x = (x - 128) * 0.95 + 128 + 4                                         # slightly softer contrast
    paper = np.array([250, 232, 200], np.float32)
    x = x * 0.88 + paper * 0.12                                            # warm paper wash
    return np.clip(x, 0, 255).astype(np.uint8)


class PuppetScene:
    def __init__(self, scene, bg_path, words, dur, seed=0):
        self.dur, self.seed = dur, seed
        self.bg = _unify(np.asarray(Image.open(bg_path).convert("RGB").resize((int(W * 1.16), int(H * 1.16)), Image.LANCZOS)))
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
            look = puppet.Look(**puppet.HOST) if ai == 0 else puppet.Look(tunic=tun, hair=tuple(sp.get("hair", (96, 62, 40))), hat=hat, seed=ai * 5 + seed)
            xf = float(keys[0]["x"])
            xf = min(0.84, max(0.16, xf))
            xf = self._xs.get(ai, xf)
            facing = keys[0].get("facing", 1)
            a_ = puppet.Actor(look, _timeline(keys, dur), mouth_cues(mine), x_frac=xf, scale=0.82 if len(specs) < 2 else 0.78 if len(specs) < 3 else 0.68, facing=facing, seed=ai * 3 + seed, mood_track=moods or None)
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


class PresenterScene:
    """The picture fills the frame and shows what the narrator is talking about. The host is the only animated character: he stands in a lower corner, faces the picture,
    gestures at it and talks. The side changes from scene to scene so he is never stuck in the middle."""

    def __init__(self, scene, pic_path, words, dur, seed=0, index=0):
        self.dur = dur
        from . import stills
        self.bg = np.asarray(Image.fromarray(stills._composite(pic_path)).resize((int(W * 1.10), int(H * 1.10)), Image.LANCZOS))     # the whole square picture centred, soft bars either side
        side = -1 if (seed + index) % 2 == 0 else 1                         # -1 left bar, +1 right bar
        xf = 0.105 if side < 0 else 0.895
        facing = 1 if side < 0 else -1
        tl = [(0.0, "idle"), (0.5, "smug")]
        t, k = 1.3, 0
        cyc = ["point", "open", "explain", "point", "count", "explain2", "lean", "point", "shrug", "open"]
        while t < dur - 0.6:
            tl.append((t, cyc[(k + seed) % len(cyc)])); t += 2.3 + (k % 3) * 0.4; k += 1
        look = puppet.Look(**puppet.HOST)
        self.actor = puppet.Actor(look, tl, mouth_cues([(w[0], w[1], w[2]) for w in words]), x_frac=xf, scale=0.70, facing=facing, seed=seed + 3, mood_track=[(0.0, "neutral")])
        self.side = side

    def frame(self, t):
        u = min(1.0, max(0.0, t / max(self.dur, 1e-6)))
        e = u * u * (3 - 2 * u)
        bh, bw = self.bg.shape[:2]
        zb = 1.0 + 0.05 * e
        ww, hh = bw / 1.10 / zb, bh / 1.10 / zb
        drift = (e - .5) * bw * 0.025 * (-self.side)
        x0 = max(0, min(bw - ww, bw / 2 - ww / 2 + drift))
        y0 = max(0, min(bh - hh, bh / 2 - hh / 2))
        M = np.float32([[W / ww, 0, -x0 * W / ww], [0, H / hh, -y0 * H / hh]])
        out = cv2.warpAffine(self.bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        out *= np.clip(1 - 0.14 * (((xx - W / 2) / (W * .62)) ** 2 + ((yy - H / 2) / (H * .62)) ** 2), 0.78, 1)[..., None]
        tile, x0_, y0_ = self.actor.render(t, self.dur, W, H)
        ta = np.asarray(tile).astype(np.float32)
        th, tw = ta.shape[:2]
        xa, ya = max(0, x0_), max(0, y0_)
        xb, yb = min(W, x0_ + tw), min(H, y0_ + th)
        if xb > xa and yb > ya:
            sub = ta[ya - y0_:yb - y0_, xa - x0_:xb - x0_]
            al = sub[..., 3:4] / 255.0
            shade = np.exp(-(((xx[ya:yb, xa:xb] - (x0_ + tw / 2)) / (tw * .30)) ** 2 + ((yy[ya:yb, xa:xb] - (H * .965)) / (H * .022)) ** 2)) * 0.30
            out[ya:yb, xa:xb] *= (1 - shade[..., None])
            out[ya:yb, xa:xb] = out[ya:yb, xa:xb] * (1 - al) + sub[..., :3] * al
        return np.clip(out, 0, 255).astype(np.uint8)
