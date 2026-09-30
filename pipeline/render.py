"""Frame-by-frame renderer: cinematic camera moves, cuts with flash, grade, grain and word-pop captions."""
import math
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import ffmpeg
from .config import ROOT


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _find_font(preferred):
    import os
    for p in ([str(ROOT / preferred)] if preferred else []) + [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        if os.path.exists(p):
            return p
    raise RuntimeError("No caption font found. Put a .ttf at the path in channel.yaml captions.font")


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


class Scene:
    """One image plus its camera move."""

    MOVES = ["in", "pan_r", "out", "pan_l", "in_up", "out_down"]

    def __init__(self, idx, path, start, dur, W, H, cfg):
        self.start, self.dur = start, dur
        self.W, self.H = W, H
        self.zoom = cfg["visuals"].get("zoom_strength", 0.16)
        self.shake = cfg["visuals"].get("shake", 0.6)
        self.move = self.MOVES[idx % len(self.MOVES)]
        rnd = np.random.default_rng(idx + 11)
        self.ph = rnd.uniform(0, 6.28, 6)
        im = Image.open(path).convert("RGB")
        # Upscale once so the deepest push-in still has pixels to spare.
        need = max(W / im.width, H / im.height) * (1 + self.zoom) * 1.12
        if need > 1:
            im = im.resize((int(im.width * need), int(im.height * need)), Image.LANCZOS)
        self.src = np.asarray(im)
        self.base = max(W / im.width, H / im.height)

    def frame(self, t):
        W, H = self.W, self.H
        ih, iw = self.src.shape[:2]
        p = _smooth(t / max(self.dur, 1e-3))
        z = self.zoom
        m = self.move
        s = 1 + z * (p if m.startswith("in") or m.startswith("pan") else 1 - p)
        if m.startswith("pan"):
            s = 1 + z * 0.45
        slack_x = iw - W / (self.base * s)
        slack_y = ih - H / (self.base * s)
        fx = 0.5 + (0.5 * (p - 0.5) * 1.6 * (1 if m == "pan_r" else -1 if m == "pan_l" else 0))
        fy = 0.5 + {"in_up": -0.18 * (p - 0.5), "out_down": 0.18 * (p - 0.5)}.get(m, 0)
        cx = iw / 2 + (fx - 0.5) * slack_x
        cy = ih / 2 + (fy - 0.5) * slack_y
        tt = self.start + t
        sh = self.shake
        cx += sh * (6 * math.sin(tt * 2.1 + self.ph[0]) + 3 * math.sin(tt * 5.3 + self.ph[1]))
        cy += sh * (6 * math.sin(tt * 1.7 + self.ph[2]) + 3 * math.sin(tt * 4.6 + self.ph[3]))
        ang = sh * 0.35 * math.sin(tt * 1.3 + self.ph[4])
        M = cv2.getRotationMatrix2D((cx, cy), ang, self.base * s)
        M[0, 2] += W / 2 - cx
        M[1, 2] += H / 2 - cy
        return cv2.warpAffine(self.src, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


class Captions:
    def __init__(self, words, cfg, W, H):
        c = cfg["captions"]
        self.W, self.H = W, H
        self.font_path = _find_font(c.get("font"))
        self.size = c.get("size", 96)
        self.col, self.hi = _hex(c["color"]), _hex(c["highlight"])
        self.y = c.get("y_position", 0.66)
        n = c.get("words_per_chunk", 3)
        self.chunks, cur = [], []
        for w in words:
            w = [re.sub(r"\s+", "", w[0]).upper(), w[1], w[2]]
            if not w[0]:
                continue
            if cur and w[1] - cur[-1][2] > 0.5:  # a pause (or a different speaker) starts a new caption
                self.chunks.append(cur)
                cur = []
            cur.append(w)
            if len(cur) >= n or (len(cur) >= 2 and re.search(r"[.!?…]$", w[0])):
                self.chunks.append(cur)
                cur = []
        if cur:
            self.chunks.append(cur)
        self.cache = {}
        self.font = ImageFont.truetype(self.font_path, self.size)

    def _draw(self, chunk, active):
        size = self.size
        font = self.font
        while True:
            font = ImageFont.truetype(self.font_path, size)
            widths = [font.getlength(w[0]) for w in chunk]
            gap = font.getlength(" ")
            total = sum(widths) + gap * (len(chunk) - 1)
            if total <= self.W - 140 or size < 50:
                break
            size -= 6
        pad = 26
        h = int(size * 1.35)
        img = Image.new("RGBA", (int(total) + pad * 2, h + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = pad
        for k, w in enumerate(chunk):
            color = self.hi if k == active else self.col
            d.text((x + 4, pad + 6), w[0], font=font, fill=(0, 0, 0, 170), stroke_width=9, stroke_fill=(0, 0, 0, 170))
            d.text((x, pad), w[0], font=font, fill=color + (255,), stroke_width=9, stroke_fill=(0, 0, 0, 255))
            x += widths[k] + gap
        return np.asarray(img)

    def overlay(self, frame, t):
        ci = None
        for i, ch in enumerate(self.chunks):
            end = self.chunks[i + 1][0][1] if i + 1 < len(self.chunks) else ch[-1][2] + 0.2
            if ch[0][1] - 0.02 <= t < end:
                ci = i
                break
        if ci is None:
            return
        ch = self.chunks[ci]
        active = max([k for k, w in enumerate(ch) if w[1] <= t] or [0])
        age = t - ch[0][1]
        pop = 1.0 + 0.16 * math.exp(-age * 16) * math.cos(age * 30) if age >= 0 else 0.85
        pop_q = round(pop * 20) / 20
        key = (ci, active)
        if key not in self.cache:
            self.cache[key] = self._draw(ch, active)
        base = self.cache[key]
        if abs(pop_q - 1.0) > 1e-6:
            base = cv2.resize(base, None, fx=pop_q, fy=pop_q, interpolation=cv2.INTER_AREA if pop_q < 1 else cv2.INTER_LINEAR)
        h, w = base.shape[:2]
        x0 = (self.W - w) // 2
        y0 = int(self.H * self.y - h / 2)
        _blend(frame, base, x0, y0)


def _blend(frame, rgba, x0, y0):
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[..., 3:4].astype(np.float32) / 255
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + sub[..., :3] * a).astype(np.uint8)


def make_hook_image(hook_text, W, font_path):
    """Big stroked headline shown in the first seconds of a video (RGBA array, W wide)."""
    hf = ImageFont.truetype(font_path, 88)
    lines, cur = [], ""
    for w in hook_text.upper().split():
        if hf.getlength((cur + " " + w).strip()) > W - 160 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    im = Image.new("RGBA", (W, int(88 * 1.3) * len(lines) + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for k, ln in enumerate(lines):
        d.text(((W - hf.getlength(ln)) / 2, 20 + k * 114), ln, font=hf, fill=(255, 255, 255, 255),
               stroke_width=9, stroke_fill=(0, 0, 0, 255))
    return np.asarray(im)


def render(scene_specs, hook_text, audio, out_path, cfg, log=print):
    """scene_specs: [{path, start, dur, words}]; audio: float32 mono @44.1k."""
    import wave

    vcfg = cfg["video"]
    W, H, fps = vcfg["width"], vcfg["height"], vcfg["fps"]
    vis = cfg["visuals"]
    total = scene_specs[-1]["start"] + scene_specs[-1]["dur"]
    nframes = int(round(total * fps))

    wav_path = str(out_path) + ".wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    scenes = [Scene(i, s["path"], s["start"], s["dur"], W, H, cfg) for i, s in enumerate(scene_specs)]
    all_words = [w for s in scene_specs for w in s["words"]]
    caps = Captions(all_words, cfg, W, H)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
    vig = (1 - vis.get("vignette", 0.5) * np.clip(r, 0, 1) ** 2.2)[..., None].astype(np.float32)
    tint = np.array(vis.get("tint", [1, 1, 1]), dtype=np.float32)[None, None, :]
    rng = np.random.default_rng(5)
    g = vis.get("grain", 0.05) * 255
    grain = [(rng.standard_normal((H, W, 1)) * g).astype(np.float32) for _ in range(10)]

    hook_img = make_hook_image(hook_text, W, caps.font_path) if hook_text else None

    cmd = [ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "pipe:0", "-i", wav_path, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-maxrate", "9M", "-bufsize", "18M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-movflags", "+faststart", "-shortest", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    X = 0.14  # crossfade seconds
    stills = {}
    for n in range(nframes):
        t = n / fps
        i = next((k for k in range(len(scenes) - 1, -1, -1) if scenes[k].start <= t + 1e-6), 0)
        sc = scenes[i]
        f = sc.frame(min(t - sc.start, sc.dur)).astype(np.float32)
        end = sc.start + sc.dur
        if i + 1 < len(scenes) and t > end - X:
            a = _smooth((t - (end - X)) / X)
            f = f * (1 - a) + scenes[i + 1].frame(0).astype(np.float32) * a
        f = f * tint * vig + grain[n % 10]
        since_cut = t - sc.start
        if i > 0 and since_cut < 0.12:
            f += 34 * (1 - since_cut / 0.12)  # punch flash on the cut
        fr = np.clip(f, 0, 255).astype(np.uint8)
        caps.overlay(fr, t)
        if hook_img is not None and t < 2.4:
            a = 1.0 if t < 1.8 else max(0.0, 1 - (t - 1.8) / 0.6)
            ov = hook_img.copy()
            ov[..., 3] = (ov[..., 3] * a).astype(np.uint8)
            _blend(fr, ov, 0, int(H * 0.14))
        if n % int(fps * 1.0) == 0 and t - sc.start > sc.dur * 0.4 and i not in stills:
            stills[i] = fr.copy()
        proc.stdin.write(fr.tobytes())
        if n % (fps * 5) == 0:
            log(f"    rendered {n}/{nframes} frames")
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return stills
