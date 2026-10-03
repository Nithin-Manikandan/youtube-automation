"""Illustrated-stills mode: one hand-drawn-style picture per scene (Cloudflare Workers AI free tier, FLUX schnell), animated with a slow zoom and pan.

Each scene's narration is turned into a plain description of what to draw; every picture shares one locked style and character look so the video stays consistent.
Pictures are drawn at plan time and saved next to the plan; the render step only moves the camera over them."""
import base64
import io
import json
import math
import os
import pathlib
import random
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

STYLE = ("simple hand-drawn webcomic cartoon, stick figure people with round white heads, tiny black dot eyes, messy brown hair, thin sketchy black ink lines, "
         "plain rough clothes, flat warm pastel colours, detailed hand-drawn background, cozy warm light, the same simple character design in every picture, "
         "the characters and the main action stay in the middle horizontal band of the picture, absolutely no text, no letters, no words, no logo, no signature, no watermark, no speech bubbles")
MODEL = "@cf/black-forest-labs/flux-1-schnell"


def enabled():
    return os.environ.get("STUDIO_STILLS") == "1" and bool(os.environ.get("CF_ACCOUNT_ID")) and bool(os.environ.get("CF_API_TOKEN"))


def _accounts():
    """Cloudflare accounts to use in order; when one has used its daily free allowance the next one takes over."""
    acc = [(os.environ.get("CF_ACCOUNT_ID"), os.environ.get("CF_API_TOKEN")), (os.environ.get("CF_ACCOUNT_ID_2"), os.environ.get("CF_API_TOKEN_2"))]
    return [(a, t) for a, t in acc if a and t]


_DEAD = set()


def generate(prompt, tries=4):
    """One 1024x1024 PIL image, or None when every try failed."""
    for acct, tok in _accounts():
        if acct in _DEAD:
            continue
        img = _generate_on(acct, tok, prompt, tries)
        if img == "quota":
            _DEAD.add(acct)
            continue
        return img
    return None


def _generate_on(acct, tok, prompt, tries):
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{MODEL}"
    for k in range(tries):
        try:
            body = json.dumps({"prompt": prompt, "steps": 6 if k < 2 else 8}).encode()
            req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return Image.open(io.BytesIO(base64.b64decode(d["result"]["image"]))).convert("RGB")
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            print("image error", e.code, msg[:120], flush=True)
            if e.code == 429 and "daily free allocation" in msg:
                return "quota"
            if e.code == 429 or e.code >= 500:
                time.sleep(4 + 4 * k)
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


def scene_prompts(llm, topic, narrations):
    """Plain 'what to draw' sentences, one per narration line (batched)."""
    out = [None] * len(narrations)
    for b0 in range(0, len(narrations), 14):
        chunk = narrations[b0:b0 + 14]
        q = (f"You are the art director of a funny stick-figure history channel. VIDEO TOPIC: {topic}.\n"
             "For each numbered voiceover line write ONE plain sentence describing exactly what the picture should show: who is there (at most 2 people), what they are physically doing, "
             "the key object or place, and the mood. Show the literal thing the line says (funny, concrete, specific). Era-correct clothes and objects. "
             "No text, no signs, no speech bubbles, no on-screen words.\n"
             + "\n".join(f"{b0 + i}: {t}" for i, t in enumerate(chunk))
             + '\nReturn JSON: {"scenes": [{"i": 0, "draw": "..."}]}')
        try:
            d = llm(q, 0.5)
            for s in d.get("scenes", []):
                j = int(s.get("i", -1))
                if 0 <= j < len(out) and s.get("draw"):
                    out[j] = str(s["draw"])
        except Exception:
            pass
    return [o or n for o, n in zip(out, narrations)]


def draw_all(pdir, jobs, workers=3):
    """jobs: [(scene_index, prompt)] -> {scene_index: filename}. Saves JPEGs to pdir/stills."""
    import concurrent.futures as cf
    d = pathlib.Path(pdir) / "stills"
    d.mkdir(exist_ok=True)

    def one(job):
        i, p = job
        img = generate(f"{STYLE}. {p}")
        if img is None:
            return i, None
        fn = f"s{i:03d}.jpg"
        w_, h_ = img.size
        img = img.crop((0, 0, w_, int(h_ * 0.93))).resize((w_, w_), Image.LANCZOS)      # signatures and stray words live in the bottom strip: crop it away
        img.save(d / fn, quality=93)
        return i, fn
    res = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fn in ex.map(one, jobs):
            if fn:
                res[i] = fn
    return res


_CACHE = {}


def _composite(path, W=1920, H=1080):
    """The whole square picture centred in a 16:9 frame; the side bars are a blurred, slightly darkened extension of the same art (no cropping)."""
    import cv2
    if path not in _CACHE:
        _CACHE.clear()
        im = Image.open(path).convert("RGB")
        a = np.asarray(im)
        bg = cv2.resize(a, (W, W), interpolation=cv2.INTER_AREA)[(W - H) // 2:(W - H) // 2 + H]
        bg = cv2.GaussianBlur(bg, (0, 0), 90)
        mean = bg.reshape(-1, 3).mean(axis=0)
        bg = (bg.astype(np.float32) * 0.45 + mean * 0.55).astype(np.uint8)        # calm, low-contrast side bars in the picture's own paper colour
        fg = cv2.resize(a, (H, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        x0 = (W - H) // 2
        mask = np.ones((H, H), np.float32)
        f = 36                                                            # feather the picture's left and right edges into the blurred bars
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        mask[:, :f] *= ramp[None, :]
        mask[:, -f:] *= ramp[::-1][None, :]
        out = bg.astype(np.float32)
        out[:, x0:x0 + H] = out[:, x0:x0 + H] * (1 - mask[..., None]) + fg * mask[..., None]
        _CACHE[path] = out.astype(np.uint8)
    return _CACHE[path]


def frame(path, t, dur, W, H, seed=0):
    """Slow push-in with a gentle drift over the whole picture."""
    import cv2
    a = _composite(path)
    CH, CW = a.shape[:2]
    u = min(1.0, max(0.0, t / max(dur, 1e-6)))
    e = u * u * (3 - 2 * u)
    rnd = random.Random(seed)
    z0, z1 = (1.0, 1.10) if rnd.random() < .6 else (1.10, 1.0)
    z = z0 + (z1 - z0) * e
    ww = CW / z
    hh = ww * H / W
    cx = CW / 2 + rnd.choice([-1, 1]) * 0.03 * CW * (e - .5)
    cy = CH / 2 + rnd.choice([-1, 1]) * 0.03 * CH * (e - .5)
    x0 = max(0, min(CW - ww, cx - ww / 2))
    y0 = max(0, min(CH - hh, cy - hh / 2))
    M = np.float32([[W / ww, 0, -x0 * W / ww], [0, H / hh, -y0 * H / hh]])
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_AREA if z < 1.01 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _cutout(img):
    """Subject on a plain light background -> RGBA with the background removed (flood fill from the borders)."""
    import cv2
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a.astype(np.float32) - bg, axis=2)
    near = (dist < 34).astype(np.uint8)
    flood = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for sx, sy in ((2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3), (w // 2, 2), (w // 2, h - 3), (2, h // 2), (w - 3, h // 2)):
        if near[sy, sx]:
            m2 = near.copy()
            cv2.floodFill(m2, flood.copy(), (sx, sy), 2)
            mask |= (m2 == 2).astype(np.uint8)
    subject = (1 - mask).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(subject)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        subject = (lab == keep).astype(np.uint8)
    subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    subject = cv2.GaussianBlur(subject.astype(np.float32), (0, 0), 1.2)
    rgba = np.dstack([a, (np.clip(subject, 0, 1) * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def thumbnail(text, scene_prompt, out_path, variant=0, accent=(255, 240, 30), arrow=False):
    """Professional-style thumbnail: sticker-outlined cutout of one funny subject on a saturated sunburst, huge outlined hook text, a red circle on the funny detail."""
    import cv2
    from PIL import ImageDraw, ImageFilter, ImageFont
    p = (f"{STYLE}. ONE single character shown from the waist up, huge and centred, an extremely exaggerated funny expression (eyes huge, mouth wide open, sweat drops), "
         f"{scene_prompt}. Drawn on a completely plain pure white background with nothing else in the picture, no ground, no shadow, no other people, no objects except the ones named. Bold thick outlines, bright saturated colours.")
    img = None
    for k in range(3):
        img = generate(p)
        if img is not None:
            break
    if img is None:
        return None
    img = img.crop((0, 0, img.width, int(img.height * 0.93)))
    cut = _cutout(img)
    W, H = 1280, 720
    palette = [((255, 214, 10), (255, 120, 0)), ((255, 72, 72), (190, 0, 60)), ((40, 200, 255), (20, 90, 230))][variant % 3]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W * 0.68, H * 0.52
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (W * 0.75)
    ang = np.arctan2(yy - cy, xx - cx)
    rays = 0.5 + 0.5 * np.sign(np.sin(ang * 12))
    t = np.clip(r, 0, 1)[..., None]
    bg = np.array(palette[0], np.float32) * (1 - t) + np.array(palette[1], np.float32) * t
    bg = bg * (0.88 + 0.12 * rays[..., None])
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
    # subject: scale to fill the right two thirds, add a thick white sticker outline and a thin black outer line
    bb = cut.getbbox()
    cut = cut.crop(bb)
    k = min(H * 0.98 / cut.height, W * 0.66 / cut.width)
    cut = cut.resize((int(cut.width * k), int(cut.height * k)), Image.LANCZOS)
    al = np.asarray(cut.split()[3])
    white = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    black = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47, 47)))
    px, py = int(W - cut.width - W * 0.02), int(H - cut.height + H * 0.01)
    pad = 30
    layer = Image.new("RGBA", (cut.width + 2 * pad, cut.height + 2 * pad), (0, 0, 0, 0))
    def put(mask, color):
        m = Image.fromarray(np.pad(mask, pad)).convert("L")
        solid = Image.new("RGBA", layer.size, color)
        layer.paste(solid, (0, 0), m)
    put(black, (18, 12, 20, 255)); put(white, (255, 255, 255, 255))
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    sh.paste(Image.new("RGBA", layer.size, (0, 0, 0, 140)), (0, 0), Image.fromarray(np.pad(black, pad)).convert("L").filter(ImageFilter.GaussianBlur(10)))
    canvas.alpha_composite(sh, (px - pad + 10, py - pad + 14))
    canvas.alpha_composite(layer, (px - pad, py - pad))
    canvas.alpha_composite(cut, (px, py))
    d = ImageDraw.Draw(canvas)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    lines = [w for w in text.upper().split() if w][:4]
    size = 330
    while size > 80:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.44 and size * .92 * len(lines) <= H * 0.9:
            break
        size -= 6
    y = (H - size * .92 * len(lines)) / 2 - size * .03
    cols = [(255, 255, 255), accent, (255, 255, 255), accent]
    for i, l in enumerate(lines):
        x = 28
        d.text((x + 10, y + 12), l, font=f, fill=(0, 0, 0), stroke_width=20, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=cols[i % 4] if len(lines) > 1 else accent, stroke_width=17, stroke_fill=(18, 12, 20))
        y += size * .92
    canvas.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path
