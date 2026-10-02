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


def _img(path):
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = np.asarray(Image.open(path).convert("RGB"))
    return _CACHE[path]


def frame(path, t, dur, W, H, seed=0):
    """Slow push-in with a gentle drift across the square picture, 16:9 window."""
    a = _img(path)
    S = a.shape[0]
    u = min(1.0, max(0.0, t / max(dur, 1e-6)))
    e = u * u * (3 - 2 * u)
    rnd = random.Random(seed)
    z0, z1 = (1.0, 1.16) if rnd.random() < .6 else (1.16, 1.0)
    z = z0 + (z1 - z0) * e
    ww = S / z
    hh = ww * H / W
    dx, dy = rnd.choice([-1, 1]) * 0.05 * S, rnd.choice([-1, 1]) * 0.04 * S
    cx = S / 2 + dx * (e - .5)
    cy = S * 0.5 + dy * (e - .5)
    x0 = max(0, min(S - ww, cx - ww / 2))
    y0 = max(0, min(S - hh, cy - hh / 2))
    M = np.float32([[W / ww, 0, -x0 * W / ww], [0, H / hh, -y0 * H / hh]])
    import cv2
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)


def thumbnail(text, scene_prompt, out_path, variant=0, accent=(255, 226, 40)):
    """A clickable thumbnail in the same illustrated style as the video: a clear drawing on the left, a bold colour panel with huge text on the right."""
    from PIL import ImageDraw, ImageFont
    emo = ("wide-eyed shocked face with mouth open and sweat drops", "desperate squirming face, eyes squeezed shut", "huge panicked face with raised eyebrows")[variant % 3]
    p = (f"{STYLE}. Thumbnail drawing, medium shot showing one character from the knees up with plenty of space around him, big expressive face: {emo}. "
         f"Scene: {scene_prompt}. Bold simple shapes, high contrast, bright warm colours, clean uncluttered background.")
    img = None
    for k in range(3):
        img = generate(p)
        if img is not None:
            break
    if img is None:
        return None
    W, H = 1280, 720
    PW = 600                                                            # text panel width
    canvas = Image.new("RGB", (W, H), (235, 72, 52))
    art = img.crop((0, 0, img.width, int(img.height * 0.93))).resize((H, H), Image.LANCZOS)
    canvas.paste(art, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.rectangle([H - 6, 0, H + 6, H], fill=(20, 16, 22))
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    words = [w for w in text.upper().split() if w][:3]
    lines = [w for w in words]
    size = 260
    while size > 70:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= PW - 60 and size * .96 * len(lines) <= H - 80:
            break
        size -= 6
    total = size * .96 * len(lines)
    y = (H - total) / 2 - size * .04
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x = H + 6 + (W - H - 6 - tw) / 2
        d.text((x + 6, y + 8), l, font=f, fill=(0, 0, 0), stroke_width=12, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=accent if i == len(lines) - 1 else (255, 255, 255), stroke_width=11, stroke_fill=(20, 16, 22))
        y += size * .96
    canvas.save(out_path, "JPEG", quality=94)
    return out_path
