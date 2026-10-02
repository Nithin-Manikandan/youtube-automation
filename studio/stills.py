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

STYLE = ("simple cartoon drawing in the style of the Ink Explainer YouTube channel, stick figure people with round white heads, tiny black dot eyes, messy brown hair, "
         "plain rough clothes, thin sketchy black ink lines, flat warm pastel colours, hand-drawn detailed background, cozy warm light, "
         "the characters and the main action stay in the middle horizontal band of the picture, no text, no letters, no speech bubbles, no watermark")
MODEL = "@cf/black-forest-labs/flux-1-schnell"


def enabled():
    return os.environ.get("STUDIO_STILLS") == "1" and bool(os.environ.get("CF_ACCOUNT_ID")) and bool(os.environ.get("CF_API_TOKEN"))


def generate(prompt, tries=4):
    """One 1024x1024 PIL image, or None when every try failed."""
    acct, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{MODEL}"
    for k in range(tries):
        try:
            body = json.dumps({"prompt": prompt, "steps": 6 if k < 2 else 8}).encode()
            req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=120))
            return Image.open(io.BytesIO(base64.b64decode(d["result"]["image"]))).convert("RGB")
        except urllib.error.HTTPError as e:
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
    """A clickable thumbnail in the same illustrated style as the video: one drawn picture with a huge expressive face plus chunky headline text."""
    from PIL import ImageDraw, ImageFont
    emo = ("wide-eyed shocked face with mouth open", "disgusted grimace holding his nose", "huge surprised grin with raised eyebrows")[variant % 3]
    p = (f"{STYLE}. Close-up thumbnail composition: ONE stick figure with a very large round white head filling the left half, {emo}, messy brown hair, "
         f"plus the key funny object or place of the story on the right: {scene_prompt}. Bold simple shapes, high contrast, bright warm colours.")
    img = None
    for k in range(3):
        img = generate(p)
        if img is not None:
            break
    if img is None:
        return None
    W, H = 1280, 720
    a = np.asarray(img)
    S = a.shape[0]
    hh = int(S * 9 / 16)
    y0 = int(S * 0.20)
    crop = Image.fromarray(a[y0:y0 + hh, :]).resize((W, H), Image.LANCZOS)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    words = [w for w in text.upper().split() if w][:3]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + len(w) < 8:
            cur += " " + w
        else:
            if cur:
                lines.append(cur)
            cur = w
    lines.append(cur)
    d = ImageDraw.Draw(crop)
    size = 300
    while size > 80:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.56 and size * .92 * len(lines) <= H * 0.62:
            break
        size -= 8
    y = 24
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x = W - 36 - tw
        d.text((x + 8, y + 10), l, font=f, fill=(0, 0, 0), stroke_width=16, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=accent if i == len(lines) - 1 else (255, 255, 255), stroke_width=13, stroke_fill=(20, 16, 22))
        y += size * .92
    crop.save(out_path, "JPEG", quality=94)
    return out_path
