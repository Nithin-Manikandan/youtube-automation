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


def thumbnail(text, scene_prompt, out_path, variant=0, accent=(255, 226, 40), arrow=False):
    """Full-bleed illustrated thumbnail: one huge funny subject on the right, big outlined text on the left, punched-up colour and contrast, a hand-drawn arrow."""
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    p = (f"{STYLE}. YouTube thumbnail illustration: ONE single large subject, close-up, filling the right two thirds of the picture, a very funny exaggerated expression, "
         f"a plain simple pale background; the subject is placed entirely in the RIGHT half of the picture and the LEFT 45 percent of the picture is completely empty background. {scene_prompt}. Bold simple shapes, thick outlines, bright high-contrast warm colours.")
    img = None
    for k in range(3):
        img = generate(p)
        if img is not None:
            break
    if img is None:
        return None
    W, H = 1280, 720
    a = img.crop((0, 0, img.width, int(img.height * 0.93)))
    S = a.width
    hh = int(S * 9 / 16)
    y0 = max(0, int((a.height - hh) * 0.35))
    crop = a.crop((0, y0, S, y0 + hh)).resize((W, H), Image.LANCZOS)
    crop = ImageEnhance.Color(crop).enhance(1.35)
    crop = ImageEnhance.Contrast(crop).enhance(1.18)
    crop = ImageEnhance.Brightness(crop).enhance(1.04)
    arr = np.asarray(crop).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.28 * np.clip(((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .8)) ** 2, 0, 1)
    arr *= vig[..., None]
    left = np.clip(1 - xx / (W * 0.46), 0, 1) ** 1.2                    # gentle warm glow behind the text so it pops
    arr = arr * (1 - 0.30 * left[..., None]) + np.array([255, 196, 96], np.float32) * 0.30 * left[..., None]
    crop = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(crop)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    lines = [w for w in text.upper().split() if w][:4]
    size = 300
    while size > 80:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.46 and size * .93 * len(lines) <= H * 0.92:
            break
        size -= 6
    total = size * .93 * len(lines)
    y = (H - total) / 2 - size * .02
    for i, l in enumerate(lines):
        x = 34
        d.text((x + 9, y + 11), l, font=f, fill=(0, 0, 0), stroke_width=18, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=accent if i == len(lines) - 1 else (255, 255, 255), stroke_width=15, stroke_fill=(18, 14, 20))
        y += size * .93
    if arrow:
        ax, ay = int(W * 0.50), int(H * 0.86)
        pts = [(ax - 110, ay + 6), (ax - 40, ay - 26), (ax + 36, ay - 70)]
        d.line(pts, fill=(20, 14, 20), width=22, joint="curve")
        d.line(pts, fill=(235, 52, 44), width=12, joint="curve")
        tip = pts[-1]
        d.polygon([(tip[0] + 34, tip[1] - 24), (tip[0] - 14, tip[1] - 6), (tip[0] + 18, tip[1] + 34)], fill=(20, 14, 20))
        d.polygon([(tip[0] + 26, tip[1] - 18), (tip[0] - 4, tip[1] - 6), (tip[0] + 16, tip[1] + 22)], fill=(235, 52, 44))
    crop.save(out_path, "JPEG", quality=95)
    return out_path
