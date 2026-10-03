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


def generate_wide(prompt, w=1280, h=720, tries=3):
    """A wide 16:9 background picture from Cloudflare's SDXL-Lightning (it accepts any size, unlike FLUX)."""
    for acct, tok in _accounts():
        if acct in _DEAD:
            continue
        url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/bytedance/stable-diffusion-xl-lightning"
        for k in range(tries):
            try:
                body = json.dumps({"prompt": prompt, "width": w, "height": h, "num_steps": 8, "guidance": 2}).encode()
                req = urllib.request.Request(url, body, {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
                raw = urllib.request.urlopen(req, timeout=150).read()
                try:
                    raw = base64.b64decode(json.loads(raw)["result"]["image"])
                except Exception:
                    pass
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except urllib.error.HTTPError as e:
                msg = e.read()[:300].decode(errors="replace")
                if e.code == 429 and "daily free allocation" in msg:
                    _DEAD.add(acct)
                    break
                time.sleep(3)
            except Exception:
                time.sleep(3)
    return None


def thumbnail(text, scene_prompt, out_path, variant=0, accent=(255, 240, 30), arrow=False, background=""):
    """Channel-style thumbnail: a detailed wide scene behind, one funny stick-figure subject standing in it, and the hook across the top in big outlined text."""
    import cv2
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    W, H = 1280, 720
    bgp = (f"flat colour hand-drawn cartoon illustration, thin black ink outlines, warm bright colours, detailed background scene only, {background}, "
           "absolutely no people, no characters, no text, the upper fifth of the picture is calm open sky or ceiling, wide cinematic composition")
    bg = generate_wide(bgp, W, H)
    p = (f"{STYLE}. ONE single character shown full body, large and centred, an extremely exaggerated funny expression (eyes huge, mouth wide open, sweat drops), "
         f"{scene_prompt}. Drawn on a completely plain pure white background with nothing else in the picture, no ground, no shadow, no other people. Bold thick outlines, bright colours.")
    img = None
    for k in range(3):
        img = generate(p)
        if img is not None:
            break
    if img is None or bg is None:
        return None
    bg = ImageEnhance.Color(bg).enhance(1.25)
    bg = ImageEnhance.Contrast(bg).enhance(1.08)
    canvas = bg.convert("RGBA")
    cut = _cutout(img.crop((0, 0, img.width, int(img.height * 0.93))))
    bb = cut.getbbox()
    cut = cut.crop(bb)
    k = min(H * 0.80 / cut.height, W * 0.46 / cut.width)
    cut = cut.resize((int(cut.width * k), int(cut.height * k)), Image.LANCZOS)
    al = np.asarray(cut.split()[3])
    edge = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    px, py = int(W * (0.60 if variant % 2 == 0 else 0.30) - cut.width / 2), int(H - cut.height - H * 0.02)
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ell = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(ell).ellipse([px - 10, py + cut.height - 30, px + cut.width + 10, py + cut.height + 24], fill=130)
    sh.paste(Image.new("RGBA", canvas.size, (0, 0, 0, 255)), (0, 0), ell.filter(ImageFilter.GaussianBlur(14)))
    canvas.alpha_composite(sh)
    ol = Image.new("RGBA", cut.size, (18, 12, 20, 255)); ol.putalpha(Image.fromarray(edge))
    canvas.alpha_composite(ol, (px, py))
    canvas.alpha_composite(cut, (px, py))
    d = ImageDraw.Draw(canvas)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    words = [w for w in text.upper().split() if w][:4]
    size = 260
    while size > 90:
        f = ImageFont.truetype(str(FONT), size)
        if d.textlength(" ".join(words), font=f) <= W * 0.94:
            break
        size -= 6
    tw = d.textlength(" ".join(words), font=f)
    x, y = (W - tw) / 2, 14
    # first part white, the last word yellow, like the reference channels
    head, last = (" ".join(words[:-1]) + " ") if len(words) > 1 else "", words[-1]
    d.text((x + 8, y + 10), head + last, font=f, fill=(0, 0, 0), stroke_width=16, stroke_fill=(0, 0, 0))
    d.text((x, y), head, font=f, fill=(255, 255, 255), stroke_width=14, stroke_fill=(18, 12, 20))
    d.text((x + d.textlength(head, font=f), y), last, font=f, fill=accent, stroke_width=14, stroke_fill=(18, 12, 20))
    if arrow:
        pts = [(int(W * 0.30), int(H * 0.62)), (int(W * 0.36), int(H * 0.50)), (int(W * 0.44), int(H * 0.44))]
        d.line(pts, fill=(18, 12, 20), width=20, joint="curve"); d.line(pts, fill=(255, 255, 255), width=11, joint="curve")
    canvas.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path
