"""Image generation with a fallback chain of free providers.

Order (first that works wins): Pollinations (no key) -> Cloudflare Workers AI
(CF_ACCOUNT_ID + CF_API_TOKEN, free daily allowance) -> Hugging Face (HF_TOKEN).
Override the order with IMAGE_PROVIDERS="cloudflare,pollinations".
"""
import base64
import hashlib
import io
import os
import pathlib
import random
import time
import urllib.parse

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter

W, H = 768, 1344  # native generation size, 9:16


def _open(data):
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert("RGB")


def pollinations(prompt, negative, seed):
    q = urllib.parse.quote(prompt)
    params = {"width": W, "height": H, "seed": seed, "model": os.environ.get("POLLINATIONS_MODEL", "flux"),
              "nologo": "true", "enhance": "false", "negative": negative}
    headers = {}
    if os.environ.get("POLLINATIONS_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['POLLINATIONS_TOKEN']}"
    r = requests.get(f"https://image.pollinations.ai/prompt/{q}", params=params, headers=headers, timeout=180)
    r.raise_for_status()
    return _open(r.content)


def cloudflare(prompt, negative, seed):
    acc, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell",
        headers={"Authorization": f"Bearer {tok}"},
        json={"prompt": prompt[:2000], "steps": 6, "seed": seed}, timeout=120)
    r.raise_for_status()
    return _open(base64.b64decode(r.json()["result"]["image"]))


def huggingface(prompt, negative, seed):
    r = requests.post(
        "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell",
        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
        json={"inputs": prompt, "parameters": {"width": 720, "height": 1280, "seed": seed}}, timeout=180)
    r.raise_for_status()
    return _open(r.content)


def placeholder(prompt, negative, seed):
    """Procedural moody scene. Only used for offline engine tests, never for real videos."""
    rnd = random.Random(seed)
    yy = np.linspace(0, 1, H)[:, None]
    base = np.array([rnd.uniform(8, 30), rnd.uniform(14, 40), rnd.uniform(22, 55)])
    top = base * 1.8 + np.array([0, 10, 20])
    img = (top[None, None, :] * (1 - yy[..., None]) + base[None, None, :] * yy[..., None])
    img = np.repeat(img, W, axis=1).astype(np.float32)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    mx, my = rnd.randint(150, W - 150), rnd.randint(200, 500)
    for r_, a in ((190, 18), (120, 30), (70, 220)):
        d.ellipse((mx - r_, my - r_, mx + r_, my + r_), fill=(230, 220, 190, a))
    for layer in range(4):
        col = (4 + layer * 5, 6 + layer * 6, 10 + layer * 8, 255)
        y0 = 700 + layer * 140
        pts = [(0, H), (0, y0)]
        x = 0
        while x < W:
            x += rnd.randint(40, 110)
            pts.append((x, y0 - rnd.randint(0, 260 - layer * 40)))
        pts += [(W, H)]
        d.polygon(pts, fill=col)
    fog = im.filter(ImageFilter.GaussianBlur(40))
    return Image.blend(im, fog, 0.35)


PROVIDERS = {"pollinations": pollinations, "cloudflare": cloudflare, "huggingface": huggingface,
             "placeholder": placeholder}


def _available(name):
    return {"cloudflare": bool(os.environ.get("CF_ACCOUNT_ID") and os.environ.get("CF_API_TOKEN")),
            "huggingface": bool(os.environ.get("HF_TOKEN"))}.get(name, True)


def generate(prompt, negative, seed, cache_dir, order=None, log=print):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    order = order or os.environ.get("IMAGE_PROVIDERS", "pollinations,cloudflare,huggingface").split(",")
    key = hashlib.sha1(f"{prompt}|{seed}|{order}".encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.png"
    if path.exists():
        return path
    errors = []
    for name in order:
        name = name.strip()
        if name not in PROVIDERS or not _available(name):
            continue
        for attempt in range(3):
            try:
                im = PROVIDERS[name](prompt, negative, seed)
                if min(im.size) < 256:
                    raise RuntimeError("image too small")
                im.save(path)
                log(f"    image via {name}")
                return path
            except Exception as e:
                errors.append(f"{name}: {str(e)[:120]}")
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("All image providers failed: " + " | ".join(errors[-6:]))
