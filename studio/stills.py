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
    acc = [(os.environ.get("CF_ACCOUNT_ID" + sfx), os.environ.get("CF_API_TOKEN" + sfx)) for sfx in ("", "_2", "_3")]
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
                print("wide image error", e.code, msg[:140], flush=True)
                if e.code == 429 and "daily free allocation" in msg:
                    _DEAD.add(acct)
                    break
                time.sleep(3)
            except Exception:
                time.sleep(3)
    return None


def thumbnail(text, scene_prompt, out_path, variant=0, accent=(255, 240, 30), arrow=False, background=""):
    """Channel-style thumbnail: ONE huge funny face/character on one side, a calm softly blurred background, a short hook in big outlined text on the other side."""
    import cv2
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    W, H = 1280, 720
    bgp = (f"flat colour hand-drawn cartoon illustration, thin black ink outlines, warm bright colours, simple scene only, {background}, "
           "absolutely no people, no characters, no text, very simple and uncluttered")
    bg = generate_wide(bgp, W, H)
    p = (f"{STYLE}. Big close-up, shown from the chest up, ONE single character with a gigantic head, an extremely exaggerated funny expression (huge round eyes, wide open mouth, sweat drops), "
         f"{scene_prompt}. Drawn on a completely plain pure white background with nothing else in the picture, no ground, no shadow, no other people. Bold thick outlines, bright colours.")
    cands = []
    for k in range(4):
        im_ = generate(p)
        if im_ is None:
            continue
        c_ = _cutout(im_.crop((0, 0, im_.width, int(im_.height * 0.93))))
        b_ = c_.getbbox()
        if b_:
            cands.append(((b_[2] - b_[0]) / max(1, b_[3] - b_[1]), c_.crop(b_)))
        if len(cands) >= 3:
            break
    if not cands or bg is None:
        return None
    cut = min(cands, key=lambda c: c[0])[1]
    # calm background: strong blur, lifted and slightly desaturated so the subject and text pop
    bg = bg.filter(ImageFilter.GaussianBlur(14))
    bg = ImageEnhance.Color(bg).enhance(0.9)
    bg = ImageEnhance.Brightness(bg).enhance(1.12)
    canvas = bg.convert("RGBA")
    k = max(H * 1.02 / cut.height, 0.1)
    k = min(k, W * 0.56 / cut.width)
    cut = cut.resize((int(cut.width * k), int(cut.height * k)), Image.LANCZOS)
    al = np.asarray(cut.split()[3])
    edge = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    px, py = int(W * 0.30 - cut.width / 2), int(H - cut.height + H * 0.04)
    ol = Image.new("RGBA", cut.size, (18, 12, 20, 255)); ol.putalpha(Image.fromarray(edge))
    canvas.alpha_composite(ol, (px, py))
    canvas.alpha_composite(cut, (px, py))
    d = ImageDraw.Draw(canvas)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    lines = [w for w in text.upper().replace(" IN ", " IN\n").split("\n")][:3] if "\n" in text.upper().replace(" IN ", " IN\n") else [text.upper()]
    if len(lines) == 1:
        ws = lines[0].split()
        lines = [" ".join(ws[:len(ws) // 2 or 1]), " ".join(ws[len(ws) // 2 or 1:])] if len(ws) > 1 else ws
    size = 300
    while size > 80:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.50 and size * .94 * len(lines) <= H * 0.70:
            break
        size -= 6
    y = H * 0.10
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x = W - 30 - tw
        d.text((x + 9, y + 11), l, font=f, fill=(0, 0, 0), stroke_width=18, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=accent if i == len(lines) - 1 else (255, 255, 255), stroke_width=15, stroke_fill=(18, 12, 20))
        y += size * .94
    canvas.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path


def make_layers(bg_prompt, char_prompt, out_dir, name):
    """Two separate pictures for a layered scene: a wide background (no people) and one character cut out on transparency. Saves PNGs and returns their paths."""
    import cv2
    bg = generate_wide(f"flat colour hand-drawn cartoon illustration, thin black ink outlines, warm bright colours, detailed background scene only, {bg_prompt}, absolutely no people, no characters, no text", 1280, 720)
    p = (f"{STYLE}. ONE single character shown full body, large and centred, {char_prompt}. Drawn on a completely plain pure white background with nothing else in the picture, no ground, no shadow, no other people. Bold thick outlines, bright colours.")
    cands = []
    for k in range(4):
        im_ = generate(p)
        if im_ is None:
            continue
        c_ = _cutout(im_.crop((0, 0, im_.width, int(im_.height * 0.93))))
        b_ = c_.getbbox()
        if b_:
            cands.append(((b_[2] - b_[0]) / max(1, b_[3] - b_[1]), c_.crop(b_)))
        if len(cands) >= 3:
            break
    if bg is None or not cands:
        return None
    cut = min(cands, key=lambda c: c[0])[1]
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bg.save(out_dir / f"{name}_bg.jpg", quality=93)
    cut.save(out_dir / f"{name}_char.png")
    return out_dir / f"{name}_bg.jpg", out_dir / f"{name}_char.png"


def bg_prompts(llm, topic, narrations):
    """One setting description per line for the painted backgrounds: where the scene happens, no people."""
    out = [None] * len(narrations)
    for b0 in range(0, len(narrations), 16):
        chunk = narrations[b0:b0 + 16]
        q = (f"You are the background artist of a funny cartoon history channel. VIDEO TOPIC: {topic}.\n"
             "For each numbered voiceover line write ONE plain sentence describing only the PLACE where it happens (era-correct building, room, field or street, key props, time of day, light). "
             "No people, no text, no signs. Match the line literally: if it talks about the sea, a ship, a beach, a plank or a flag, show the ship deck, open sea or beach (outdoors, bright sky), not a room. Vary the places across lines; only repeat a place when the story stays there.\n"
             + "\n".join(f"{b0 + i}: {t}" for i, t in enumerate(chunk)) + '\nReturn JSON: {"scenes": [{"i": 0, "place": "..."}]}')
        try:
            d = llm(q, 0.4)
            for s_ in d.get("scenes", []):
                j = int(s_.get("i", -1))
                if 0 <= j < len(out) and s_.get("place"):
                    out[j] = str(s_["place"])
        except Exception:
            pass
    return [o or "a plain medieval stone room with warm light" for o in out]


def draw_backgrounds(pdir, jobs, workers=3):
    """jobs: [(scene_index, place)] -> {scene_index: filename} saved in pdir/stills as bgNNN.jpg. Identical places share one picture."""
    import concurrent.futures as cf
    d = pathlib.Path(pdir) / "stills"
    d.mkdir(exist_ok=True)
    uniq = {}
    for i, pl in jobs:
        uniq.setdefault(pl.strip().lower(), []).append(i)

    def one(item):
        place, idxs = item
        img = generate_wide(f"flat colour hand-drawn cartoon illustration, thin black ink outlines, warm bright colours, detailed background scene only, {place}, "
                            "absolutely no people, no characters, no text, wide cinematic composition", 1280, 720)
        if img is None:
            return idxs, None
        fn = f"bg{idxs[0]:03d}.jpg"
        img.save(d / fn, quality=93)
        return idxs, fn
    res = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for idxs, fn in ex.map(one, list(uniq.items())):
            if fn:
                for i in idxs:
                    res[i] = fn
    return res


def host_thumbnail(text, background, out_path, mood="shock", mouth="D", pose="shock", hat=None, accent=(255, 240, 30), side="left"):
    """Thumbnail starring the channel host, drawn by the puppet rig at poster size over a calm, softly blurred painted background, with the hook in huge outlined text."""
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    from . import puppet
    W, H = 1280, 720
    bg = generate_wide(f"flat colour hand-drawn cartoon illustration, thin black ink outlines, warm bright colours, simple scene only, {background}, "
                       "absolutely no people, no characters, no text, very simple and uncluttered", W, H)
    if bg is None:
        return None
    bg = bg.resize((int(W * 1.12), int(H * 1.12)), Image.LANCZOS).filter(ImageFilter.GaussianBlur(4)).crop((int(W * .06), int(H * .06), int(W * .06) + W, int(H * .06) + H))     # blur then trim, so the soft edge never shows
    bg = ImageEnhance.Color(bg).enhance(1.15)
    bg = ImageEnhance.Brightness(bg).enhance(1.08)
    canvas = bg.convert("RGBA")
    look = puppet.Look(**{**puppet.HOST, "hat": hat})
    p = {**puppet.POSES["idle"], **puppet.POSES.get(pose, {}), "facing": 1, "mouth": mouth, "mood": mood, "t": 0.0}
    big = 2300
    tile = puppet.draw_character(p, look, big, 1900, 2300, 950, 2200)
    bb = tile.getbbox()
    face_top = bb[1]
    crop = tile.crop((120, bb[1], 1600, bb[1] + int((bb[3] - bb[1]) * 0.62)))               # head, shoulders and chest around the head's centre: the face is the hook
    k = min(H * 0.80 / crop.height, (W * 0.50) / crop.width)
    crop = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
    import cv2
    al = np.asarray(crop.split()[3])
    edge = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    px = int(W * (0.30 if side == "left" else 0.70) - crop.width / 2)
    py = int(H - crop.height + H * 0.02)
    ol = Image.new("RGBA", crop.size, (18, 12, 20, 255)); ol.putalpha(Image.fromarray(edge))
    canvas.alpha_composite(ol, (px, py)); canvas.alpha_composite(crop, (px, py))
    d = ImageDraw.Draw(canvas)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    lines = text.upper().split("|")
    size = 330
    while size > 90:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.43 and size * .93 * len(lines) <= H * 0.86:
            break
        size -= 6
    y = (H - size * .93 * len(lines)) / 2
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x = (W - 28 - tw) if side == "left" else 28
        d.text((x + 10, y + 12), l, font=f, fill=(0, 0, 0), stroke_width=20, stroke_fill=(0, 0, 0))
        d.text((x, y), l, font=f, fill=accent if i == len(lines) - 1 else (255, 255, 255), stroke_width=17, stroke_fill=(18, 12, 20))
        y += size * .93
    canvas.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path


WIDE_STYLE = ("flat colour hand-drawn cartoon illustration in a simple webcomic style, stick figure people with round white heads and tiny dot eyes, thin black ink outlines, "
              "warm bright colours, detailed painted background, wide cinematic composition with the main subject in the centre and calmer edges, absolutely no text, no letters, no signs")


def draw_scene_pictures(pdir, jobs, workers=3):
    """jobs: [(scene_index, what_to_draw)] -> {scene_index: filename}. One FLUX picture per scene (it follows the description far better than the wide model), saved as picNNN.jpg."""
    import concurrent.futures as cf
    d = pathlib.Path(pdir) / "stills"
    d.mkdir(exist_ok=True)

    def one(job):
        i, what = job
        img = generate(f"{STYLE}. {what}")
        if img is None:
            return i, None
        w_, h_ = img.size
        img = img.crop((0, 0, w_, int(h_ * 0.93))).resize((w_, w_), Image.LANCZOS)
        fn = f"pic{i:03d}.jpg"
        img.save(d / fn, quality=93)
        return i, fn
    res = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fn in ex.map(one, jobs):
            if fn:
                res[i] = fn
    return res


def scene_thumbnail(text, scene, out_path, mood="shock", mouth="D", pose="shock", side="left", accent=(255, 240, 30)):
    """High-CTR layout: a sharp, saturated scene that shows the actual mystery fills the frame; the host's big shocked face sits in a lower corner; the hook runs huge across the top."""
    import cv2
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    from . import puppet
    W, H = 1280, 720
    bg = generate_wide(f"{WIDE_STYLE}. {scene}. Bright saturated colours, strong contrast, clear and readable even when tiny, the upper fifth is plain calm sky.", W, H)
    if bg is None:
        return None
    bg = ImageEnhance.Color(bg).enhance(1.35)
    bg = ImageEnhance.Contrast(bg).enhance(1.15)
    bg = ImageEnhance.Brightness(bg).enhance(1.05)
    canvas = bg.convert("RGBA")
    look = puppet.Look(**puppet.HOST)
    p = {**puppet.POSES["idle"], **puppet.POSES.get(pose, {}), "facing": 1 if side == "left" else -1, "mouth": mouth, "mood": mood, "t": 0.0}
    tile = puppet.draw_character(p, look, 2300, 1900, 2300, 950, 2200)
    bb = tile.getbbox()
    crop = tile.crop((150, bb[1], 1600, bb[1] + int((bb[3] - bb[1]) * 0.58)))
    k = min(H * 0.62 / crop.height, W * 0.34 / crop.width * 1.0)
    crop = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
    al = np.asarray(crop.split()[3])
    edge = cv2.dilate(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    px = int(W * (0.17 if side == "left" else 0.83) - crop.width / 2)
    py = H - crop.height + int(H * 0.03)
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ell = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(ell).ellipse([px, py + crop.height - 40, px + crop.width, py + crop.height + 30], fill=120)
    sh.paste(Image.new("RGBA", canvas.size, (0, 0, 0, 255)), (0, 0), ell.filter(ImageFilter.GaussianBlur(12)))
    canvas.alpha_composite(sh)
    ol = Image.new("RGBA", crop.size, (18, 12, 20, 255)); ol.putalpha(Image.fromarray(edge))
    canvas.alpha_composite(ol, (px, py)); canvas.alpha_composite(crop, (px, py))
    # hook across the top on a dark soft band so it is always readable
    band = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(band).rectangle([0, 0, W, int(H * 0.33)], fill=(0, 0, 0, 120))
    band = band.filter(ImageFilter.GaussianBlur(26))
    canvas.alpha_composite(band)
    d = ImageDraw.Draw(canvas)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    words = [w for w in text.upper().split() if w]
    size = 300
    while size > 90:
        f = ImageFont.truetype(str(FONT), size)
        if d.textlength(" ".join(words), font=f) <= W * 0.93:
            break
        size -= 6
    head, last = (" ".join(words[:-1]) + " ") if len(words) > 1 else "", words[-1]
    tw = d.textlength(head + last, font=f)
    x, y = (W - tw) / 2, 8
    d.text((x + 8, y + 10), head + last, font=f, fill=(0, 0, 0), stroke_width=18, stroke_fill=(0, 0, 0))
    d.text((x, y), head, font=f, fill=(255, 255, 255), stroke_width=15, stroke_fill=(18, 12, 20))
    d.text((x + d.textlength(head, font=f), y), last, font=f, fill=accent, stroke_width=15, stroke_fill=(18, 12, 20))
    canvas.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path


def epic_thumbnail(text, scene, out_path, accent=(255, 224, 40), arrow_to=None):
    """Entertainment-style thumbnail: one epic, clear, mysterious image graded warm and punchy, one huge word, an arrow at the mystery. No faces."""
    import cv2
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    img = None
    for k in range(3):
        img = generate(f"{STYLE}. Epic dramatic composition, strong lighting, big simple shapes, very clear even when tiny. {scene}")
        if img is not None:
            break
    if img is None:
        return None
    W, H = 1280, 720
    a = img.crop((0, 0, img.width, int(img.height * 0.93)))
    S = a.width
    hh = int(S * 9 / 16)
    y0 = int((a.height - hh) * 0.45)
    pic = a.crop((0, y0, S, y0 + hh)).resize((W, H), Image.LANCZOS)
    pic = ImageEnhance.Color(pic).enhance(1.45)
    pic = ImageEnhance.Contrast(pic).enhance(1.22)
    arr = np.asarray(pic).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    arr *= (1 - 0.30 * np.clip(((xx - W / 2) / (W * .70)) ** 2 + ((yy - H / 2) / (H * .75)) ** 2, 0, 1))[..., None]
    left = np.clip(1 - xx / (W * 0.55), 0, 1) ** 1.3
    arr *= (1 - 0.35 * left)[..., None]                                      # darken the text side for contrast
    pic = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(pic)
    FONT = pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf"
    words = text.upper().split("|")
    size = 460
    while size > 120:
        f = ImageFont.truetype(str(FONT), size)
        if max(d.textlength(w, font=f) for w in words) <= W * 0.52 and size * .90 * len(words) <= H * 0.86:
            break
        size -= 8
    y = (H - size * .90 * len(words)) / 2 - size * .02
    for i, w in enumerate(words):
        d.text((34 + 12, y + 14), w, font=f, fill=(0, 0, 0), stroke_width=24, stroke_fill=(0, 0, 0))
        d.text((34, y), w, font=f, fill=accent if i == len(words) - 1 else (255, 255, 255), stroke_width=20, stroke_fill=(18, 12, 20))
        y += size * .90
    if arrow_to:
        ax, ay = int(arrow_to[0] * W), int(arrow_to[1] * H)
        sx, sy = int(W * 0.60), int(H * 0.88)
        d.line([sx, sy, ax + 40, ay + 40], fill=(18, 12, 20), width=34)
        d.line([sx, sy, ax + 40, ay + 40], fill=(235, 40, 40), width=20)
        d.ellipse([ax - 55, ay - 55, ax + 55, ay + 55], outline=(18, 12, 20), width=22)
        d.ellipse([ax - 55, ay - 55, ax + 55, ay + 55], outline=(235, 40, 40), width=12)
    pic.convert("RGB").save(out_path, "JPEG", quality=95)
    return out_path
