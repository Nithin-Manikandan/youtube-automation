"""Click-optimised stickman thumbnails (1280x720 JPEG).

Built from the published CTR rules: one giant expressive face, 2-3 huge words, extreme contrast, a dramatic backdrop,
a date badge and a ?/! mark, legible at phone size. Three layouts so you can pick.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pipeline.config import ROOT
from . import recipes, stick

W, H = 1280, 720
SS = stick.SS
MOODS = {
    "fire": ((255, 150, 36), (160, 16, 24)), "ice": ((90, 190, 255), (10, 30, 96)), "gold": ((255, 222, 110), (150, 64, 16)),
    "storm": ((150, 160, 190), (14, 18, 34)), "blood": ((240, 70, 58), (60, 6, 14)), "night": ((140, 110, 240), (14, 10, 50)),
}
FONT_BIG = str(ROOT / "assets/fonts/BigShoulders-Bold.ttf")


def _bg(mood, cx_frac):
    a, b = MOODS.get(mood, MOODS["fire"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W * cx_frac
    r = np.sqrt(((xx - cx) / (W * 0.55)) ** 2 + ((yy - H * 0.40) / (H * 0.85)) ** 2)
    t = np.clip(r, 0, 1)[..., None] ** 0.9
    img = np.array(a, np.float32) * (1 - t) + np.array(b, np.float32) * t
    ang = np.arctan2(yy - H * 0.40, xx - cx)
    img *= (0.88 + 0.12 * (0.5 + 0.5 * np.sin(ang * 16)))[..., None]
    vig = 1 - 0.5 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41, 0, 1)[..., None] ** 2
    return np.clip(img * vig, 0, 255).astype(np.uint8)


def _scenery(img, objs, flip):
    """Dark blurred silhouettes of the story's setting behind the figure (castle, pyramid, ship...)."""
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    gy = H * SS * 0.96
    for i, o in enumerate(objs[:2] or ["castle"]):
        if o not in recipes.OBJECTS or o in ("cloud", "torch"):
            o = "castle"
        x = (0.52 if not flip else 0.48) + (0.12 if i else 0)
        stick.draw_object(d, dict(type=o, x=x, scale=2.2), W, H, gy, 0)
    sil = layer.resize((W, H), Image.LANCZOS)
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dark.putalpha(sil.split()[3].point(lambda v: int(v * 0.62)))
    dark = dark.filter(ImageFilter.GaussianBlur(3))
    img.alpha_composite(dark)


def _fit_text(d, lines, max_w, max_h):
    size = 300
    while size > 80:
        f = ImageFont.truetype(FONT_BIG, size)
        if max(d.textlength(l, font=f) for l in lines) <= max_w and size * 0.92 * len(lines) <= max_h:
            break
        size -= 6
    return f, size


def render(text, recipe, out_path, variant=0):
    layout = variant % 3
    flip = layout == 1
    center = layout == 2
    mood = recipe.get("mood", "fire")
    base = Image.fromarray(_bg(mood, 0.5 if center else (0.72 if not flip else 0.28))).convert("RGBA")
    _scenery(base, recipe.get("objects") or [], flip)

    role = recipe.get("role", "citizen")
    role = role if role in recipes.ROLES else "citizen"
    props, tun = recipes.ROLES[role]
    col = recipes.COLORS.get(recipe.get("color") or tun, recipes.COLORS[tun])
    pose = recipe.get("action", "scared")
    pose = pose if pose in ("scared", "point", "sword_up", "proud", "shrug", "cheer", "crouch", "slump") else "scared"
    emo = recipe.get("emotion", "shock")
    emo = emo if emo in recipes.EMOTIONS else "shock"
    x = 0.5 if center else (0.74 if not flip else 0.26)
    scale = 1.9 if center else 3.0
    spec = dict(id="thumb", color=col, tunic=col, props=props, scale=scale,
                keys=[dict(t=0, x=x, pose=pose, face=emo, facing=(-1 if not flip else 1) if not center else 1)])
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    actor = stick.Actor(spec, W, H)
    gy = H * SS * (1.25 if not center else 1.08)
    actor.draw(d, 0.4, gy)
    layer = layer.resize((W, H), Image.LANCZOS)
    rim = layer.filter(ImageFilter.GaussianBlur(16))
    sil = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    sil.putalpha(rim.split()[3].point(lambda v: min(255, v * 3)))
    base.alpha_composite(sil)
    base.alpha_composite(layer)

    d2 = ImageDraw.Draw(base)
    words = [w for w in text.upper().split() if w][:4]
    if len(words) <= 2:
        lines = [" ".join(words)]
    else:
        cut = 1 if len(words) == 3 else 2
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    if center:
        f, size = _fit_text(d2, lines, W * 0.92, H * 0.34)
        y = H * 0.03
        for i, l in enumerate(lines):
            w = d2.textlength(l, font=f)
            _draw_text(d2, ((W - w) / 2, y), l, f, (255, 214, 64) if i == len(lines) - 1 else (255, 255, 255))
            y += size * 0.92
    else:
        f, size = _fit_text(d2, lines, W * 0.54, H * 0.80)
        y = (H - size * 0.92 * len(lines)) / 2 - size * 0.06
        xs = 44 if not flip else W - 44
        for i, l in enumerate(lines):
            w = d2.textlength(l, font=f)
            _draw_text(d2, (xs if not flip else xs - w, y), l, f, (255, 214, 64) if i == len(lines) - 1 else (255, 255, 255))
            y += size * 0.92

    badge = str(recipe.get("badge") or "").upper()[:10]
    if badge:
        bf = ImageFont.truetype(FONT_BIG, 74)
        bw = d2.textlength(badge, font=bf) + 50
        bx, by = (30 if not flip else W - bw - 30), 28
        d2.rounded_rectangle([bx + 6, by + 8, bx + bw + 6, by + 100], 18, fill=(0, 0, 0, 150))
        d2.rounded_rectangle([bx, by, bx + bw, by + 92], 18, fill=(255, 214, 64), outline=(10, 10, 14), width=5)
        d2.text((bx + 25, by + 2), badge, font=bf, fill=(20, 16, 10))
    mark = recipe.get("mark", "!" if emo in ("shock", "angry") else "?")
    if mark in ("!", "?", "!?"):
        mf = ImageFont.truetype(FONT_BIG, 200)
        mx = int(W * (x + (0.17 if not center else 0.26) * (1 if not flip else -1))) if not center else int(W * 0.78)
        my = int(H * 0.06) if (center or flip) else int(H * 0.06)
        _draw_text(d2, (mx, my), mark, mf, (255, 70, 60), sw=10)
    base.convert("RGB").save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


def _draw_text(d, xy, text, font, fill, sw=12):
    x, y = xy
    d.text((x + 9, y + 10), text, font=font, fill=(0, 0, 0, 170), stroke_width=sw + 2, stroke_fill=(0, 0, 0, 170))
    d.text((x, y), text, font=font, fill=fill, stroke_width=sw, stroke_fill=(10, 10, 14))
