"""Click-optimised stickman thumbnails (1280x720 JPEG), built like the ones that do well:

- a dramatic illustrated SCENE (glowing horizon, armies, burning skyline, fog, embers), not a flat backdrop
- scale contrast: a small lit hero against something huge (a looming silhouette, an army, a ruin)
- 2-3 giant words with a coloured key word, placed in a free zone and checked so they never cover the character
- one focal point, max one extra cue (date badge), colour-graded with bloom and vignette
Concepts: "looming" (hero vs giant), "ruin" (burning skyline), "versus" (two sides clash).
"""
import math
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from pipeline.config import ROOT
from . import recipes, stick

W, H = 1280, 720
SS = stick.SS
FONT_BIG = str(ROOT / "assets/fonts/BigShoulders-Bold.ttf")
MOODS = {  # top, mid, glow, highlight word colour
    "fire": ((34, 6, 8), (150, 30, 18), (255, 150, 40), (255, 214, 64)),
    "ice": ((5, 14, 40), (28, 88, 160), (150, 222, 255), (120, 236, 255)),
    "gold": ((36, 18, 6), (150, 80, 20), (255, 222, 110), (255, 224, 90)),
    "storm": ((8, 12, 24), (58, 72, 104), (176, 196, 236), (255, 214, 64)),
    "blood": ((28, 0, 6), (130, 10, 22), (255, 96, 72), (255, 226, 120)),
    "night": ((6, 5, 28), (60, 40, 132), (170, 142, 255), (255, 214, 64)),
}


def _bg(mood, gx, gy):
    top, mid, glow, _ = MOODS.get(mood, MOODS["fire"])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.clip(yy / (H * 0.78), 0, 1)[..., None]
    img = np.array(top, np.float32) * (1 - v) + np.array(mid, np.float32) * v
    cx, cy = gx * W, gy * H
    r = np.sqrt(((xx - cx) / (W * 0.45)) ** 2 + ((yy - cy) / (H * 0.55)) ** 2)
    g = np.exp(-(r ** 1.6) * 2.4)[..., None]
    img = img * (1 - g * 0.85) + np.array(glow, np.float32) * g * 0.95
    ang = np.arctan2(yy - cy, xx - cx)
    rays = (0.5 + 0.5 * np.sin(ang * 11 + 0.6)) ** 2
    img += (np.array(glow, np.float32) * 0.10 * rays[..., None] * np.exp(-r * 0.9)[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")


def _silhouette(layer, color, alpha_mul=1.0):
    a = layer.split()[3].point(lambda v: int(v * alpha_mul))
    out = Image.new("RGBA", layer.size, color + (0,))
    out.putalpha(a)
    return out


def _army(d, x0, x1, base_y, h, n, col, rng):
    for i in range(n):
        x = x0 + (x1 - x0) * (i + rng.random() * 0.8) / n
        hh = h * (0.85 + 0.3 * rng.random())
        lw = max(2, hh * 0.085)
        hr = hh * 0.12
        top = base_y - hh
        d.ellipse([x - hr, top, x + hr, top + 2 * hr], fill=col)
        d.line([(x, top + 2 * hr), (x, base_y - hh * 0.42)], fill=col, width=int(lw))
        for s in (-1, 1):
            d.line([(x, base_y - hh * 0.42), (x + s * hh * 0.14, base_y)], fill=col, width=int(lw))
        d.line([(x, top + hh * 0.3), (x + hh * 0.22, top + hh * 0.45)], fill=col, width=int(lw))
        if rng.random() < 0.7:
            d.line([(x + hh * 0.2, top + hh * 0.5), (x + hh * 0.2, top - hh * 0.55)], fill=col, width=max(2, int(lw * 0.6)))


def _flames(base, xs, y, size, rng, glow):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x in xs:
        for _ in range(7):
            fx = x + rng.uniform(-size * .6, size * .6)
            fh = size * rng.uniform(.5, 1.5)
            fw = size * rng.uniform(.25, .5)
            d.polygon([(fx - fw, y), (fx, y - fh), (fx + fw, y)], fill=(255, int(rng.uniform(90, 170)), 30, 235))
            d.polygon([(fx - fw * .5, y), (fx, y - fh * .6), (fx + fw * .5, y)], fill=(255, 226, 120, 240))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(9)).point(lambda v: min(255, int(v * 1.6))))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.5)))
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    for x in xs:
        hd.ellipse([x - size * 2.4, y - size * 2.2, x + size * 2.4, y + size * .8], fill=glow + (70,))
    base.alpha_composite(halo.filter(ImageFilter.GaussianBlur(40)))


def _embers(base, rng, n, col):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H * 0.85)
        r = rng.uniform(1.5, 4.5)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (int(rng.uniform(120, 255)),))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.2)))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(5)))


def _fog(base, y, strength=70):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i in range(3):
        d.ellipse([-200 + i * 380, y - 70 + i * 18, 560 + i * 380, y + 90 + i * 18], fill=(235, 230, 240, strength))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(46)))


def _skyline(base, objs, flip, y, scale=1.6):
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    xs = []
    for i, o in enumerate(objs[:3] or ["castle", "tower"]):
        o = o if o in recipes.OBJECTS and o not in ("cloud", "torch") else "castle"
        x = (0.14 + 0.2 * i) if not flip else (0.86 - 0.2 * i)
        stick.draw_object(d, dict(type=o, x=x, scale=scale), W, H, y * SS, 0)
        xs.append(x * W)
    sil = _silhouette(layer.resize((W, H), Image.LANCZOS), (10, 7, 12), 0.96)
    base.alpha_composite(sil)
    return xs


def _grade(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    arr = np.clip((arr - 0.5) * 1.22 + 0.5, 0, 1)                      # contrast
    gray = arr.mean(axis=2, keepdims=True)
    arr = np.clip(gray + (arr - gray) * 1.22, 0, 1)                    # saturation
    bright = np.clip(arr - 0.72, 0, 1) * 3.2                            # bloom
    bl = np.asarray(Image.fromarray((bright * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(22))).astype(np.float32) / 255
    arr = np.clip(arr + bl * 0.38, 0, 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.42 * np.clip(np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41, 0, 1) ** 2.2
    arr = arr * vig[..., None]
    out = Image.fromarray((arr * 255).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=2))


def _hero(role, color, pose, emo, x, scale, facing, gy_frac=1.0, boss=False):
    role = role if role in recipes.ROLES else "citizen"
    props, tun = recipes.ROLES[role]
    col = recipes.COLORS.get(color or tun, recipes.COLORS[tun])
    pose = pose if pose in ("scared", "point", "sword_up", "proud", "shrug", "cheer", "crouch", "slump", "swing") else "scared"
    emo = emo if emo in recipes.EMOTIONS else "shock"
    spec = dict(id="thumb", color=col, tunic=None if boss else col, props=props, scale=scale, keys=[dict(t=0, x=x, pose=pose, face=emo, facing=facing)])
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    stick.Actor(spec, W, H).draw(d, 0.4, H * SS * gy_frac)
    return layer.resize((W, H), Image.LANCZOS)


def _text_layer(lines, font, hi, flip, size, x_left, y_top, tilt=-2.2):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = y_top
    maxw = max(d.textlength(l, font=font) for l in lines)
    for i, l in enumerate(lines):
        w = d.textlength(l, font=font)
        x = x_left if not flip else x_left + (maxw - w)
        last = i == len(lines) - 1
        fill = hi if last else (255, 255, 255)
        d.text((x + 10, y + 12), l, font=font, fill=(0, 0, 0, 175), stroke_width=16, stroke_fill=(0, 0, 0, 175))
        d.text((x, y), l, font=font, fill=fill, stroke_width=13, stroke_fill=(12, 8, 10))
        y += size * 0.93
    glow = layer.filter(ImageFilter.GaussianBlur(10))
    g = Image.new("RGBA", layer.size, hi + (0,))
    g.putalpha(glow.split()[3].point(lambda v: int(v * 0.35)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(g)
    out.alpha_composite(layer)
    return out.rotate(tilt, resample=Image.BICUBIC, center=(x_left + maxw / 2, y_top + size))


SUBJECT_BG = {"submarine": "underwater", "torpedo": "underwater", "depth_charge": "underwater", "warship": "sea", "ship": "sea", "wave": "sea", "plane": "sea",
              "volcano": "volcanic", "ash_cloud": "ashen", "explosion": "night", "fire": "night", "missile": "sea", "pyramid": "desert", "castle": "battlefield",
              "building": "city_modern", "house": "countryside", "tower": "battlefield", "column": "palace"}


def _subject_scene(recipe, flip):
    """The story's actual subject drawn big and lit by the real scene engine, graded and darkened on the text side."""
    obj = str(recipe.get("backdrop") or (recipe.get("objects") or ["castle"])[0]).lower()
    obj = obj if obj in recipes.OBJECTS else "castle"
    bg = str(recipe.get("scene") or SUBJECT_BG.get(obj, "battlefield")).lower()
    bg = bg if bg in recipes.BACKGROUNDS else "battlefield"
    rnd = random.Random(5)
    extra = [o for o in (recipe.get("objects") or []) if o in recipes.OBJECTS and o != obj][:1]
    vis = recipes.clean_visual({"type": "stage", "background": bg, "actors": [], "objects": [obj] + extra,
                                "effects": ["bubbles"] if bg == "underwater" else ["embers"] if bg in ("volcanic", "night") else []}, rnd)
    vis["objects"] = [obj] + extra
    sc = recipes.build_stage(vis, 6.0, rnd)
    for o in sc["objects"]:
        if o["type"] == obj:
            o["x"] = 0.62 if not flip else 0.38
            o["scale"] = 1.2 if obj in ("submarine", "warship", "ship") else 1.15
            o["t0"] = 0.3
            o["dur"] = 6.0
        elif o["type"] != "seascape":
            o["x"] = 0.18 if not flip else 0.82
    sc["shots"] = []
    sc["hits"] = []
    sc["speed"] = False
    sc["blur"] = False
    sc["shake"] = []
    sc["text"] = []
    sc["zoom"] = [1.05, 1.05]
    sc["focus"] = (0.5, 0.58)
    sc.pop("focus_to", None)
    sc["fx"] = [f for f in sc.get("fx", []) if f["type"] != "sparks"]
    stick.prepare(sc, W, H)
    t_show = {"wave": 3.1, "explosion": 1.0, "torpedo": 1.6, "missile": 1.6}.get(obj, 1.9)
    frame = Image.fromarray(stick.render_frame(sc, t_show, W, H)[..., :3]).convert("RGBA")
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for x in range(W):                                    # darken the headline side so the text pops
        k = (1 - x / (W * 0.46)) if not flip else (1 - (W - x) / (W * 0.46))
        if k > 0:
            sd.line([(x, 0), (x, H)], fill=(0, 0, 0, int(165 * k ** 1.4)))
    frame.alpha_composite(shade)
    return frame


def render(text, recipe, out_path, variant=0):
    concept = recipe.get("concept") or ("looming", "ruin", "versus")[variant % 3]
    concept = concept if concept in ("looming", "ruin", "versus", "subject") else "looming"
    flip = bool(recipe.get("flip", variant % 3 == 1))
    mood = recipe.get("mood", "fire")
    if mood not in MOODS:
        mood = "fire"
    _, _, glow, hi = MOODS[mood]
    rng = np.random.default_rng(variant * 7 + 3)
    hx = 0.73 if not flip else 0.27
    if concept == "subject":
        hx = 0.88 if not flip else 0.12
    base = _bg(mood, hx, 0.62)
    horizon = 0.80

    if concept == "subject":
        base = _subject_scene(recipe, flip)
        _embers(base, rng, 22, (255, 220, 150))
    elif concept == "ruin":
        xs = _skyline(base, recipe.get("objects") or ["castle", "tower", "column"], flip, H * horizon, 1.9)
        _flames(base, [x + rng.uniform(-20, 20) for x in xs for _ in range(1)] + [W * (0.5 if not flip else 0.5)], H * 0.62, 34, rng, glow)
        _embers(base, rng, 70, (255, 190, 70))
    else:
        army = str(recipe.get("army", True)).lower() not in ("false", "0", "no")
        if army:
            layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer, "RGBA")
            _army(d, -20, W * SS * 0.62 if not flip else W * SS + 20, H * SS * horizon, H * SS * 0.13, 46, (12, 8, 14, 255), rng)
            if flip:
                _army(d, W * SS * 0.38, W * SS + 20, H * SS * horizon, H * SS * 0.13, 46, (12, 8, 14, 255), rng)
            base.alpha_composite(layer.resize((W, H), Image.LANCZOS))
        backdrop = str(recipe.get("backdrop", "")).lower()
        if concept == "looming" and backdrop in recipes.OBJECTS and backdrop not in ("cloud", "torch"):
            big = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            bd = ImageDraw.Draw(big, "RGBA")
            stick.draw_object(bd, dict(type=backdrop, x=hx - 0.13 if not flip else hx + 0.13, scale=1.75, y=0.3), W, H, H * SS * horizon, 0)
            base.alpha_composite(_silhouette(big.resize((W, H), Image.LANCZOS), (8, 5, 12), 0.94))
        elif concept == "looming":
            boss = _hero(recipe.get("enemy_role", "warrior"), "black", "sword_up", "angry", hx - (0.02 if not flip else -0.02), 4.6, -1 if not flip else 1, gy_frac=1.62, boss=True)
            sil = _silhouette(boss, (8, 5, 12), 0.93)
            bb = boss.split()[3].getbbox()
            base.alpha_composite(sil)
            if bb:
                ey = bb[1] + (bb[3] - bb[1]) * 0.085
                ex = (bb[0] + bb[2]) / 2
                eyes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                ed = ImageDraw.Draw(eyes)
                for dx in (-26, 26):
                    ed.ellipse([ex + dx - 13, ey - 8, ex + dx + 13, ey + 8], fill=(255, 60, 40, 255))
                base.alpha_composite(eyes.filter(ImageFilter.GaussianBlur(9)).point(lambda v: min(255, int(v * 2.2))))
                base.alpha_composite(eyes)
        _embers(base, rng, 30, (255, 190, 70))
    if concept != "subject":
        _fog(base, H * (horizon - 0.02))
        ground = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(ground)
        for i in range(int(H * (1 - horizon)) + 2):
            gd.line([(0, H * horizon + i), (W, H * horizon + i)], fill=(8, 6, 10, min(255, 120 + i * 3)))
        base.alpha_composite(ground)

    role = recipe.get("role", "emperor")
    facing = -1 if not flip else 1
    if concept == "versus":
        a = _hero(role, recipe.get("color"), "sword_up", recipe.get("emotion", "angry"), 0.64 if not flip else 0.36, 1.5, 1 if not flip else -1, 1.0)
        b = _hero(recipe.get("enemy_role", "warrior"), recipe.get("enemy_color", "red"), "swing", "angry", 0.86 if not flip else 0.14, 1.5, -1 if not flip else 1, 1.0)
        hero = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        hero.alpha_composite(b)
        hero.alpha_composite(a)
        spark = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(spark)
        cx, cy = W * (0.75 if not flip else 0.25), H * 0.5
        for k in range(16):
            ang = k * math.tau / 16 + rng.random() * .3
            r1, r2 = 30 + rng.random() * 30, 100 + rng.random() * 140
            sd.line([(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1), (cx + math.cos(ang) * r2, cy + math.sin(ang) * r2)], fill=(255, 226, 120, 255), width=6)
        sd.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(255, 250, 220, 255))
        base.alpha_composite(spark.filter(ImageFilter.GaussianBlur(10)).point(lambda v: min(255, int(v * 2))))
        base.alpha_composite(spark)
    else:
        if concept == "subject":        # a big close reaction shot beside the story's subject: face and shoulders only
            hero = _hero(role, recipe.get("color"), recipe.get("action", "scared"), recipe.get("emotion", "shock"), hx, 2.1, facing, 1.3)
        else:
            hero = _hero(role, recipe.get("color"), recipe.get("action", "scared"), recipe.get("emotion", "shock"), hx, 1.7 if concept == "looming" else 1.75, facing, 1.0)
    rim = hero.filter(ImageFilter.GaussianBlur(14))
    rim_c = Image.new("RGBA", (W, H), glow + (0,))
    rim_c.putalpha(rim.split()[3].point(lambda v: min(255, int(v * 2.6))))
    base.alpha_composite(rim_c)
    base.alpha_composite(hero)

    # ---- headline in the free zone, verified not to cover the character
    words = [w for w in text.upper().split() if w][:3] or ["HISTORY"]
    lines = []
    for w in words:  # keep short words together ("WHY IT" / "COLLAPSED")
        if lines and len(lines[-1]) < 6:
            lines[-1] += " " + w
        else:
            lines.append(w)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    zone_w = W * (0.44 if concept == "subject" else 0.54)
    hero_mask = np.asarray(hero.split()[3]) > 40
    size = 290
    while size > 70:
        f = ImageFont.truetype(FONT_BIG, size)
        if max(tmp.textlength(l, font=f) for l in lines) <= zone_w and size * 0.93 * len(lines) <= H * 0.80:
            tl = _text_layer(lines, f, hi, flip, size, 44 if not flip else W - 44 - zone_w, (H - size * 0.93 * len(lines)) / 2 - size * 0.04)
            tm = np.asarray(tl.split()[3]) > 128
            if tm.sum() and (tm & hero_mask).sum() / tm.sum() < 0.012:
                break
        size -= 8
    f = ImageFont.truetype(FONT_BIG, size)
    tl = _text_layer(lines, f, hi, flip, size, 44 if not flip else W - 44 - zone_w, (H - size * 0.93 * len(lines)) / 2 - size * 0.04)
    base.alpha_composite(tl)

    badge = str(recipe.get("badge") or "").upper()[:9]
    if badge:
        bf = ImageFont.truetype(FONT_BIG, 66)
        d2 = ImageDraw.Draw(base)
        bw = d2.textlength(badge, font=bf) + 44
        bx, by = (34 if not flip else W - bw - 34), 26
        d2.rounded_rectangle([bx + 5, by + 7, bx + bw + 5, by + 88], 16, fill=(0, 0, 0, 160))
        d2.rounded_rectangle([bx, by, bx + bw, by + 80], 16, fill=hi, outline=(12, 8, 10), width=5)
        d2.text((bx + 22, by + 1), badge, font=bf, fill=(24, 14, 8))
    if recipe.get("mark") in ("?", "!"):
        mf = ImageFont.truetype(FONT_BIG, 170)
        d3 = ImageDraw.Draw(base)
        mx = int(W * (0.90 if not flip else 0.06))
        d3.text((mx + 6, 40 + 8), recipe["mark"], font=mf, fill=(0, 0, 0, 150), stroke_width=12, stroke_fill=(0, 0, 0, 150))
        d3.text((mx, 40), recipe["mark"], font=mf, fill=(255, 72, 60), stroke_width=10, stroke_fill=(12, 8, 10))
    _grade(base).save(out_path, "JPEG", quality=92, optimize=True)
    return out_path
