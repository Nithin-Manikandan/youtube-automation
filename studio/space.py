"""Shaded space art: a lit, rotating Earth/Moon sphere and an Apollo-style stack drawn as a gradient-shaded sprite."""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

INK = (28, 24, 32)
_TEX = {}


def _evict():
    keys = [k for k in _TEX if isinstance(k, tuple)]
    for k in keys[:-24]:
        del _TEX[k]


def _fbm(h, w, seed, octaves=5, base=4):
    """Horizontally tileable fractal noise in 0..1."""
    rng = np.random.RandomState(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = base * 2 ** o, base * 2 ** (o + 1)
        g = rng.rand(gh, gw).astype(np.float32)
        g = np.concatenate([g, g[:, :1]], axis=1)
        g = cv2.resize(g, (w + w // gw, h), interpolation=cv2.INTER_CUBIC)[:, :w]
        out += g * amp
        tot += amp
        amp *= 0.5
    out /= tot
    half = out[:, :w // 2]
    out = np.concatenate([half, half[:, ::-1]], axis=1)[:, :w]          # mirrored so the wrap-around has no seam
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def _earth_tex():
    if "earth" not in _TEX:
        h, w = 256, 512
        land = _fbm(h, w, 7, 5, 3)
        cloud = _fbm(h, w, 21, 5, 4)
        lat = np.abs(np.linspace(-1, 1, h))[:, None]
        _TEX["earth"] = (land, cloud, lat)
    return _TEX["earth"]


def _moon_tex():
    if "moon" not in _TEX:
        h, w = 256, 512
        base = _fbm(h, w, 3, 5, 4)
        rng = random.Random(11)
        cr = np.zeros((h, w), np.float32)
        for _ in range(70):
            cx, cy, r = rng.randrange(w), rng.randrange(20, h - 20), rng.choice([4, 6, 9, 14, 20])
            tmp = np.zeros((h, w), np.float32)
            for ox in (-w, 0, w):                                        # draw wrapped so craters cross the seam cleanly
                cv2.circle(tmp, (cx + ox, cy), r, -0.55, -1)
                cv2.circle(tmp, (cx + ox, cy), r, 0.7, max(1, r // 5))
            cr += cv2.GaussianBlur(tmp, (0, 0), 1.2)
        _TEX["moon"] = (base, cr)
    return _TEX["moon"]


def _sphere_maps(n, rot):
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (xs - n / 2) / (n / 2)
    y = (ys - n / 2) / (n / 2)
    rr = x * x + y * y
    inside = rr < 1.0
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    lon = np.arctan2(x, z) + rot
    lat = np.arcsin(np.clip(y, -1, 1))
    return x, y, z, inside, lon, lat, np.sqrt(rr)


def _lambert(x, y, z, light=(-.55, -.45, .70)):
    lx, ly, lz = light
    nrm = math.sqrt(lx * lx + ly * ly + lz * lz)
    return np.clip((x * lx + y * ly + z * lz) / nrm, 0, 1)


def planet_sprite(kind, r, t):
    key = ("p", kind, int(r), int(t * 0.05 / 0.004))
    if key not in _TEX:
        _TEX[key] = _planet_sprite(kind, r, int(t * 0.05 / 0.004) * 0.004 / 0.05)
        _evict()
    return _TEX[key]


def _planet_sprite(kind, r, t):
    """RGBA sprite, side 2.6r, planet centred. Spherical mapping of a procedural texture, lambert shading, atmosphere rim."""
    n = max(64, int(2 * r))
    x, y, z, inside, lon, lat, dist = _sphere_maps(n, t * 0.05)
    lam = _lambert(x, y, z)
    if kind == "moon":
        base, cr = _moon_tex()
        mx = ((lon / (2 * math.pi)) % 1.0 * 512).astype(np.float32)
        my = ((lat / math.pi + .5) * 255).astype(np.float32)
        b = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        c = cv2.remap(cr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        g = 150 + 60 * (b - .5) + 45 * np.clip(c, -1, 1)
        col = np.stack([g, g, g * 1.03], -1)
    else:
        land, cloud, latt = _earth_tex()
        mx = ((lon / (2 * math.pi)) % 1.0 * 511).astype(np.float32)
        my = ((lat / math.pi + .5) * 255).astype(np.float32)
        ld = cv2.remap(land, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        cl = cv2.remap(cloud, mx + 60, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        lt = np.abs(lat / (math.pi / 2))
        ocean = np.stack([np.full_like(ld, 18), 78 + 40 * (1 - lt), 160 + 40 * (1 - lt)], -1)
        landc = np.stack([86 + 70 * ld, 128 + 40 * ld, 62 + 20 * ld], -1)
        isl = np.clip((ld - .54) * 14, 0, 1)[..., None]
        col = ocean * (1 - isl) + landc * isl
        ice = np.clip((lt - .84) * 12, 0, 1)[..., None]
        col = col * (1 - ice) + np.array([236, 244, 250], np.float32) * ice
        ca = np.clip((cl - .5) * 3.2, 0, .9)[..., None]
        col = col * (1 - ca) + 250 * ca
    shade = (0.16 + 0.95 * lam)[..., None]
    col = col * shade
    rim = (np.clip(dist, 0, 1) ** 7)[..., None]
    if kind != "moon":
        col = col + rim * np.array([60, 110, 170], np.float32) * (0.35 + lam[..., None])
    rgb = np.clip(col, 0, 255).astype(np.uint8)
    sprite = np.zeros((int(n * 1.3), int(n * 1.3), 4), np.uint8)
    o = (sprite.shape[0] - n) // 2
    a = np.where(inside, 255, 0).astype(np.uint8)
    a = cv2.GaussianBlur(a, (0, 0), 0.9)
    sprite[o:o + n, o:o + n, :3] = rgb
    sprite[o:o + n, o:o + n, 3] = a
    if kind != "moon":                                    # atmosphere glow outside the limb
        big = sprite.shape[0]
        yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)
        dd = np.sqrt((xx - big / 2) ** 2 + (yy - big / 2) ** 2) / (n / 2)
        glow = np.exp(-np.clip(dd - 1, 0, None) / 0.05) * (dd >= 1) * 0.65
        lit = np.clip(0.35 + 0.65 * _lambert((xx - big / 2) / (n / 2), (yy - big / 2) / (n / 2), np.full_like(xx, .5)), 0, 1)
        ga = (glow * lit * 255).astype(np.uint8)
        spr = sprite.copy()
        spr[..., :3] = np.where((sprite[..., 3:4] > 0), sprite[..., :3], np.array([120, 180, 255], np.uint8))
        spr[..., 3] = np.maximum(sprite[..., 3], ga)
        sprite = spr
    return Image.fromarray(sprite, "RGBA")


def paste_sprite(d, sp, cx, cy, deg=0.0):
    if deg:
        sp = sp.rotate(deg, resample=Image.BICUBIC, expand=True)
    img = d._image
    img.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)


def _gpoly(sp, pts, stops, outline=INK, ow=3):
    """Polygon filled with a vertical colour gradient (stops = [(pos 0..1, rgb)...]) and an ink outline."""
    mask = Image.new("L", sp.size, 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    ys = [p[1] for p in pts]
    y0, y1 = min(ys), max(ys)
    rows = np.arange(sp.size[1], dtype=np.float32)
    pos = np.clip((rows - y0) / max(1.0, y1 - y0), 0, 1)
    ps = [s[0] for s in stops]
    ch = [np.interp(pos, ps, [s[1][k] for s in stops]) for k in range(3)]
    grad = np.stack([np.repeat(c[:, None], sp.size[0], 1) for c in ch], -1).astype(np.uint8)
    sp.paste(Image.fromarray(grad, "RGB"), mask=mask)
    if outline:
        ImageDraw.Draw(sp).line(list(pts) + [pts[0]], fill=outline, width=ow, joint="curve")


def spacecraft_sprite(L, R, roll, venting=False):
    key = ("s", int(L), int(R), int(roll / 0.03), venting)
    if key not in _TEX:
        _TEX[key] = _spacecraft_sprite(L, R, int(roll / 0.03) * 0.03, venting)
        _evict()
    return _TEX[key]


def _spacecraft_sprite(L, R, roll, venting=False):
    """Apollo-style stack facing +x: lit cylinder with wrapping panels, shaded cone, windows, bell, RCS quads, dish."""
    W, H = int(2.2 * L), int(4.4 * R)
    sp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cx, cy = W / 2, H / 2
    ow = max(3, int(R * .06))
    P = lambda pts: [(cx + px * L, cy + py * R) for px, py in pts]
    cyl = [(-.5, -.8), (.15, -.8), (.15, .8), (-.5, .8)]
    _gpoly(sp, P(cyl), [(0, (252, 246, 226)), (.35, (226, 214, 184)), (.75, (166, 154, 132)), (1, (96, 90, 84))], None)
    ov = Image.new("RGBA", sp.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for k in range(9):                                                         # panel seams wrap round the cylinder as it rolls
        ph = roll + k * math.pi / 4.5
        c_ = math.cos(ph)
        if c_ <= 0.1:
            continue
        yy = .8 * math.sin(ph)
        hh = .06 * c_ + .01
        dr.polygon(P([(-.5, yy - hh), (.15, yy - hh), (.15, yy + hh), (-.5, yy + hh)]), fill=(120, 110, 96, 70))
    for k in range(3):
        dr.line(P([(-.26 + k * .22, -.8), (-.26 + k * .22, .8)]), fill=(120, 110, 96, 150), width=2)
    dr.line(P([(-.49, -.7), (.14, -.7)]), fill=(255, 255, 255, 160), width=max(3, int(R * .05)))   # specular streak
    sp.alpha_composite(ov)
    dr = ImageDraw.Draw(sp)
    dr.line(list(P(cyl)) + [P(cyl)[0]], fill=INK, width=ow, joint="curve")
    cone = [(.15, -.8), (.50, -.38), (.50, .38), (.15, .8)]
    _gpoly(sp, P(cone), [(0, (236, 240, 248)), (.4, (196, 202, 214)), (1, (104, 110, 124))], INK, ow)
    dr.polygon(P([(.15, -.8), (.19, -.8), (.19, .8), (.15, .8)]), fill=(52, 44, 42))                     # heat-shield ring
    for k in range(3):                                                                                   # windows with a glint
        wx = .27 + k * .065
        _gpoly(sp, P([(wx, -.13), (wx + .035, -.13), (wx + .035, .13), (wx, .13)]), [(0, (110, 160, 220)), (1, (24, 40, 80))], INK, 2)
        dr.line(P([(wx + .008, -.1), (wx + .008, -.02)]), fill=(255, 255, 255, 220), width=2)
    _gpoly(sp, P([(.5, -.05), (.64, -.05), (.64, .05), (.5, .05)]), [(0, (220, 222, 228)), (1, (110, 112, 120))], INK, 2)
    _gpoly(sp, P([(-.5, -.3), (-.68, -.5), (-.68, .5), (-.5, .3)]), [(0, (150, 152, 160)), (.5, (96, 98, 106)), (1, (50, 52, 60))], INK, ow)
    for y0_, y1_ in ((-1.0, -.8), (.8, 1.0)):                                                            # RCS quads
        _gpoly(sp, P([(-.2, y0_), (-.1, y0_), (-.1, y1_), (-.2, y1_)]), [(0, (190, 190, 196)), (1, (110, 110, 118))], INK, 2)
    dr.line(P([(-.12, -.8), (-.12, -1.35)]), fill=INK, width=3)                                          # antenna mast and dish
    (dx0, dy0), (dx1, dy1) = P([(-.2, -1.4), (-.04, -1.3)])
    dr.ellipse([dx0, dy0, dx1, dy1], fill=(214, 214, 220), outline=INK, width=2)
    if venting:
        dr.polygon(P([(-.28, .8), (-.12, .8), (-.1, .45), (-.2, .55), (-.3, .4)]), fill=(30, 26, 28))
    return sp
