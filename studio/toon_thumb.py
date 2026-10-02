"""Flat cartoon thumbnails (entertainment style): clean sky and grass, one huge expressive stick-figure face, the story's key prop drawn big, chunky headline.
Drawn directly with bold ink outlines at 2x and downsampled: no glow, grade or blur, so it stays crisp and readable at phone size."""
import math
import random

from PIL import Image, ImageDraw, ImageFont

from pipeline.config import ROOT
from . import props

W, H = 1280, 720
K = 2
INK = (24, 22, 30)
FONT = str(ROOT / "assets/fonts/BigShoulders-Bold.ttf")
SKINS = (254, 252, 248)
HAIR = ((96, 62, 40), (52, 36, 30), (150, 98, 48), (176, 60, 40))
SHIRT = {"peasant": (150, 98, 56), "citizen": (210, 110, 52), "merchant": (66, 150, 84), "worker": (236, 168, 40), "scholar": (120, 90, 170), "king": (200, 56, 52)}


def _ell(d, cx, cy, rx, ry, fill, w):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=INK, width=w)


def _cloud(d, cx, cy, r, w):
    for dx, dy, rr in ((-1.1, .25, .75), (0, 0, 1.0), (1.2, .25, .8), (.15, .45, .8)):
        d.ellipse([cx + dx * r - rr * r, cy + dy * r - rr * r, cx + dx * r + rr * r, cy + dy * r + rr * r], fill=(255, 255, 255))
    for dx, dy, rr in ((-1.1, .25, .75), (0, 0, 1.0), (1.2, .25, .8), (.15, .45, .8)):
        d.arc([cx + dx * r - rr * r, cy + dy * r - rr * r, cx + dx * r + rr * r, cy + dy * r + rr * r], 200, 340, fill=INK, width=w)


def _background(d, rnd, scene):
    gy = int(H * K * 0.66)
    sky = {"night": ((38, 42, 110), (110, 100, 190)), "gloom": ((120, 140, 150), (190, 200, 196))}.get(scene, ((104, 190, 244), (214, 240, 252)))
    for y in range(gy):
        u = y / gy
        d.line([(0, y), (W * K, y)], fill=tuple(int(sky[0][i] + (sky[1][i] - sky[0][i]) * u) for i in range(3)))
    w = 6 * K // 2
    if scene == "night":
        d.ellipse([W * K * .72 - 90 * K, 70 * K - 90 * K + 40 * K, W * K * .72 + 90 * K, 70 * K + 90 * K + 40 * K], fill=(250, 244, 200), outline=INK, width=w)
        for _ in range(40):
            x, y = rnd.random() * W * K, rnd.random() * gy * .7
            d.ellipse([x - 3 * K, y - 3 * K, x + 3 * K, y + 3 * K], fill=(255, 255, 255))
    else:
        sx = W * K * (0.56 if rnd.random() < .5 else 0.44)
        d.ellipse([sx - 56 * K, 90 * K - 56 * K, sx + 56 * K, 90 * K + 56 * K], fill=(255, 224, 70), outline=INK, width=w)
        for cx, cy, r in ((.08, .13, 38), (.50, .20, 30), (.90, .10, 34)):
            _cloud(d, cx * W * K, cy * H * K, r * K, w)
    hill = (122, 190, 92) if scene != "night" else (40, 60, 110)
    pts = [(0, gy)]
    for i in range(0, 41):
        x = i / 40 * W * K
        pts.append((x, gy - 70 * K - math.sin(i / 40 * 6.3 + 1) * 34 * K - math.sin(i / 40 * 15) * 10 * K))
    pts.append((W * K, gy))
    d.polygon(pts, fill=hill)
    d.line(pts[1:-1], fill=INK, width=w, joint="curve")
    grass = (142, 206, 78) if scene != "night" else (52, 84, 92)
    d.rectangle([0, gy, W * K, H * K], fill=grass)
    d.line([(0, gy), (W * K, gy)], fill=INK, width=w + 2)
    for _ in range(46):
        x, y = rnd.random() * W * K, gy + 24 * K + rnd.random() * (H * K - gy - 30 * K)
        for t in (-1, 0, 1):
            d.line([(x + t * 8 * K, y), (x + t * 14 * K, y - 22 * K)], fill=INK, width=max(3, w // 2))


def _stink(d, cx, cy, s, w, k=3):
    for i in range(k):
        x = cx + (i - 1) * s * .5
        pts = [(x + math.sin(j * .9 + i) * s * .12, cy - j * s * .12) for j in range(8)]
        d.line(pts, fill=(96, 170, 70), width=w * 2, joint="curve")


def _hero(img, role, emotion, cx, cy, R, facing, look, hair_i, w):
    d = ImageDraw.Draw(img)
    shirt = SHIRT.get(role, (210, 110, 52))
    # body: tunic from the neck down to the bottom edge, arms as ink lines with fists
    nx, ny = cx - facing * R * .05, cy + R * .96
    d.polygon([(nx - R * .55, ny + R * .02), (nx + R * .55, ny + R * .02), (nx + R * .72, H * K), (nx - R * .72, H * K)], fill=shirt, outline=INK)
    d.line([(nx - R * .55, ny), (nx - R * .72, H * K)], fill=INK, width=w)
    d.line([(nx + R * .55, ny), (nx + R * .72, H * K)], fill=INK, width=w)
    d.arc([nx - R * .34, ny - R * .22, nx + R * .34, ny + R * .22], 10, 170, fill=INK, width=w)
    # head
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=SKINS, outline=INK, width=w + 2)
    # messy hair: a cap over the top of the head plus spikes
    hc = HAIR[hair_i % len(HAIR)]
    d.chord([cx - R, cy - R, cx + R, cy + R], 214, 326, fill=hc)
    spikes = []
    for i in range(11):
        a = math.radians(216 + 108 * i / 10)
        r = R * (1.0 if (i % 2 or i in (0, 10)) else 1.3 + 0.1 * math.sin(i * 2.1))
        spikes.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    d.polygon(spikes, fill=hc)
    d.line(spikes, fill=INK, width=w, joint="curve")
    d.chord([cx - R, cy - R, cx + R, cy + R], 214, 326, outline=INK, width=w)
    # face
    ex, ey = R * .34, R * .02
    er_x, er_y = R * .2, R * .27
    shock = emotion in ("shock", "scared")
    disgust = emotion in ("worried", "disgust", "sad")
    grin = emotion in ("smile", "cheer")
    if disgust:
        er_y = R * .2
    if shock:
        er_x, er_y = R * .24, R * .31
    for sgn in (-1, 1):
        x0 = cx + sgn * ex + facing * R * .06
        d.ellipse([x0 - er_x, cy + ey - er_y, x0 + er_x, cy + ey + er_y], fill=(255, 255, 255), outline=INK, width=w)
        pr = R * (.07 if shock else .105)
        px = x0 + look[0] * (er_x - pr) * .8
        py = cy + ey + look[1] * (er_y - pr) * .8
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=INK)
        d.ellipse([px - pr * .1, py - pr * .65, px + pr * .5, py - pr * .05], fill=(255, 255, 255))
        by = cy + ey - er_y - R * (.2 if shock else .13)
        tilt = {"shock": 0, "scared": 0}.get(emotion, .2 if disgust else (-.05 if grin else 0))
        d.line([(x0 - er_x * 1.1, by - sgn * tilt * R * .5), (x0 + er_x * 1.1, by + sgn * tilt * R * .5)], fill=INK, width=w + 4)
        if disgust:                                                    # heavy half-closed lids
            d.rectangle([x0 - er_x, cy + ey - er_y, x0 + er_x, cy + ey - er_y * .25], fill=SKINS)
            d.line([(x0 - er_x, cy + ey - er_y * .25), (x0 + er_x, cy + ey - er_y * .25)], fill=INK, width=w)
            d.ellipse([px - pr, cy + ey - er_y * .1 - pr * .6, px + pr, cy + ey - er_y * .1 + pr * .6], fill=INK)
    my = cy + R * .56
    mx = cx + facing * R * .1
    if shock:
        d.ellipse([mx - R * .17, my - R * .1, mx + R * .17, my + R * .3], fill=(120, 30, 44), outline=INK, width=w)
    elif grin:
        d.chord([mx - R * .36, my - R * .22, mx + R * .36, my + R * .3], 0, 180, fill=(120, 30, 44), outline=INK, width=w)
        d.rectangle([mx - R * .3, my + R * .005, mx + R * .3, my + R * .06], fill=(255, 255, 255))
    elif disgust:
        pts = [(mx - R * .32 + i * R * .08, my + math.sin(i * 1.8) * R * .05) for i in range(9)]
        d.line(pts, fill=INK, width=w + 2, joint="curve")
        d.ellipse([cx + facing * R * .62 - R * .07, cy + R * .1, cx + facing * R * .62 + R * .07, cy + R * .3], fill=(120, 200, 240), outline=INK, width=w // 2)   # sweat drop
    else:
        d.arc([mx - R * .3, my - R * .12, mx + R * .3, my + R * .16], 20, 160, fill=INK, width=w + 2)
    return (nx, ny)


def _icon(name, box):
    """Bold poster icons for the story's key object, drawn straight into box=(x0, y0, x1, y1). Returns a layer, or None for objects without one."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    lay = Image.new("RGBA", (W * K, H * K), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    lw = 9 * K
    WOOD, WOOD2, WOOD3 = (176, 122, 70), (140, 92, 50), (214, 168, 112)
    if name in ("barrel", "bucket", "basket"):                                # a big wooden vat full of something
        bw, bh = min(w, h * .95), min(h, w * 1.02)
        L, R_, T, B = cx - bw / 2, cx + bw / 2, y1 - bh, y1
        d.polygon([(L + bw * .06, B), (L, B - bh * .5), (L + bw * .05, T + bh * .14), (R_ - bw * .05, T + bh * .14), (R_, B - bh * .5), (R_ - bw * .06, B)], fill=WOOD, outline=INK)
        for f in (.3, .62):
            d.line([(L + bw * .02, T + bh * f), (R_ - bw * .02, T + bh * f)], fill=INK, width=lw)
        for f in (.25, .5, .75):
            d.line([(L + bw * f, T + bh * .14), (L + bw * (f * .85 + .075), B)], fill=WOOD2, width=lw // 2)
        d.ellipse([L + bw * .02, T, R_ - bw * .02, T + bh * .28], fill=(236, 214, 70), outline=INK, width=lw)      # the liquid
        d.ellipse([L + bw * .16, T + bh * .06, L + bw * .42, T + bh * .15], fill=(255, 248, 150))
        for k in range(3):
            d.ellipse([cx - bw * .2 + k * bw * .2, T - bh * (.06 + .03 * k), cx - bw * .2 + k * bw * .2 + bw * .1, T - bh * (.06 + .03 * k) + bw * .1], fill=(255, 255, 255), outline=INK, width=lw // 2)
    elif name == "stool":                                                      # a padded royal toilet seat
        bw, bh = min(w, h * 1.3), min(h, w * .8)
        L, R_, T, B = cx - bw / 2, cx + bw / 2, y1 - bh, y1
        d.rounded_rectangle([L, T + bh * .38, R_, B], radius=int(bh * .08), fill=(120, 60, 140), outline=INK, width=lw)
        d.rounded_rectangle([L - bw * .03, T + bh * .1, R_ + bw * .03, T + bh * .46], radius=int(bh * .12), fill=(214, 60, 70), outline=INK, width=lw)
        d.ellipse([cx - bw * .22, T + bh * .17, cx + bw * .22, T + bh * .4], fill=INK)
        for f in (.12, .3, .5, .7, .88):
            d.ellipse([L + bw * f - 7 * K, T + bh * .66 - 7 * K, L + bw * f + 7 * K, T + bh * .66 + 7 * K], fill=(250, 214, 80))
    elif name == "toilet":
        bw, bh = min(w, h * .95), min(h, w * 1.05)
        L, R_, T, B = cx - bw / 2, cx + bw / 2, y1 - bh, y1
        d.rounded_rectangle([L + bw * .08, T, R_ - bw * .08, T + bh * .46], radius=int(bh * .06), fill=(246, 248, 252), outline=INK, width=lw)          # cistern
        d.rectangle([R_ - bw * .3, T + bh * .06, R_ - bw * .2, T + bh * .12], fill=(190, 200, 214), outline=INK, width=lw // 2)
        d.ellipse([L, T + bh * .42, R_, T + bh * .64], fill=(246, 248, 252), outline=INK, width=lw)                                                   # seat rim
        d.ellipse([L + bw * .12, T + bh * .46, R_ - bw * .12, T + bh * .6], fill=(70, 60, 56), outline=INK, width=lw // 2)
        d.polygon([(L + bw * .14, T + bh * .56), (R_ - bw * .14, T + bh * .56), (R_ - bw * .22, B - bh * .02), (L + bw * .22, B - bh * .02)], fill=(236, 240, 246), outline=INK)
        d.line([(L + bw * .14, T + bh * .56), (L + bw * .22, B - bh * .02), (R_ - bw * .22, B - bh * .02), (R_ - bw * .14, T + bh * .56)], fill=INK, width=lw, joint="curve")
    elif name == "poop":
        bw, bh = min(w, h * 1.05), min(h, w * .95)
        T, B = y1 - bh, y1
        for k, (f, ww) in enumerate(((0.0, .96), (.3, .72), (.58, .48))):
            top, bot = B - bh * (f + .38), B - bh * f
            d.ellipse([cx - bw * ww / 2, top, cx + bw * ww / 2, bot], fill=(124, 78, 40), outline=INK, width=lw)
        d.polygon([(cx - bw * .08, T + bh * .06), (cx + bw * .12, T + bh * .26), (cx - bw * .12, T + bh * .3)], fill=(124, 78, 40), outline=INK)
        for sx in (-1, 1):
            ex = cx + sx * bw * .13
            d.ellipse([ex - bw * .07, B - bh * .5, ex + bw * .07, B - bh * .3], fill=(255, 255, 255), outline=INK, width=lw // 2)
            d.ellipse([ex - bw * .025, B - bh * .44, ex + bw * .025, B - bh * .36], fill=INK)
        d.arc([cx - bw * .14, B - bh * .34, cx + bw * .14, B - bh * .14], 20, 160, fill=INK, width=lw // 2)
    elif name == "coins":
        bw, bh = min(w, h * 1.1), min(h, w * .9)
        T, B = y1 - bh, y1
        for row, (n, yy) in enumerate(((4, .74), (3, .5), (2, .26))):
            for i in range(n):
                px = cx + (i - (n - 1) / 2) * bw * .24
                py = B - bh * (1 - yy) * 1.0 - bh * .12
                d.ellipse([px - bw * .14, py - bh * .13, px + bw * .14, py + bh * .13], fill=(252, 208, 40), outline=INK, width=lw)
                d.ellipse([px - bw * .08, py - bh * .07, px + bw * .08, py + bh * .07], outline=(200, 150, 20), width=lw // 2)
        d.text((cx - bw * .08, T + bh * .02), "$", font=ImageFont.truetype(FONT, int(bh * .34)), fill=(60, 170, 70), stroke_width=lw // 2, stroke_fill=INK)
    elif name == "skull":
        bw, bh = min(w, h), min(h, w)
        T, B = y1 - bh, y1
        d.ellipse([cx - bw * .46, T, cx + bw * .46, T + bh * .72], fill=(246, 244, 236), outline=INK, width=lw)
        d.rounded_rectangle([cx - bw * .26, T + bh * .6, cx + bw * .26, B], radius=int(bh * .06), fill=(246, 244, 236), outline=INK, width=lw)
        for sx in (-1, 1):
            d.ellipse([cx + sx * bw * .2 - bw * .13, T + bh * .26, cx + sx * bw * .2 + bw * .13, T + bh * .56], fill=INK)
        d.polygon([(cx, T + bh * .55), (cx - bw * .06, T + bh * .66), (cx + bw * .06, T + bh * .66)], fill=INK)
        for f in (-.15, -.05, .05, .15):
            d.line([(cx + bw * f, T + bh * .72), (cx + bw * f, B - bh * .04)], fill=INK, width=lw // 2)
    elif name == "leech":
        bw, bh = min(w, h * 1.5), min(h, w * .7)
        cx0, T, B = cx, y1 - bh, y1
        d.ellipse([cx0 - bw * .5, B - bh * .5, cx0 + bw * .5, B], fill=(72, 150, 96), outline=INK, width=lw)               # murky pond
        d.ellipse([cx0 - bw * .36, B - bh * .4, cx0 + bw * .36, B - bh * .1], outline=(150, 210, 150), width=lw // 2)
        pts = [(cx0 - bw * .22 + i * bw * .055, T + bh * .1 + math.sin(i * .75) * bh * .1 + (i * i) * bh * .0035) for i in range(10)]   # a fat leech
        d.line(pts, fill=(36, 24, 26), width=int(bh * .2), joint="curve")
        for p_ in (pts[0], pts[-1]):
            d.ellipse([p_[0] - bh * .1, p_[1] - bh * .1, p_[0] + bh * .1, p_[1] + bh * .1], fill=(36, 24, 26))
        d.line(pts, fill=(130, 40, 50), width=int(bh * .06), joint="curve")
        d.ellipse([pts[-1][0] - bh * .03, pts[-1][1] - bh * .04, pts[-1][0] + bh * .03, pts[-1][1] + bh * .0], fill=(255, 255, 255))
    elif name in ("bed",):
        bw, bh = w, min(h, w * .6)
        L, R_, T, B = x0, x1, y1 - bh, y1
        d.rectangle([L, T + bh * .1, L + bw * .05, B], fill=WOOD2, outline=INK, width=lw)
        d.rectangle([R_ - bw * .05, T + bh * .4, R_, B], fill=WOOD2, outline=INK, width=lw)
        d.rounded_rectangle([L + bw * .04, T + bh * .38, R_ - bw * .04, T + bh * .7], radius=int(bh * .1), fill=WOOD, outline=INK, width=lw)
        d.rounded_rectangle([L + bw * .05, T + bh * .2, R_ - bw * .05, T + bh * .5], radius=int(bh * .15), fill=(118, 150, 214), outline=INK, width=lw)
        d.rounded_rectangle([L + bw * .08, T + bh * .06, L + bw * .3, T + bh * .32], radius=int(bh * .1), fill=(250, 246, 236), outline=INK, width=lw)
        for k in range(3):
            tx, ty = R_ - bw * (.34 - .08 * k), T - bh * .02 - k * bh * .1
            d.line([(tx, ty), (tx + 22 * K, ty), (tx, ty + 22 * K), (tx + 22 * K, ty + 22 * K)], fill=INK, width=lw // 2)
    elif name == "bell":
        bw, bh = min(w, h * .9), min(h, w * 1.1)
        T, B = y1 - bh, y1
        d.polygon([(cx - bw * .5, B - bh * .1), (cx - bw * .3, T + bh * .3), (cx - bw * .12, T + bh * .04), (cx + bw * .12, T + bh * .04), (cx + bw * .3, T + bh * .3), (cx + bw * .5, B - bh * .1)], fill=(244, 196, 50), outline=INK)
        d.line([(cx - bw * .5, B - bh * .1), (cx - bw * .3, T + bh * .3), (cx - bw * .12, T + bh * .04), (cx + bw * .12, T + bh * .04), (cx + bw * .3, T + bh * .3), (cx + bw * .5, B - bh * .1), (cx - bw * .5, B - bh * .1)], fill=INK, width=lw, joint="curve")
        d.ellipse([cx - bw * .12, B - bh * .14, cx + bw * .12, B + bh * .02], fill=(150, 110, 40), outline=INK, width=lw)
        d.line([(cx - bw * .3, T + bh * .4), (cx - bw * .22, B - bh * .2)], fill=(255, 240, 160), width=lw)
    elif name == "sack":
        bw, bh = min(w, h * 1.0), min(h, w * 1.1)
        T, B = y1 - bh, y1
        d.polygon([(cx - bw * .14, T), (cx + bw * .14, T), (cx + bw * .2, T + bh * .22), (cx + bw * .5, B - bh * .2), (cx + bw * .42, B), (cx - bw * .42, B), (cx - bw * .5, B - bh * .2), (cx - bw * .2, T + bh * .22)], fill=(214, 180, 118), outline=INK)
        d.line([(cx - bw * .14, T), (cx + bw * .14, T), (cx + bw * .2, T + bh * .22), (cx + bw * .5, B - bh * .2), (cx + bw * .42, B), (cx - bw * .42, B), (cx - bw * .5, B - bh * .2), (cx - bw * .2, T + bh * .22), (cx - bw * .14, T)], fill=INK, width=lw, joint="curve")
        d.line([(cx - bw * .2, T + bh * .2), (cx + bw * .2, T + bh * .2)], fill=(150, 60, 50), width=lw)
        d.ellipse([cx - bw * .2, B - bh * .62, cx + bw * .2, B - bh * .22], fill=(250, 214, 60), outline=INK, width=lw)
        d.line([(cx - bw * .05, B - bh * .52), (cx - bw * .05, B - bh * .32)], fill=INK, width=lw)
        d.line([(cx + bw * .05, B - bh * .52), (cx + bw * .05, B - bh * .32)], fill=INK, width=lw)
    elif name == "chest":
        bw, bh = min(w, h * 1.5), min(h, w * .75)
        L, R_, T, B = cx - bw / 2, cx + bw / 2, y1 - bh, y1
        d.rectangle([L, T + bh * .4, R_, B], fill=WOOD, outline=INK, width=lw)
        d.chord([L, T, R_, T + bh * .8], 180, 360, fill=WOOD2, outline=INK, width=lw)
        d.rectangle([L, T + bh * .38, R_, T + bh * .48], fill=(120, 124, 134), outline=INK, width=lw // 2)
        d.rectangle([cx - bw * .07, T + bh * .34, cx + bw * .07, T + bh * .62], fill=(250, 214, 60), outline=INK, width=lw)
        for k in range(4):
            d.ellipse([L + bw * (.2 + .2 * k) - 12 * K, T - bh * .05 - 12 * K, L + bw * (.2 + .2 * k) + 12 * K, T - bh * .05 + 12 * K], fill=(250, 214, 60), outline=INK, width=lw // 2) if k % 2 == 0 else None
    elif name == "table":
        bw, bh = w, min(h, w * .7)
        L, R_, T, B = x0, x1, y1 - bh, y1
        d.rectangle([L, T + bh * .38, R_, T + bh * .5], fill=WOOD3, outline=INK, width=lw)
        for f in (.08, .88):
            d.rectangle([L + bw * f, T + bh * .5, L + bw * (f + .05), B], fill=WOOD2, outline=INK, width=lw)
        d.ellipse([L + bw * .12, T + bh * .06, L + bw * .46, T + bh * .4], fill=(232, 160, 88), outline=INK, width=lw)
        d.rectangle([L + bw * .6, T + bh * .1, L + bw * .72, T + bh * .38], fill=(244, 232, 190), outline=INK, width=lw)
        d.arc([L + bw * .68, T + bh * .14, L + bw * .8, T + bh * .32], 270, 90, fill=INK, width=lw)
    else:
        return None
    return lay


def _big_prop(img, name, box):
    """The story's key object: a bold poster icon when there is one, otherwise the scene prop scaled to fit the box."""
    ic = _icon(name, box)
    if ic is not None:
        img.alpha_composite(ic)
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    props.DRAW[name](d, {"scale": 2.4, "dur": 6.0}, img.width / 2, H * K * .9, H * K * .9, 1.2, W * K)
    bb = layer.getbbox()
    if not bb:
        return
    crop = layer.crop(bb)
    x0, y0, x1, y1 = box
    k = min((x1 - x0) / crop.width, (y1 - y0) / crop.height)
    crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS)
    img.alpha_composite(crop, (int(x0 + (x1 - x0 - crop.width) / 2), int(y1 - crop.height)))


def _headline(img, text, flip, w):
    words = [x for x in text.upper().split() if x][:3] or ["JOBS?"]
    lines = []
    for x in words:
        if lines and len(lines[-1]) + len(x) < 7:
            lines[-1] += " " + x
        else:
            lines.append(x)
    d = ImageDraw.Draw(img)
    zone_w = W * K * .54
    size = 330 * K
    while size > 60 * K:
        f = ImageFont.truetype(FONT, int(size))
        if max(d.textlength(l, font=f) for l in lines) <= zone_w and size * .92 * len(lines) <= H * K * .44:
            break
        size -= 6 * K
    f = ImageFont.truetype(FONT, int(size))
    y = 22 * K
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f)
        x = (W * K - 40 * K - tw) if not flip else 40 * K
        d.text((x + 8 * K, y + 10 * K), l, font=f, fill=(0, 0, 0, 160), stroke_width=14 * K, stroke_fill=(0, 0, 0, 160))
        d.text((x, y), l, font=f, fill=(255, 238, 34) if i == len(lines) - 1 or len(lines) == 1 else (255, 255, 255), stroke_width=13 * K, stroke_fill=INK)
        y += size * .92


def render(text, recipe, out_path, variant=0):
    rnd = random.Random(31 + variant * 7)
    flip = bool(recipe.get("flip", variant % 2 == 1))
    img = Image.new("RGBA", (W * K, H * K), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    _background(d, rnd, recipe.get("scene") if recipe.get("scene") in ("night", "gloom") else "day")
    w = 6 * K // 2 + 2
    ICONS = {"leech", "toilet", "poop", "coins", "skull", "barrel", "bucket", "basket", "bed", "bell", "sack", "chest", "table", "stool"}
    objs = [o for o in (recipe.get("objects") or []) if o in props.DRAW or o in ICONS] or ["barrel"]
    gy = int(H * K * .90)
    hx = (0.27 if not flip else 0.73) * W * K
    big_x = (0.70 if not flip else 0.30) * W * K
    # the story's key prop sits under the headline on the side away from the face; a second one smaller beside the face
    bx0, bx1 = ((0.50, 0.99) if not flip else (0.01, 0.50))
    _big_prop(img, objs[0], (bx0 * W * K, H * K * .52, bx1 * W * K, H * K * .985))
    if len(objs) > 1:
        sx0, sx1 = ((0.20, 0.42) if not flip else (0.58, 0.80))
        _big_prop(img, objs[1], (sx0 * W * K, H * K * .74, sx1 * W * K, H * K * .97))
    dd = ImageDraw.Draw(img)
    ax = (0.62 if not flip else 0.38) * W * K
    pts = [(ax - 34 * K, H * K * .50), (ax + 34 * K, H * K * .50), (ax + 34 * K, H * K * .56), (ax + 70 * K, H * K * .56), (ax, H * K * .66), (ax - 70 * K, H * K * .56), (ax - 34 * K, H * K * .56)] if False else None
    emotion = recipe.get("emotion", "shock")
    look = ((.8 if not flip else -.8), .25)
    _hero(img, recipe.get("role", "citizen"), emotion, hx, H * K * .46, H * K * .31, 1 if not flip else -1, look, variant, w)
    if emotion in ("worried", "sad", "disgust"):
        _stink(ImageDraw.Draw(img), hx + (1 if not flip else -1) * H * K * .42, H * K * .44, H * K * .2, w // 2)
    if emotion in ("shock", "scared"):
        dd = ImageDraw.Draw(img)
        cxm, cym = hx + (-1 if not flip else 1) * H * K * .38, H * K * .12
        for k in range(5):
            a = -1.9 + k * .45
            dd.line([(cxm + math.cos(a) * 54 * K, cym + math.sin(a) * 54 * K), (cxm + math.cos(a) * 110 * K, cym + math.sin(a) * 110 * K)], fill=INK, width=w + 2)
    _headline(img, text, flip, w)
    out = img.convert("RGB").resize((W, H), Image.LANCZOS)
    out.save(out_path, "JPEG", quality=94, optimize=True)
    return out_path
