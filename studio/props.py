"""Everyday props for stories about how people lived and worked: pits, barrels, carts, beds, bells, tables, stools, ladders, swamps, chests, stalls.

All are drawn flat with an ink outline so they sit in the same hand-drawn world as the stick figures. Signature: fn(d, o, x, S, gy, t, WS) where x is the
prop's horizontal centre in pixels, S the picture height and gy the ground line; sizes are fractions of S (a standing person is about .52 S tall).
"""
import math
import random

INK = (27, 27, 32)
WOOD, WOOD_D, WOOD_L = (150, 104, 62), (112, 76, 44), (186, 140, 90)
IRON = (96, 100, 110)


def _lw(S, k=.004):
    return max(3, int(S * k))


def _ell(d, cx, cy, rx, ry, fill, w=4):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=INK, width=w)


def _rect(d, x0, y0, x1, y1, fill, w=4):
    d.rectangle([x0, y0, x1, y1], fill=fill, outline=INK, width=w)


def pit(d, o, x, S, gy, t, WS):
    """A cesspit / trench: stone rim, murky brown depths with slow bubbles and a ladder."""
    sc = o.get("scale", 1)
    hw, dp = S * .30 * sc, S * .075 * sc
    _rect(d, x - hw, gy, x + hw, gy + dp, (92, 62, 36), _lw(S))
    d.rectangle([x - hw + 6, gy + 6, x + hw - 6, gy + dp * .55], fill=(118, 84, 46))
    rng = random.Random(3)
    for k in range(9):                                                              # bubbles rise and pop
        ph = (t * .5 + k * .37) % 1.0
        bx = x - hw * .8 + (k / 8) * hw * 1.6 + math.sin(k * 2.1) * S * .01
        by = gy + dp * (.85 - .85 * ph)
        r = S * (.006 + .008 * (k % 3)) * (1 - ph * .4)
        d.ellipse([bx - r, by - r, bx + r, by + r], outline=(60, 40, 22), width=3)
    for side in (-1, 1):                                                           # stone lip
        for j in range(3):
            _rect(d, x + side * (hw + S * (.02 + .035 * j)) - S * .02, gy - S * .02, x + side * (hw + S * (.02 + .035 * j)) + S * .02, gy + S * .004, (176, 170, 156), 3)
    lx = x + hw * .72                                                              # ladder leaning into the pit
    d.line([(lx - S * .02, gy - S * .10), (lx - S * .05, gy + dp)], fill=INK, width=_lw(S, .007))
    d.line([(lx + S * .02, gy - S * .10), (lx - S * .01, gy + dp)], fill=INK, width=_lw(S, .007))
    for r in range(5):
        yy = gy - S * .08 + r * (S * .075 / 2.2)
        d.line([(lx - S * .028 - r * S * .004, yy), (lx + S * .018 - r * S * .004, yy)], fill=WOOD_D, width=_lw(S, .006))


def barrel(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * .055 * sc, S * .10 * sc
    pts = [(x - w * .8, gy), (x - w, gy - h * .5), (x - w * .8, gy - h), (x + w * .8, gy - h), (x + w, gy - h * .5), (x + w * .8, gy)]
    d.polygon(pts, fill=WOOD, outline=INK)
    d.line(pts + [pts[0]], fill=INK, width=_lw(S))
    for yy in (.22, .5, .78):
        d.line([(x - w * (.82 if yy != .5 else 1.0), gy - h * yy), (x + w * (.82 if yy != .5 else 1.0), gy - h * yy)], fill=IRON, width=_lw(S, .006))
    _ell(d, x, gy - h, w * .8, h * .08, WOOD_L, 3)


def bucket(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * .035 * sc, S * .06 * sc
    d.polygon([(x - w, gy - h), (x + w, gy - h), (x + w * .75, gy), (x - w * .75, gy)], fill=(176, 150, 112), outline=INK)
    d.line([(x - w, gy - h), (x + w, gy - h), (x + w * .75, gy), (x - w * .75, gy), (x - w, gy - h)], fill=INK, width=_lw(S))
    d.arc([x - w, gy - h * 1.9, x + w, gy - h * .1], 180, 360, fill=INK, width=_lw(S))
    d.line([(x - w * .9, gy - h * .55), (x + w * .9, gy - h * .55)], fill=IRON, width=_lw(S, .005))
    d.rectangle([x - w * .85, gy - h, x + w * .85, gy - h * .8], fill=(110, 78, 44))


def basket(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * .05 * sc, S * .055 * sc
    d.polygon([(x - w, gy - h), (x + w, gy - h), (x + w * .8, gy), (x - w * .8, gy)], fill=(206, 170, 98), outline=INK)
    d.line([(x - w, gy - h), (x + w, gy - h), (x + w * .8, gy), (x - w * .8, gy), (x - w, gy - h)], fill=INK, width=_lw(S))
    for k in range(1, 5):                                                          # wicker weave
        d.line([(x - w * (1 - .08 * k), gy - h + k * h / 5), (x + w * (1 - .08 * k), gy - h + k * h / 5)], fill=(150, 112, 56), width=3)
    d.arc([x - w * .9, gy - h * 2.0, x + w * .9, gy - h * .2], 180, 360, fill=INK, width=_lw(S))


def sack(d, o, x, S, gy, t, WS):
    """A fat sack of coins with a few coins spilled in front."""
    sc = o.get("scale", 1)
    w, h = S * .05 * sc, S * .075 * sc
    d.polygon([(x - w * .35, gy - h), (x + w * .35, gy - h), (x + w, gy - h * .3), (x + w * .9, gy), (x - w * .9, gy), (x - w, gy - h * .3)], fill=(196, 168, 112), outline=INK)
    d.line([(x - w * .35, gy - h), (x + w * .35, gy - h), (x + w, gy - h * .3), (x + w * .9, gy), (x - w * .9, gy), (x - w, gy - h * .3), (x - w * .35, gy - h)], fill=INK, width=_lw(S))
    d.line([(x - w * .4, gy - h * .88), (x + w * .4, gy - h * .88)], fill=(120, 80, 40), width=_lw(S, .007))
    _ell(d, x, gy - h * .45, w * .32, w * .32, (240, 200, 70), 3)
    for k, dx in enumerate((.9, 1.25, 1.55)):
        _ell(d, x + w * dx, gy - S * .006, S * .014, S * .006, (240, 200, 70), 3)


def chest(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * .075 * sc, S * .06 * sc
    _rect(d, x - w, gy - h * .62, x + w, gy, WOOD, _lw(S))
    d.pieslice([x - w, gy - h * 1.2, x + w, gy - h * .05], 180, 360, fill=WOOD_D, outline=INK, width=_lw(S))
    for dx in (-.55, 0, .55):
        d.line([(x + w * dx, gy - h * 1.1), (x + w * dx, gy)], fill=IRON, width=_lw(S, .006))
    _rect(d, x - S * .01, gy - h * .75, x + S * .01, gy - h * .45, (240, 200, 70), 3)


def bed(d, o, x, S, gy, t, WS):
    """A wooden bed with a lumpy blanket and a pillow; a sleeper's 'z' drifts up."""
    sc = o.get("scale", 1)
    w, h = S * .20 * sc, S * .075 * sc
    _rect(d, x - w, gy - h * 1.9, x - w + S * .014, gy, WOOD_D, 3)
    _rect(d, x + w - S * .014, gy - h * 1.3, x + w, gy, WOOD_D, 3)
    _rect(d, x - w, gy - h * 1.0, x + w, gy - h * .55, WOOD, _lw(S))
    d.rounded_rectangle([x - w * .98, gy - h * 1.55, x + w * .98, gy - h * .98], radius=int(h * .35), fill=(122, 148, 196), outline=INK, width=_lw(S))
    d.rounded_rectangle([x - w * .92, gy - h * 1.78, x - w * .55, gy - h * 1.3], radius=int(h * .3), fill=(244, 240, 228), outline=INK, width=3)
    for k in range(3):                                                               # a sleeper's Zs drift up and fade
        age = (t * .5 + k / 3) % 1.0
        zx, zy, zs = x - w * .6 + age * S * .05, gy - h * 2.0 - age * S * .12, S * (.012 + .01 * age)
        col = INK + (int(255 * (1 - age)),)
        d.line([(zx - zs, zy - zs), (zx + zs, zy - zs), (zx - zs, zy + zs), (zx + zs, zy + zs)], fill=col, width=3)


def bell(d, o, x, S, gy, t, WS):
    """A big hand-swung bell on a post, swinging while it rings."""
    sc = o.get("scale", 1)
    h = S * .24 * sc
    d.line([(x, gy), (x, gy - h)], fill=INK, width=_lw(S, .012))
    d.line([(x - S * .06 * sc, gy - h), (x + S * .06 * sc, gy - h)], fill=INK, width=_lw(S, .012))
    sw = math.sin(t * 7) * .22
    cx, cy = x + S * .035 * sc, gy - h
    pts = []
    for a in range(0, 181, 15):
        r = S * .04 * sc
        pts.append((cx + math.cos(math.radians(a)) * r * math.cos(sw) - 0, cy + S * .01 + math.sin(math.radians(a)) * r * 1.0 - 0))
    pts = [(cx - S * .05 * sc, cy + S * .075 * sc), (cx - S * .028 * sc, cy + S * .018 * sc), (cx + S * .028 * sc, cy + S * .018 * sc), (cx + S * .05 * sc, cy + S * .075 * sc)]
    pts = [(cx + (px - cx) * math.cos(sw) - (py - cy) * math.sin(sw), cy + (px - cx) * math.sin(sw) + (py - cy) * math.cos(sw)) for px, py in pts]
    d.polygon(pts, fill=(222, 178, 64), outline=INK)
    d.line(pts + [pts[0]], fill=INK, width=_lw(S))


def table(d, o, x, S, gy, t, WS):
    """A plain wooden table with a mug and a loaf."""
    sc = o.get("scale", 1)
    w, h = S * .15 * sc, S * .115 * sc
    _rect(d, x - w, gy - h, x + w, gy - h + S * .016, WOOD_L, _lw(S))
    for dx in (-.82, .82):
        _rect(d, x + w * dx - S * .008, gy - h + S * .016, x + w * dx + S * .008, gy, WOOD_D, 3)
    _rect(d, x - w * .5, gy - h - S * .035, x - w * .5 + S * .028, gy - h, (226, 196, 140), 3)
    d.arc([x - w * .5 + S * .018, gy - h - S * .03, x - w * .5 + S * .045, gy - h - S * .008], 270, 90, fill=INK, width=3)
    _ell(d, x + w * .35, gy - h - S * .012, S * .034, S * .015, (206, 150, 84), 3)


def stool(d, o, x, S, gy, t, WS):
    """A three-legged stool with a round seat and a cloth over it (the 'groom of the stool')."""
    sc = o.get("scale", 1)
    w, h = S * .045 * sc, S * .075 * sc
    for dx in (-.8, 0, .8):
        d.line([(x + w * dx * .85, gy), (x + w * dx, gy - h)], fill=INK, width=_lw(S, .016))
    _ell(d, x, gy - h, w * 1.2, S * .012, WOOD_L, 4)
    d.polygon([(x - w * 1.2, gy - h), (x + w * 1.2, gy - h), (x + w * 1.0, gy - h + S * .03), (x + w * .3, gy - h + S * .022), (x - w * .3, gy - h + S * .034), (x - w * 1.0, gy - h + S * .026)], fill=(176, 60, 70), outline=INK)


def ladder(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    h = S * .30 * sc
    for dx in (-S * .025, S * .025):
        d.line([(x + dx - h * .05, gy - h), (x + dx + h * .08, gy)], fill=INK, width=_lw(S, .008))
    for k in range(1, 7):
        yy = gy - h * k / 7
        d.line([(x - S * .025 + (gy - yy) / h * h * .08 - h * .05 * (1 - (gy - yy) / h) * 0, yy), (x + S * .025 + (gy - yy) / h * h * .08, yy)], fill=WOOD_D, width=_lw(S, .007))


def swamp(d, o, x, S, gy, t, WS):
    """A green pond with reeds and a few leeches; ripples spread."""
    sc = o.get("scale", 1)
    rx = S * .38 * sc
    d.ellipse([x - rx, gy - S * .018, x + rx, gy + S * .075], fill=(70, 112, 72), outline=INK, width=_lw(S))
    for k in range(3):
        ph = (t * .4 + k / 3) % 1.0
        rr = rx * (.2 + .6 * ph)
        d.ellipse([x - rr, gy + S * .028 - rr * .11, x + rr, gy + S * .028 + rr * .11], outline=(150, 196, 150, int(200 * (1 - ph))), width=3)
    rng = random.Random(8)
    for k in range(5):                                                             # leeches wriggling in the water
        lx = x + (rng.random() - .5) * rx * 1.3
        ly = gy + S * (.02 + .03 * rng.random())
        pts = [(lx + i * S * .008, ly + math.sin(t * 4 + k + i) * S * .004) for i in range(6)]
        d.line(pts, fill=(34, 24, 26), width=_lw(S, .008))
    for k in range(7):                                                             # reeds
        rx0 = x - rx * 1.05 + (k / 6) * rx * 2.1
        d.line([(rx0, gy + S * .01), (rx0 + math.sin(t * 1.5 + k) * S * .006, gy - S * (.07 + .03 * (k % 3)))], fill=(60, 96, 52), width=_lw(S, .006))
        _ell(d, rx0 + math.sin(t * 1.5 + k) * S * .006, gy - S * (.07 + .03 * (k % 3)), S * .006, S * .016, (112, 78, 44), 3)


def cart(d, o, x, S, gy, t, WS):
    """A two-wheeled wooden cart piled with barrels."""
    sc = o.get("scale", 1)
    w, h = S * .18 * sc, S * .05 * sc
    for k, dx in enumerate((-.45, .1)):
        barrel(d, {"scale": sc * 0.5}, x + w * dx, S, gy - S * .06 * sc, t, WS)
    _rect(d, x - w, gy - S * .06 * sc, x + w, gy - S * .06 * sc + h * .6, WOOD, _lw(S))
    d.line([(x + w, gy - S * .045 * sc), (x + w * 1.45, gy - S * .02)], fill=INK, width=_lw(S, .007))
    wr = S * .05 * sc
    for dx in (-.55, .55):
        cxw = x + w * dx
        d.ellipse([cxw - wr, gy - wr * 2, cxw + wr, gy], fill=WOOD_L, outline=INK, width=_lw(S))
        spin = t * .8
        for k in range(6):
            a = spin + k * math.pi / 3
            d.line([(cxw, gy - wr), (cxw + math.cos(a) * wr, gy - wr + math.sin(a) * wr)], fill=INK, width=3)
        d.ellipse([cxw - wr * .15, gy - wr * 1.15, cxw + wr * .15, gy - wr * .85], fill=INK)


def stall(d, o, x, S, gy, t, WS):
    """A market stall with a striped awning and goods on the counter."""
    sc = o.get("scale", 1)
    w, h = S * .17 * sc, S * .20 * sc
    for dx in (-1, 1):
        d.line([(x + w * dx * .92, gy), (x + w * dx * .92, gy - h)], fill=INK, width=_lw(S, .008))
    _rect(d, x - w, gy - h * .45, x + w, gy - h * .35, WOOD, 3)
    _rect(d, x - w * .95, gy - h * .35, x + w * .95, gy, WOOD_D, _lw(S))
    n = 8
    for k in range(n):
        x0, x1 = x - w + k * 2 * w / n, x - w + (k + 1) * 2 * w / n
        d.polygon([(x0, gy - h), (x1, gy - h), (x1 + (x1 - x) * .05, gy - h * .78), (x0 + (x0 - x) * .05, gy - h * .78)], fill=(200, 60, 56) if k % 2 == 0 else (244, 238, 224), outline=INK)
    for k in range(5):
        _ell(d, x - w * .6 + k * w * .3, gy - h * .5, S * .018, S * .018, [(224, 70, 56), (240, 190, 60), (110, 170, 70)][k % 3], 3)


def pole(d, o, x, S, gy, t, WS):
    """A long pole leaning against the wall (the knocker-upper's tapping stick)."""
    sc = o.get("scale", 1)
    h = S * .36 * sc
    d.line([(x - S * .02, gy), (x + S * .05, gy - h)], fill=WOOD_D, width=_lw(S, .011))
    d.line([(x - S * .02, gy), (x + S * .05, gy - h)], fill=WOOD, width=_lw(S, .006))
    d.ellipse([x + S * .035, gy - h - S * .012, x + S * .065, gy - h + S * .012], fill=IRON, outline=INK, width=3)


_FIT = {"pit": 1.15, "barrel": 2.0, "bucket": 1.9, "basket": 1.9, "sack": 1.8, "chest": 1.6, "bed": 1.45, "bell": 1.0, "table": 1.8, "stool": 2.5, "ladder": 1.5, "swamp": 1.0,
        "cart": 1.6, "stall": 1.5, "pole": 1.5}                      # real-world proportions against a person who stands about .52 of the picture tall


def _fit(name, fn):
    k = _FIT.get(name, 1.0)
    return lambda d, o, x, S, gy, t, WS: fn(d, {**o, "scale": o.get("scale", 1) * k}, x, S, gy, t, WS)


_RAW = {"pit": pit, "barrel": barrel, "bucket": bucket, "basket": basket, "sack": sack, "chest": chest, "bed": bed, "bell": bell, "table": table, "stool": stool,
        "ladder": ladder, "swamp": swamp, "cart": cart, "stall": stall, "pole": pole}
DRAW = {k: _fit(k, f) for k, f in _RAW.items()}
# where each prop likes to stand (fraction of the picture width); characters usually stand at .38 and .62
HOME_X = {"pit": .5, "barrel": .15, "bucket": .24, "basket": .78, "sack": .22, "chest": .82, "bed": .5, "bell": .82, "table": .5, "stool": .5,
          "ladder": .86, "swamp": .5, "cart": .2, "stall": .8, "pole": .9}
