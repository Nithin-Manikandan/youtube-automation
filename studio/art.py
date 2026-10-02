"""Hand-built scenery art for the stickman engine: layered, shaded, animated.

Every drawer is draw(d, o, x, S, gy, t, WS) with d an RGBA ImageDraw on the supersampled canvas,
x the object's centre in pixels, S the canvas height, gy the ground line, t the scene time, WS the canvas width."""
import math
import random

INK = (28, 28, 34)


def _clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def _rect(d, x0, y0, x1, y1, **kw):
    d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], **kw)


def _circle(d, cx, cy, r, fill, outline=None, width=0):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)


# ------------------------------------------------------------------------------------------------------ waves
def wave(d, o, x, S, gy, t, WS):
    dur = max(o.get("dur", 6.0), 1.0)
    u = _clamp(t / (dur * 0.8))
    cx = WS * (1.12 - 0.78 * u)                        # the wave sweeps in from the right
    h = S * (0.30 + 0.22 * u)
    L = S * 0.85
    rng = random.Random(4)

    def profile(r, k=1.0):
        # steep face on the left of the crest, long gentle back to the right
        if r < 0:
            return k * (1 - min(1.0, (r / 0.85) ** 2)) ** 1.15
        return k * math.exp(-(r / 1.2) ** 2) ** 0.8

    for layer, (col, off, k) in enumerate(((( 38, 92, 150), 0.55, 0.78), ((48, 118, 178), 0.2, 0.92), ((74, 150, 204), 0.0, 1.0))):
        pts = [(cx + off * L - 0.95 * L, gy)]
        for i in range(0, 61):
            r = -0.95 + 2.8 * i / 60
            pts.append((cx + off * L + r * L, gy - h * profile(r, k) + math.sin(t * 2 + i * .6 + layer) * S * .004))
        pts.append((cx + off * L + 1.85 * L, gy))
        d.polygon(pts, fill=col)
    # curling lip: a lighter crescent tucked under the crest
    crest = [(cx - 0.05 * L - 0.55 * L * math.cos(a) * 0.45, gy - h * 0.98 + 0.22 * h * math.sin(a)) for a in [i * math.pi / 14 for i in range(0, 15)]]
    d.polygon(crest + [(cx + 0.1 * L, gy - h * 0.9)], fill=(126, 188, 226))
    d.line([(px, py) for px, py in crest], fill=(236, 248, 255), width=max(4, int(S * 0.008)))
    # foam: clusters riding the crest line and flung forward
    for i in range(70):
        r = -0.62 + rng.random() * 0.75
        fy = gy - h * profile(r) - rng.random() * h * 0.06
        fx = cx + r * L + math.sin(t * 3 + i) * S * 0.006
        _circle(d, fx, fy + math.sin(t * 4 + i * 1.7) * S * 0.006, S * (0.006 + 0.012 * rng.random()), (246, 252, 255, 235))
    for i in range(36):                                  # spray thrown ahead of the crest
        a = rng.random()
        age = (t * 0.9 + a) % 1.0
        sx = cx - 0.45 * L - age * L * 0.45 * (0.4 + rng.random())
        sy = gy - h * 0.98 - math.sin(age * math.pi) * h * 0.28 + age * h * 0.2
        _circle(d, sx, sy, S * 0.005 * (1.2 - age), (240, 250, 255, int(210 * (1 - age))))
    d.rectangle([0, gy - 3, WS, gy + S], fill=(44, 108, 168, 0))


# --------------------------------------------------------------------------------------------------- explosion
def explosion(d, o, x, S, gy, t, WS):
    t0 = o.get("t0", 0.4)
    T = 2.6
    if not (t0 <= t <= t0 + T):
        return
    u = (t - t0) / T
    rng = random.Random(int(t0 * 100) + 11)
    ease = 1 - (1 - min(1.0, u * 1.8)) ** 3
    R = S * 0.17 * (0.35 + ease)
    cy = gy - R * 0.9
    from . import stick as _st
    if _st.INK_STYLE:                                     # a cartoon starburst: flat colours, ink outline, puffs of smoke rising
        Rb = R * (1.0 if u < 0.55 else max(0.25, 1.0 - (u - 0.55) / 0.45 * 0.75))
        spin = u * 0.6
        def star(r, n, k, fill, w):
            pts = []
            for i in range(n * 2):
                rr = r * (1.0 if i % 2 == 0 else k) * (1 + 0.08 * math.sin(i * 2.3 + t * 9))
                a = spin + i * math.pi / n
                pts.append((x + math.cos(a) * rr, cy + math.sin(a) * rr * 0.9))
            d.polygon(pts, fill=fill, outline=INK)
            d.line(pts + [pts[0]], fill=INK, width=w, joint="curve")
        for i in range(5):                                # smoke puffs
            ph = min(1.0, max(0.0, u * 1.3 - i * 0.1))
            _circle(d, x + math.sin(i * 2.2) * S * 0.05, cy - R * 0.4 - ph * S * (0.10 + 0.04 * i), S * (0.03 + 0.012 * i) * (0.5 + ph), (110, 108, 114), INK, 4)
        star(Rb, 9, 0.62, (232, 86, 40), 6)
        star(Rb * 0.78, 9, 0.62, (255, 166, 40), 4)
        star(Rb * 0.5, 8, 0.65, (255, 226, 90), 4)
        _circle(d, x, cy, Rb * 0.2, (255, 252, 236), INK, 3)
        for i in range(10):                               # chunks flying out
            a = i * 0.63 + 0.2
            dist = ease * S * (0.16 + 0.07 * (i % 3))
            px_, py_ = x + math.cos(a) * dist, cy + math.sin(a) * dist * 0.8 + u * u * S * 0.12
            d.polygon([(px_, py_ - S * 0.012), (px_ + S * 0.01, py_ + S * 0.008), (px_ - S * 0.01, py_ + S * 0.008)], fill=INK)
        return
    if u > 0.12:                                          # rising smoke column + mushroom cap
        for i in range(16):
            f = i / 15
            rise = (u - 0.12) * S * 0.55 * (0.4 + f)
            sx = x + math.sin(i * 1.7 + u * 2) * S * 0.04 * (0.4 + f)
            _circle(d, sx, gy - R * 0.5 - rise * (0.3 + 0.7 * f), S * (0.035 + 0.05 * f) * (0.6 + u),
                    (70 + int(40 * f), 66 + int(36 * f), 70 + int(34 * f), int(215 * (1 - u) ** 0.7)))
    for i in range(14):                                   # lumpy fireball, not a perfect circle
        a = rng.random() * 6.283
        rr = R * (0.15 + 0.7 * rng.random())
        r_i = R * (0.35 + 0.35 * rng.random())
        c = ((255, 232, 150), (255, 170, 60), (232, 96, 36), (160, 50, 30))[min(3, int(u * 3 + rng.random() * 1.2))]
        _circle(d, x + math.cos(a) * rr, cy + math.sin(a) * rr * 0.75, r_i, c + (int(235 * (1 - u * 0.85)),))
    if u < 0.3:
        _circle(d, x, cy, R * 0.55, (255, 252, 230, int(255 * (1 - u / 0.3))))
    rw = S * 0.8 * u                                       # shockwave along the ground
    d.ellipse([x - rw, gy - rw * 0.12, x + rw, gy + rw * 0.12], outline=(255, 244, 220, int(220 * (1 - u))), width=max(3, int(S * 0.006)))
    for i in range(14):                                    # debris thrown on arcs
        a = (rng.random() - 0.5) * 2.4
        v = 0.3 + rng.random() * 0.5
        dx = math.sin(a) * v * u * S * 0.9
        dy = -(math.cos(a) * v * u * S * 1.1) + 0.9 * (u ** 2) * S * 0.5
        _circle(d, x + dx, gy - R * 0.4 + dy, S * 0.006, (40, 34, 30, int(255 * (1 - u))))


# ---------------------------------------------------------------------------------------------------- vehicles
def submarine(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * 0.62 * sc, S * 0.1 * sc
    yb = gy - h * 0.15 + math.sin(t * 1.2) * S * 0.004
    if o.get("afloat"):
        yb = gy - S * 0.26 + math.sin(t * 1.2) * S * 0.006
    x += (t - o.get("dur", 6) / 2) * S * 0.018
    d.ellipse([x - w / 2, yb - h * 1.05, x + w / 2, yb + h * 1.05], fill=(54, 64, 76), outline=INK, width=4)           # dark underside
    d.ellipse([x - w / 2 + 3, yb - h * 1.05 + 3, x + w / 2 - 3, yb + h * 0.55], fill=(92, 106, 122))                    # lit upper hull
    d.ellipse([x - w * .42, yb - h * .92, x + w * .38, yb - h * .45], fill=(126, 142, 158, 150))                         # highlight strip
    d.rectangle([x - w * .30, yb + h * .42, x + w * .34, yb + h * .5], fill=(196, 52, 44))                              # red waterline band
    sx = x - w * .02
    d.polygon([(sx - w * .1, yb - h * .7), (sx + w * .12, yb - h * .7), (sx + w * .09, yb - h * 1.75), (sx - w * .06, yb - h * 1.75)], fill=(82, 96, 112), outline=INK)
    for k in range(3):
        d.rectangle([sx - w * .045 + k * w * .04, yb - h * 1.5, sx - w * .02 + k * w * .04, yb - h * 1.38], fill=(250, 226, 150))
    d.line([(sx + w * .02, yb - h * 1.75), (sx + w * .02, yb - h * 2.3), (sx + w * .1, yb - h * 2.3)], fill=INK, width=6)
    d.line([(sx - w * .03, yb - h * 1.75), (sx - w * .03, yb - h * 2.1)], fill=INK, width=4)
    for i in range(5):
        cx = x - w * .30 + i * w * .13
        _circle(d, cx, yb - h * .1, S * .0075, (222, 232, 242), INK, 2)
    px = x - w / 2 - w * .015                                # spinning propeller + fins
    d.polygon([(px + w * .05, yb - h * .5), (px - w * .03, yb - h * 1.15), (px + w * .09, yb - h * .75)], fill=(70, 82, 96), outline=INK)
    ang = t * 14
    for k in range(3):
        a = ang + k * 2.094
        d.line([(px, yb), (px + math.cos(a) * h * .22, yb + math.sin(a) * h * .75)], fill=(40, 44, 52), width=6)
    d.polygon([(x + w * .22, yb + h * .6), (x + w * .3, yb + h * .6), (x + w * .26, yb + h * 1.05)], fill=(70, 82, 96), outline=INK)
    if o.get("afloat"):                                       # wake of bubbles behind the stern
        for i in range(10):
            age = (t * 0.7 + i / 10) % 1.0
            _circle(d, px - age * w * .35, yb + math.sin(i * 2 + t * 4) * h * .3, S * 0.004 + S * 0.007 * age, (226, 244, 255, int(170 * (1 - age))), (240, 250, 255, int(200 * (1 - age))), 1)


def warship(d, o, x, S, gy, t, WS):
    sc = o.get("scale", 1)
    w, h = S * 0.66 * sc, S * 0.075 * sc
    yb = gy - S * 0.004 + math.sin(t * 1.1) * S * 0.004
    x += (t - o.get("dur", 6) / 2) * S * 0.02
    d.polygon([(x - w * .5, yb - h * 1.1), (x + w * .46, yb - h * 1.1), (x + w * .56, yb - h * 1.45), (x + w * .5, yb), (x - w * .42, yb)], fill=(118, 128, 138), outline=INK)
    d.polygon([(x - w * .42, yb), (x + w * .5, yb), (x + w * .49, yb - h * .35), (x - w * .45, yb - h * .35)], fill=(176, 56, 48))       # red boot-topping
    d.line([(x - w * .5, yb - h * .7), (x + w * .52, yb - h * .7)], fill=(150, 160, 168), width=3)
    d.rectangle([x - w * .16, yb - h * 2.5, x + w * .16, yb - h * 1.1], fill=(150, 160, 168), outline=INK, width=3)
    d.rectangle([x - w * .1, yb - h * 3.3, x + w * .08, yb - h * 2.5], fill=(160, 170, 178), outline=INK, width=3)
    for k in range(4):
        d.rectangle([x - w * .12 + k * w * .06, yb - h * 2.15, x - w * .09 + k * w * .06, yb - h * 1.95], fill=(70, 90, 110))
    d.rectangle([x + w * .02, yb - h * 4.0, x + w * .06, yb - h * 3.3], fill=(98, 106, 112), outline=INK, width=3)
    d.line([(x - w * .01, yb - h * 3.3), (x - w * .01, yb - h * 4.5)], fill=INK, width=4)
    ra = t * 3
    d.line([(x - w * .01 - math.cos(ra) * h * .6, yb - h * 4.5), (x - w * .01 + math.cos(ra) * h * .6, yb - h * 4.5)], fill=INK, width=5)
    d.rounded_rectangle([x + w * .24, yb - h * 1.9, x + w * .36, yb - h * 1.1], 14, fill=(138, 148, 156), outline=INK, width=3)
    d.line([(x + w * .34, yb - h * 1.65), (x + w * .48, yb - h * 1.8)], fill=INK, width=8)
    fl = math.sin(t * 6) * h * .08
    d.polygon([(x - w * .4, yb - h * 1.1), (x - w * .4, yb - h * 2.1), (x - w * .3, yb - h * 1.95 + fl), (x - w * .4, yb - h * 1.8)], fill=(220, 220, 224), outline=INK)
    for k in range(9):                                     # bow wave and wake
        age = (t * .8 + k / 9) % 1.0
        _circle(d, x - w * .42 - age * w * .35, yb + h * .05, S * (.006 + .012 * age), (240, 248, 255, int(210 * (1 - age))))
    _circle(d, x + w * .55, yb - h * .1, S * .012, (240, 248, 255, 220))


def plane(d, o, x, S, gy, t, WS):
    dur = max(o.get("dur", 6.0), 1.0)
    u = _clamp(t / dur)
    px = (-0.15 + 1.3 * u) * WS
    py = o.get("y", 0.24) * S * 1.1 + math.sin(t * 1.4) * S * 0.008 + (u - .5) * S * .06
    sc = 1.4 * o.get("scale", 1)
    L, H_ = S * 0.2 * sc, S * 0.032 * sc
    for k in range(14):                                    # contrail
        age = k / 14
        _circle(d, px - L * .5 - k * L * .12, py + k * 2, S * (.006 + .01 * age), (250, 250, 252, int(170 * (1 - age))))
    d.polygon([(px - L * .5, py - H_ * .3), (px - L * .56, py - H_ * 2.1), (px - L * .4, py - H_ * 2.1), (px - L * .3, py - H_ * .3)], fill=(170, 176, 182), outline=INK)
    d.ellipse([px - L * .5, py - H_, px + L * .5, py + H_], fill=(196, 202, 208), outline=INK, width=3)
    d.polygon([(px - L * .12, py), (px - L * .28, py + H_ * 3.2), (px - L * .02, py + H_ * 3.2), (px + L * .12, py)], fill=(160, 168, 176), outline=INK)
    d.polygon([(px - L * .1, py - H_ * .2), (px - L * .22, py - H_ * 2.3), (px, py - H_ * 2.3), (px + L * .1, py - H_ * .2)], fill=(176, 182, 190), outline=INK)
    d.ellipse([px + L * .2, py - H_ * .8, px + L * .36, py - H_ * .1], fill=(120, 170, 210), outline=INK, width=2)
    d.ellipse([px + L * .47, py - H_ * 1.6, px + L * .52, py + H_ * 1.6], fill=(60, 60, 66, 130))   # prop disc
    d.rectangle([px - L * .5, py + H_ * .1, px + L * .5, py + H_ * .24], fill=(196, 52, 44))


# ------------------------------------------------------------------------------------------------- underwater
def seascape(d, o, x, S, gy, t, WS):
    rng = random.Random(21)
    for i in range(5):                                      # light shafts from the surface
        bx = WS * (0.1 + 0.2 * i) + math.sin(t * .35 + i) * S * .05
        d.polygon([(bx, 0), (bx + S * .09, 0), (bx + S * .42 + math.sin(t * .3 + i) * S * .04, gy), (bx + S * .2, gy)], fill=(190, 240, 255, 26))
    for i in range(4):                                      # distant rock stacks
        rx = WS * (0.04 + 0.3 * i + rng.random() * 0.1)
        rw, rh = S * (.12 + rng.random() * .1), S * (.05 + rng.random() * .08)
        d.polygon([(rx, gy), (rx + rw * .2, gy - rh), (rx + rw * .55, gy - rh * .7), (rx + rw, gy)], fill=(70, 84, 82), outline=INK)
    for i, kx in enumerate((.04, .1, .17, .84, .91, .97)):  # swaying kelp
        for strand in range(2):
            px = WS * kx + strand * S * .02
            hgt = S * (.28 + .1 * ((i * 3 + strand) % 3) / 2)
            pts = [(px + math.sin(t * 1.3 + i + k * .5) * S * .012 * k, gy - hgt * k / 8) for k in range(9)]
            d.line(pts, fill=(52, 130, 84), width=int(S * .012), joint="curve")
    for sch in range(2):                                    # fish schools crossing
        dirn = 1 if sch == 0 else -1
        base = ((t * (.05 + .02 * sch) + sch * .45) % 1.3) - .15
        bx = (base if dirn > 0 else 1 - base) * WS
        by = S * (.38 + .17 * sch)
        for f in range(9):
            fx = bx - dirn * (f % 3) * S * .05 - dirn * (f // 3) * S * .02
            fy = by + (f // 3 - 1) * S * .035 + math.sin(t * 2.3 + f) * S * .01
            fl = S * .018
            col = (232, 196, 90) if sch == 0 else (180, 210, 226)
            d.ellipse([fx - fl, fy - fl * .5, fx + fl, fy + fl * .5], fill=col)
            tw = math.sin(t * 12 + f) * fl * .3
            d.polygon([(fx - dirn * fl, fy), (fx - dirn * fl * 1.7, fy - fl * .6 + tw), (fx - dirn * fl * 1.7, fy + fl * .6 + tw)], fill=col)
    for i in range(40):                                     # marine snow
        sx = (rng.random() * WS + math.sin(t * .4 + i) * 8) % WS
        sy = (rng.random() * S + t * S * 0.03 * (0.4 + rng.random())) % S
        _circle(d, sx, sy, 2 + 3 * rng.random(), (220, 240, 250, 90))


def crowd(d, o, x, S, gy, t, WS):
    """Two ranks of small marching figures, charging in from the edges; ancient armies carry spears, modern ones rifles."""
    dur = max(o.get("dur", 6.0), 1.0)
    u = _clamp(t / (dur * 0.85))
    sides = ((1, (170, 54, 46)), (-1, (52, 96, 170))) if not o.get("one_side") else ((1, (170, 54, 46)),)
    for sd, col in sides:
        rng = random.Random(7 + sd)
        for row in range(5):
            for i in range(8):
                depth = 1 - row * 0.1
                h = S * 0.2 * depth
                lane = (i + (0.5 if row % 2 else 0) + rng.random() * .5) / 7.0
                start = WS * (0.97 + 0.3 * lane) if sd > 0 else WS * (0.03 - 0.3 * lane)
                goal = WS * (0.54 + 0.34 * lane + 0.012 * row) if sd > 0 else WS * (0.46 - 0.34 * lane - 0.012 * row)
                e = u ** 0.7
                fx = start + (goal - start) * e
                moving = u < 0.995
                ph = t * 9 + i * 1.3 + row
                bob = abs(math.sin(ph)) * h * 0.04 if moving else 0
                fy = gy - row * S * 0.03 - bob
                face = -sd
                lw = max(3, int(h * 0.04))
                hip = (fx, fy - h * 0.46)
                sh = (fx, fy - h * 0.8)
                for k in (-1, 1):                                    # legs
                    sw = math.sin(ph + (0 if k < 0 else math.pi)) * (h * 0.2 if moving else 0)
                    d.line([hip, (hip[0] + sw * face, fy)], fill=INK, width=lw)
                d.polygon([(sh[0] - h * .09, sh[1]), (sh[0] + h * .09, sh[1]), (hip[0] + h * .07, hip[1]), (hip[0] - h * .07, hip[1])], fill=col, outline=INK)
                _circle(d, sh[0], sh[1] - h * .1, h * .1, (240, 214, 184), INK, 2)
                d.rectangle([sh[0] - h * .1, sh[1] - h * .2, sh[0] + h * .1, sh[1] - h * .12], fill=(60, 60, 66))
                tip_y = sh[1] - h * (0.45 if not o.get("modern") else 0.0)
                hand = (sh[0] + face * h * .14, sh[1] + h * .12)
                d.line([(sh[0], sh[1] + h * .04), hand], fill=INK, width=lw)
                if o.get("modern"):
                    d.line([(hand[0] - face * h * .1, hand[1] + h * .02), (hand[0] + face * h * .3, hand[1] - h * .04)], fill=(70, 62, 56), width=lw + 1)
                else:
                    d.line([(hand[0], hand[1] + h * .2), (hand[0] + face * h * .1, tip_y)], fill=(120, 96, 70), width=lw)
                    d.polygon([(hand[0] + face * h * .1, tip_y), (hand[0] + face * h * .07, tip_y + h * .08), (hand[0] + face * h * .13, tip_y + h * .08)], fill=(200, 204, 210))
    if u > 0.95:                                               # the lines collide in a haze of dust
        for k in range(12):
            _circle(d, WS * .5 + math.sin(k * 2.1 + t * 5) * S * .15, gy - S * (.02 + .06 * (k % 4)), S * (.03 + .015 * (k % 3)), (196, 176, 146, 46))


def ambient(d, bg, t, S, gy, WS):
    """Small constant background life so no scene is ever a still photograph."""
    rng = random.Random(33)
    if bg in ("city_day", "countryside", "sea", "palace", "desert", "forest", "snow", "battlefield"):
        for i in range(2):                                             # distant gulls / birds
            bx = ((t * (.03 + .01 * i) + i * .5) % 1.2 - .1) * WS
            by = S * (.18 + .08 * i) + math.sin(t * .8 + i) * S * .01
            for k in range(3 if i == 0 else 2):
                fx, fy = bx - k * S * .035, by + abs(k) * S * .012
                fl = math.sin(t * 7 + k + i * 2) * S * .008
                d.line([(fx - S * .012, fy + fl), (fx, fy), (fx + S * .012, fy + fl)], fill=(40, 44, 54, 190), width=3)
    if bg in ("city_day", "city_modern", "countryside", "palace", "desert"):                # tiny people going about their day, far away
        for i in range(5):
            dirn = 1 if i % 2 == 0 else -1
            fx = ((t * (.018 + .006 * (i % 3)) * dirn + i * .21) % 1.2 - .1) * WS
            fy = gy - S * (.004 + .006 * (i % 2))
            h = S * .075
            ph = t * 5 + i
            col = [(150, 90, 70), (70, 100, 150), (110, 130, 80), (140, 110, 60), (120, 80, 120)][i]
            d.line([(fx, fy - h * .45), (fx + math.sin(ph) * h * .12, fy)], fill=(40, 40, 46, 200), width=3)
            d.line([(fx, fy - h * .45), (fx - math.sin(ph) * h * .12, fy)], fill=(40, 40, 46, 200), width=3)
            d.rectangle([fx - h * .09, fy - h * .85, fx + h * .09, fy - h * .42], fill=col + (210,))
            _circle(d, fx, fy - h * .95, h * .1, (232, 205, 175, 230))
    if bg in ("night", "storm", "ashen", "volcanic", "city_modern", "space", "moon"):
        drift = bg in ("space", "moon")
        for i in range(36):                                            # twinkling stars / lights (they drift slowly in space, in three depth layers)
            sx, sy = rng.random() * WS, rng.random() * gy * .55
            if drift:
                sx = (sx - t * S * (.006 + .012 * (i % 3))) % WS
            a = 90 + 120 * (0.5 + 0.5 * math.sin(t * (1.5 + rng.random() * 2) + i))
            _circle(d, sx, sy, 2 + rng.random() * 2, (255, 248, 220, int(a)))
    if bg == "city_modern":
        for i in range(3):                                             # cars crossing the far street
            cx = ((t * (.06 + .02 * i) * (1 if i % 2 == 0 else -1) + i * .37) % 1.2 - .1) * WS
            cy = gy - S * .012
            d.rounded_rectangle([cx - S * .04, cy - S * .022, cx + S * .04, cy], 8, fill=(60 + 50 * i, 70, 90, 220))
            d.rectangle([cx - S * .02, cy - S * .036, cx + S * .02, cy - S * .02], fill=(110, 130, 150, 220))
    if bg == "submarine_interior":
        for i, lx in enumerate((.18, .5, .82)):                        # warning lamps pulsing, steam venting
            on = 0.5 + 0.5 * math.sin(t * 3.2 + i * 2)
            _circle(d, WS * lx, S * .1, S * .014, (255, 70, 50, int(90 + 160 * on)))
            _circle(d, WS * lx, S * .1, S * .035, (255, 70, 50, int(50 * on)))
        for i in range(10):
            age = (t * .45 + i / 10) % 1.0
            _circle(d, WS * (.08 + .84 * ((i * 7) % 10) / 10) + math.sin(age * 6 + i) * 10, gy - S * .5 * age, S * (.012 + .02 * age), (220, 226, 230, int(70 * (1 - age))))
    if bg in ("sea", "storm", "harbor"):
        for i in range(14):                                            # sparkles / whitecaps on the water
            wx = (rng.random() * WS + t * S * .02 * (1 + i % 3)) % WS
            wy = gy + S * .01 + rng.random() * (S - gy) * .8
            d.line([(wx, wy), (wx + S * .03, wy)], fill=(235, 246, 255, int(60 + 80 * abs(math.sin(t * 2 + i)))), width=3)


def interior(d, o, x, S, gy, t, WS):
    """Inside a submarine: riveted wall, bunks, a blinking control panel, valve wheels, sagging cables and swaying lamp light."""
    for i in range(1, 9):                                              # wall seams and rivets
        sx = WS * i / 9
        d.line([(sx, S * .06), (sx, gy)], fill=(20, 26, 28, 120), width=4)
        for k in range(8):
            _circle(d, sx + 10, S * .12 + k * (gy - S * .12) / 8, 4, (90, 100, 100, 150))
    for i, lx in enumerate((.2, .5, .8)):                              # swinging lamp cones
        sway = math.sin(t * .9 + i * 1.7) * S * .03
        d.polygon([(WS * lx - 8, S * .055), (WS * lx + 8, S * .055), (WS * lx + S * .2 + sway, gy), (WS * lx - S * .2 + sway, gy)], fill=(255, 226, 150, 20))
        _circle(d, WS * lx, S * .06, S * .012, (255, 236, 180, 255))
    for i, (cx0, cx1) in enumerate(((.12, .34), (.46, .7), (.62, .9))):    # cables sagging along the ceiling
        pts = [(WS * (cx0 + (cx1 - cx0) * k / 12), S * (.19 + .035 * math.sin(math.pi * k / 12) + .004 * math.sin(t * 1.4 + k + i))) for k in range(13)]
        d.line(pts, fill=(30, 32, 34, 220), width=int(S * .008))
    for bx, bw in ((0.0, .13),):                                       # bunks on the left edge
        for lvl in range(2):
            y0 = gy - S * (.2 + .2 * lvl)
            d.rectangle([WS * bx, y0, WS * (bx + bw), y0 + S * .045], fill=(96, 84, 66), outline=INK, width=3)
            d.rectangle([WS * bx, y0 - S * .035, WS * (bx + bw) * .9, y0], fill=(120, 134, 150), outline=INK, width=3)
        d.line([(WS * (bx + bw), gy), (WS * (bx + bw), gy - S * .42)], fill=INK, width=6)
    px0 = WS * .87                                                     # control panel on the right
    d.rectangle([px0, gy - S * .5, WS, gy], fill=(52, 62, 66), outline=INK, width=4)
    for r in range(4):
        for c in range(3):
            on = (math.sin(t * (1.5 + (r * 3 + c) % 4) + r + c) > 0.1)
            col = [(110, 230, 120), (255, 190, 60), (255, 80, 70)][(r + c) % 3] + ((255,) if on else (60,))
            _circle(d, px0 + S * (.03 + .035 * c), gy - S * (.44 - .045 * r), S * .011, col)
    for k in range(2):
        cx, cy = px0 + S * (.045 + .055 * k), gy - S * .22
        _circle(d, cx, cy, S * .03, (230, 232, 226), INK, 3)
        a = -2.2 + 1.6 * (0.5 + 0.5 * math.sin(t * .8 + k * 2))
        d.line([(cx, cy), (cx + math.cos(a) * S * .024, cy + math.sin(a) * S * .024)], fill=(200, 40, 34), width=4)
    for k, vx in enumerate((.3, .7)):                                  # valve wheels
        cx, cy = WS * vx, gy - S * .42
        _circle(d, cx, cy, S * .045, None, (110, 118, 120), 8)
        for a in range(3):
            ang = a * 1.047 + (0.2 if k else 0)
            d.line([(cx - math.cos(ang) * S * .045, cy - math.sin(ang) * S * .045), (cx + math.cos(ang) * S * .045, cy + math.sin(ang) * S * .045)], fill=(110, 118, 120), width=6)
        d.rectangle([cx - 7, cy + S * .04, cx + 7, gy], fill=(80, 88, 90))


def liner(d, o, x, S, gy, t, WS):
    """A four-funnel passenger liner: black hull, white decks with window rows, buff funnels, drifting smoke, lifeboats."""
    sc = o.get("scale", 1)
    w, h = S * 0.72 * sc, S * 0.07 * sc
    yb = gy - S * 0.004 + math.sin(t * .9) * S * 0.004
    x += (t - o.get("dur", 6) / 2) * S * 0.015
    d.polygon([(x - w * .5, yb - h * 1.15), (x + w * .5, yb - h * 1.15), (x + w * .6, yb - h * 1.65), (x + w * .5, yb), (x - w * .46, yb)], fill=(24, 28, 40), outline=INK)
    _rect(d, x - w * .48, yb - h * .55, x + w * .5, yb - h * .45, fill=(176, 56, 48))
    for k in range(3):                                                  # tiered white superstructure
        tw = w * (.84 - .13 * k)
        y0 = yb - h * (1.15 + 1.0 * (k + 1))
        _rect(d, x - tw / 2 - w * .02, y0, x + tw / 2 - w * .02, y0 + h * 1.0, fill=(240, 240, 234), outline=INK, width=3)
        for q in range(int(tw / (S * .022))):
            _rect(d, x - tw / 2 + q * S * .022, y0 + h * .3, x - tw / 2 + q * S * .022 + S * .012, y0 + h * .55, fill=(70, 96, 130))
    top = yb - h * 4.15
    for k in range(4):                                                   # funnels with smoke
        fx = x - w * .22 + k * w * .14
        d.polygon([(fx - w * .035, top), (fx + w * .035, top), (fx + w * .03, top - h * 1.7), (fx - w * .03, top - h * 1.7)], fill=(222, 170, 80), outline=INK)
        _rect(d, fx - w * .03, top - h * 1.7, fx + w * .03, top - h * 1.95, fill=INK)
        for j in range(4):
            age = (t * .5 + j / 4 + k * .13) % 1.0
            _circle(d, fx + age * w * .25, top - h * 2.0 - age * h * 1.8, S * (.012 + .02 * age), (80, 80, 86, int(150 * (1 - age))))
    for mx in (x - w * .38, x + w * .36):
        d.line([(mx, top + h * .6), (mx, top - h * 1.5)], fill=INK, width=4)
    for q in range(8):                                                   # lifeboats
        d.ellipse([x - w * .3 + q * w * .075, yb - h * 2.2, x - w * .3 + q * w * .075 + S * .02, yb - h * 2.2 + S * .008], fill=(200, 120, 60), outline=INK, width=2)
    for k in range(8):
        age = (t * .7 + k / 8) % 1.0
        _circle(d, x + w * .56 + age * S * .02, yb - h * .1, S * (.006 + .01 * age), (240, 248, 255, int(210 * (1 - age))))


def airship(d, o, x, S, gy, t, WS):
    """A rigid airship: silver envelope with girder rings, tail fins, gondola and propeller pods; burning tail and nose-up tilt when o['burning']."""
    sc = o.get("scale", 1)
    L, H_ = S * 0.66 * sc, S * 0.082 * sc
    burning = bool(o.get("burning"))
    cx = WS * .5 + math.sin(t * .35) * S * .02 + (t - o.get("dur", 6) / 2) * S * .012
    cy = S * (.16 if not burning else .19) + math.sin(t * .8) * S * .008
    tilt = -.18 if burning else 0.0
    def P(px, py):
        c_, s_ = math.cos(tilt), math.sin(tilt)
        return (cx + px * c_ - py * s_, cy + px * s_ + py * c_)
    pts = [P(math.cos(a / 24 * 6.2832) * L / 2, math.sin(a / 24 * 6.2832) * H_) for a in range(24)]
    for sgn in (-1, 1):                                                  # tail fins
        d.polygon([P(-L * .5, 0), P(-L * .5 - L * .06, sgn * H_ * 1.55), P(-L * .34, sgn * H_ * .8)], fill=(168, 174, 182), outline=INK)
    d.polygon([P(-L * .5, 0), P(-L * .56, -H_ * 1.5), P(-L * .34, -H_ * .8)], fill=(176, 182, 190), outline=INK)
    d.polygon(pts, fill=(196, 202, 210), outline=INK)
    for k in range(-5, 6):                                               # girder rings
        a = k / 6 * 1.35
        xx = math.cos(a + 1.5708) * 0
        px_ = k * L * .082
        hh = H_ * math.sqrt(max(0.0, 1 - (px_ / (L / 2)) ** 2))
        d.line([P(px_, -hh), P(px_, hh)], fill=(150, 158, 168), width=2)
    d.line([P(-L * .48, -H_ * .12), P(L * .48, -H_ * .12)], fill=(150, 158, 168), width=2)
    d.polygon([P(-L * .45, H_ * .62), P(L * .45, H_ * .62), P(L * .4, H_ * .78), P(-L * .4, H_ * .78)], fill=(150, 158, 168))
    d.polygon([P(-L * .1, H_ * .95), P(L * .12, H_ * .95), P(L * .1, H_ * 1.35), P(-L * .08, H_ * 1.35)], fill=(88, 92, 100), outline=INK)   # gondola
    for px_ in (-L * .22, L * .2):
        _circle(d, *P(px_, H_ * 1.05), S * .012, (70, 74, 82), INK, 2)
    if burning:
        rng = random.Random(5)
        for k in range(16):
            age = (t * .9 + k / 16) % 1.0
            fx, fy = P(-L * (.3 + .2 * rng.random()), -H_ * (.2 + .5 * rng.random()))
            _circle(d, fx, fy - age * S * .1, S * (.02 + .035 * (1 - age)), (255, int(150 - 90 * age), 30, int(230 * (1 - age * .6))))
        for k in range(10):
            age = (t * .5 + k / 10) % 1.0
            fx, fy = P(-L * .4, -H_ * .3)
            _circle(d, fx - age * S * .05, fy - age * S * .22, S * (.03 + .04 * age), (60, 58, 62, int(160 * (1 - age))))


def iceberg(d, o, x, S, gy, t, WS):
    """A jagged berg: lit white face, blue shadow face, and the huge pale mass hidden under the surface."""
    sc = o.get("scale", 1)
    w, h = S * 0.55 * sc, S * 0.34 * sc
    bob = math.sin(t * .8) * S * .004
    pts = [(-.5, 0), (-.36, -.42), (-.22, -.3), (-.1, -.78), (.04, -.52), (.16, -1.0), (.3, -.6), (.42, -.7), (.5, 0)]
    d.polygon([(x + px * w, gy + py * h + bob + S * .004) for px, py in [(-.5, 0), (-.4, .4), (-.15, .75), (.1, .7), (.4, .45), (.5, 0)]], fill=(150, 200, 230, 90))
    d.polygon([(x + px * w, gy + py * h + bob) for px, py in pts], fill=(236, 246, 252), outline=(90, 130, 160))
    d.polygon([(x + px * w, gy + py * h + bob) for px, py in [(.04, -.52), (.16, -1.0), (.3, -.6), (.42, -.7), (.5, 0), (.12, 0)]], fill=(168, 206, 232))
    d.line([(x + .16 * w, gy - h + bob), (x + .12 * w, gy + bob)], fill=(120, 164, 196), width=4)
    for k in range(6):
        age = (t * .4 + k / 6) % 1.0
        _circle(d, x + (-.4 + age * .8) * w, gy + S * .004, S * .008 * (1 - age) + 2, (240, 250, 255, int(200 * (1 - age))))


def burning_town(d, o, x, S, gy, t, WS):
    """A row of timber houses far back, engulfed: tall flames, glowing windows, rolling smoke and an orange sky glow."""
    sc = o.get("scale", .6)
    rng = random.Random(17)
    d.polygon([(0, gy), (WS, gy), (WS, gy - S * .42), (0, gy - S * .42)], fill=(255, 110, 30, 26))             # sky glow
    xs = [WS * (.06 + .11 * i) + rng.random() * S * .02 for i in range(9)]
    for i, hx in enumerate(xs):
        hw, hh = S * (.07 + .02 * rng.random()) * sc * 2.6, S * (.08 + .05 * rng.random()) * sc * 2.4
        top = gy - hh
        d.polygon([(hx - hw / 2, gy), (hx + hw / 2, gy), (hx + hw / 2, top), (hx, top - hh * .5), (hx - hw / 2, top)], fill=(58, 44, 40, 235), outline=(20, 16, 16))
        for wx in (-.2, .2):
            on = .6 + .4 * math.sin(t * 5 + i + wx * 9)
            _rect(d, hx + wx * hw - hw * .07, gy - hh * .6, hx + wx * hw + hw * .07, gy - hh * .35, fill=(255, int(150 + 80 * on), 40, 255))
        for k in range(4):                                                 # flames licking up the roofline
            ph = t * (6 + k) + i * 1.7 + k
            fh = hh * (.55 + .35 * math.sin(ph) ** 2 + .15 * (k % 2))
            fx = hx - hw * .35 + k * hw * .23
            d.polygon([(fx - hw * .13, top + hh * .2), (fx + hw * .13, top + hh * .2), (fx + math.sin(ph * 1.3) * hw * .08, top - fh)], fill=(255, 120, 30, 235))
            d.polygon([(fx - hw * .07, top + hh * .2), (fx + hw * .07, top + hh * .2), (fx + math.sin(ph * 1.3) * hw * .05, top - fh * .6)], fill=(255, 220, 90, 245))
        for j in range(5):                                                 # smoke columns
            age = (t * .3 + j / 5 + i * .11) % 1.0
            _circle(d, hx + math.sin(age * 5 + i) * hw * .6 + age * S * .04, top - hh * .3 - age * S * .45, S * (.02 + .045 * age), (60, 56, 58, int(170 * (1 - age) ** .8)))
    for k in range(40):                                                    # embers drifting up
        age = (t * .4 + k / 40) % 1.0
        _circle(d, WS * (k * .0251 % 1) + math.sin(t + k) * S * .02, gy - age * S * .5, 2 + 2 * (k % 3), (255, int(160 + 80 * (1 - age)), 50, int(230 * (1 - age))))


def ship(d, o, x, S, gy, t, WS):
    """A wooden sailing ship (carrack): curved hull, two masts with billowing sails, banner, oars dipping, bow wave."""
    sc = o.get("scale", 1)
    w, h = S * 0.5 * sc, S * 0.07 * sc
    yb = gy - S * 0.006 + math.sin(t * 1.0) * S * 0.006
    x += (t - o.get("dur", 6) / 2) * S * 0.014
    d.polygon([(x - w * .5, yb - h * 1.6), (x - w * .32, yb - h * 1.25), (x + w * .38, yb - h * 1.25), (x + w * .55, yb - h * 1.7), (x + w * .44, yb - h * .35), (x + w * .1, yb), (x - w * .36, yb)], fill=(104, 70, 44), outline=INK)
    d.line([(x - w * .4, yb - h * .95), (x + w * .45, yb - h * .95)], fill=(176, 130, 70), width=5)
    for q in range(6):
        _circle(d, x - w * .3 + q * w * .12, yb - h * .62, S * .006, (30, 24, 20))
    bil = math.sin(t * 1.4) * S * .006
    for mx, mh, sw in ((x - w * .12, h * 4.8, w * .3), (x + w * .22, h * 4.0, w * .25)):
        d.line([(mx, yb - h * 1.25), (mx, yb - h * 1.25 - mh)], fill=(70, 50, 34), width=7)
        top = yb - h * 1.25 - mh
        d.polygon([(mx - sw / 2, top + mh * .08), (mx + sw / 2, top + mh * .08), (mx + sw / 2 + bil, top + mh * .55), (mx, top + mh * .62 + bil), (mx - sw / 2 + bil, top + mh * .55)], fill=(240, 232, 214), outline=INK)
        d.line([(mx - sw / 2, top + mh * .08), (mx + sw / 2, top + mh * .08)], fill=(70, 50, 34), width=5)
        d.polygon([(mx, top), (mx + w * .1, top + h * .25 + bil), (mx, top + h * .5)], fill=(176, 56, 48))
    for q in range(5):                                                   # oars rowing
        ang = math.sin(t * 2.2 + q) * .3
        ox = x - w * .3 + q * w * .16
        d.line([(ox, yb - h * .6), (ox + math.sin(ang) * h * 1.2 - h * .5, yb + h * 1.0)], fill=(80, 56, 38), width=4)
    for k in range(8):
        age = (t * .7 + k / 8) % 1.0
        _circle(d, x + w * .55 + age * S * .02, yb - h * .1, S * (.006 + .01 * age), (240, 248, 255, int(210 * (1 - age))))


def _rot(pts, cx, cy, a):
    c, s_ = math.cos(a), math.sin(a)
    return [(cx + (px - cx) * c - (py - cy) * s_, cy + (px - cx) * s_ + (py - cy) * c) for px, py in pts]


def spacecraft(d, o, x, S, gy, t, WS):
    """Apollo-style stack: cone command module, cylindrical service module with engine bell, slow tumble, venting gas when damaged."""
    sc = o.get("scale", 1) * 1.7
    cx, cy = x + math.sin(t * .25) * S * .02, S * .47 + math.sin(t * .4) * S * .012
    a = math.sin(t * .3) * .12 + o.get("tilt", -.12)
    L, R = S * .30 * sc, S * .075 * sc
    from . import space
    roll = t * 0.55 + o.get("roll0", 0.0)
    sprite = space.spacecraft_sprite(L, R, roll, bool(o.get("venting")))
    space.paste_sprite(d, sprite, cx, cy, -math.degrees(a))
    wy = o.get("venting") or (int(t * 1.2) % 5 == 0)
    if o.get("venting"):                                                                              # oxygen streaming out of the side
        for k in range(26):
            age = (t * .8 + k / 26) % 1.0
            vx0, vy0 = cx - L * .1, cy + R * .8
            _circle(d, vx0 + math.sin(a) * 0 + age * L * .6 * (1 + .3 * math.sin(k)), vy0 + age * R * 5 * (.5 + (k % 5) * .2), S * (.006 + .025 * age), (235, 244, 255, int(210 * (1 - age))))
    for k in range(6):                                                                                # faint exhaust shimmer from the bell
        age = (t * 2 + k / 6) % 1.0
        d.line([_rot([(cx - L * .66 - age * L * .25, cy)], cx, cy, a)[0], _rot([(cx - L * .7 - age * L * .3, cy)], cx, cy, a)[0]], fill=(255, 210, 120, int(120 * (1 - age))), width=4)


def planet(d, o, x, S, gy, t, WS):
    kind = o.get("kind", "earth")
    r = S * .30 * o.get("scale", 1)
    cx, cy = WS * (.84 if o.get("side", 1) > 0 else .16), max(S * .22, r * 1.08)          # the whole globe stays inside the picture so no straight cut shows when the camera moves
    from . import space
    space.paste_sprite(d, space.planet_sprite(kind, r, t), cx, cy)


def capsule_interior(d, o, x, S, gy, t, WS):
    """Cramped command module: padded wall, two round windows onto stars and Earth, a dense blinking panel, warning light."""
    d.rectangle([0, 0, WS, gy], fill=(58, 64, 66))
    for i in range(1, 10):
        d.line([(WS * i / 10, S * .04), (WS * i / 10, gy)], fill=(34, 40, 42, 150), width=4)
    for wx in (.22, .78):
        _circle(d, WS * wx, S * .30, S * .13, (8, 10, 26), (150, 154, 158), 12)
        rng = random.Random(int(wx * 100))
        for k in range(26):
            _circle(d, WS * wx + (rng.random() - .5) * S * .22, S * .30 + (rng.random() - .5) * S * .22, 2 + 2 * rng.random() * (0.6 + .4 * math.sin(t * 2 + k)), (255, 250, 230, 220))
    _circle(d, WS * .78 + S * .06, S * .36, S * .06, (44, 110, 200), None)                        # Earth sliver in a window
    d.rectangle([0, gy - S * .22, WS, gy], fill=(44, 48, 50), outline=INK, width=4)                 # control panel band
    rng = random.Random(7)
    for r_ in range(3):
        for c in range(34):
            on = math.sin(t * (1 + rng.random() * 3) + c + r_) > (-0.1 if r_ < 2 else .5)
            col = ((120, 230, 130), (255, 190, 60), (255, 80, 70))[(c + r_) % 3] + ((255,) if on else (46,))
            _circle(d, WS * (.02 + c * .029), gy - S * (.17 - .05 * r_), S * .008, col)
    for k, dx in enumerate((.3, .5, .7)):                                                          # dials
        _circle(d, WS * dx, gy - S * .06, S * .026, (220, 224, 218), INK, 3)
        a_ = -2.2 + 1.7 * (.5 + .5 * math.sin(t * .7 + k))
        d.line([(WS * dx, gy - S * .06), (WS * dx + math.cos(a_) * S * .022, gy - S * .06 + math.sin(a_) * S * .022)], fill=(200, 40, 34), width=4)
    al = .5 + .5 * math.sin(t * 4)
    _circle(d, WS * .5, S * .08, S * .02, (255, 60, 40, int(70 + 185 * al)))                       # master alarm
    _circle(d, WS * .5, S * .08, S * .06, (255, 60, 40, int(40 * al)))


def mission_control(d, o, x, S, gy, t, WS):
    """Houston: a wall-size screen with live traces, rows of consoles with glowing monitors."""
    d.rectangle([0, 0, WS, gy], fill=(20, 26, 40))
    sx0, sx1, sy0, sy1 = WS * .12, WS * .88, S * .07, S * .42
    d.rectangle([sx0, sy0, sx1, sy1], fill=(8, 18, 34), outline=(90, 130, 180), width=6)
    for g in range(1, 5):
        d.line([(sx0, sy0 + (sy1 - sy0) * g / 5), (sx1, sy0 + (sy1 - sy0) * g / 5)], fill=(40, 70, 110, 140), width=2)
    for tr, col in enumerate(((90, 255, 150), (255, 200, 80), (120, 190, 255))):
        pts = [(sx0 + (sx1 - sx0) * k / 60, sy0 + (sy1 - sy0) * (.3 + .2 * tr) + math.sin(k * .5 + t * 2 + tr * 2) * S * .03 * (1 + .5 * math.sin(t * .3 + tr))) for k in range(61)]
        d.line(pts, fill=col + (230,), width=4)
    for row in range(3):                                                                              # console rows
        yy = gy - S * (.06 + .11 * row)
        for c in range(7 - row):
            cx = WS * (.08 + (c + .5 * (row % 2)) * (.86 / (7 - row)))
            w = S * (.17 - .02 * row)
            d.rectangle([cx - w / 2, yy - S * .045, cx + w / 2, yy + S * .015], fill=(36, 44, 58), outline=(12, 16, 24), width=3)
            on = .6 + .4 * math.sin(t * 2 + c + row)
            d.rectangle([cx - w * .35, yy - S * .035, cx + w * .35, yy - S * .008], fill=(60, int(150 + 80 * on), 200, 255))


def reactor(d, o, x, S, gy, t, WS):
    """Soviet-style nuclear plant: blocky turbine hall, the striped ventilation stack, a cooling tower, a faint green glow and a drifting plume."""
    sc = o.get("scale", 1)
    hw, hh = S * .42 * sc, S * .16 * sc
    d.rectangle([x - hw / 2, gy - hh, x + hw / 2, gy], fill=(142, 146, 150), outline=INK, width=4)                 # turbine hall
    for k in range(7):
        _rect(d, x - hw / 2 + hw * .06 + k * hw * .13, gy - hh * .7, x - hw / 2 + hw * .06 + k * hw * .13 + hw * .06, gy - hh * .35, fill=(70, 96, 120))
    rx, rh = x + hw * .3, S * .30 * sc
    d.rectangle([rx - hw * .18, gy - hh - rh * .5, rx + hw * .18, gy - hh], fill=(168, 170, 172), outline=INK, width=4)    # reactor block
    sx = x - hw * .72                                                                                              # red and white ventilation stack
    sh, sw = S * .72 * sc, S * .035 * sc
    d.polygon([(sx - sw * 1.1, gy), (sx + sw * 1.1, gy), (sx + sw * .6, gy - sh), (sx - sw * .6, gy - sh)], fill=(238, 238, 234), outline=INK)
    for k in range(4):
        y0 = gy - sh * (.18 + .22 * k)
        d.polygon([(sx - sw * (1.05 - .45 * (.18 + .22 * k)), y0), (sx + sw * (1.05 - .45 * (.18 + .22 * k)), y0), (sx + sw * (1.05 - .45 * (.18 + .22 * k + .1)), y0 - sh * .1), (sx - sw * (1.05 - .45 * (.18 + .22 * k + .1)), y0 - sh * .1)], fill=(196, 52, 44))
    tx, tw, th = x + hw * .95, S * .2 * sc, S * .42 * sc                                                          # cooling tower
    d.polygon([(tx - tw * .55, gy), (tx + tw * .55, gy), (tx + tw * .32, gy - th * .55), (tx + tw * .42, gy - th), (tx - tw * .42, gy - th), (tx - tw * .32, gy - th * .55)], fill=(184, 186, 188), outline=INK)
    for k in range(10):                                                                                           # plume
        age = (t * .22 + k / 10) % 1.0
        _circle(d, tx + math.sin(age * 4 + k) * tw * .3 + age * S * .1, gy - th - age * S * .5, S * (.03 + .06 * age), (210, 212, 214, int(170 * (1 - age))))
    for k in range(5):                                                                                            # eerie green glow over the reactor
        _circle(d, rx, gy - hh - rh * .2, S * (.1 + .04 * k) * sc, (120, 255, 140, 14 + int(8 * math.sin(t * 2))))


def tank(d, o, x, S, gy, t, WS):
    """A pressure tank: steel cylinder with domed ends, bands, a valve wheel, a gauge with a trembling needle and frost; vents gas when o['venting']."""
    sc = o.get("scale", 1)
    w, h = S * 0.30 * sc, S * 0.20 * sc
    yb = gy - S * 0.03
    for k in range(2):                                                    # legs
        _rect(d, x - w * (.34 - .6 * k), yb - h * .05, x - w * (.28 - .6 * k), yb + S * .03, fill=(70, 74, 84))
    d.rounded_rectangle([x - w / 2, yb - h, x + w / 2, yb], radius=h * .5, fill=(176, 184, 196), outline=INK, width=4)
    d.rounded_rectangle([x - w / 2 + w * .06, yb - h * .94, x + w / 2 - w * .06, yb - h * .62], radius=h * .2, fill=(214, 222, 232))
    for bx in (-.22, .22):
        _rect(d, x + bx * w - w * .02, yb - h, x + bx * w + w * .02, yb, fill=(96, 104, 118))
    gx, gy_ = x, yb - h - S * .02                                         # gauge on top
    _rect(d, gx - w * .02, gy_, gx + w * .02, yb - h, fill=(70, 74, 84))
    _circle(d, gx, gy_ - S * .035, S * .04, (246, 246, 240), INK, 3)
    ang = -2.2 + 1.6 * (0.5 + 0.5 * math.sin(t * 1.4)) + (math.sin(t * 30) * .08 if o.get("venting") else 0)
    d.line([(gx, gy_ - S * .035), (gx + math.cos(ang) * S * .032, gy_ - S * .035 + math.sin(ang) * S * .032)], fill=(200, 40, 40), width=4)
    _circle(d, x + w * .5, yb - h * .55, S * .022, (190, 60, 50), INK, 3)   # valve wheel
    if o.get("venting"):
        for k in range(9):
            age = (t * .9 + k / 9) % 1.0
            _circle(d, x + w * .5 + S * .02 + age * S * .22, yb - h * .55 - age * S * .12, S * (.012 + .03 * age), (230, 240, 250, int(190 * (1 - age))))


def parachute(d, o, x, S, gy, t, WS):
    """Three big striped canopies lowering a capsule on lines, swinging gently as it descends."""
    dur = max(o.get("dur", 6.0), 1.0)
    u = min(1.0, t / dur)
    sc = o.get("scale", 1)
    cy = S * (0.16 + 0.34 * u) + math.sin(t * 1.3) * S * .006
    sway = math.sin(t * 1.1) * S * .02
    cap = (x + sway * 1.6, cy + S * .30 * sc)
    for k, off in enumerate((-.19, 0.0, .19)):
        cx = x + off * S * sc + sway
        r = S * .12 * sc
        top = cy - S * (.04 if k == 1 else 0) * sc
        for i in range(6):                                              # alternating gores
            a0, a1 = math.pi + i * math.pi / 6, math.pi + (i + 1) * math.pi / 6
            pts = [(cx, top)] + [(cx + math.cos(a0 + (a1 - a0) * j / 4) * r, top + math.sin(a0 + (a1 - a0) * j / 4) * r * .8) for j in range(5)]
            d.polygon(pts, fill=(238, 118, 40) if i % 2 == 0 else (248, 246, 240), outline=INK)
        for sx_ in (-r, 0, r):
            d.line([(cx + sx_, top), (cap[0], cap[1] - S * .02)], fill=(60, 60, 66), width=2)
    d.polygon([(cap[0] - S * .045 * sc, cap[1] + S * .06 * sc), (cap[0] + S * .045 * sc, cap[1] + S * .06 * sc), (cap[0] + S * .02 * sc, cap[1] - S * .035 * sc), (cap[0] - S * .02 * sc, cap[1] - S * .035 * sc)], fill=(206, 198, 184), outline=INK)


def flag(d, o, x, S, gy, t, WS):
    """A tall pole with a flag rippling in the wind."""
    col = o.get("color")
    if not col or len(col) == 4 or tuple(col[:3]) == (255, 255, 255):
        col = (196, 57, 43)
    col = tuple(col[:3])
    sc = o.get("scale", 1)
    h = S * .52 * sc
    _rect(d, x - S * .006, gy - h, x + S * .006, gy, fill=(110, 80, 52), outline=INK, width=2)
    _circle(d, x, gy - h, S * .012, (226, 176, 40), INK, 2)
    w, hh = S * .26 * sc, S * .15 * sc
    top = gy - h + S * .02
    n = 14
    up, down = [], []
    for i in range(n + 1):
        u = i / n
        wave = math.sin(t * 4 - u * 5) * S * .018 * u
        up.append((x + S * .006 + u * w, top + wave))
        down.append((x + S * .006 + u * w, top + hh + wave))
    d.polygon(up + down[::-1], fill=col, outline=INK)
    mid = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(up, down)]
    d.line(mid[2:-2], fill=(248, 246, 238), width=max(3, int(S * .01)))


def cannon(d, o, x, S, gy, t, WS):
    """An old iron cannon on a wooden carriage; fires with a flash and smoke at o['t0']."""
    sc = o.get("scale", 1)
    f = -1 if o.get("flip") else 1
    L = S * .30 * sc
    cx, cy = x, gy - S * .07 * sc
    t0 = o.get("t0", 1.0)
    rec = 0.0
    if 0 <= t - t0 < .5:
        rec = math.exp(-(t - t0) * 7) * S * .02 * f
    ang = -.22
    bx, by = cx - f * L * .45 - rec, cy
    tx, ty = bx + f * math.cos(ang) * L, by + math.sin(ang) * L
    d.line([(bx, by), (tx, ty)], fill=(44, 46, 52), width=int(S * .055 * sc))
    d.line([(bx, by - S * .01), (tx, ty - S * .01)], fill=(86, 90, 100), width=int(S * .012 * sc))
    _circle(d, bx, by, S * .032 * sc, (44, 46, 52), INK, 3)
    d.polygon([(cx - S * .08 * sc, cy + S * .01), (cx + S * .08 * sc, cy + S * .01), (cx + S * .06 * sc, cy + S * .045 * sc), (cx - S * .06 * sc, cy + S * .045 * sc)], fill=(120, 84, 52), outline=INK)
    _circle(d, cx - f * S * .02, gy - S * .045 * sc, S * .05 * sc, (140, 100, 62), INK, 4)
    for k in range(6):
        a_ = k * math.pi / 3 + 0.3
        d.line([(cx - f * S * .02, gy - S * .045 * sc), (cx - f * S * .02 + math.cos(a_) * S * .05 * sc, gy - S * .045 * sc + math.sin(a_) * S * .05 * sc)], fill=INK, width=2)
    if 0 <= t - t0 < 1.4:
        u = (t - t0) / 1.4
        if u < .12:
            _circle(d, tx + f * S * .03, ty, S * (.05 + .06 * u / .12), (255, 220, 120, 235))
        for k in range(6):
            _circle(d, tx + f * (S * .04 + u * S * (.10 + .05 * k)), ty - u * S * .05 * k * .4, S * (.025 + .05 * u + .01 * k), (210, 210, 214, int(170 * (1 - u))))


def clock(d, o, x, S, gy, t, WS):
    """A big clock hanging in the sky of the scene: the second hand ticks, the hour hand creeps."""
    r = S * .15 * o.get("scale", 1)
    cy = S * .27
    _circle(d, x, cy, r * 1.08, (60, 52, 46), INK, 4)
    _circle(d, x, cy, r, (248, 244, 232), INK, 3)
    for k in range(12):
        a_ = k * math.pi / 6
        d.line([(x + math.sin(a_) * r * .84, cy - math.cos(a_) * r * .84), (x + math.sin(a_) * r * .94, cy - math.cos(a_) * r * .94)], fill=INK, width=4 if k % 3 == 0 else 2)
    sec = int(t * 2) / 2 * 6
    hr = 0.5 * (o.get("t0", 0) + t) * .3
    for ang, ln, wd, col in ((hr * 12, .5, 7, INK), (hr * 1.0 * 6, .72, 5, INK), (sec * 1.0, .82, 3, (200, 40, 40))):
        a_ = math.radians(ang)
        d.line([(x, cy), (x + math.sin(a_) * r * ln, cy - math.cos(a_) * r * ln)], fill=col, width=wd)
    _circle(d, x, cy, r * .06, (200, 40, 40), INK, 2)


DRAW = {"tank": tank, "parachute": parachute, "flag": flag, "cannon": cannon, "clock": clock, "airship": airship, "reactor": reactor, "spacecraft": spacecraft, "planet": planet, "capsule_interior": capsule_interior, "mission_control": mission_control, "ship": ship, "burning_town": burning_town, "liner": liner, "iceberg": iceberg, "interior": interior, "crowd": crowd, "wave": wave, "explosion": explosion, "submarine": submarine, "warship": warship, "plane": plane, "seascape": seascape}
from . import props as _props
DRAW.update(_props.DRAW)
