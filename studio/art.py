"""Hand-built scenery art for the stickman engine: layered, shaded, animated.

Every drawer is draw(d, o, x, S, gy, t, WS) with d an RGBA ImageDraw on the supersampled canvas,
x the object's centre in pixels, S the canvas height, gy the ground line, t the scene time, WS the canvas width."""
import math
import random

INK = (28, 28, 34)


def _clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


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
    if bg in ("night", "storm", "ashen", "volcanic", "city_modern"):
        for i in range(36):                                            # twinkling stars / lights
            sx, sy = rng.random() * WS, rng.random() * gy * .55
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
    if bg in ("sea", "storm"):
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


DRAW = {"interior": interior, "crowd": crowd, "wave": wave, "explosion": explosion, "submarine": submarine, "warship": warship, "plane": plane, "seascape": seascape}
