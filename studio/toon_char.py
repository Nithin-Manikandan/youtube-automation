"""Flat cartoon puppet: big expressive head, chunky outlined limbs, mitten hands, squash and stretch. Driven by the same pose channels and face extras as the other renderers,
so every acting beat, mocap clip and lip-sync cue carries over. Everything is drawn at the supersampled frame size with bold ink outlines and flat colours."""
import math

from PIL import ImageDraw

INK = (24, 22, 30)
SKIN = (255, 252, 246)
HAIRS = ((96, 62, 40), (52, 36, 30), (150, 98, 48), (176, 60, 40), (30, 30, 36))


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c[:3])


def _limb(d, pts, w_in, w_out, fill):
    """A chunky limb: ink round-capped stroke, then the fill stroke on top."""
    d.line(pts, fill=INK, width=int(w_out), joint="curve")
    for p in pts:
        d.ellipse([p[0] - w_out / 2, p[1] - w_out / 2, p[0] + w_out / 2, p[1] + w_out / 2], fill=INK)
    d.line(pts, fill=fill, width=int(w_in), joint="curve")
    for p in pts:
        d.ellipse([p[0] - w_in / 2, p[1] - w_in / 2, p[0] + w_in / 2, p[1] + w_in / 2], fill=fill)


def draw(a, img, t, gy, scene, ctx):
    from . import stick as K
    SS = K.SS
    S = a.unit * 1.2
    ent = min(1.0, t / 0.35)                                           # entrance pop: a quick overshoot so characters land with life
    pop = 1.0 + 0.07 * math.sin(ent * math.pi) * (1 - ent) * 2 if ent < 1.0 else 1.0
    S *= pop
    pose, xr, f, face = a.key_state(t)
    a._look_dx = 0.40 * f
    ex = a.extras(t, scene)
    pose = dict(pose)
    prev = a.key_state(max(0.0, t - 0.07))[0]
    for ch in ("a1", "a2", "b1", "b2"):
        pose[ch] = pose[ch] * 0.62 + prev[ch] * 0.38
    for ch, ph in (("a1", 0.0), ("b1", 1.9)):
        pose[ch] += 2.2 * math.sin(t * 1.3 + a.seed + ph)
    pose["torso"] = max(-5.0, min(4.0, pose["torso"]))                 # a cartoon body leans a little; deep bends make the heads collide and the body read as broken
    pose["head"] = max(-8.0, min(6.0, pose["head"]))
    if abs(pose["l1"] - pose["m1"]) < 38:                                  # standing: a relaxed, nearly straight stance; only real strides keep their swing
        pose["l1"] = max(-3.0, min(5.0, pose["l1"])); pose["m1"] = max(-5.0, min(3.0, pose["m1"]))
        pose["l2"] = max(-10.0, min(2.0, pose["l2"])); pose["m2"] = max(-10.0, min(2.0, pose["m2"]))
    else:
        pose["l1"] = max(-40.0, min(40.0, pose["l1"])); pose["m1"] = max(-40.0, min(40.0, pose["m1"]))
        pose["l2"] = max(-60.0, min(3.0, pose["l2"])); pose["m2"] = max(-60.0, min(3.0, pose["m2"]))
    x = xr * a.W * SS
    d = ImageDraw.Draw(img, "RGBA")
    Lb = dict(torso=0.27, upper=0.165, fore=0.15, thigh=0.19, shin=0.19)
    add = lambda p, q: (p[0] + q[0], p[1] + q[1])
    props = a.s.get("props", [])
    suit = "spacehelmet" in props
    shirt = a.s.get("tunic") or (210, 110, 52)
    if suit:
        shirt = (236, 238, 242)
    pants = _shade(shirt, 0.55) if not suit else (214, 218, 224)
    hair = a.s.get("hair") or HAIRS[a.seed % len(HAIRS)]
    wi, wo = S * 0.066, S * 0.066 + S * 0.019                        # limb fill / limb with outline
    l1 = K.seg((0, 0), pose["l1"], Lb["thigh"] * S, f); l2 = K.seg(l1, pose["l1"] + pose["l2"], Lb["shin"] * S, f)
    m1 = K.seg((0, 0), pose["m1"], Lb["thigh"] * S, f); m2 = K.seg(m1, pose["m1"] + pose["m2"], Lb["shin"] * S, f)
    drop = max(l2[1], m2[1])
    mouth_ = a.extras(t, scene).get("mouth", 0.0)
    bob = math.sin(t * 2.4 + a.seed) * S * 0.006 - mouth_ * S * 0.012      # a tiny bounce on every spoken syllable and a breath in between
    sway = math.sin(t * 0.9 + a.seed * 1.3) * S * 0.012
    hip = (x + sway, gy - drop - pose.get("lift", 0) * S + bob)
    tr = math.radians(pose["torso"])
    up = (math.sin(tr) * f, -math.cos(tr))
    tl = Lb["torso"] * S
    sh = (hip[0] + up[0] * tl * .92, hip[1] + up[1] * tl * .92)
    neck = (hip[0] + up[0] * tl, hip[1] + up[1] * tl)
    hr = math.radians(pose["torso"] + pose["head"])
    hup = (math.sin(hr) * f, -math.cos(hr))
    R = S * 0.150
    head = (neck[0] + hup[0] * R * .80, neck[1] + hup[1] * R * .80)
    spr = S * 0.080                                                     # shoulders sit either side of the body, so arms hang at the sides instead of in a clump in front
    shA, shB = (sh[0] + spr, sh[1] + S * .012), (sh[0] - spr, sh[1] + S * .012)
    if f < 0:
        shA, shB = shB, shA
    a1 = add(shA, K.seg((0, 0), pose["a1"] + 5 * f, Lb["upper"] * S, f)); a2 = add(a1, K.seg((0, 0), pose["a1"] + 5 * f + pose["a2"], Lb["fore"] * S, f))
    b1 = add(shB, K.seg((0, 0), pose["b1"] - 5 * f, Lb["upper"] * S, f)); b2 = add(b1, K.seg((0, 0), pose["b1"] - 5 * f + pose["b2"], Lb["fore"] * S, f))
    hsp = S * 0.062
    hipA, hipB = (hip[0] + hsp * f, hip[1]), (hip[0] - hsp * f, hip[1])
    sw = S * 0.15
    d.ellipse([x - sw, gy - S * 0.014, x + sw, gy + S * 0.02], fill=(0, 0, 0, 60))
    def foot(p, shade=1.0):
        fx, fy = p
        x0_, x1_ = sorted((fx - f * S * 0.045, fx + f * S * 0.12))
        d.ellipse([x0_, fy - S * 0.04, x1_, fy + S * 0.035], fill=_shade((70, 50, 44), shade), outline=INK, width=int(S * 0.010))
    # far side first
    _limb(d, [hipB, add(hipB, m1), add(hipB, m2)], wi, wo, _shade(pants, .8)); foot(add(hipB, m2), .8)
    _limb(d, [shB, b1], wi * .95, wo * .95, _shade(shirt, .8)); _limb(d, [b1, b2], wi * .78, wo * .78, _shade(shirt, .8))
    # body: a rounded, slightly tapered tunic with shoulder caps, a belt and a hem
    nx, ny = -up[1], up[0]
    bw_t, bw_b = S * 0.115, S * 0.100
    pts_l, pts_r = [], []
    for i in range(9):
        u = i / 8
        w_ = bw_t * (1 - u) + bw_b * u
        w_ *= 1 + 0.05 * math.sin(math.pi * u)                      # a little belly curve
        cxp, cyp = sh[0] + (hip[0] - sh[0]) * u, sh[1] + (hip[1] - sh[1]) * u
        pts_l.append((cxp + nx * w_, cyp + ny * w_)); pts_r.append((cxp - nx * w_, cyp - ny * w_))
    body = pts_l + [(hip[0] + nx * bw_b * 1.05, hip[1] + ny * bw_b * 1.05 + S * .035), (hip[0] - nx * bw_b * 1.05, hip[1] - ny * bw_b * 1.05 + S * .035)] + pts_r[::-1]
    d.polygon(body, fill=shirt)
    d.line(body + [body[0]], fill=INK, width=int(S * 0.012), joint="curve")
    d.ellipse([sh[0] - bw_t, sh[1] - bw_t * .75, sh[0] + bw_t, sh[1] + bw_t * .75], fill=shirt)
    d.arc([sh[0] - bw_t, sh[1] - bw_t * .75, sh[0] + bw_t, sh[1] + bw_t * .75], 185, 355, fill=INK, width=int(S * 0.012))
    bl = (hip[0] + (sh[0] - hip[0]) * .10, hip[1] + (sh[1] - hip[1]) * .10)
    d.line([(bl[0] + nx * bw_b, bl[1] + ny * bw_b), (bl[0] - nx * bw_b, bl[1] - ny * bw_b)], fill=_shade(shirt, .45), width=int(S * 0.026))
    d.rectangle([bl[0] - S * .016, bl[1] - S * .016, bl[0] + S * .016, bl[1] + S * .016], fill=(236, 200, 80), outline=INK, width=int(S * 0.005))
    ctr = (sh[0] + (hip[0] - sh[0]) * .06, sh[1] + (hip[1] - sh[1]) * .06)
    d.line([ctr, (ctr[0] + (hip[0] - sh[0]) * .35, ctr[1] + (hip[1] - sh[1]) * .35)], fill=_shade(shirt, .75), width=int(S * 0.006))      # centre seam
    _limb(d, [hipA, add(hipA, l1), add(hipA, l2)], wi, wo, pants); foot(add(hipA, l2))
    _limb(d, [shA, a1], wi * .95, wo * .95, shirt); _limb(d, [a1, a2], wi * .78, wo * .78, shirt)
    if "cape" in props:
        d.polygon([sh, (sh[0] - f * S * .26, sh[1] + S * .38), (sh[0] - f * S * .04, sh[1] + S * .42)], fill=a.s.get("color", (196, 57, 43)), outline=INK)
    for hp, e0 in ((b2, b1), (a2, a1)):                              # mitten hands with a thumb
        ax_, ay_ = hp[0] - e0[0], hp[1] - e0[1]
        an = math.hypot(ax_, ay_) or 1.0
        ux, uy = ax_ / an, ay_ / an
        hr_ = S * 0.040
        cx_, cy_ = hp[0] + ux * hr_ * .4, hp[1] + uy * hr_ * .4
        d.ellipse([cx_ - hr_, cy_ - hr_ * .9, cx_ + hr_, cy_ + hr_ * .9], fill=SKIN, outline=INK, width=int(S * 0.010))
        tx, ty = hp[0] - uy * hr_ * .85, hp[1] + ux * hr_ * .85
        d.ellipse([tx - hr_ * .45, ty - hr_ * .45, tx + hr_ * .45, ty + hr_ * .45], fill=SKIN, outline=INK, width=int(S * 0.008))
    # neck and head
    d.line([sh, neck, head], fill=INK, width=int(S * 0.058), joint="curve")
    d.line([sh, neck, (head[0] - hup[0] * R * .3, head[1] - hup[1] * R * .3)], fill=SKIN, width=int(S * 0.038), joint="curve")
    d.ellipse([head[0] - R, head[1] - R, head[0] + R, head[1] + R], fill=SKIN, outline=INK, width=int(S * 0.014))
    # hair / hat props
    if "hair" in props or not props or props == ["beard"]:
        d.chord([head[0] - R, head[1] - R, head[0] + R, head[1] + R], 205, 335, fill=hair)
        sp = [(head[0] + math.cos(math.radians(a_)) * R * r_, head[1] + math.sin(math.radians(a_)) * R * r_) for a_, r_ in ((212, 1.0), (222, 1.22), (234, 1.0), (246, 1.28), (258, 1.0), (270, 1.3), (282, 1.0), (294, 1.24), (306, 1.0), (318, 1.18), (328, 1.0))]
        d.polygon(sp, fill=hair); d.line(sp, fill=INK, width=int(S * 0.010), joint="curve")
        d.chord([head[0] - R, head[1] - R, head[0] + R, head[1] + R], 205, 335, outline=INK, width=int(S * 0.010))
    if "hardhat" in props:
        d.pieslice([head[0] - R * 1.05, head[1] - R * 1.08, head[0] + R * 1.05, head[1] - R * .1], 180, 360, fill=(246, 200, 40), outline=INK, width=int(S * 0.012))
        d.rectangle([head[0] - R * 1.22, head[1] - R * .58, head[0] + R * 1.22, head[1] - R * .40], fill=(226, 176, 28), outline=INK, width=int(S * 0.010))
    if "crown" in props:
        w_, h_ = R * .8, R * .7
        cy_ = head[1] - R * .84
        d.polygon([(head[0] - w_, cy_), (head[0] - w_, cy_ - h_), (head[0] - w_ / 2, cy_ - h_ / 2), (head[0], cy_ - h_ * 1.1), (head[0] + w_ / 2, cy_ - h_ / 2), (head[0] + w_, cy_ - h_), (head[0] + w_, cy_)], fill=(240, 196, 50), outline=INK)
    if "helmet" in props:
        d.pieslice([head[0] - R * 1.06, head[1] - R * 1.1, head[0] + R * 1.06, head[1] + R * .1], 180, 360, fill=a.s.get("color", (150, 146, 138)), outline=INK, width=int(S * 0.012))
    if "navycap" in props or "hat" in props:
        d.pieslice([head[0] - R, head[1] - R * 1.1, head[0] + R, head[1] + R * .1], 180, 360, fill=(240, 240, 236), outline=INK, width=int(S * 0.012))
    # face
    _face(d, head, R, f, face, ex, S)
    if suit:
        d.ellipse([head[0] - R * 1.3, head[1] - R * 1.3, head[0] + R * 1.3, head[1] + R * 1.3], outline=(220, 230, 240), width=int(S * 0.02))


def _face(d, head, R, f, face, ex, S):
    look = ex.get("look", (.4 * f, 0.0))
    blink = ex.get("blink", 0.0)
    mo = ex.get("mouth", 0.0)
    brow_up = ex.get("brow", 0.0)
    w = max(3, int(S * 0.011))
    shock = face in ("shock",)
    sad = face in ("worried", "sad")
    ang = face == "angry"
    smile = face == "smile"
    ex_off, ey = R * .38, R * .0
    er_x, er_y = R * (.30 if shock else .27), R * (.38 if shock else .34 if not sad else .29)
    for sgn in (-1, 1):
        x0 = head[0] + sgn * ex_off + f * R * .20
        y0 = head[1] + ey
        lid = max(blink, 0.22 if sad else .18 if ang else 0.0)
        ery = er_y * (1 - .88 * blink)
        d.ellipse([x0 - er_x, y0 - ery, x0 + er_x, y0 + ery], fill=(255, 255, 255), outline=INK, width=w)
        pr = R * (.09 if shock else .135)
        px = x0 + max(-1, min(1, look[0] * f * 0.6)) * f * (er_x - pr) * .8
        py = y0 + look[1] * (ery - pr) * .8
        if blink < .6:
            d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=INK)
            d.ellipse([px - pr * .1, py - pr * .65, px + pr * .5, py - pr * .05], fill=(255, 255, 255))
        if lid > .2 and blink < .6:
            d.rectangle([x0 - er_x, y0 - ery, x0 + er_x, y0 - ery + 2 * ery * lid], fill=SKIN)
            d.line([(x0 - er_x, y0 - ery + 2 * ery * lid), (x0 + er_x, y0 - ery + 2 * ery * lid)], fill=INK, width=w)
        by = y0 - er_y - R * (.14 + (.14 if shock else 0) + .10 * brow_up)
        tilt = (-.22 if ang else .22 if sad else 0.0) * sgn * f
        d.line([(x0 - er_x * 1.1, by - tilt * R), (x0 + er_x * 1.1, by + tilt * R)], fill=INK, width=w + 3)
    mx, my = head[0] + f * R * .22, head[1] + R * .60
    if mo > .12 or shock:
        o = max(mo, .5 if shock else 0)
        wd, ht = R * (.26 + .12 * o), R * (.08 + .42 * o)
        d.ellipse([mx - wd, my - ht * .4, mx + wd, my + ht], fill=(120, 30, 44), outline=INK, width=w)
        if o > .35:
            d.ellipse([mx - wd * .6, my + ht * .25, mx + wd * .6, my + ht * .92], fill=(214, 96, 104))
    elif smile:
        d.chord([mx - R * .34, my - R * .2, mx + R * .34, my + R * .28], 0, 180, fill=(120, 30, 44), outline=INK, width=w)
    elif sad:
        d.arc([mx - R * .28, my - R * .02, mx + R * .28, my + R * .3], 200, 340, fill=INK, width=w + 1)
    elif ang:
        d.line([(mx - R * .3, my + R * .06), (mx + R * .3, my - R * .04)], fill=INK, width=w + 1)
    else:
        d.arc([mx - R * .28, my - R * .12, mx + R * .28, my + R * .14], 20, 160, fill=INK, width=w + 1)
