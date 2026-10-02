"""Character renderer v2: tapered limbs and a clothed torso, a silhouette outline, toon shading with rim light, a cast shadow and a big expressive face.

Everything is drawn flat on a small per-actor RGBA layer first; the outline, shading and shadow are then derived from that layer's silhouette,
so hats, helmets, beards and held props all get the same treatment for free. The face is drawn last so shading never muddies it.
"""
import math
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

INK = (27, 27, 32)
SKIN_DARK = (214, 176, 142)
DARK_BG = {"space", "capsule", "mission_control", "night", "underwater", "submarine_interior", "moon"}


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c[:3])


def _tcap(d, p, q, r1, r2, fill):
    """A tapered capsule from p (radius r1) to q (radius r2)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / n, dx / n
    d.polygon([(p[0] + nx * r1, p[1] + ny * r1), (q[0] + nx * r2, q[1] + ny * r2), (q[0] - nx * r2, q[1] - ny * r2), (p[0] - nx * r1, p[1] - ny * r1)], fill=fill)
    d.ellipse([p[0] - r1, p[1] - r1, p[0] + r1, p[1] + r1], fill=fill)
    d.ellipse([q[0] - r2, q[1] - r2, q[0] + r2, q[1] + r2], fill=fill)


def _shift(a, dx, dy):
    h, w = a.shape[:2]
    return cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _post(L, S, light, rim, lw_px):
    """Outline + toon shading + rim light on the flat layer. L is an RGBA uint8 array; returns (processed RGBA array, (x0, y0) of the crop in L).
    Shading maps are computed at half resolution on the silhouette's own bounding box, which is what keeps this cheap."""
    h, w = L.shape[:2]
    ys, xs = np.nonzero(L[..., 3] > 8)
    if len(xs) == 0:
        return L[:1, :1] * 0, (0, 0)
    r = max(3.0, S * 0.026)
    m = int(r * 4 + lw_px + 4)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
    Lc = L[y0:y1, x0:x1]
    A8 = Lc[..., 3]
    # --- shading maps at half resolution
    Ah = cv2.resize(A8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lx, ly = light
    rh = r * 0.5
    sh = Ah * (1.0 - _shift(Ah, lx * rh, ly * rh))
    sh = cv2.GaussianBlur(sh, (0, 0), rh * 0.55)
    sh2 = Ah * (1.0 - _shift(Ah, lx * rh * 2.4, ly * rh * 2.4))
    sh2 = cv2.GaussianBlur(sh2, (0, 0), rh * 1.1)
    hi = Ah * (1.0 - _shift(Ah, -lx * rh * 0.7, -ly * rh * 0.7))
    hi = cv2.GaussianBlur(hi, (0, 0), rh * 0.35)
    dsz = (Lc.shape[1], Lc.shape[0])
    shade = cv2.resize(np.clip(sh * 0.55 + sh2 * 0.35, 0, 0.75), dsz, interpolation=cv2.INTER_LINEAR)[..., None] * 0.62
    hi = cv2.resize(np.clip(hi * 0.55, 0, 0.6), dsz, interpolation=cv2.INTER_LINEAR)[..., None]
    # --- apply at full resolution on the crop only
    out = Lc[..., :3].astype(np.float32)
    out *= (1.0 - shade)
    out += np.array([20, 18, 44], np.float32) * (shade * 0.32)
    out += (np.array(rim, np.float32) - out) * hi
    np.clip(out, 0, 255, out=out)
    # --- silhouette outline, drawn underneath
    k = int(max(2, lw_px))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    dil = cv2.dilate(A8, kern)
    res = np.empty_like(Lc)
    body = (A8.astype(np.float32) / 255.0)[..., None]
    ring = np.clip((dil.astype(np.float32) - A8.astype(np.float32)) / 255.0, 0, 1)[..., None]
    res[..., :3] = np.clip(out * body + np.array(INK, np.float32) * ring, 0, 255).astype(np.uint8)
    res[..., 3] = dil
    return res, (x0, y0)


def light_for(scene):
    """(light direction toward the source, rim colour) from the scene's sun / mood."""
    sun = scene.get("sun")
    if sun:
        sx = sun[0]
        c = tuple(sun[2])
        lx = -0.78 if sx < 0.5 else 0.78
        return (lx, -0.62), (min(255, c[0] + 10), min(255, c[1] + 6), min(255, c[2] + 40))
    bg = scene.get("bg_name", "")
    if bg in ("space", "capsule", "mission_control", "night", "underwater", "submarine_interior"):
        return (-0.72, -0.60), (150, 190, 255)
    if bg in ("volcanic", "battlefield", "ashen"):
        return (0.72, -0.60), (255, 168, 90)
    return (-0.72, -0.62), (255, 236, 190)


def draw_v2(a, img, t, gy, scene, ctx):
    """Draw actor `a` onto the PIL RGB image `img` (supersampled frame space)."""
    from . import stick as K
    SSc = K.SS
    S = a.unit
    pose, xr, f, face = a.key_state(t)
    a._look_dx = 0.40 * f
    ex = a.extras(t, scene)
    x_world = xr * a.W * SSc
    # ---- local layer (the actor's own box) -------------------------------------------------------
    bw, bh = int(S * 1.9), int(S * 1.5)
    ox, oy = x_world - bw / 2, gy - S * 1.38
    x, gl = bw / 2, S * 1.38
    L = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(L, "RGBA")
    Lb = dict(torso=0.30, upper=0.17, fore=0.17, thigh=0.22, shin=0.22, head=0.132, neck=0.035)
    add = lambda p, q: (p[0] + q[0], p[1] + q[1])
    l1 = K.seg((0, 0), pose["l1"], Lb["thigh"] * S, f)
    l2 = K.seg(l1, pose["l1"] + pose["l2"], Lb["shin"] * S, f)
    m1 = K.seg((0, 0), pose["m1"], Lb["thigh"] * S, f)
    m2 = K.seg(m1, pose["m1"] + pose["m2"], Lb["shin"] * S, f)
    drop = max(l2[1], m2[1])
    sq = ex.get("squash", 0.0)                                           # landing / impact squash
    hip = (x, gl - drop - pose.get("lift", 0) * S + sq * S * 0.02)
    tr = math.radians(pose["torso"])
    up = (math.sin(tr) * f, -math.cos(tr))
    tl = Lb["torso"] * S * (1 - sq * 0.05)
    neck_base = (hip[0] + up[0] * tl, hip[1] + up[1] * tl)
    sh = (hip[0] + up[0] * tl * 0.9, hip[1] + up[1] * tl * 0.9)
    hr = math.radians(pose["torso"] + pose["head"])
    hup = (math.sin(hr) * f, -math.cos(hr))
    rad = Lb["head"] * S * (1.2 if ctx.get("ink") else 1.0)
    head = (neck_base[0] + hup[0] * (Lb["neck"] * S + rad * 0.92), neck_base[1] + hup[1] * (Lb["neck"] * S + rad * 0.92))
    skin = a.s.get("skin", (247, 222, 190))
    tunic = a.s.get("tunic")
    col = a.s.get("color", INK)
    props = a.s.get("props", [])
    shirt = tunic if tunic else (52, 54, 66)
    pants = a.s.get("pants") or (shirt if "spacehelmet" in props else _shade(shirt, 0.50) if tunic else (36, 38, 50))
    shoe = (66, 46, 36)
    wide = pose["a1"], pose["a2"], pose["b1"], pose["b2"]
    a1 = add(sh, K.seg((0, 0), wide[0], Lb["upper"] * S, f)); a2 = add(a1, K.seg((0, 0), wide[0] + wide[1], Lb["fore"] * S, f))
    b1 = add(sh, K.seg((0, 0), wide[2], Lb["upper"] * S, f)); b2 = add(b1, K.seg((0, 0), wide[2] + wide[3], Lb["fore"] * S, f))
    # cape sits behind everything
    if "cape" in props:
        flap = math.sin(t * 5 + a.seed) * S * 0.03
        d.polygon([sh, (sh[0] - f * S * 0.30 + flap, sh[1] + S * 0.42), (sh[0] - f * S * 0.05, sh[1] + S * 0.46)], fill=col)
    ink = bool(ctx.get("ink"))
    lc = (236, 238, 240) if (ink and scene.get("bg_name", "") in DARK_BG and "spacehelmet" not in props) else INK   # light ink lines on dark sets
    shield_at = None
    if ink:
        skin = (251, 251, 249)                                             # stick-figure style: white heads and hands, black ink lines
        bk = 1.38 if "spacehelmet" in props else 1.0
        shield_at = _ink_body(d, a, t, S, f, up, hip, sh, neck_base, hup, l1, l2, m1, m2, a1, a2, b1, b2, wide, props, shirt, col, bk, add, lc)
    else:
        far = 0.74
        bk = 1.38 if "spacehelmet" in props else 1.0                       # a pressure suit is bulky: fatter torso, limbs, gloves
        _tc = lambda d_, p_, q_, r1_, r2_, fill_: _tcap(d_, p_, q_, r1_ * bk, r2_ * bk, fill_)
        if "spacehelmet" in props:                                          # life-support backpack behind the shoulders
            bpk = (sh[0] - f * S * 0.085 - up[0] * S * 0.02, sh[1] + S * 0.10)
            d.rounded_rectangle([bpk[0] - S * 0.05, bpk[1] - S * 0.11, bpk[0] + S * 0.05, bpk[1] + S * 0.11], radius=int(S * 0.03), fill=_shade(shirt, 0.82))
        # far leg, far arm (darker so the body reads as having depth)
        _tc(d, hip, add(hip, m1), S * 0.036, S * 0.027, _shade(pants, far)); _tc(d, add(hip, m1), add(hip, m2), S * 0.027, S * 0.021, _shade(pants, far))
        d.line([add(hip, m2), (add(hip, m2)[0] + f * S * 0.07, add(hip, m2)[1])], fill=_shade(shoe, far), width=int(S * 0.045))
        _tc(d, sh, b1, S * 0.026, S * 0.021, _shade(shirt, far)); _tc(d, b1, b2, S * 0.021, S * 0.017, _shade(shirt, far))
        _hand(d, b2, wide[2] + wide[3], f, S * bk, _shade(shirt if bk > 1 else skin, far))
        if "shield" in props:                                            # round shield strapped to the forearm, face to the viewer
            sr = 0.105 * S
            sc_ = ((b1[0] + b2[0]) / 2 + f * S * 0.03, (b1[1] + b2[1]) / 2 + S * 0.01)
            d.ellipse([sc_[0] - sr, sc_[1] - sr, sc_[0] + sr, sc_[1] + sr], fill=col)
            shield_at = (sc_, sr)
        # near leg
        _tc(d, hip, add(hip, l1), S * 0.038, S * 0.028, pants); _tc(d, add(hip, l1), add(hip, l2), S * 0.028, S * 0.022, pants)
        foot = add(hip, l2)
        d.line([foot, (foot[0] + f * S * 0.075, foot[1])], fill=shoe, width=int(S * 0.048))
        d.ellipse([foot[0] + f * S * 0.045 - S * 0.03, foot[1] - S * 0.026, foot[0] + f * S * 0.045 + S * 0.03, foot[1] + S * 0.026], fill=shoe)
        # torso (clothed, tapered), belt, collar
        _tc(d, sh, hip, S * 0.072, S * 0.064, shirt)
        nrm = (-up[1], up[0])
        belt_a = (hip[0] - nrm[0] * S * 0.064 + up[0] * S * 0.012, hip[1] - nrm[1] * S * 0.064 + up[1] * S * 0.012)
        belt_b = (hip[0] + nrm[0] * S * 0.064 + up[0] * S * 0.012, hip[1] + nrm[1] * S * 0.064 + up[1] * S * 0.012)
        d.line([belt_a, belt_b], fill=(58, 42, 32), width=max(3, int(S * 0.016)))
        # neck
        _tcap(d, sh, add(neck_base, (hup[0] * S * 0.02, hup[1] * S * 0.02)), S * 0.026, S * 0.024, _shade(skin, 0.92))
        # near arm over the torso
        _tc(d, sh, a1, S * 0.028, S * 0.022, shirt); _tc(d, a1, a2, S * 0.022, S * 0.018, shirt)
        _hand(d, a2, wide[0] + wide[1], f, S * bk, shirt if bk > 1 else skin)
    # head
    d.ellipse([head[0] - rad, head[1] - rad, head[0] + rad, head[1] + rad], fill=skin)
    lw = max(4, S * 0.042)
    if "beard" in props and ink:
        pass                                                         # beards are switched off in the stick-figure style for now
    elif False:                                     # chin beard that follows the jaw, plus a moustache; the face stays clear above it
        hrc = a.s.get("hair", (70, 48, 30))
        bcol = hrc if sum(hrc) < 420 else (150, 150, 146)
        bx = head[0] + f * rad * 0.30
        d.polygon([(bx - rad * 0.62, head[1] + rad * 0.62), (bx + rad * 0.62, head[1] + rad * 0.62), (bx + rad * 0.40, head[1] + rad * 1.15), (bx + rad * 0.06, head[1] + rad * 1.72), (bx - rad * 0.36, head[1] + rad * 1.2)], fill=bcol)   # pointed chin beard
        d.arc([bx - rad * 0.50, head[1] + rad * 0.18, bx + rad * 0.50, head[1] + rad * 0.62], 200, 340, fill=bcol, width=max(4, int(rad * 0.14)))               # drooping moustache
    elif "beard" in props:
        hrc = a.s.get("hair", (70, 48, 30))
        bcol = hrc if sum(hrc) < 420 else (150, 150, 146)
        d.pieslice([head[0] - rad * .98, head[1] - rad * .35, head[0] + rad * .98, head[1] + rad * 1.3], 5, 175, fill=bcol)
        d.ellipse([head[0] - rad * .72, head[1] - rad * .55, head[0] + rad * .72, head[1] + rad * .62], fill=skin)
        d.rectangle([head[0] - rad * .5, head[1] + rad * .26, head[0] + rad * .5, head[1] + rad * .38], fill=bcol)
    if "hair" in props:
        d.pieslice([head[0] - rad * 1.06, head[1] - rad * 1.10, head[0] + rad * 1.06, head[1] + rad * .22], 180, 360, fill=a.s.get("hair", (70, 48, 30)))
    # props that belong to the silhouette (outlined and shaded together with the body)
    hand, dirv = a2, wide[0] + wide[1]
    if "helmet" in props:
        d.pieslice([head[0] - rad * 1.08, head[1] - rad * 1.12, head[0] + rad * 1.08, head[1] + rad * 0.10], 180, 360, fill=col)
    if "crown" in props:
        fall = a.s.get("crown_fall")
        cx, cy = head[0], head[1] - rad * 0.95
        if fall is not None and t > fall:
            dt = t - fall
            cy = min(cy + 0.5 * S * 3.2 * dt * dt, gl - S * 0.03)
            cx = cx + f * S * 0.10 * min(dt, 0.6) * 1.6
        w, h = rad * 0.9, rad * 0.75
        d.polygon([(cx - w, cy), (cx - w, cy - h), (cx - w / 2, cy - h / 2), (cx, cy - h * 1.1), (cx + w / 2, cy - h / 2), (cx + w, cy - h), (cx + w, cy)], fill=K.GOLD)
    if "sword" in props:                                             # a tapered steel blade with a cross-guard, grip and pommel
        def _clear(ang):                                             # does a blade at this angle stay out of the wielder's own head?
            tp = add(hand, K.seg((0, 0), ang, 0.40 * S, f))
            vx, vy = tp[0] - hand[0], tp[1] - hand[1]
            ll = vx * vx + vy * vy or 1.0
            u_ = max(0.0, min(1.0, ((head[0] - hand[0]) * vx + (head[1] - hand[1]) * vy) / ll))
            return math.hypot(hand[0] + vx * u_ - head[0], hand[1] + vy * u_ - head[1]) > rad * 1.22
        sw_ang = dirv
        if not _clear(sw_ang):
            for dlt in (12, -12, 24, -24, 36, -36, 48, -48, 60, -60, 80, -80, 100, -100):
                if _clear(dirv + dlt):
                    sw_ang = dirv + dlt
                    break
        tip = add(hand, K.seg((0, 0), sw_ang, 0.40 * S, f))
        ux, uy = tip[0] - hand[0], tip[1] - hand[1]
        un = math.hypot(ux, uy) or 1.0
        ux, uy = ux / un, uy / un
        nx_, ny_ = -uy, ux
        bw_ = S * 0.017
        base_ = (hand[0] + ux * S * 0.05, hand[1] + uy * S * 0.05)
        d.polygon([(base_[0] + nx_ * bw_, base_[1] + ny_ * bw_), tip, (base_[0] - nx_ * bw_, base_[1] - ny_ * bw_)], fill=(206, 212, 224))
        d.line([base_, (tip[0] - ux * S * 0.03, tip[1] - uy * S * 0.03)], fill=(244, 247, 252), width=max(2, int(S * 0.006)))
        d.line([(base_[0] + nx_ * S * 0.05, base_[1] + ny_ * S * 0.05), (base_[0] - nx_ * S * 0.05, base_[1] - ny_ * S * 0.05)], fill=(186, 150, 60), width=max(3, int(S * 0.016)))
        d.line([hand, (hand[0] - ux * S * 0.06, hand[1] - uy * S * 0.06)], fill=(98, 66, 44), width=max(3, int(S * 0.018)))
        d.ellipse([hand[0] - ux * S * 0.075 - S * 0.011, hand[1] - uy * S * 0.075 - S * 0.011, hand[0] - ux * S * 0.075 + S * 0.011, hand[1] - uy * S * 0.075 + S * 0.011], fill=(186, 150, 60))
    if "spear" in props:                                             # held upright beside the body, leaning a little forward: never across the face
        p1 = add(hand, K.seg((0, 0), 172, 0.62 * S, f))
        if p1[1] < S * 0.10:                                     # keep the spear head inside the character's box
            k_ = (hand[1] - S * 0.10) / max(hand[1] - p1[1], 1e-6)
            p1 = (hand[0] + (p1[0] - hand[0]) * k_, hand[1] + (p1[1] - hand[1]) * k_)
        p0 = (hand[0] - (p1[0] - hand[0]) * 0.45, hand[1] - (p1[1] - hand[1]) * 0.45)
        K.line(d, [p0, p1], lw * .5, (130, 96, 62))
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        un = math.hypot(ux, uy) or 1.0
        ux, uy = ux / un, uy / un
        tipp = (p1[0] + ux * S * 0.085, p1[1] + uy * S * 0.085)
        d.polygon([(p1[0] - uy * S * 0.022, p1[1] + ux * S * 0.022), tipp, (p1[0] + uy * S * 0.022, p1[1] - ux * S * 0.022)], fill=(206, 212, 224))   # steel head
    if "scroll" in props and math.hypot(hand[0] - head[0], hand[1] - head[1]) > rad * 2.3:     # not while the hand is up at the face (it would hide it)                                            # a rolled parchment held in the fist, with curled ends and lines of script
        sw_, sh_ = S * .045, S * .075
        d.rounded_rectangle([hand[0] - sw_, hand[1] - sh_, hand[0] + sw_, hand[1] + sh_], radius=int(S * .02), fill=(238, 222, 176))
        for yy_ in (-.045, -.015, .015, .045):
            d.line([hand[0] - sw_ * .62, hand[1] + S * yy_, hand[0] + sw_ * .62, hand[1] + S * yy_], fill=(150, 120, 80), width=max(2, int(S * .006)))
        for sgn_ in (-1, 1):
            d.ellipse([hand[0] - sw_ * 1.1, hand[1] + sgn_ * sh_ - S * .014, hand[0] + sw_ * 1.1, hand[1] + sgn_ * sh_ + S * .014], fill=(214, 190, 140))
        _hand(d, hand, dirv, f, S, skin)
    if "hardhat" in props:
        d.pieslice([head[0] - rad * 1.1, head[1] - rad * 1.25, head[0] + rad * 1.1, head[1] + rad * .15], 180, 360, fill=(246, 200, 40))
        d.rectangle([head[0] - rad * 1.3, head[1] - rad * .66, head[0] + rad * 1.3, head[1] - rad * .50], fill=(226, 176, 28))
    if "navycap" in props:
        d.rectangle([head[0] - rad * 1.0, head[1] - rad * 1.12, head[0] + rad * 1.0, head[1] - rad * 0.52], fill=(242, 242, 238))
        d.rectangle([head[0] - rad * 1.0, head[1] - rad * 0.62, head[0] + rad * 1.0, head[1] - rad * 0.52], fill=(30, 40, 70))
        d.rectangle([min(head[0], head[0] + f * rad * 1.45), head[1] - rad * 0.58, max(head[0], head[0] + f * rad * 1.45), head[1] - rad * 0.44], fill=INK)
    if "hat" in props:
        d.ellipse([head[0] - rad * 1.45, head[1] - rad * 0.98, head[0] + rad * 1.45, head[1] - rad * 0.58], fill=(64, 54, 48))
        d.rectangle([head[0] - rad * 0.78, head[1] - rad * 1.6, head[0] + rad * 0.78, head[1] - rad * 0.78], fill=(64, 54, 48))
    if "tie" in props:
        d.polygon([(sh[0], sh[1] + S * .01), (sh[0] - S * .022, sh[1] + S * .09), (sh[0] + S * .022, sh[1] + S * .09)], fill=K.RED)
    if "rifle" in props:
        K.line(d, [add(hand, K.seg((0, 0), dirv + 180, .12 * S, f)), add(hand, K.seg((0, 0), dirv, .30 * S, f))], lw * .7, (92, 80, 68))
    if "flag" in props:
        pole = (hand[0], hand[1] - S * 0.45)
        K.line(d, [hand, pole], lw * .5, (130, 96, 62))
        d.polygon([pole, (pole[0] + f * S * .22, pole[1] + S * .06 + math.sin(t * 6) * S * .015), (pole[0], pole[1] + S * .13)], fill=col)
    # ---- derive outline / shading / shadow from the silhouette ------------------------------------
    light, rim = ctx["light"], ctx["rim"]
    arr_c, (cx0, cy0) = _post_ink(np.asarray(L), S, light, rim, max(3, S * 0.0072 * ctx.get("ring", 1.0)), lc) if ink else _post(np.asarray(L), S, light, rim, max(3, S * 0.0075))
    arr = np.zeros((bh, bw, 4), np.uint8)
    arr[cy0:cy0 + arr_c.shape[0], cx0:cx0 + arr_c.shape[1]] = arr_c
    Lp = Image.fromarray(arr, "RGBA")
    d2 = ImageDraw.Draw(Lp, "RGBA")
    # glass helmet is outline-only and sits over the face, so it is drawn after shading
    if "spacehelmet" in props:
        hr_ = rad * 1.5
        d2.ellipse([head[0] - hr_, head[1] - hr_, head[0] + hr_, head[1] + hr_], outline=(236, 242, 248), width=max(4, int(S * 0.03)))
        d2.arc([head[0] - hr_ * .8, head[1] - hr_ * .8, head[0] + hr_ * .8, head[1] + hr_ * .8], 200, 260, fill=(255, 255, 255, 200), width=max(3, int(S * 0.02)))
        d2.rectangle([head[0] - rad * .9, head[1] + rad * 1.05, head[0] + rad * .9, head[1] + rad * 1.4], fill=(214, 218, 222), outline=INK, width=max(2, int(S * 0.012)))
    if "spacehelmet" in props:                                       # chest control box with indicator lights, and a mission patch on the arm
        cc = (sh[0] - up[0] * S * 0.085 + f * S * 0.03, sh[1] - up[1] * S * 0.085)
        d2.rounded_rectangle([cc[0] - S * 0.05, cc[1] - S * 0.035, cc[0] + S * 0.05, cc[1] + S * 0.035], radius=int(S * 0.012), fill=(shirt if ink else (52, 58, 66)), outline=INK, width=max(2, int(S * 0.006)))
        for k_, c_ in enumerate(((120, 230, 130), (255, 190, 60), (255, 80, 70))):
            d2.ellipse([cc[0] - S * 0.036 + k_ * S * 0.032, cc[1] - S * 0.012, cc[0] - S * 0.022 + k_ * S * 0.032, cc[1] + S * 0.002], fill=c_)
        d2.rectangle([cc[0] - S * 0.034, cc[1] + S * 0.01, cc[0] + S * 0.034, cc[1] + S * 0.022], fill=(30, 34, 40))
    if shield_at:                                                    # rim, boss and straps on top of the shaded disc
        (scx, scy), sr = shield_at
        d2.ellipse([scx - sr * .86, scy - sr * .86, scx + sr * .86, scy + sr * .86], outline=_shade(col, 0.55), width=max(3, int(S * 0.012)))
        d2.line([scx - sr * .86, scy, scx + sr * .86, scy], fill=_shade(col, 0.62), width=max(3, int(S * 0.010)))
        d2.line([scx, scy - sr * .86, scx, scy + sr * .86], fill=_shade(col, 0.62), width=max(3, int(S * 0.010)))
        d2.ellipse([scx - sr * .26, scy - sr * .26, scx + sr * .26, scy + sr * .26], fill=(206, 176, 84), outline=INK, width=max(2, int(S * 0.006)))
        d2.ellipse([scx - sr * .12 - sr * .06, scy - sr * .16, scx - sr * .02, scy - sr * .06], fill=(250, 236, 170))
    _face(d2, a, K, head, rad, f, face, ex, S, t, skin, props, ink)
    # ---- cast shadow on the ground, then composite ------------------------------------------------
    A = arr[..., 3]
    shadow = _ink_shadow(S, x, gl, bh, bw) if ink else _cast_shadow(A, light, S, gl, bh, bw)
    if img.mode == "RGBA":                                             # a transparent layer (thumbnails): composite properly with clipping
        if shadow is not None and ctx.get("shadow", True):
            sh_im = Image.new("RGBA", Lp.size, (10, 10, 24, 0))
            sh_im.putalpha(shadow)
            _blit(img, sh_im, ox, oy)
        _blit(img, Lp, ox, oy)
        return
    if shadow is not None:
        img.paste((10, 10, 24), (int(ox), int(oy)), shadow)
    img.paste(Lp.convert("RGB"), (int(ox), int(oy)), Lp.split()[3])


def _post_ink(L, S, light, rim, lw_px, ring_col=INK):
    """Outline only: every ink-style figure is a flat white/black drawing with a clean line round the whole silhouette."""
    h, w = L.shape[:2]
    ys, xs = np.nonzero(L[..., 3] > 8)
    if len(xs) == 0:
        return L[:1, :1] * 0, (0, 0)
    m = int(lw_px * 3 + 4)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
    Lc = L[y0:y1, x0:x1]
    A8 = Lc[..., 3]
    k = int(max(2, lw_px))
    dil = cv2.dilate(A8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    res = np.empty_like(Lc)
    body = (A8.astype(np.float32) / 255.0)[..., None]
    ring = np.clip((dil.astype(np.float32) - A8.astype(np.float32)) / 255.0, 0, 1)[..., None]
    res[..., :3] = np.clip(Lc[..., :3] * body + np.array(ring_col, np.float32) * ring, 0, 255).astype(np.uint8)
    res[..., 3] = dil
    return res, (x0, y0)


def _ink_shadow(S, x, gl, bh, bw):
    sm = np.zeros((bh, bw), np.uint8)
    cv2.ellipse(sm, (int(x), int(gl)), (int(S * 0.20), max(2, int(S * 0.028))), 0, 0, 360, 255, -1)
    sm = cv2.GaussianBlur(sm, (0, 0), max(2.0, S * 0.012))
    return Image.fromarray((sm.astype(np.float32) * 0.30).astype(np.uint8), "L")


def _ink_body(d, a, t, S, f, up, hip, sh, neck_base, hup, l1, l2, m1, m2, a1, a2, b1, b2, wide, props, shirt, col, bk, add, lc=INK):
    """Stick-figure body: constant-width ink lines for legs, spine and arms, small white fists, a coloured scarf. The line boils a little every 1/8 s, like hand-drawn animation."""
    fr = int(t * 8)
    amp = S * 0.0032

    def jt(key):
        h_ = zlib.crc32(("%s|%s|%d" % (a.seed, key, fr)).encode())
        return (((h_ & 0xFFFF) / 32767.5 - 1.0) * amp, (((h_ >> 16) & 0xFFFF) / 32767.5 - 1.0) * amp)

    def J(p, key):
        j = jt(key)
        return (p[0] + j[0], p[1] + j[1])

    suit = "spacehelmet" in props
    w0 = S * (0.0065 if not suit else 0.05)
    fill = lc if not suit else (226, 230, 236)

    def ln(pts, w=None, key="", c=None):
        w = w or w0
        c = c or fill
        pts = [J(p, key + str(i)) for i, p in enumerate(pts)]
        d.line(pts, fill=c, width=int(w), joint="curve")
        for p in pts:
            d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=c)

    if suit:                                                         # backpack behind the shoulders, in the character's colour
        bpk = (sh[0] - f * S * 0.085 - up[0] * S * 0.02, sh[1] + S * 0.10)
        d.rounded_rectangle([bpk[0] - S * 0.05, bpk[1] - S * 0.11, bpk[0] + S * 0.05, bpk[1] + S * 0.11], radius=int(S * 0.03), fill=_shade(shirt, 0.9))
    shield_at = None
    ln([sh, b1, b2], key="b")                                       # far arm
    ln([hip, add(hip, m1), add(hip, m2)], key="m")                  # far leg
    if "shield" in props:
        sr = 0.105 * S
        sc_ = ((b1[0] + b2[0]) / 2 + f * S * 0.03, (b1[1] + b2[1]) / 2 + S * 0.01)
        d.ellipse([sc_[0] - sr, sc_[1] - sr, sc_[0] + sr, sc_[1] + sr], fill=col)
        shield_at = (sc_, sr)
    ln([hip, add(hip, l1), add(hip, l2)], key="l")                  # near leg
    for ft in (add(hip, m2), add(hip, l2)):                         # feet: a short line forward
        ln([ft, (ft[0] + f * S * 0.062, ft[1])], key="ft%d" % int(ft[0]))
    ln([hip, sh, add(neck_base, (hup[0] * S * 0.02, hup[1] * S * 0.02))], w=w0 * (1.0 if not suit else 1.6), key="t")
    ln([sh, a1, a2], key="a")                                       # near arm
    nrm = (-up[1], up[0])
    nb = neck_base                                                  # scarf in the role colour: the only colour on a plain stick figure
    tail = (nb[0] - f * S * 0.075 + math.sin(t * 5 + a.seed) * S * 0.012, nb[1] + S * 0.06)
    if not suit:
        d.polygon([(nb[0] + nrm[0] * S * 0.036, nb[1] + nrm[1] * S * 0.036 + S * 0.01), (nb[0] - nrm[0] * S * 0.036, nb[1] - nrm[1] * S * 0.036 + S * 0.01), tail], fill=shirt)
        d.ellipse([nb[0] - S * 0.04, nb[1] - S * 0.012, nb[0] + S * 0.04, nb[1] + S * 0.024], fill=shirt)
    hr_ = S * 0.034 * bk
    for hp, (e0, e1) in ((b2, (b1, b2)), (a2, (a1, a2))):
        hp = J(hp, "h")
        fc = (240, 242, 244) if suit else (251, 251, 249)
        ax_, ay_ = hp[0] - e0[0], hp[1] - e0[1]                      # forearm direction: the hand is a mitten pointing along it, with a thumb on top
        an = math.hypot(ax_, ay_) or 1.0
        ux, uy = ax_ / an, ay_ / an
        d.ellipse([hp[0] + ux * hr_ * .35 - hr_ * 1.08, hp[1] + uy * hr_ * .35 - hr_ * .92, hp[0] + ux * hr_ * .35 + hr_ * 1.08, hp[1] + uy * hr_ * .35 + hr_ * .92], fill=fc, outline=lc, width=max(2, int(S * 0.0045)))
        tx, ty = hp[0] + (-uy) * hr_ * .78 + ux * hr_ * .2, hp[1] + ux * hr_ * .78 + uy * hr_ * .2
        d.ellipse([tx - hr_ * .42, ty - hr_ * .42, tx + hr_ * .42, ty + hr_ * .42], fill=fc, outline=lc, width=max(2, int(S * 0.004)))
    return shield_at


def _blit(img, Lp, ox, oy):
    ox, oy = int(ox), int(oy)
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + Lp.width, img.width), min(oy + Lp.height, img.height)
    if x1 <= x0 or y1 <= y0:
        return
    img.alpha_composite(Lp.crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy)), (x0, y0))


def _hand(d, p, ang, f, S, col):
    """A rounded mitten with a thumb on the upper side."""
    a = math.radians(ang)
    dx, dy = math.sin(a) * f, math.cos(a)
    nx, ny = -dy, dx
    sgn = -1.0 if ny > 0 else 1.0
    d.ellipse([p[0] - S * 0.029, p[1] - S * 0.029, p[0] + S * 0.029, p[1] + S * 0.029], fill=col)
    tx, ty = p[0] + dx * S * 0.016 + nx * sgn * S * 0.026, p[1] + dy * S * 0.016 + ny * sgn * S * 0.026
    d.ellipse([tx - S * 0.013, ty - S * 0.013, tx + S * 0.013, ty + S * 0.013], fill=col)


def _cast_shadow(A, light, S, gl, bh, bw):
    """Flatten the silhouette onto the ground away from the light and blur it."""
    h, w = A.shape
    lx = light[0]
    k = 0.085                                                          # ground squash
    sh = -lx * 0.55                                                    # lean away from the light
    M = np.float32([[1, sh * 0.5, -sh * 0.5 * gl], [0, k, gl * (1 - k)]])
    sm = cv2.warpAffine(A, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    sm = cv2.GaussianBlur(sm, (0, 0), max(2.0, S * 0.012))
    sm = (sm.astype(np.float32) * 0.42).astype(np.uint8)
    # a tight dark contact patch under the feet
    return Image.fromarray(sm, "L")


def _face(d, a, K, head, rad, f, face, ex, S, t, skin, props, ink=False):
    lw = max(4, S * 0.042)
    look = ex.get("look", (0.35 * f, 0.0))                             # (dx, dy) in eye radii, signed to world x
    blink = ex.get("blink", 0.0)                                       # 0 open .. 1 shut
    mo = ex.get("mouth", 0.0)                                          # 0 closed .. 1 wide open
    talk_brow = ex.get("brow", 0.0)
    er_x, er_y = rad * 0.245, rad * 0.33
    big = 1.18 if face == "shock" else 1.0
    eyes = [(head[0] + f * rad * 0.24, head[1] - rad * 0.10), (head[0] + f * rad * 0.76, head[1] - rad * 0.10)]
    far = 0
    # nose: a small rounded bump on the facing side
    nx, ny = head[0] + f * rad * 1.0, head[1] + rad * 0.12
    if not ink:
      d.ellipse([nx - rad * .13, ny - rad * .11, nx + rad * .13, ny + rad * .13], fill=(236, 190, 152), outline=(150, 100, 80), width=max(1, int(S * 0.004)))
    # cheeks
    if face == "smile" and not ink:
        for cx_ in (head[0] + f * rad * 0.52,):
            d.ellipse([cx_ - rad * .2, head[1] + rad * .16, cx_ + rad * .2, head[1] + rad * .38], fill=(244, 170, 160, 120))
    lid_base = {"worried": 0.12, "sad": 0.24, "angry": 0.14, "smile": 0.08, "shock": 0.0}.get(face, 0.0)
    lid = max(lid_base, blink)
    if ink:                                                              # two plain black eyes with a pinprick of light; a blink is a line
        for i_, (ex_, ey_) in enumerate(eyes):
            erx, ery = rad * (0.115 if i_ == 0 else 0.10) * big, rad * 0.17 * big
            ox_, oy_ = look[0] * rad * 0.045, ey_ * 0 + look[1] * rad * 0.04
            sq_ = max(lid, 0.0)
            ery2 = max(ery * (1 - 0.92 * sq_), rad * 0.018)
            d.ellipse([ex_ + ox_ - erx, ey_ + oy_ - ery2, ex_ + ox_ + erx, ey_ + oy_ + ery2], fill=INK)
            if sq_ < 0.5:
                d.ellipse([ex_ + ox_ - erx * .15, ey_ + oy_ - ery2 * .65, ex_ + ox_ + erx * .55, ey_ + oy_ - ery2 * .15], fill=(255, 255, 255))
        eyes_done = True
    else:
        eyes_done = False
    for i_, (ex_, ey_) in enumerate(eyes if not eyes_done else []):
        erx, ery = er_x * big, er_y * big
        if i_ == 1:
            erx *= 0.88
        d.ellipse([ex_ - erx, ey_ - ery, ex_ + erx, ey_ + ery], fill=(255, 255, 255), outline=None if "glasses" in props else INK, width=max(2, int(S * 0.006)))
        pr = erx * (0.72 if face != "shock" else 0.46)
        px_ = ex_ + look[0] * erx * 0.55
        py_ = ey_ + ery * 0.14 + look[1] * ery * 0.55
        d.ellipse([px_ - pr, py_ - pr * 1.12, px_ + pr, py_ + pr * 1.12], fill=(30, 26, 34))
        d.ellipse([px_ - pr * 0.42 - pr * 0.1, py_ - pr * 0.62, px_ + pr * 0.12 - pr * 0.1, py_ - pr * 0.1], fill=(255, 255, 255))   # specular
        if lid > 0.02:                                                  # eyelid: skin-coloured cap that closes down over the eye
            ly = ey_ - ery + 2 * ery * lid
            slope = {"angry": 0.15, "worried": -0.15, "sad": -0.2}.get(face, 0.0)
            d.polygon([(ex_ - erx * 1.12, ey_ - ery * 1.15), (ex_ + erx * 1.12, ey_ - ery * 1.15),
                       (ex_ + erx * 1.12, ly + slope * ery * f * (1 if i_ == 0 else -1) * 0.8), (ex_ - erx * 1.12, ly - slope * ery * f * (1 if i_ == 0 else -1) * 0.8)], fill=skin)
            d.line([(ex_ - erx * 1.05, ly), (ex_ + erx * 1.05, ly)], fill=INK, width=max(2, int(S * 0.006)))
    # brows
    bw = max(4, int(S * 0.021))
    by = head[1] - rad * 0.66 - talk_brow * rad * 0.14
    tilt = {"angry": (-1, 0.17), "worried": (1, 0.22), "sad": (1, 0.22), "shock": (0, 0.0), "smile": (0, 0.0)}.get(face, (0, 0))
    lift = rad * (0.22 if face == "shock" else 0.10 if face == "smile" else 0.0)
    for k_, (ex_, ey_) in enumerate(eyes):
        sgn = 1 if k_ == 0 else -1
        din = tilt[0] * rad * tilt[1] * sgn * f * -1
        arch = rad * (0.06 if face in ("smile", "shock") else 0.0)
        d.line([(ex_ - rad * .21, by - lift + din), (ex_, by - lift - arch), (ex_ + rad * .21, by - lift - din)], fill=(34, 26, 22), width=bw, joint="curve")
    # mouth
    bearded = "beard" in props and not ink
    mx, my = head[0] + f * rad * 0.46, head[1] + rad * (0.76 if bearded else 0.56)
    mw = max(4, int(S * 0.016))
    MC = (238, 168, 156) if bearded else INK
    if mo > 0.12:
        wd = rad * (0.24 + 0.12 * mo)
        ht = rad * (0.07 + 0.42 * mo)
        d.ellipse([mx - wd, my - ht * 0.45, mx + wd, my + ht], fill=(96, 34, 44), outline=INK, width=max(2, int(S * 0.006)))
        if mo > 0.35:
            d.ellipse([mx - wd * 0.6, my + ht * 0.2, mx + wd * 0.6, my + ht * 0.92], fill=(214, 96, 104))
            d.rectangle([mx - wd * 0.7, my - ht * 0.42, mx + wd * 0.7, my - ht * 0.18], fill=(250, 250, 244))
    elif face == "smile":
        d.arc([mx - rad * .42, my - rad * .30, mx + rad * .42, my + rad * .26], 10, 170, fill=MC, width=mw)
    elif face in ("sad", "worried"):
        d.arc([mx - rad * .36, my - rad * .02, mx + rad * .36, my + rad * .46], 200, 340, fill=MC, width=mw)
    elif face == "angry":
        d.line([mx - rad * .36, my + rad * .08, mx + rad * .36, my - rad * .04], fill=MC, width=mw)
    elif face == "shock":
        d.ellipse([mx - rad * .17, my - rad * .08, mx + rad * .17, my + rad * .34], fill=(96, 34, 44), outline=INK, width=max(2, int(S * 0.006)))
    else:
        d.arc([mx - rad * .32, my - rad * .14, mx + rad * .32, my + rad * .14], 20, 160, fill=MC, width=mw)
    if "glasses" in props:
        for ex_, ey_ in eyes:
            d.ellipse([ex_ - er_x * 1.45, ey_ - er_x * 1.45, ex_ + er_x * 1.45, ey_ + er_x * 1.45], outline=INK, width=max(3, int(S * 0.008)))
        d.line([eyes[0][0] + er_x * 1.45, eyes[0][1], eyes[1][0] - er_x * 1.45, eyes[1][1]], fill=INK, width=max(3, int(S * 0.008)))
    # emotion marks above the head
    exx, eyy = head[0], head[1] - rad * 1.7
    pulse = 0.5 + 0.5 * math.sin(t * 6)
    if face == "shock":
        d.rectangle([exx - rad * .1, eyy - rad * .6, exx + rad * .1, eyy + rad * .1], fill=(255, 210, 60), outline=INK, width=3)
        d.ellipse([exx - rad * .12, eyy + rad * .22, exx + rad * .12, eyy + rad * .46], fill=(255, 210, 60), outline=INK, width=3)
    elif face == "worried":
        for k_ in range(2):
            age = (t * .9 + k_ * .5) % 1.0
            dx_, dy_ = f * rad * (1.05 + .1 * k_), -rad * .5 + age * rad * 1.1
            d.polygon([(exx + dx_, eyy + dy_ + rad * .6), (exx + dx_ - rad * .13, eyy + dy_ + rad * .9), (exx + dx_ + rad * .13, eyy + dy_ + rad * .9)], fill=(120, 190, 240, int(230 * (1 - age))))
            d.ellipse([exx + dx_ - rad * .13, eyy + dy_ + rad * .84, exx + dx_ + rad * .13, eyy + dy_ + rad * 1.1], fill=(120, 190, 240, int(230 * (1 - age))))
    elif face == "angry":
        for k_ in range(3):
            a_ = -2.4 + k_ * .6
            d.line([(exx + math.cos(a_) * rad * (.8 + .3 * pulse), eyy + rad * .5 + math.sin(a_) * rad * (.8 + .3 * pulse)), (exx + math.cos(a_) * rad * 1.4, eyy + rad * .5 + math.sin(a_) * rad * 1.4)], fill=(210, 50, 40), width=max(4, int(lw * .5)))
    elif face == "sad":
        tx0, tx1 = sorted((head[0] + f * rad * .3, head[0] + f * rad * .5))
        d.ellipse([tx0, head[1] + rad * (.1 + .6 * ((t * .8) % 1.0)), tx1, head[1] + rad * (.35 + .6 * ((t * .8) % 1.0))], fill=(120, 190, 240, 220))
