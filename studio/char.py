"""Character renderer v2: tapered limbs and a clothed torso, a silhouette outline, toon shading with rim light, a cast shadow and a big expressive face.

Everything is drawn flat on a small per-actor RGBA layer first; the outline, shading and shadow are then derived from that layer's silhouette,
so hats, helmets, beards and held props all get the same treatment for free. The face is drawn last so shading never muddies it.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

INK = (27, 27, 32)
SKIN_DARK = (214, 176, 142)


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
    Lb = dict(torso=0.30, upper=0.17, fore=0.17, thigh=0.22, shin=0.22, head=0.118, neck=0.035)
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
    rad = Lb["head"] * S
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
    far = 0.74
    # far leg, far arm (darker so the body reads as having depth)
    _tcap(d, hip, add(hip, m1), S * 0.036, S * 0.027, _shade(pants, far)); _tcap(d, add(hip, m1), add(hip, m2), S * 0.027, S * 0.021, _shade(pants, far))
    d.line([add(hip, m2), (add(hip, m2)[0] + f * S * 0.07, add(hip, m2)[1])], fill=_shade(shoe, far), width=int(S * 0.045))
    _tcap(d, sh, b1, S * 0.026, S * 0.021, _shade(shirt, far)); _tcap(d, b1, b2, S * 0.021, S * 0.017, _shade(shirt, far))
    d.ellipse([b2[0] - S * 0.026, b2[1] - S * 0.026, b2[0] + S * 0.026, b2[1] + S * 0.026], fill=_shade(skin, far))
    # near leg
    _tcap(d, hip, add(hip, l1), S * 0.038, S * 0.028, pants); _tcap(d, add(hip, l1), add(hip, l2), S * 0.028, S * 0.022, pants)
    foot = add(hip, l2)
    d.line([foot, (foot[0] + f * S * 0.075, foot[1])], fill=shoe, width=int(S * 0.048))
    d.ellipse([foot[0] + f * S * 0.045 - S * 0.03, foot[1] - S * 0.026, foot[0] + f * S * 0.045 + S * 0.03, foot[1] + S * 0.026], fill=shoe)
    # torso (clothed, tapered), belt, collar
    _tcap(d, sh, hip, S * 0.072, S * 0.064, shirt)
    nrm = (-up[1], up[0])
    belt_a = (hip[0] - nrm[0] * S * 0.064 + up[0] * S * 0.012, hip[1] - nrm[1] * S * 0.064 + up[1] * S * 0.012)
    belt_b = (hip[0] + nrm[0] * S * 0.064 + up[0] * S * 0.012, hip[1] + nrm[1] * S * 0.064 + up[1] * S * 0.012)
    d.line([belt_a, belt_b], fill=(58, 42, 32), width=max(3, int(S * 0.016)))
    # neck
    _tcap(d, sh, add(neck_base, (hup[0] * S * 0.02, hup[1] * S * 0.02)), S * 0.026, S * 0.024, _shade(skin, 0.92))
    # near arm over the torso
    _tcap(d, sh, a1, S * 0.028, S * 0.022, shirt); _tcap(d, a1, a2, S * 0.022, S * 0.018, shirt)
    d.ellipse([a2[0] - S * 0.028, a2[1] - S * 0.028, a2[0] + S * 0.028, a2[1] + S * 0.028], fill=skin)
    # head
    d.ellipse([head[0] - rad, head[1] - rad, head[0] + rad, head[1] + rad], fill=skin)
    lw = max(4, S * 0.042)
    if "beard" in props:
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
        tip = add(hand, K.seg((0, 0), dirv, 0.40 * S, f))
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
    if "spear" in props:
        p0, p1 = add(hand, K.seg((0, 0), dirv + 180, .2 * S, f)), add(hand, K.seg((0, 0), dirv, .5 * S, f))
        K.line(d, [p0, p1], lw * .5, (130, 96, 62))
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        un = math.hypot(ux, uy) or 1.0
        ux, uy = ux / un, uy / un
        tipp = (p1[0] + ux * S * 0.075, p1[1] + uy * S * 0.075)
        d.polygon([(p1[0] - uy * S * 0.02, p1[1] + ux * S * 0.02), tipp, (p1[0] + uy * S * 0.02, p1[1] - ux * S * 0.02)], fill=(206, 212, 224))   # steel head
    if "shield" in props:
        sr = 0.085 * S
        d.ellipse([b2[0] - sr, b2[1] - sr, b2[0] + sr, b2[1] + sr], fill=col)
    if "scroll" in props:
        d.rectangle([hand[0] - S * .05, hand[1] - S * .06, hand[0] + S * .05, hand[1] + S * .06], fill=(232, 215, 170))
    if "hardhat" in props:
        d.pieslice([head[0] - rad * 1.1, head[1] - rad * 1.25, head[0] + rad * 1.1, head[1] + rad * .15], 180, 360, fill=(246, 200, 40))
        d.rectangle([head[0] - rad * 1.3, head[1] - rad * .18, head[0] + rad * 1.3, head[1] - rad * .02], fill=(226, 176, 28))
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
    arr_c, (cx0, cy0) = _post(np.asarray(L), S, light, rim, max(3, S * 0.0075))
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
    _face(d2, a, K, head, rad, f, face, ex, S, t, skin, props)
    # ---- cast shadow on the ground, then composite ------------------------------------------------
    A = arr[..., 3]
    shadow = _cast_shadow(A, light, S, gl, bh, bw)
    if shadow is not None:
        img.paste((10, 10, 24), (int(ox), int(oy)), shadow)
    img.paste(Lp.convert("RGB"), (int(ox), int(oy)), Lp.split()[3])


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


def _face(d, a, K, head, rad, f, face, ex, S, t, skin, props):
    lw = max(4, S * 0.042)
    look = ex.get("look", (0.35 * f, 0.0))                             # (dx, dy) in eye radii, signed to world x
    blink = ex.get("blink", 0.0)                                       # 0 open .. 1 shut
    mo = ex.get("mouth", 0.0)                                          # 0 closed .. 1 wide open
    talk_brow = ex.get("brow", 0.0)
    er_x, er_y = rad * 0.185, rad * 0.255
    big = 1.18 if face == "shock" else 1.0
    eyes = [(head[0] + f * rad * 0.28, head[1] - rad * 0.12), (head[0] + f * rad * 0.72, head[1] - rad * 0.12)]
    far = 0
    # nose: a small rounded bump on the facing side
    nx, ny = head[0] + f * rad * 1.0, head[1] + rad * 0.12
    d.ellipse([nx - rad * .13, ny - rad * .11, nx + rad * .13, ny + rad * .13], fill=(236, 190, 152), outline=(150, 100, 80), width=max(1, int(S * 0.004)))
    # cheeks
    if face == "smile":
        for cx_ in (head[0] + f * rad * 0.52,):
            d.ellipse([cx_ - rad * .2, head[1] + rad * .16, cx_ + rad * .2, head[1] + rad * .38], fill=(244, 170, 160, 120))
    lid_base = {"worried": 0.22, "sad": 0.34, "angry": 0.28, "smile": 0.12, "shock": 0.0}.get(face, 0.06)
    lid = max(lid_base, blink)
    for i_, (ex_, ey_) in enumerate(eyes):
        erx, ery = er_x * big, er_y * big
        if i_ == 1:
            erx *= 0.88
        d.ellipse([ex_ - erx, ey_ - ery, ex_ + erx, ey_ + ery], fill=(255, 255, 255), outline=INK, width=max(2, int(S * 0.006)))
        pr = erx * (0.62 if face != "shock" else 0.42)
        px_ = ex_ + look[0] * erx * 0.55
        py_ = ey_ + look[1] * ery * 0.55
        d.ellipse([px_ - pr, py_ - pr * 1.12, px_ + pr, py_ + pr * 1.12], fill=(30, 26, 34))
        d.ellipse([px_ - pr * 0.42 - pr * 0.1, py_ - pr * 0.62, px_ + pr * 0.12 - pr * 0.1, py_ - pr * 0.1], fill=(255, 255, 255))   # specular
        if lid > 0.02:                                                  # eyelid: skin-coloured cap that closes down over the eye
            ly = ey_ - ery + 2 * ery * lid
            slope = {"angry": 0.35, "worried": -0.3, "sad": -0.35}.get(face, 0.0)
            d.polygon([(ex_ - erx * 1.12, ey_ - ery * 1.15), (ex_ + erx * 1.12, ey_ - ery * 1.15),
                       (ex_ + erx * 1.12, ly + slope * ery * f * (1 if i_ == 0 else -1) * 0.8), (ex_ - erx * 1.12, ly - slope * ery * f * (1 if i_ == 0 else -1) * 0.8)], fill=skin)
            d.line([(ex_ - erx * 1.05, ly), (ex_ + erx * 1.05, ly)], fill=INK, width=max(2, int(S * 0.006)))
    # brows
    bw = max(3, int(S * 0.012))
    by = head[1] - rad * 0.56 - talk_brow * rad * 0.10
    tilt = {"angry": (-1, 0.34), "worried": (1, 0.32), "sad": (1, 0.30), "shock": (0, 0.0), "smile": (0, 0.0)}.get(face, (0, 0))
    lift = rad * (0.14 if face == "shock" else 0.0)
    for k_, (ex_, ey_) in enumerate(eyes):
        sgn = 1 if k_ == 0 else -1
        din = tilt[0] * rad * tilt[1] * sgn * f * -1
        d.line([(ex_ - rad * .20, by - lift + din), (ex_ + rad * .20, by - lift - din)], fill=(46, 36, 30), width=bw)
    # mouth
    mx, my = head[0] + f * rad * 0.46, head[1] + rad * 0.52
    mw = max(3, int(S * 0.011))
    if mo > 0.12:
        wd = rad * (0.19 + 0.10 * mo)
        ht = rad * (0.05 + 0.34 * mo)
        d.ellipse([mx - wd, my - ht * 0.45, mx + wd, my + ht], fill=(96, 34, 44), outline=INK, width=max(2, int(S * 0.006)))
        if mo > 0.35:
            d.ellipse([mx - wd * 0.6, my + ht * 0.2, mx + wd * 0.6, my + ht * 0.92], fill=(214, 96, 104))
            d.rectangle([mx - wd * 0.7, my - ht * 0.42, mx + wd * 0.7, my - ht * 0.18], fill=(250, 250, 244))
    elif face == "smile":
        d.arc([mx - rad * .34, my - rad * .26, mx + rad * .34, my + rad * .22], 15, 165, fill=INK, width=mw)
    elif face in ("sad", "worried"):
        d.arc([mx - rad * .30, my - rad * .02, mx + rad * .30, my + rad * .42], 200, 340, fill=INK, width=mw)
    elif face == "angry":
        d.line([mx - rad * .30, my + rad * .06, mx + rad * .30, my - rad * .02], fill=INK, width=mw)
    elif face == "shock":
        d.ellipse([mx - rad * .17, my - rad * .08, mx + rad * .17, my + rad * .34], fill=(96, 34, 44), outline=INK, width=max(2, int(S * 0.006)))
    else:
        d.line([mx - rad * .24, my, mx + rad * .24, my], fill=INK, width=mw)
    if "glasses" in props:
        for ex_, ey_ in eyes:
            d.ellipse([ex_ - er_x * 1.7, ey_ - er_x * 1.7, ex_ + er_x * 1.7, ey_ + er_x * 1.7], outline=INK, width=max(2, int(S * 0.005)))
        d.line([eyes[0][0] + er_x * 1.7, eyes[0][1], eyes[1][0] - er_x * 1.7, eyes[1][1]], fill=INK, width=max(2, int(S * 0.005)))
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
