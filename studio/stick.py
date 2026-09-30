"""Code-drawn stickman animation. Free forever: no AI video model, just maths and drawing.

A scene is plain data (actors with keyframes, objects, on-screen text), so an LLM can write it.
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.render import _find_font

PAPER, INK = (244, 238, 224), (27, 27, 32)
GOLD, RED, BLUE, PURPLE, GREY = (226, 176, 40), (196, 57, 43), (47, 99, 176), (122, 59, 140), (150, 146, 138)
SS = 2  # supersampling for smooth lines


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


# ---- poses: angles in degrees, measured from "straight down", positive = toward the way the figure faces ----
BASE = dict(torso=0, head=0, a1=8, a2=10, b1=-8, b2=10, l1=4, l2=0, m1=-4, m2=0, hip=0, lift=0)
POSES = {
    "stand": {},
    "proud": dict(a1=155, a2=0, b1=-155, b2=0, torso=-4, head=-6),
    "point": dict(a1=92, a2=4, b1=-10, b2=10, torso=3),
    "shrug": dict(a1=40, a2=75, b1=-40, b2=75, head=10),
    "scared": dict(a1=120, a2=-70, b1=-120, b2=-70, torso=-8, head=6, l1=14, l2=-10, m1=-14, m2=-10),
    "sword_up": dict(a1=150, a2=-15, b1=25, b2=40, torso=-4),
    "slump": dict(torso=24, head=26, a1=-5, a2=6, b1=-20, b2=6, l1=18, l2=-28, m1=-8, m2=-18),
    "crouch": dict(torso=14, l1=55, l2=-95, m1=40, m2=-90, a1=30, a2=50, b1=10, b2=60),
}


def pose_at(name, t, seed=0.0):
    p = dict(BASE)
    if name in ("walk", "run"):
        f, amp, lean = (1.7, 30, 3) if name == "walk" else (2.7, 52, 16)
        ph = 2 * math.pi * f * t + seed
        s_, c_ = math.sin(ph), math.cos(ph)
        bend = 38 if name == "walk" else 105
        # a leg bends its knee while it swings forward (thigh angle rising)
        p.update(l1=amp * s_, m1=-amp * s_,
                 l2=-bend * max(0.0, c_) - 3, m2=-bend * max(0.0, -c_) - 3,
                 torso=lean, a1=-amp * 0.8 * s_, b1=amp * 0.8 * s_)
        if name == "run":
            p.update(a2=85, b2=85, lift=0.055 * abs(c_), head=-4)
    elif name == "sprint_hit":
        p.update(POSES["sword_up"])
    elif name == "cheer":
        w = math.sin(2 * math.pi * 2.2 * t + seed)
        p.update(a1=150 + 14 * w, b1=-150 - 14 * w, a2=0, b2=0, head=-8, l2=-8 * abs(w), m2=-8 * abs(w))
    elif name == "swing":
        w = 0.5 + 0.5 * math.sin(2 * math.pi * 1.4 * t + seed)
        p.update(a1=lerp(165, 55, w), a2=-10, b1=-25, b2=50, torso=lerp(-6, 10, w), l1=18, m1=-14, l2=-8)
    elif name == "tremble":
        w = math.sin(2 * math.pi * 9 * t + seed)
        p.update(POSES["scared"])
        p.update(head=6 + 3 * w)
    else:
        p.update(POSES.get(name, {}))
    return p


def blend(pa, pb, u):
    return {k: lerp(pa[k], pb[k], u) for k in pa}


# ---- drawing helpers -------------------------------------------------------------------------
def line(d, pts, w, col=INK):
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=col, width=int(w))
    for p in pts:
        d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=col)


def seg(p, ang, L, f):
    a = math.radians(ang)
    return (p[0] + math.sin(a) * L * f, p[1] + math.cos(a) * L)


class Actor:
    def __init__(self, spec, W, H):
        self.s = spec
        self.W, self.H = W, H
        self.unit = H * 0.40 * spec.get("scale", 1.0) * SS
        self.seed = hash(spec["id"]) % 7

    def key_state(self, t):
        ks = self.s["keys"]
        k0, k1 = ks[0], ks[-1]
        for a, b in zip(ks, ks[1:]):
            if a["t"] <= t <= b["t"]:
                k0, k1 = a, b
                break
        else:
            if t < ks[0]["t"]:
                k1 = k0
            else:
                k0 = k1
        u = smooth((t - k0["t"]) / max(k1["t"] - k0["t"], 1e-6)) if k1 is not k0 else 0
        pose = blend(pose_at(k0["pose"], t, self.seed), pose_at(k1["pose"], t, self.seed), u)
        x = lerp(k0["x"], k1["x"], u)
        facing = k0.get("facing", 1) if u < 0.5 else k1.get("facing", k0.get("facing", 1))
        face = k0.get("face", "neutral") if u < 0.5 else k1.get("face", "neutral")
        return pose, x, facing, face

    def draw(self, d, t, gy):
        S, W = self.unit, self.W * SS
        pose, xr, f, face = self.key_state(t)
        x = xr * W
        L = dict(torso=0.30, upper=0.17, fore=0.17, thigh=0.22, shin=0.22, head=0.085, neck=0.035)
        def vec(a, length):
            return seg((0, 0), a, length * S, f)
        l1 = seg((0, 0), pose["l1"], L["thigh"] * S, f)
        l2 = seg(l1, pose["l1"] + pose["l2"], L["shin"] * S, f)
        m1 = seg((0, 0), pose["m1"], L["thigh"] * S, f)
        m2 = seg(m1, pose["m1"] + pose["m2"], L["shin"] * S, f)
        drop = max(l2[1], m2[1])
        hip = (x, gy - drop - pose.get("lift", 0) * S)
        sw = S * 0.15 * (1 - min(0.5, pose.get("lift", 0) * 5))
        d.ellipse([x - sw, gy - S * 0.012, x + sw, gy + S * 0.022], fill=(0, 0, 0, 70))
        tr = math.radians(pose["torso"])
        up = (math.sin(tr) * f, -math.cos(tr))
        neck_base = (hip[0] + up[0] * L["torso"] * S, hip[1] + up[1] * L["torso"] * S)
        sh = (hip[0] + up[0] * L["torso"] * 0.9 * S, hip[1] + up[1] * L["torso"] * 0.9 * S)
        hr = math.radians(pose["torso"] + pose["head"])
        hup = (math.sin(hr) * f, -math.cos(hr))
        rad = L["head"] * S
        head = (neck_base[0] + hup[0] * (L["neck"] * S + rad), neck_base[1] + hup[1] * (L["neck"] * S + rad))
        lw = max(3, S * 0.034)
        col = self.s.get("color", INK)
        skin = self.s.get("skin", (247, 222, 190))
        tunic = self.s.get("tunic")

        # cape (behind body)
        if "cape" in self.s.get("props", []):
            flap = math.sin(t * 5 + self.seed) * S * 0.03
            d.polygon([sh, (sh[0] - f * S * 0.30 + flap, sh[1] + S * 0.42), (sh[0] - f * S * 0.05, sh[1] + S * 0.46)], fill=col)
        add = lambda p, q: (p[0] + q[0], p[1] + q[1])
        wide = pose["a1"], pose["a2"], pose["b1"], pose["b2"]
        a1 = add(sh, seg((0, 0), wide[0], L["upper"] * S, f)); a2 = add(a1, seg((0, 0), wide[0] + wide[1], L["fore"] * S, f))
        b1 = add(sh, seg((0, 0), wide[2], L["upper"] * S, f)); b2 = add(b1, seg((0, 0), wide[2] + wide[3], L["fore"] * S, f))
        line(d, [hip, add(hip, m1), add(hip, m2)], lw)
        for pts in ([add(hip, m1), add(hip, m2)], [add(hip, l1), add(hip, l2)]):  # boots
            foot = pts[-1]
            d.line([foot, (foot[0] + f * S * 0.055, foot[1])], fill=(62, 44, 34), width=int(lw * 1.25))
        line(d, [hip, add(hip, l1), add(hip, l2)], lw)
        nrm = (-up[1], up[0])
        if tunic:
            tw, bw = S * 0.055, S * 0.075
            d.polygon([(sh[0] + nrm[0] * tw, sh[1] + nrm[1] * tw), (sh[0] - nrm[0] * tw, sh[1] - nrm[1] * tw),
                       (hip[0] - nrm[0] * bw, hip[1] - nrm[1] * bw + S * 0.05), (hip[0] + nrm[0] * bw, hip[1] + nrm[1] * bw + S * 0.05)],
                      fill=tunic, outline=INK)
            d.line([(hip[0] - nrm[0] * bw * .9, hip[1] - nrm[1] * bw * .9 + S * .012), (hip[0] + nrm[0] * bw * .9, hip[1] + nrm[1] * bw * .9 + S * .012)], fill=(70, 50, 34), width=int(lw * .55))
        else:
            line(d, [hip, neck_base], lw)
        line(d, [sh, b1, b2], lw)
        line(d, [sh, a1, a2], lw)
        for hnd in (a2, b2):
            d.ellipse([hnd[0] - lw * .75, hnd[1] - lw * .75, hnd[0] + lw * .75, hnd[1] + lw * .75], fill=skin, outline=INK, width=2)
        d.ellipse([head[0] - rad, head[1] - rad, head[0] + rad, head[1] + rad], fill=skin, outline=INK, width=int(lw * .85))
        if "beard" in self.s.get("props", []):
            d.pieslice([head[0] - rad * .95, head[1] - rad * .5, head[0] + rad * .95, head[1] + rad * 1.35], 10, 170, fill=(236, 232, 224), outline=INK, width=int(lw * .5))
        if "hair" in self.s.get("props", []):
            d.pieslice([head[0] - rad * 1.05, head[1] - rad * 1.08, head[0] + rad * 1.05, head[1] + rad * .9], 180, 360, fill=self.s.get("hair", (70, 48, 30)), outline=INK, width=int(lw * .5))
        # face
        ex = head[0] + f * rad * 0.35
        ey = head[1] - rad * 0.12
        er = max(2, rad * 0.10)
        for dx in (0.0, 0.42):
            ox = ex + f * rad * dx * 0.6 - f * rad * 0.15
            d.ellipse([ox - er, ey - er, ox + er, ey + er], fill=INK)
        my, mx = head[1] + rad * 0.42, head[0] + f * rad * 0.22
        if face == "smile":
            d.arc([mx - rad * .3, my - rad * .3, mx + rad * .3, my + rad * .15], 10, 170, fill=INK, width=int(lw * .6))
        elif face in ("sad", "angry", "worried"):
            d.arc([mx - rad * .3, my - rad * .05, mx + rad * .3, my + rad * .45], 190, 350, fill=INK, width=int(lw * .6))
            if face == "angry":
                d.line([ex - er * 2, ey - er * 2.6, ex + er * 3.5, ey - er * 1.4], fill=INK, width=int(lw * .6))
        elif face == "shock":
            d.ellipse([mx - rad * .16, my - rad * .16, mx + rad * .16, my + rad * .22], outline=INK, width=int(lw * .6))
        else:
            d.line([mx - rad * .25, my, mx + rad * .25, my], fill=INK, width=int(lw * .6))
        # props
        props = self.s.get("props", [])
        if "helmet" in props:
            d.pieslice([head[0] - rad * 1.08, head[1] - rad * 1.1, head[0] + rad * 1.08, head[1] + rad * 1.05], 180, 360, fill=col, outline=INK, width=int(lw * .6))
        if "crown" in props:
            fall = self.s.get("crown_fall")
            cx, cy = head[0], head[1] - rad * 0.95
            if fall is not None and t > fall:
                dt = t - fall
                cy = min(cy + 0.5 * S * 3.2 * dt * dt, gy - S * 0.03)
                cx = cx + f * S * 0.10 * min(dt, 0.6) * 1.6
            w, h = rad * 0.9, rad * 0.75
            pts = [(cx - w, cy), (cx - w, cy - h), (cx - w / 2, cy - h / 2), (cx, cy - h * 1.1), (cx + w / 2, cy - h / 2), (cx + w, cy - h), (cx + w, cy)]
            d.polygon(pts, fill=GOLD, outline=INK)
        hand, dirv = add(sh, add(seg((0, 0), wide[0], L["upper"] * S, f), seg((0, 0), wide[0] + wide[1], L["fore"] * S, f))), wide[0] + wide[1]
        if "sword" in props:
            tip = add(hand, seg((0, 0), dirv, 0.34 * S, f))
            line(d, [hand, tip], lw * 0.6, (90, 90, 100))
            gx = seg((0, 0), dirv + 90, 0.05 * S, f)
            d.line([add(hand, gx), (hand[0] - gx[0], hand[1] - gx[1])], fill=INK, width=int(lw * .7))
        if "spear" in props:
            line(d, [add(hand, seg((0, 0), dirv + 180, .2 * S, f)), add(hand, seg((0, 0), dirv, .5 * S, f))], lw * .5, (110, 80, 50))
        if "shield" in props:
            sr = 0.085 * S
            d.ellipse([b2[0] - sr, b2[1] - sr, b2[0] + sr, b2[1] + sr], fill=col, outline=INK, width=int(lw * .6))
        if "scroll" in props:
            d.rectangle([hand[0] - S * .05, hand[1] - S * .06, hand[0] + S * .05, hand[1] + S * .06], fill=(232, 215, 170), outline=INK)
        if "flag" in props:
            top = add(hand, seg((0, 0), dirv + 180, .05 * S, f)); pole = (hand[0], hand[1] - S * 0.45)
            line(d, [hand, pole], lw * .5, (110, 80, 50))
            d.polygon([pole, (pole[0] + f * S * .22, pole[1] + S * .06 + math.sin(t * 6) * S * .015), (pole[0], pole[1] + S * .13)], fill=col)


def draw_object(d, o, W, H, gy, t):
    x = o["x"] * W * SS
    S = H * SS
    k = o["type"]
    if k == "castle":
        w, h = S * 0.30 * o.get("scale", 1), S * 0.36 * o.get("scale", 1)
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(198, 190, 172), outline=INK, width=4)
        tw = w / 7
        for i in range(4):
            d.rectangle([x - w / 2 + i * 2 * tw, gy - h - tw, x - w / 2 + i * 2 * tw + tw, gy - h], fill=(198, 190, 172), outline=INK, width=3)
        d.rounded_rectangle([x - w * 0.12, gy - h * 0.42, x + w * 0.12, gy], 40, fill=INK)
    elif k == "pedestal":
        w, h = S * .16, S * .06
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(190, 183, 168), outline=INK, width=4)
    elif k == "cloud":
        cx, cy, r = x + math.sin(t * .6) * S * .01, o.get("y", .18) * S, S * o.get("r", .06)
        for dx, dy, rr in ((-1.2, .3, .8), (0, 0, 1.1), (1.2, .3, .9), (.2, .55, .9)):
            d.ellipse([cx + dx * r - rr * r, cy + dy * r - rr * r, cx + dx * r + rr * r, cy + dy * r + rr * r], fill=o.get("color", (200, 196, 186)))
    elif k == "sun":
        d.ellipse([x - S * .07, o["y"] * S - S * .07, x + S * .07, o["y"] * S + S * .07], fill=(250, 214, 120), outline=INK, width=4)
    elif k == "column":
        w, h = S * .05, S * .30
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(225, 219, 204), outline=INK, width=4)
        d.rectangle([x - w, gy - h - S * .02, x + w, gy - h], fill=(225, 219, 204), outline=INK, width=4)


def font(size):
    return ImageFont.truetype(_find_font("assets/fonts/BricolageGrotesque-Bold.ttf"), size)


def _hills(d, W, H, gy, layer, off):
    n = 40
    base, amp, fr, seed = layer["base"] * H * SS, layer["amp"] * H * SS, layer.get("freq", 3.0), layer.get("seed", 0)
    pts = [(-10, H * SS)]
    for i in range(n + 1):
        x = i / n
        y = base - amp * (math.sin(x * fr + seed) + 0.5 * math.sin(x * fr * 2.3 + seed * 2) + 0.25 * math.sin(x * fr * 5 + seed * 3))
        pts.append((x * W * SS - off * layer.get("par", 0.2), y))
    pts.append((W * SS + 10, H * SS))
    d.polygon(pts, fill=layer["color"])


def _sky(img_arr_h, W, H, top, bottom):
    t = np.linspace(0, 1, H * SS)[:, None, None]
    return np.clip(np.array(top) * (1 - t) + np.array(bottom) * t, 0, 255).astype(np.uint8).repeat(W * SS, axis=1)


def draw_map(d, scene, W, H, t):
    S = H * SS
    w = W * SS
    d.rectangle([0, 0, w, S], fill=(214, 190, 142))
    for k in range(18):  # worn parchment edges
        d.rectangle([k * 9, k * 9, w - k * 9, S - k * 9], outline=(120, 86, 40, 10 + k * 3), width=16)
    for poly in scene["land"]:
        pts = [(x * w, y * S) for x, y in poly["pts"]]
        d.polygon(pts, fill=poly.get("color", (236, 222, 184)), outline=INK)
        d.line(pts + [pts[0]], fill=INK, width=5)
        if poly.get("label"):
            f_ = font(int(S * 0.042))
            lx, ly = poly["label_at"]
            d.text((lx * w - d.textlength(poly["label"], font=f_) / 2, ly * S), poly["label"], font=f_, fill=poly.get("label_color", (60, 44, 26)))
    for c in scene.get("cities", []):
        cx, cy = c["x"] * w, c["y"] * S
        d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=GOLD, outline=INK, width=4)
        f_ = font(int(S * 0.034))
        d.text((cx - d.textlength(c["name"], font=f_) / 2, cy + 24), c["name"], font=f_, fill=INK, stroke_width=3, stroke_fill=(236, 204, 132))
    for ar in scene.get("arrows", []):
        pr = smooth((t - ar["t0"]) / max(ar["t1"] - ar["t0"], 1e-6))
        if pr <= 0:
            continue
        pts = [(x * w, y * S) for x, y in ar["pts"]]
        lens = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        goal, run, cur = sum(lens) * pr, 0.0, [pts[0]]
        for i, L_ in enumerate(lens):
            if run + L_ <= goal:
                cur.append(pts[i + 1]); run += L_
            else:
                u = (goal - run) / max(L_, 1e-6)
                cur.append((lerp(pts[i][0], pts[i + 1][0], u), lerp(pts[i][1], pts[i + 1][1], u)))
                break
        col = ar.get("color", RED)
        for i in range(len(cur) - 1):
            d.line([cur[i], cur[i + 1]], fill=col, width=int(S * 0.016))
        (x1, y1), (x2, y2) = cur[-2], cur[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        hs = S * 0.04
        d.polygon([(x2 + math.cos(ang) * hs, y2 + math.sin(ang) * hs),
                   (x2 + math.cos(ang + 2.5) * hs, y2 + math.sin(ang + 2.5) * hs),
                   (x2 + math.cos(ang - 2.5) * hs, y2 + math.sin(ang - 2.5) * hs)], fill=col)
        if ar.get("label") and pr > 0.15:
            f_ = font(int(S * 0.04))
            lx, ly = ar["label_at"]
            d.text((lx * w, ly * S), ar["label"], font=f_, fill=col)


def draw_fx(d, scene, t, W, H, gy, actors):
    S = H * SS
    for fx in scene.get("fx", []):
        k = fx["type"]
        if k == "rain" and fx["t0"] <= t <= fx["t1"]:
            rng = np.random.default_rng(3)
            xs, ys, sp = rng.random(160), rng.random(160), 0.9 + rng.random(160) * 0.8
            for i in range(160):
                x = ((xs[i] + t * 0.12 * sp[i]) % 1.0) * W * SS
                y = ((ys[i] + t * 1.6 * sp[i]) % 1.0) * S
                d.line([(x, y), (x - S * 0.012, y + S * 0.04)], fill=(60, 70, 100, 120), width=3)
        elif k == "sparks" and fx["t"] <= t <= fx["t"] + 0.4:
            age = (t - fx["t"]) / 0.4
            cx, cy = fx["x"] * W * SS, fx.get("y", 0.6) * S
            rng = np.random.default_rng(int(fx["t"] * 100))
            for i in range(14):
                a_ = rng.random() * 6.283
                r0, r1 = S * 0.02 * age, S * (0.03 + 0.09 * rng.random()) * (0.3 + age)
                d.line([(cx + math.cos(a_) * r0, cy + math.sin(a_) * r0), (cx + math.cos(a_) * r1, cy + math.sin(a_) * r1)],
                       fill=(255, 210, 90, int(255 * (1 - age))), width=5)
        elif k == "dust":
            for a in actors:
                if a.s["id"] != fx["actor"]:
                    continue
                for j in range(9):
                    tk = t - j * 0.055
                    if tk < 0:
                        continue
                    x0, x1 = a.key_state(tk)[1], a.key_state(max(tk - 0.1, 0))[1]
                    if abs(x0 - x1) * W < 18:
                        continue
                    age = j / 9
                    r = S * (0.012 + 0.03 * age)
                    cx, cy = x0 * W * SS, gy - r * 0.6 - age * S * 0.02
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 130, 100, int(110 * (1 - age))))
        elif k == "flash" and fx["t"] <= t <= fx["t"] + 0.5:
            pass  # drawn later over the whole frame


def render_frame(scene, t, W, H):
    gyr = scene.get("ground", 0.80)
    gy = gyr * H * SS
    fx0, fx1 = scene.get("focus", (0.5, 0.55)), scene.get("focus_to", None)
    u = smooth(t / max(scene["duration"], 1e-6))
    fxc = lerp(fx0[0], (fx1 or fx0)[0], u)
    off = (fxc - 0.5) * W * SS
    if scene.get("kind") == "map":
        img = Image.new("RGB", (W * SS, H * SS), (214, 190, 142))
        d = ImageDraw.Draw(img, "RGBA")
        draw_map(d, scene, W, H, t)
    else:
        top, bot = scene.get("sky", ((196, 214, 226), (246, 236, 214)))
        img = Image.fromarray(_sky(None, W, H, top, bot))
        d = ImageDraw.Draw(img, "RGBA")
        if scene.get("sun"):
            sx, sy, sc = scene["sun"]
            for r, al in ((0.22, 18), (0.14, 30), (0.085, 255)):
                d.ellipse([sx * W * SS - r * H * SS, sy * H * SS - r * H * SS, sx * W * SS + r * H * SS, sy * H * SS + r * H * SS],
                          fill=sc + (al,))
        for layer in scene.get("hills", []):
            _hills(d, W, H, gy, layer, off)
        gcol = scene.get("ground_color", (176, 158, 120))
        d.rectangle([0, gy, W * SS, H * SS], fill=gcol)
        for i in range(10):  # ground gradient darker toward the bottom
            d.rectangle([0, gy + (H * SS - gy) * i / 10, W * SS, H * SS], fill=(0, 0, 0, 10))
        d.line([(0, gy), (W * SS, gy)], fill=INK, width=5)
        rng = np.random.default_rng(11)
        for _ in range(70):
            x, y = rng.random() * W * SS, gy + 14 + rng.random() * (H * SS - gy) * 0.8
            d.line([(x, y), (x + 26 + rng.random() * 30, y)], fill=(0, 0, 0, 38), width=3)
        for o in scene.get("objects", []):
            draw_object(d, o, W, H, gy, t)
        for a in sorted(scene["_actors"], key=lambda a: a.s.get("z", 0)):
            a.draw(d, t, gy)
        draw_fx(d, scene, t, W, H, gy, scene["_actors"])
        if scene.get("dim"):
            d.rectangle([0, 0, W * SS, H * SS], fill=(20, 24, 50, int(scene["dim"] * 255)))
        for fx in scene.get("fx", []):
            if fx["type"] == "flash" and fx["t"] <= t <= fx["t"] + 0.5:
                d.rectangle([0, 0, W * SS, H * SS], fill=(255, 255, 255, int(200 * (1 - (t - fx["t"]) / 0.5))))
    # camera: push-in, pan and impact shake
    z0, z1 = scene.get("zoom", [1.0, 1.06])
    z = lerp(z0, z1, u)
    cx = lerp(fx0[0], (fx1 or fx0)[0], u)
    cy = fx0[1]
    sx = sy = 0.0
    for sh_ in scene.get("shake", []):
        if sh_["t"] <= t <= sh_["t"] + sh_.get("dur", 0.4):
            k = 1 - (t - sh_["t"]) / sh_.get("dur", 0.4)
            sx = math.sin(t * 90) * sh_.get("amp", 14) * k * SS
            sy = math.cos(t * 77) * sh_.get("amp", 14) * k * SS
    M = cv2.getRotationMatrix2D((cx * W * SS, cy * H * SS), 0, z / SS)
    M[0, 2] += W / 2 - cx * W * SS + sx / SS
    M[1, 2] += H / 2 - cy * H * SS + sy / SS
    out = cv2.warpAffine(np.asarray(img), M, (W, H), flags=cv2.INTER_AREA if z / SS < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    out = _grade(out, W, H, t)
    pil = Image.fromarray(out)
    dd = ImageDraw.Draw(pil, "RGBA")
    for tx in scene.get("text", []):
        if tx["t"] <= t < tx.get("end", scene["duration"]):
            age = t - tx["t"]
            a = smooth(age / 0.35) * (1 - smooth((t - (tx.get("end", scene["duration"]) - 0.3)) / 0.3))
            size = int(tx.get("size", 0.085) * H * (0.9 + 0.1 * smooth(age / 0.35)))
            f_ = font(size)
            w = dd.textlength(tx["text"], font=f_)
            px, py = (W - w) / 2, tx.get("y", 0.12) * H
            dd.text((px + 4, py + 5), tx["text"], font=f_, fill=(0, 0, 0, int(70 * a)))
            dd.text((px, py), tx["text"], font=f_, fill=tx.get("color", INK) + (int(255 * a),), stroke_width=1, stroke_fill=INK + (int(255 * a),))
            ub = smooth((age - 0.1) / 0.5)  # ink underline that draws itself in
            dd.line([(px, py + size * 1.12), (px + w * ub, py + size * 1.12)], fill=tx.get("color", INK) + (int(230 * a),), width=max(4, size // 14))
    return np.asarray(pil)


_VIG = {}


def _grade(arr, W, H, t):
    key = (W, H)
    if key not in _VIG:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
        _VIG[key] = (1 - 0.38 * np.clip(r, 0, 1) ** 2.3)[..., None]
    f = arr.astype(np.float32) * _VIG[key] * np.array([1.03, 1.0, 0.93], dtype=np.float32)
    rng = np.random.default_rng(int(t * 30) % 12)
    f += rng.standard_normal((H, W, 1)).astype(np.float32) * 4.0
    return np.clip(f, 0, 255).astype(np.uint8)


def prepare(scene, W, H):
    scene["_actors"] = [Actor(a, W, H) for a in scene.get("actors", [])]
    return scene
