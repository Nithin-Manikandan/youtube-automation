"""Code-drawn stickman animation. Free forever: no AI video model, just maths and drawing.

A scene is plain data (actors with keyframes, objects, on-screen text), so an LLM can write it.
"""
import math
from . import art

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
    "think": dict(a1=62, a2=128, b1=-8, b2=12, torso=3, head=8),
    "salute": dict(a1=150, a2=118, b1=-8, b2=10, torso=-2, head=-3),
    "armscross": dict(a1=34, a2=112, b1=-34, b2=112, head=2),
    "facepalm": dict(a1=110, a2=132, b1=-10, b2=10, torso=9, head=16),
    "demand": dict(a1=98, a2=10, b1=-98, b2=10, torso=5, head=-2),
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
    if name not in ("walk", "run"):                       # breathing / weight shifts: nobody is ever frozen
        w = 2 * math.pi * t
        p["torso"] += 1.4 * math.sin(w * 0.33 + seed)
        p["head"] += 2.0 * math.sin(w * 0.47 + seed * 2)
        p["a1"] += 2.6 * math.sin(w * 0.41 + seed * 3)
        p["b1"] -= 2.6 * math.sin(w * 0.37 + seed * 4)
        p["l1"] += 1.2 * math.sin(w * 0.29 + seed)
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
        self.unit = H * 0.52 * spec.get("scale", 1.0) * SS
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
        L = dict(torso=0.30, upper=0.17, fore=0.17, thigh=0.22, shin=0.22, head=0.095, neck=0.03)
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
        lw = max(4, S * 0.042)
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
            hr = self.s.get("hair", (70, 48, 30))
            bcol = hr if sum(hr) < 420 else (150, 150, 146)           # dark beard, or a grey one for old characters, never a white bib
            d.pieslice([head[0] - rad * .98, head[1] - rad * .4, head[0] + rad * .98, head[1] + rad * 1.35], 5, 175, fill=bcol, outline=INK, width=int(lw * .5))
            d.ellipse([head[0] - rad * .72, head[1] - rad * .55, head[0] + rad * .72, head[1] + rad * .66], fill=skin)   # face shows through; beard remains as a jaw band
            d.rectangle([head[0] - rad * .5, head[1] + rad * .2, head[0] + rad * .5, head[1] + rad * .3], fill=bcol)       # moustache
        if "hair" in self.s.get("props", []):
            d.pieslice([head[0] - rad * 1.05, head[1] - rad * 1.10, head[0] + rad * 1.05, head[1] + rad * .25], 180, 360, fill=self.s.get("hair", (70, 48, 30)), outline=INK, width=int(lw * .5))
        # face: eyes with pupils, expressive brows and mouth
        eyes = [(head[0] + f * rad * 0.18, head[1] - rad * 0.10), (head[0] + f * rad * 0.62, head[1] - rad * 0.10)]
        er = max(3.0, rad * 0.20)
        look = 0.35 if face != "sad" else 0.1
        for ex_, ey_ in eyes:
            d.ellipse([ex_ - er, ey_ - er, ex_ + er, ey_ + er], fill=(255, 255, 255), outline=INK, width=2)
            pr = er * (0.42 if face != "shock" else 0.28)
            d.ellipse([ex_ + f * er * look - pr, ey_ - pr, ex_ + f * er * look + pr, ey_ + pr], fill=INK)
        by = head[1] - rad * 0.42
        bw = max(3, int(lw * .55))
        tilt = {"angry": (-1, 0.30), "worried": (1, 0.32), "sad": (1, 0.28), "shock": (0, -0.30), "smile": (0, -0.10)}.get(face, (0, 0))
        for k_, (ex_, ey_) in enumerate(eyes):
            sgn = 1 if k_ == 0 else -1
            dy_in, dy_out = tilt[0] * rad * tilt[1] * sgn * f * -1, -tilt[0] * rad * tilt[1] * sgn * f * -1
            d.line([(ex_ - rad * .22, by + dy_in + tilt[1] * 0 * rad), (ex_ + rad * .22, by + dy_out)], fill=INK, width=bw)
        my, mx = head[1] + rad * 0.50, head[0] + f * rad * 0.40
        mw = int(lw * .55)
        if face == "smile":
            d.arc([mx - rad * .34, my - rad * .30, mx + rad * .34, my + rad * .18], 15, 165, fill=INK, width=mw)
        elif face in ("sad", "worried"):
            d.arc([mx - rad * .30, my - rad * .02, mx + rad * .30, my + rad * .42], 200, 340, fill=INK, width=mw)
        elif face == "angry":
            d.line([mx - rad * .30, my + rad * .06, mx + rad * .30, my - rad * .02], fill=INK, width=mw)
        elif face == "shock":
            d.ellipse([mx - rad * .17, my - rad * .10, mx + rad * .17, my + rad * .30], fill=(90, 40, 40), outline=INK, width=2)
        else:
            d.line([mx - rad * .26, my, mx + rad * .26, my], fill=INK, width=mw)
        # props
        props = self.s.get("props", [])
        if "helmet" in props:
            d.pieslice([head[0] - rad * 1.08, head[1] - rad * 1.12, head[0] + rad * 1.08, head[1] + rad * 0.10], 180, 360, fill=col, outline=INK, width=int(lw * .6))
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
        ex_, ey_ = head[0], head[1] - rad * 1.55                       # emotion shown above the head so reactions read at a glance
        pulse = 0.5 + 0.5 * math.sin(t * 6)
        if face == "shock":
            d.rectangle([ex_ - rad * .1, ey_ - rad * .6, ex_ + rad * .1, ey_ + rad * .1], fill=(255, 210, 60), outline=INK, width=3)
            d.ellipse([ex_ - rad * .12, ey_ + rad * .22, ex_ + rad * .12, ey_ + rad * .46], fill=(255, 210, 60), outline=INK, width=3)
        elif face == "worried":
            for k_ in range(2):
                age = (t * .9 + k_ * .5) % 1.0
                dx_, dy_ = f * rad * (1.05 + .1 * k_), -rad * .5 + age * rad * 1.1
                d.polygon([(ex_ + dx_, ey_ + dy_ + rad * .9 - rad * .3), (ex_ + dx_ - rad * .13, ey_ + dy_ + rad * .9), (ex_ + dx_ + rad * .13, ey_ + dy_ + rad * .9)], fill=(120, 190, 240, int(230 * (1 - age))))
                d.ellipse([ex_ + dx_ - rad * .13, ey_ + dy_ + rad * .84, ex_ + dx_ + rad * .13, ey_ + dy_ + rad * 1.1], fill=(120, 190, 240, int(230 * (1 - age))))
        elif face == "angry":
            for k_ in range(3):
                a_ = -2.4 + k_ * .6
                d.line([(ex_ + math.cos(a_) * rad * (.8 + .3 * pulse), ey_ + rad * .5 + math.sin(a_) * rad * (.8 + .3 * pulse)), (ex_ + math.cos(a_) * rad * 1.4, ey_ + rad * .5 + math.sin(a_) * rad * 1.4)], fill=(210, 50, 40), width=max(4, int(lw * .5)))
        elif face == "sad":
            tx0, tx1 = sorted((head[0] + f * rad * .3, head[0] + f * rad * .5))
            d.ellipse([tx0, head[1] + rad * (.1 + .6 * ((t * .8) % 1.0)), tx1, head[1] + rad * (.35 + .6 * ((t * .8) % 1.0))], fill=(120, 190, 240, 220))
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
        if "navycap" in props:
            d.rectangle([head[0] - rad * 1.0, head[1] - rad * 1.12, head[0] + rad * 1.0, head[1] - rad * 0.52], fill=(242, 242, 238), outline=INK, width=int(lw * .5))
            d.rectangle([head[0] - rad * 1.0, head[1] - rad * 0.62, head[0] + rad * 1.0, head[1] - rad * 0.52], fill=(30, 40, 70))
            d.rectangle([min(head[0], head[0] + f * rad * 1.45), head[1] - rad * 0.58, max(head[0], head[0] + f * rad * 1.45), head[1] - rad * 0.44], fill=INK)
        if "hat" in props:
            d.ellipse([head[0] - rad * 1.45, head[1] - rad * 0.98, head[0] + rad * 1.45, head[1] - rad * 0.58], fill=(64, 54, 48), outline=INK, width=int(lw * .5))
            d.rectangle([head[0] - rad * 0.78, head[1] - rad * 1.6, head[0] + rad * 0.78, head[1] - rad * 0.78], fill=(64, 54, 48), outline=INK, width=int(lw * .5))
        if "glasses" in props:
            for ex_, ey_ in eyes:
                d.ellipse([ex_ - er * 1.7, ey_ - er * 1.7, ex_ + er * 1.7, ey_ + er * 1.7], outline=INK, width=max(2, int(lw * .35)))
            d.line([eyes[0][0] + er * 1.7, eyes[0][1], eyes[1][0] - er * 1.7, eyes[1][1]], fill=INK, width=max(2, int(lw * .3)))
        if "tie" in props:
            d.polygon([(sh[0], sh[1] + S * .01), (sh[0] - S * .022, sh[1] + S * .09), (sh[0] + S * .022, sh[1] + S * .09)], fill=RED, outline=INK)
        if "rifle" in props:
            line(d, [add(hand, seg((0, 0), dirv + 180, .12 * S, f)), add(hand, seg((0, 0), dirv, .30 * S, f))], lw * .7, (70, 60, 50))
        if "flag" in props:
            top = add(hand, seg((0, 0), dirv + 180, .05 * S, f)); pole = (hand[0], hand[1] - S * 0.45)
            line(d, [hand, pole], lw * .5, (110, 80, 50))
            d.polygon([pole, (pole[0] + f * S * .22, pole[1] + S * .06 + math.sin(t * 6) * S * .015), (pole[0], pole[1] + S * .13)], fill=col)


def draw_object(d, o, W, H, gy, t):
    x = o["x"] * W * SS
    S = H * SS
    k = o["type"]
    if k in art.DRAW:
        if o.get("harbor") and k in ("ship", "warship", "liner", "iceberg", "wave"):
            gy = gy - S * 0.075                                  # vessels float out on the water band, people stand on the shore
        art.DRAW[k](d, o, x, S, gy, t, W * SS)
        return
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
    elif k == "submarine":
        w, h = S * 0.62 * o.get("scale", 1), S * 0.11 * o.get("scale", 1)
        yb = gy - h * 0.15 + math.sin(t * 1.2) * S * 0.004
        if o.get("afloat"):
            yb = gy - S * 0.26 + math.sin(t * 1.2) * S * 0.006
        x += (t - o.get("dur", 6) / 2) * S * 0.018
        d.ellipse([x - w / 2, yb - h, x + w / 2, yb + h * .35], fill=(74, 84, 96), outline=INK, width=4)
        d.rectangle([x - w * .06, yb - h * 1.55, x + w * .1, yb - h * .75], fill=(84, 94, 106), outline=INK, width=4)
        d.line([(x + w * .06, yb - h * 1.55), (x + w * .06, yb - h * 2.05), (x + w * .14, yb - h * 2.05)], fill=INK, width=6)
        for i in range(4):
            cx = x - w * .28 + i * w * .14
            d.ellipse([cx - 7, yb - h * .55 - 7, cx + 7, yb - h * .55 + 7], fill=(200, 210, 220), outline=INK, width=2)
    elif k == "warship":
        w, h = S * 0.66 * o.get("scale", 1), S * 0.07 * o.get("scale", 1)
        yb = gy - S * 0.004 + math.sin(t * 1.1) * S * 0.004
        x += (t - o.get("dur", 6) / 2) * S * 0.02
        d.polygon([(x - w / 2, yb - h), (x + w / 2, yb - h), (x + w * .38, yb), (x - w * .42, yb)], fill=(118, 126, 134), outline=INK)
        d.rectangle([x - w * .1, yb - h * 2.4, x + w * .12, yb - h], fill=(136, 144, 152), outline=INK, width=3)
        d.rectangle([x + w * .02, yb - h * 3.4, x + w * .07, yb - h * 2.4], fill=(92, 98, 104), outline=INK, width=3)
        d.line([(x - w * .3, yb - h * 1.3), (x - w * .15, yb - h * 1.55)], fill=INK, width=6)
    elif k == "depth_charge":
        t0 = o.get("t0", 1.0)
        surf, seabed = gy - S * 0.62, gy - S * 0.2
        if t < t0 + 1.3:
            u = max(0.0, (t - t0) / 1.3)
            cy = surf + (seabed - surf) * u ** 1.3
            if t >= t0 - 0.3:
                d.ellipse([x - S * .02, cy - S * .03, x + S * .02, cy + S * .03], fill=(70, 74, 70), outline=INK, width=3)
                d.line([(x, cy - S * .03), (x, cy - S * .06)], fill=INK, width=3)
                for j in range(4):
                    d.ellipse([x - 7 + math.sin(t * 9 + j) * 8, cy - S * .06 - j * S * .03, x + 7 + math.sin(t * 9 + j) * 8, cy - S * .06 - j * S * .03 + 14], outline=(220, 240, 255, 200), width=2)
        elif t < t0 + 2.6:
            u = (t - t0 - 1.3) / 1.3
            r = S * 0.34 * (1 - (1 - u) ** 2)
            a = int(230 * (1 - u))
            d.ellipse([x - r, seabed - r * .8, x + r, seabed + r * .8], fill=(210, 240, 255, a // 2), outline=(255, 255, 255, a), width=6)
            d.ellipse([x - r * .45, seabed - r * .45, x + r * .45, seabed + r * .45], fill=(255, 255, 255, a))
            d.polygon([(x - r * .3, seabed), (x - r * .2, seabed - r * 2.1 * u), (x + r * .2, seabed - r * 2.1 * u), (x + r * .3, seabed)], fill=(225, 245, 255, a // 2))
    elif k == "torpedo":
        t0, dur = o.get("t0", 1.0), o.get("dur", 6)
        u = max(0.0, min(1.0, (t - t0) / max(dur * 0.55, 1)))
        xx = (0.2 + 0.7 * u) * W * SS
        yy = gy - S * (0.26 if o.get("afloat") else 0.1)
        if t >= t0:
            d.rounded_rectangle([xx - S * .06, yy - S * .012, xx + S * .06, yy + S * .012], 8, fill=(150, 154, 150), outline=INK, width=3)
            d.polygon([(xx - S * .06, yy), (xx - S * .08, yy - S * .02), (xx - S * .08, yy + S * .02)], fill=INK)
            for j in range(14):
                bx = xx - S * (.08 + j * .02)
                d.ellipse([bx - 5 - j * .5, yy - 5 - j * .5 + math.sin(t * 8 + j) * 4, bx + 5 + j * .5, yy + 5 + j * .5 + math.sin(t * 8 + j) * 4], outline=(225, 245, 255, 220 - j * 12), width=2)
    elif k == "missile":
        h, w = S * 0.46 * o.get("scale", 1), S * 0.05
        if o.get("launch"):
            t0 = o.get("t0", 1.0)
            lift = max(0.0, t - t0) ** 1.8 * S * 0.07
            gy0 = gy
            gy = gy - lift
            d.line([(x - S * .13, gy0), (x - S * .13, gy0 - S * .45)], fill=INK, width=8)          # launch gantry
            for q in range(5):
                yy = gy0 - S * (.08 + .08 * q)
                d.line([(x - S * .13, yy), (x - S * .07, yy - S * .04)], fill=INK, width=4)
            if t > t0:
                d.polygon([(x - w * 1.4, gy0), (x + w * 1.4, gy0), (x + w * .5, gy), (x - w * .5, gy)], fill=(214, 212, 208, 95))   # smoke column left behind
            if t > t0:
                fl = S * (0.05 + 0.12 * min(1.0, (t - t0)))
                d.polygon([(x - w * .6, gy), (x, gy + fl * (1 + .2 * math.sin(t * 30))), (x + w * .6, gy)], fill=(255, 170, 40, 235))
                d.polygon([(x - w * .3, gy), (x, gy + fl * .6), (x + w * .3, gy)], fill=(255, 245, 200, 245))
                for j in range(10):
                    sy = gy0 - S * .0 + j * S * .012
                    rr = S * (.03 + .012 * j)
                    d.ellipse([x - rr * (1 + j * .2) - S * .03 * (j % 3 - 1), sy - rr * .5, x + rr * (1 + j * .2) - S * .03 * (j % 3 - 1), sy + rr * .5], fill=(210, 208, 205, max(0, 150 - j * 12)))
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(236, 236, 232), outline=INK, width=3)
        d.polygon([(x - w / 2, gy - h), (x, gy - h - S * .09), (x + w / 2, gy - h)], fill=(196, 57, 43), outline=INK)
        d.polygon([(x - w / 2, gy), (x - w * 1.1, gy + 1), (x - w / 2, gy - h * .22)], fill=(196, 57, 43), outline=INK)
        d.polygon([(x + w / 2, gy), (x + w * 1.1, gy + 1), (x + w / 2, gy - h * .22)], fill=(196, 57, 43), outline=INK)
        d.rectangle([x - w / 2, gy - h * .55, x + w / 2, gy - h * .5], fill=(196, 57, 43))
    elif k == "plane":
        yy = o.get("y", 0.24) * S
        xx = (o["x"] * W * SS + t * S * 0.03) % (W * SS + 400) - 200
        w = S * 0.22 * o.get("scale", 1)
        d.ellipse([xx - w / 2, yy - w * .07, xx + w / 2, yy + w * .07], fill=(210, 214, 220), outline=INK, width=3)
        d.polygon([(xx - w * .05, yy), (xx - w * .22, yy + w * .28), (xx + w * .05, yy)], fill=(180, 186, 194), outline=INK)
        d.polygon([(xx - w * .4, yy), (xx - w * .5, yy - w * .16), (xx - w * .3, yy)], fill=(180, 186, 194), outline=INK)
    elif k == "building":
        w, h = S * 0.17 * o.get("scale", 1), S * 0.44 * o.get("scale", 1)
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(120, 126, 138), outline=INK, width=4)
        for r in range(int(h / (S * .055))):
            for c in range(3):
                lit = (r * 3 + c + int(x)) % 5 != 0
                d.rectangle([x - w / 2 + w * (.14 + c * .3), gy - h + S * .03 + r * S * .055, x - w / 2 + w * (.3 + c * .3), gy - h + S * .06 + r * S * .055],
                            fill=(250, 220, 130) if lit else (70, 76, 88))
    elif k == "hatch":
        r = S * 0.12
        cy = gy - S * 0.25
        d.ellipse([x - r, cy - r, x + r, cy + r], fill=(96, 106, 108), outline=INK, width=6)
        d.ellipse([x - r * .7, cy - r * .7, x + r * .7, cy + r * .7], outline=(60, 68, 70), width=5)
        for a_ in range(0, 360, 60):
            d.line([(x, cy), (x + math.cos(math.radians(a_)) * r * .55, cy + math.sin(math.radians(a_)) * r * .55)], fill=INK, width=5)
    elif k == "pipes":
        for j, yy in enumerate((0.10, 0.17)):
            d.rectangle([0, yy * S, W * SS, yy * S + S * .035], fill=(112, 122, 124), outline=INK, width=3)
            for i in range(0, W * SS, int(S * .22)):
                d.rectangle([i, yy * S - 4, i + 10, yy * S + S * .035 + 4], fill=(70, 78, 80))
        for px_ in (W * SS * .035, W * SS * .965):               # stanchions at the edges, never a pole through the middle of the shot
            d.rectangle([px_ - S * .018, 0, px_ + S * .018, gy], fill=(104, 114, 116), outline=INK, width=3)
    elif k == "gauge":
        yy = o.get("y", 0.4) * S
        r = S * 0.05
        d.ellipse([x - r, yy - r, x + r, yy + r], fill=(228, 230, 224), outline=INK, width=5)
        a_ = math.sin(t * 0.9 + x) * 1.0 - 0.6
        d.line([(x, yy), (x + math.sin(a_) * r * .8, yy - math.cos(a_) * r * .8)], fill=RED, width=4)
    elif k == "volcano":
        w, h = S * 0.80 * o.get("scale", 1), S * 0.46 * o.get("scale", 1)
        d.polygon([(x - w / 2, gy), (x - w * .07, gy - h), (x + w * .07, gy - h), (x + w / 2, gy)], fill=(84, 66, 60), outline=INK)
        d.polygon([(x - w * .07, gy - h), (x + w * .07, gy - h), (x + w * .03, gy - h + S * .02), (x - w * .03, gy - h + S * .02)], fill=(255, 130, 40))
        for sx, off in ((-.03, 0), (.04, .5)):   # lava streams
            d.polygon([(x + w * sx, gy - h), (x + w * (sx + .03), gy - h), (x + w * (sx + .09 + off * .05), gy - h * .35), (x + w * (sx + .04 + off * .05), gy - h * .35)], fill=(255, 120, 36, 230))
        if o.get("erupt", True):
            for i in range(9):   # rising plume of ash
                ph = (t * 0.25 + i / 9.0) % 1.0
                cy = gy - h - ph * S * 0.55
                cx = x + math.sin(i * 1.7 + t * .5) * S * .03 * (1 + ph * 3) + ph * S * 0.1
                r = S * (0.03 + 0.09 * ph)
                a = int(215 * (1 - ph * .7))
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(58, 54, 56, a))
            d.ellipse([x - w * .12, gy - h - S * .03, x + w * .12, gy - h + S * .03], fill=(255, 150, 50, 90))
    elif k == "ash_cloud":
        for i in range(11):
            cx = (x + i * S * 0.13 + t * S * 0.02) % (W * SS + S * .4) - S * .2
            cy = S * (0.11 + 0.035 * math.sin(i * 1.3 + t * .3))
            r = S * (0.085 + 0.03 * (i % 3))
            d.ellipse([cx - r, cy - r * .7, cx + r, cy + r * .7], fill=(52, 50, 54, 215))
    elif k == "wave":
        hh = S * 0.46 * o.get("scale", 1) * min(1.0, 0.3 + t / 3.5)
        pts = [(x + S * .75, gy), (x + S * .3, gy - hh * .35), (x, gy - hh * .75), (x - S * .18, gy - hh), (x - S * .34, gy - hh * .86), (x - S * .3, gy - hh * .6), (x - S * .45, gy)]
        d.polygon(pts, fill=(30, 104, 168), outline=INK)
        for i in range(9):
            fx_ = x - S * .36 + i * S * .05
            d.ellipse([fx_ - S * .02, gy - hh * (.86 - i * .02) - S * .02 + math.sin(t * 6 + i) * 4, fx_ + S * .03, gy - hh * (.86 - i * .02) + S * .03], fill=(240, 246, 250))
    elif k == "fire":
        for i in range(9):
            fx_ = x + (i - 4) * S * 0.028
            fh = S * (0.06 + 0.04 * abs(math.sin(t * 8 + i * 1.9)))
            d.polygon([(fx_ - S * .02, gy), (fx_, gy - fh), (fx_ + S * .02, gy)], fill=(255, 130 + (i * 23) % 70, 30, 235))
            d.polygon([(fx_ - S * .01, gy), (fx_, gy - fh * .55), (fx_ + S * .01, gy)], fill=(255, 226, 120, 240))
        d.ellipse([x - S * .18, gy - S * .14, x + S * .18, gy + S * .02], fill=(255, 120, 30, 50))
    elif k == "smoke":
        for i in range(8):
            ph = (t * 0.3 + i / 8.0) % 1.0
            cy = gy - ph * S * 0.5
            cx = x + math.sin(i * 2.1 + t) * S * .02 + ph * S * .06
            r = S * (0.02 + 0.05 * ph)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(70, 68, 72, int(200 * (1 - ph * .8))))
    elif k == "house":
        w, h = S * 0.11 * o.get("scale", 1), S * 0.085 * o.get("scale", 1)
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(196, 170, 130), outline=INK, width=3)
        d.polygon([(x - w * .62, gy - h), (x, gy - h - S * .055), (x + w * .62, gy - h)], fill=(150, 64, 50), outline=INK)
        d.rectangle([x - w * .1, gy - h * .6, x + w * .1, gy], fill=(84, 60, 44))
    elif k == "explosion":
        t0 = o.get("t0", 0.4)
        if t0 <= t <= t0 + 1.3:
            u = (t - t0) / 1.3
            r = S * 0.55 * (1 - (1 - u) ** 2)
            cy = gy - S * 0.22
            for rr, col in ((1.0, (255, 120, 30)), (.75, (255, 190, 70)), (.45, (255, 246, 210))):
                d.ellipse([x - r * rr, cy - r * rr, x + r * rr, cy + r * rr], fill=col + (int(235 * (1 - u)),))
    elif k == "tree":
        h = S * 0.30 * o.get("scale", 1)
        d.rectangle([x - h * .05, gy - h * .45, x + h * .05, gy], fill=(96, 70, 48), outline=INK, width=3)
        for dx, dy, r in ((0, -.62, .26), (-.17, -.48, .2), (.17, -.48, .2)):
            d.ellipse([x + dx * h - r * h, gy + dy * h - r * h, x + dx * h + r * h, gy + dy * h + r * h], fill=o.get("color", (84, 120, 70)), outline=INK, width=3)
    elif k == "tent":
        w, h = S * 0.22 * o.get("scale", 1), S * 0.17 * o.get("scale", 1)
        d.polygon([(x - w / 2, gy), (x, gy - h), (x + w / 2, gy)], fill=o.get("color", (190, 70, 56)), outline=INK)
        d.polygon([(x - w * .08, gy), (x, gy - h * .45), (x + w * .08, gy)], fill=INK)
    elif k == "pyramid":
        w, h = S * 0.5 * o.get("scale", 1), S * 0.34 * o.get("scale", 1)
        d.polygon([(x - w / 2, gy), (x, gy - h), (x + w / 2, gy)], fill=(222, 196, 140), outline=INK)
        d.polygon([(x, gy - h), (x + w / 2, gy), (x + w * .08, gy)], fill=(196, 168, 112))
    elif k == "tower":
        w, h = S * 0.10, S * 0.46
        d.rectangle([x - w / 2, gy - h, x + w / 2, gy], fill=(190, 182, 166), outline=INK, width=4)
        d.polygon([(x - w * .7, gy - h), (x, gy - h - S * .09), (x + w * .7, gy - h)], fill=(150, 60, 50), outline=INK)
    elif k == "torch":
        d.rectangle([x - 5, gy - S * .18, x + 5, gy], fill=(96, 70, 48))
        fl = math.sin(t * 14 + x) * S * 0.008
        d.ellipse([x - S * .02, gy - S * .25 + fl, x + S * .02, gy - S * .17], fill=(255, 170, 40, 230))
        d.ellipse([x - S * .05, gy - S * .28, x + S * .05, gy - S * .14], fill=(255, 140, 30, 50))
    elif k == "ship":
        w, h = S * 0.30 * o.get("scale", 1), S * 0.06 * o.get("scale", 1)
        bob = math.sin(t * 1.4) * S * 0.008
        yb = gy - S * 0.01 + bob
        d.polygon([(x - w / 2, yb - h), (x + w / 2, yb - h), (x + w * .36, yb), (x - w * .36, yb)], fill=(116, 80, 52), outline=INK)
        d.line([(x, yb - h), (x, yb - h - S * .30)], fill=INK, width=6)
        d.polygon([(x + 6, yb - h - S * .29), (x + w * .34, yb - h - S * .10), (x + 6, yb - h - S * .06)], fill=(240, 234, 220), outline=INK)
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
    for k in range(0, int(S), 22):                                   # inked sea lines, drifting slowly
        off = (t * 14 + k * 3) % 60
        for x0 in range(-60, int(w), 120):
            d.arc([x0 + off, k, x0 + off + 40, k + 14], 200, 340, fill=(150, 126, 84, 70), width=3)
    for poly in scene["land"]:
        pts = [(x * w, y * S) for x, y in poly["pts"]]
        d.polygon([(px + 12, py + 14) for px, py in pts], fill=(120, 90, 50, 80))              # drop shadow so land lifts off the page
        d.polygon(pts, fill=poly.get("color", (236, 222, 184)), outline=INK)
        d.line(pts + [pts[0]], fill=INK, width=5)
        if poly.get("label"):
            f_ = font(int(S * 0.042))
            lx, ly = poly["label_at"]
            d.text((lx * w - d.textlength(poly["label"], font=f_) / 2, ly * S), poly["label"], font=f_, fill=poly.get("label_color", (60, 44, 26)))
    for c in scene.get("cities", []):
        cx, cy = c["x"] * w, c["y"] * S
        for j in range(2):                                           # pulsing target rings
            ph = (t * .8 + j * .5) % 1.0
            d.ellipse([cx - 16 - ph * 70, cy - 16 - ph * 70, cx + 16 + ph * 70, cy + 16 + ph * 70], outline=(196, 57, 43, int(200 * (1 - ph))), width=5)
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
        for j in range(3):                                           # little units marching along the arrow
            q = (t * .35 + j / 3) % 1.0
            tot = sum(lens)
            target, run_ = tot * pr * q, 0.0
            for i, L_ in enumerate(lens):
                if run_ + L_ >= target:
                    uu = (target - run_) / max(L_, 1e-6)
                    ux, uy = lerp(pts[i][0], pts[i + 1][0], uu), lerp(pts[i][1], pts[i + 1][1], uu)
                    d.ellipse([ux - 11, uy - 11, ux + 11, uy + 11], fill=(255, 244, 220), outline=col, width=4)
                    break
                run_ += L_
        if ar.get("label") and pr > 0.15:
            f_ = font(int(S * 0.04))
            lx, ly = ar["label_at"]
            d.text((lx * w, ly * S), ar["label"], font=f_, fill=col)


def draw_fx(d, scene, t, W, H, gy, actors):
    S = H * SS
    for fx in scene.get("fx", []):
        k = fx["type"]
        if k == "ambient":
            art.ambient(d, fx["bg"], t, S, gy, W * SS)
        elif k == "rain" and fx["t0"] <= t <= fx["t1"]:
            rng = np.random.default_rng(3)
            xs, ys, sp = rng.random(160), rng.random(160), 0.9 + rng.random(160) * 0.8
            for i in range(160):
                x = ((xs[i] + t * 0.12 * sp[i]) % 1.0) * W * SS
                y = ((ys[i] + t * 1.6 * sp[i]) % 1.0) * S
                d.line([(x, y), (x - S * 0.012, y + S * 0.04)], fill=(60, 70, 100, 120), width=3)
        elif k == "bubbles" and fx["t0"] <= t <= fx["t1"]:
            rng = np.random.default_rng(11)
            xs, ys, sp = rng.random(60), rng.random(60), 0.4 + rng.random(60)
            for i in range(60):
                x = ((xs[i] + math.sin(t * 1.5 + i) * 0.012) % 1.0) * W * SS
                y = S - ((ys[i] + t * 0.18 * sp[i]) % 1.0) * S
                r = 3 + 8 * sp[i]
                d.ellipse([x - r, y - r, x + r, y + r], outline=(230, 246, 255, 190), width=2)
        elif k == "sonar" and fx["t0"] <= t <= fx["t1"]:
            cx, cy = W * SS * .5, gy - S * .26
            for j in range(3):
                u = ((t * .5 + j / 3) % 1.0)
                r = S * (.08 + .7 * u)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 255, 170, int(220 * (1 - u))), width=4)
        elif k == "ashfall":
            rng = np.random.default_rng(8)
            xs, ys, sp = rng.random(150), rng.random(150), 0.5 + rng.random(150)
            for i in range(150):
                x = ((xs[i] + math.sin(t * .7 + i) * 0.01) % 1.0) * W * SS
                y = ((ys[i] + t * 0.25 * sp[i]) % 1.0) * S
                r = 2 + 4 * sp[i]
                d.ellipse([x - r, y - r, x + r, y + r], fill=(70, 68, 72, 170))
        elif k == "embers":
            rng = np.random.default_rng(9)
            xs, ys, sp = rng.random(70), rng.random(70), 0.4 + rng.random(70)
            for i in range(70):
                x = ((xs[i] + math.sin(t + i) * 0.02) % 1.0) * W * SS
                y = S - ((ys[i] + t * 0.22 * sp[i]) % 1.0) * S
                r = 2 + 3 * sp[i]
                d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 170 + int(60 * sp[i]), 50, 210))
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


_SKYSUN = {}
_GROUND = {}


def _sky_sun(W, H, top, bot, sun):
    key = (W, H, tuple(top), tuple(bot), tuple(sun[:2]) + tuple(sun[2]) if sun else None)
    if key not in _SKYSUN:
        im = Image.fromarray(_sky(None, W, H, top, bot))
        if sun:
            d = ImageDraw.Draw(im, "RGBA")
            sx, sy, sc = sun
            for r, al in ((0.22, 18), (0.14, 30), (0.085, 255)):
                d.ellipse([sx * W * SS - r * H * SS, sy * H * SS - r * H * SS, sx * W * SS + r * H * SS, sy * H * SS + r * H * SS], fill=tuple(sc) + (al,))
        _SKYSUN[key] = im
    return _SKYSUN[key]


def _ground(W, H, gy, gcol):
    key = (W, H, int(gy), tuple(gcol))
    if key not in _GROUND:
        h = H * SS - int(gy)
        im = Image.new("RGB", (W * SS, h), tuple(gcol))
        dd = ImageDraw.Draw(im, "RGBA")
        for i in range(10):  # ground gets darker toward the bottom
            dd.rectangle([0, h * i / 10, W * SS, h], fill=(0, 0, 0, 10))
        rng = np.random.default_rng(11)
        for _ in range(70):
            x, y = rng.random() * W * SS, 14 + rng.random() * h * 0.8
            dd.line([(x, y), (x + 26 + rng.random() * 30, y)], fill=(0, 0, 0, 38), width=3)
        _GROUND[key] = im
    return _GROUND[key]


def render_card(scene, t, W, H):
    c = scene["card"]
    dark = c.get("dark", True)
    top, bot = ((16, 18, 26), (40, 38, 52)) if dark else ((240, 232, 214), (226, 214, 188))
    g = np.linspace(0, 1, H)[:, None, None]
    arr = (np.array(top) * (1 - g) + np.array(bot) * g).astype(np.uint8).repeat(W, axis=1)
    pil = Image.fromarray(arr)
    d = ImageDraw.Draw(pil, "RGBA")
    for k in range(14):                                              # slow diagonal light streaks so the card is never dead
        x0 = ((k * 137 + t * 40 * (1 + k % 3)) % (W + 400)) - 200
        d.polygon([(x0, 0), (x0 + 90, 0), (x0 - 160, H), (x0 - 250, H)], fill=(255, 255, 255, 6 if dark else 10))
    ink = (244, 240, 230) if dark else INK
    u = smooth(t / 0.45)
    big = c.get("big", "")
    if big:
        f_ = font(int(H * (0.20 if len(big) < 14 else 0.13) * (0.9 + 0.1 * u)))
        w = d.textlength(big, font=f_)
        y = H * (0.10 if c.get("bullets") else 0.34)
        d.text(((W - w) / 2 + 4, y + 5), big, font=f_, fill=(0, 0, 0, int(90 * u)))
        d.text(((W - w) / 2, y), big, font=f_, fill=ink + (int(255 * u),))
        ub = smooth((t - 0.2) / 0.6)
        d.line([((W - w) / 2, y + H * 0.215), ((W - w) / 2 + w * ub, y + H * 0.215)], fill=GOLD + (255,), width=max(5, H // 90))
    if c.get("small"):
        f2 = font(int(H * 0.06))
        a = smooth((t - 0.6) / 0.5)
        w2 = d.textlength(c["small"], font=f2)
        d.text(((W - w2) / 2, H * (0.36 if c.get("bullets") else 0.62)), c["small"], font=f2, fill=ink + (int(235 * a),))
    for i, b in enumerate(c.get("bullets", [])):
        a = smooth((t - 0.8 - i * 0.7) / 0.4)
        f3 = font(int(H * 0.055))
        x0, y0 = W * 0.22 + (1 - a) * W * 0.12, H * (0.50 + i * 0.095)          # bullets slide in from the right
        d.ellipse([x0 - 40, y0 + 14, x0 - 22, y0 + 32], fill=GOLD + (int(255 * a),))
        d.text((x0, y0), b, font=f3, fill=ink + (int(240 * a),))
    return _grade(np.asarray(pil), W, H, t)


def render_frame(scene, t, W, H):
    if scene.get("kind") == "card":
        return render_card(scene, t, W, H)
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
        img = _sky_sun(W, H, top, bot, scene.get("sun")).copy()
        d = ImageDraw.Draw(img, "RGBA")
        for layer in scene.get("hills", []):
            _hills(d, W, H, gy, layer, off)
        gcol = scene.get("ground_color", (176, 158, 120))
        img.paste(_ground(W, H, gy, gcol), (0, int(gy)))
        d.line([(0, gy), (W * SS, gy)], fill=INK, width=5)
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
    punch = 0.0
    shots = scene.get("shots")
    if shots:
        sh = shots[0]
        for cand in shots:
            if cand["t0"] <= t:
                sh = cand
        us = (t - sh["t0"]) / max(sh["t1"] - sh["t0"], 1e-6)
        z = lerp(sh["z0"], sh["z1"], us)
        cx = lerp(sh["x0"], sh["x1"], us)
        cy = lerp(sh["y0"], sh["y1"], us)
        age = t - sh["t0"]
        if sh["t0"] > 0 and age < 0.22:                # hard cut lands with a small zoom-settle
            z *= 1 + 0.07 * (1 - age / 0.22) ** 2
        for e in scene.get("hits", []):                # impacts punch the camera in
            if e <= t < e + 0.35:
                punch = (1 - (t - e) / 0.35)
                z *= 1 + 0.09 * punch
    for sh_ in scene.get("shake", []):
        if sh_["t"] <= t <= sh_["t"] + sh_.get("dur", 0.4):
            k = 1 - (t - sh_["t"]) / sh_.get("dur", 0.4)
            sx = math.sin(t * 90) * sh_.get("amp", 14) * k * SS
            sy = math.cos(t * 77) * sh_.get("amp", 14) * k * SS
    if scene.get("kind") != "map":                       # faint handheld drift: the camera is never perfectly still
        sx += (math.sin(t * 1.3) * 2.2 + math.sin(t * 3.1 + 1) * 0.8) * SS
        sy += (math.cos(t * 1.1) * 1.6 + math.sin(t * 2.7) * 0.6) * SS
    M = cv2.getRotationMatrix2D((cx * W * SS, cy * H * SS), 0, z / SS)
    M[0, 2] += W / 2 - cx * W * SS + sx / SS
    M[1, 2] += H / 2 - cy * H * SS + sy / SS
    out = cv2.warpAffine(np.asarray(img), M, (W, H), flags=cv2.INTER_AREA if z / SS < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    out = _grade(out, W, H, t)
    if punch > 0.05:                                   # chromatic aberration on impact
        k = max(1, int(6 * punch))
        out = out.copy()
        out[..., 0] = np.roll(out[..., 0], k, axis=1)
        out[..., 2] = np.roll(out[..., 2], -k, axis=1)
    pil = Image.fromarray(out)
    dd = ImageDraw.Draw(pil, "RGBA")
    if scene.get("speed") and scene.get("kind") != "map":
        rng = np.random.default_rng(int(t * 24) % 7)   # flickering radial speed lines at the frame edge
        for i in range(26):
            a_ = rng.random() * 6.283
            r0 = (0.62 + rng.random() * 0.12) * W / 2
            r1 = r0 + (0.18 + rng.random() * 0.25) * W / 2
            c_, s_ = math.cos(a_), math.sin(a_)
            dd.line([(W / 2 + c_ * r0, H / 2 + s_ * r0 * 0.9), (W / 2 + c_ * r1, H / 2 + s_ * r1 * 0.9)], fill=(255, 255, 255, 70), width=3)
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


_GRADE = {}


def _grade(arr, W, H, t):
    """Vignette + warm tint + a little film grain. Multiplier and grain bank are built once per size."""
    key = (W, H)
    if key not in _GRADE:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 1.41
        vig = (1 - 0.38 * np.clip(r, 0, 1) ** 2.3)[..., None]
        mult = (vig * np.array([1.03, 1.0, 0.93], dtype=np.float32)).astype(np.float32)
        rng = np.random.default_rng(5)
        bank = [(rng.standard_normal((H, W, 1)) * 1.6).astype(np.float32) for _ in range(12)]
        _GRADE[key] = (mult, bank)
    mult, bank = _GRADE[key]
    f = arr.astype(np.float32)
    f *= mult
    f += bank[int(t * 30) % 12]
    np.clip(f, 0, 255, out=f)
    return f.astype(np.uint8)


def prepare(scene, W, H):
    scene["_actors"] = [Actor(a, W, H) for a in scene.get("actors", [])]
    return scene


def _kx(keys, t):
    if t <= keys[0]["t"]:
        return keys[0]["x"]
    for a, b in zip(keys, keys[1:]):
        if a["t"] <= t <= b["t"]:
            return lerp(a["x"], b["x"], (t - a["t"]) / max(b["t"] - a["t"], 1e-6))
    return keys[-1]["x"]


def make_shots(scene, rnd):
    """Cut long scenes into shots: wide -> medium on a character -> close-up/insert -> wide, each with its own drift."""
    dur = scene["duration"]
    acts = scene.get("actors", [])
    objs = [o for o in scene.get("objects", []) if o["type"] not in ("cloud", "torch", "seascape")]
    bigs = [o for o in objs if o["type"] in ("submarine", "warship", "ship", "liner", "iceberg", "volcano", "wave", "pyramid", "plane", "missile", "explosion", "castle")]
    n = 1 if dur < 4.2 else 2 if dur < 8 else 3 if dur < 13 else 4
    if n == 1:
        scene["hits"] = []
    else:
        bounds = [dur * i / n for i in range(n + 1)]
        style = ["wide", "medium", "close", "wide"]
        first = rnd.choice(["wide", "medium"])
        shots = []
        for i in range(n):
            st = first if i == 0 else style[i] if rnd.random() < 0.8 else "medium"
            tm = (bounds[i] + bounds[i + 1]) / 2
            if bigs and st != "wide":                      # the subject of the scene stays in frame
                o = bigs[i % len(bigs)]
                fx, fy = o.get("x", 0.5), 0.52
                if acts and st == "close" and abs(_kx(acts[0]["keys"], tm) - fx) < .3:
                    fx = (fx + _kx(acts[0]["keys"], tm)) / 2
                z0, z1 = (1.15, 1.3) if st == "medium" else (1.35, 1.55)
            elif len(acts) >= 2 and st == "medium":          # two-shot keeps both characters in frame
                fx = sum(_kx(a["keys"], tm) for a in acts[:2]) / 2
                fy, z0, z1 = 0.52, 1.25, 1.4
            elif acts and st != "wide":
                a = acts[(i + rnd.randrange(len(acts))) % len(acts)]
                fx = _kx(a["keys"], tm)
                fy = 0.50 if st == "medium" else 0.40
                z0, z1 = (1.45, 1.6) if st == "medium" else (1.9, 2.15)
            elif objs and st != "wide":
                o = rnd.choice(objs)
                fx, fy, z0, z1 = o.get("x", 0.5), 0.5, 1.35, 1.5
            else:
                fx, fy, z0, z1 = 0.5, 0.55, 1.0, 1.1
            if rnd.random() < 0.5:
                z0, z1 = z1, z0 if st == "wide" else z1
            shots.append(dict(t0=bounds[i], t1=bounds[i + 1], z0=z0, z1=z1, x0=fx, x1=fx + rnd.choice([-.02, .02]), y0=fy, y1=fy))
        scene["shots"] = shots
        scene["hits"] = []
    for sh_ in scene.get("shake", []):
        scene["hits"].append(sh_["t"])
    for i_, fx in enumerate(f_ for f_ in scene.get("fx", []) if f_["type"] in ("sparks", "flash")):
        if fx["type"] == "flash" or i_ % 3 == 0:      # not every exchange punches the camera
            scene["hits"].append(fx["t"])
    for o in scene.get("objects", []):
        if o["type"] in ("explosion",):
            scene["hits"].append(o.get("t0", 0.4))
        if o["type"] == "depth_charge":
            scene["hits"].append(o.get("t0", 1.0) + 1.3)
    if not scene.get("shots") and scene["hits"]:   # single-shot scenes still need a camera for punches
        scene["shots"] = [dict(t0=0, t1=dur, z0=1.0, z1=1.12, x0=.5, x1=.5, y0=.55, y1=.55)]
    scene["speed"] = bool(scene.get("blur"))
