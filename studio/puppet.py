"""Layered puppet rig drawn in code, in the simple 'webcomic stick figure' look: big white round head, dot eyes, thin black limbs, a plain tunic.

Parts are drawn separately and posed with pivots, like a cut-out rig: head (3 turns), 9 mouth shapes, eye/brow sets, 4 hand shapes, torso, 2-bone arms and legs.
Motion comes from spring-damper chasers per body part (overshoot, settle and overlap), driven by hand-designed key poses and an audio-driven mouth."""
import math

import numpy as np
from PIL import Image, ImageDraw

INK = (24, 20, 26)
SKIN = (255, 252, 246)
SS = 3                                                                   # supersampling for clean lines

# ----------------------------------------------------------------------------------- the face: mouth shapes (Preston Blair style) -----------------------
MOUTHS = "XABCDEFGH"            # X rest, A closed (M B P), B slight open, C open, D wide, E round, F pucker, G teeth on lip (F V), H tongue (L)


def draw_mouth(d, shape, cx, cy, r, mood="neutral", w=3):
    s = r * 0.5
    if shape == "A":
        d.line([cx - s * .55, cy, cx + s * .55, cy], fill=INK, width=w)
    elif shape == "X":
        if mood in ("sad", "worried"):
            d.arc([cx - s * .6, cy - s * .1, cx + s * .6, cy + s * .7], 200, 340, fill=INK, width=w)
        elif mood == "smile":
            d.arc([cx - s * .7, cy - s * .55, cx + s * .7, cy + s * .35], 15, 165, fill=INK, width=w)
        else:
            d.line([cx - s * .5, cy, cx + s * .5, cy + s * .04], fill=INK, width=w)
    elif shape == "B":
        d.ellipse([cx - s * .5, cy - s * .16, cx + s * .5, cy + s * .22], fill=(120, 40, 50), outline=INK, width=w)
    elif shape == "C":
        d.ellipse([cx - s * .55, cy - s * .28, cx + s * .55, cy + s * .5], fill=(120, 40, 50), outline=INK, width=w)
    elif shape == "D":
        d.ellipse([cx - s * .65, cy - s * .4, cx + s * .65, cy + s * .75], fill=(120, 40, 50), outline=INK, width=w)
        d.ellipse([cx - s * .4, cy + s * .25, cx + s * .4, cy + s * .7], fill=(214, 100, 108))
    elif shape == "E":
        d.ellipse([cx - s * .36, cy - s * .3, cx + s * .36, cy + s * .55], fill=(120, 40, 50), outline=INK, width=w)
    elif shape == "F":
        d.ellipse([cx - s * .22, cy - s * .16, cx + s * .22, cy + s * .3], fill=(120, 40, 50), outline=INK, width=w)
    elif shape == "G":
        d.chord([cx - s * .55, cy - s * .2, cx + s * .55, cy + s * .42], 0, 180, fill=(120, 40, 50), outline=INK, width=w)
        d.rectangle([cx - s * .46, cy - s * .04, cx + s * .46, cy + s * .08], fill=(255, 255, 255))
    elif shape == "H":
        d.ellipse([cx - s * .5, cy - s * .26, cx + s * .5, cy + s * .42], fill=(120, 40, 50), outline=INK, width=w)
        d.ellipse([cx - s * .22, cy + s * .08, cx + s * .22, cy + s * .38], fill=(214, 100, 108))


def draw_eye(d, cx, cy, r, look=(0.0, 0.0), blink=0.0, lid=0.0, w=3):
    """Dot eyes like the reference: a black dot, a heavy lid when sleepy, a line when blinking."""
    rr = r * 0.115
    if blink > 0.6:
        d.line([cx - rr * 1.5, cy, cx + rr * 1.5, cy], fill=INK, width=w)
        return
    ox, oy = look[0] * rr * 0.9, look[1] * rr * 0.9
    d.ellipse([cx + ox - rr, cy + oy - rr * (1 - 0.5 * blink), cx + ox + rr, cy + oy + rr * (1 - 0.5 * blink)], fill=INK)
    if lid > 0.05:
        d.chord([cx - rr * 2.1, cy - rr * 2.1, cx + rr * 2.1, cy + rr * 2.1], 180, 360, fill=SKIN)
        d.line([cx - rr * 2.1, cy - rr * (1.8 - 2.6 * lid), cx + rr * 2.1, cy - rr * (1.8 - 2.6 * lid)], fill=INK, width=w)


def draw_brow(d, cx, cy, r, tilt=0.0, raise_=0.0, side=1, w=3):
    ln = r * 0.26
    y = cy - r * (0.20 + 0.15 * raise_)
    d.line([cx - ln, y + side * tilt * r * 0.16, cx + ln, y - side * tilt * r * 0.16], fill=INK, width=w)


def draw_hand(d, x, y, ang, shape, u, flip=1, w=3):
    """Hand shapes: 0 mitten, 1 open palm, 2 pointing, 3 fist."""
    r = u * 0.055
    a = math.radians(ang)
    dx, dy = math.sin(a), math.cos(a)
    cx, cy = x + dx * r * 0.7, y + dy * r * 0.7
    if shape == 2:                                                          # pointing finger
        d.ellipse([cx - r * .85, cy - r * .8, cx + r * .85, cy + r * .8], fill=SKIN, outline=INK, width=w)
        fx, fy = cx + dx * r * 2.1, cy + dy * r * 2.1
        d.line([cx, cy, fx, fy], fill=INK, width=int(r * .55 + w))
        d.line([cx, cy, fx, fy], fill=SKIN, width=int(r * .55))
        return
    if shape == 1:                                                          # open palm: bigger mitten with a thumb out
        d.ellipse([cx - r * 1.05, cy - r * .95, cx + r * 1.05, cy + r * .95], fill=SKIN, outline=INK, width=w)
        tx, ty = cx - dy * r * 1.1 * flip, cy + dx * r * 1.1 * flip
        d.ellipse([tx - r * .42, ty - r * .42, tx + r * .42, ty + r * .42], fill=SKIN, outline=INK, width=w)
        return
    d.ellipse([cx - r * (0.95 if shape == 0 else 0.8), cy - r * 0.85, cx + r * (0.95 if shape == 0 else 0.8), cy + r * 0.85], fill=SKIN, outline=INK, width=w)
    tx, ty = cx - dy * r * .8 * flip, cy + dx * r * .8 * flip
    d.ellipse([tx - r * .38, ty - r * .38, tx + r * .38, ty + r * .38], fill=SKIN, outline=INK, width=max(2, w - 1))


# ----------------------------------------------------------------------------------- the character ---------------------------------------------------
class Look:
    def __init__(self, tunic=(150, 100, 60), pants=(110, 80, 56), hair=(96, 62, 40), fur=True, hat=None, seed=0):
        self.tunic, self.pants, self.hair, self.fur, self.hat, self.seed = tunic, pants, hair, fur, hat, seed


def _dark(c, k=.62):
    return tuple(int(v * k) for v in c[:3])


def _limb(d, p0, p1, p2, w):
    d.line([p0, p1, p2], fill=INK, width=w, joint="curve")
    for p in (p0, p1, p2):
        d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=INK)


def draw_character(pose, look, u, W, H, ox, oy):
    """Draw one character into an RGBA layer of size (W, H). u = character height in output px; (ox, oy) = feet position. pose is a dict of floats/strings."""
    S = SS
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    U = u * S
    X, Y = ox * S, oy * S
    lw = max(2, int(U * 0.014))                                            # thin ink line like the reference
    limb = max(2, int(U * 0.016))
    f = pose.get("facing", 1)
    R = U * 0.215                                                          # head radius (big head, like the reference)
    sq = pose.get("squash", 0.0)
    hip = (X + pose.get("x", 0.0) * S, Y - U * 0.30 * (1 - 0.04 * sq) + pose.get("y", 0.0) * S)
    tr = math.radians(pose.get("torso", 0.0))
    up = (math.sin(tr) * f, -math.cos(tr))
    tl = U * 0.25 * (1 - 0.05 * sq)
    sh = (hip[0] + up[0] * tl * .92, hip[1] + up[1] * tl * .92)
    neck = (hip[0] + up[0] * tl, hip[1] + up[1] * tl)
    ht = math.radians(pose.get("torso", 0.0) + pose.get("head", 0.0))
    head = (neck[0] + math.sin(ht) * f * R * .80, neck[1] - math.cos(ht) * R * .80)

    def seg(p, ang, L):
        a = math.radians(ang)
        return (p[0] + math.sin(a) * f * L, p[1] + math.cos(a) * L)
    upa, loa = U * 0.17, U * 0.165
    sp = U * 0.075
    shA, shB = (sh[0] + sp * f, sh[1] + U * .01), (sh[0] - sp * f, sh[1] + U * .01)
    a1 = seg(shA, pose.get("a1", 8), upa); a2 = seg(a1, pose.get("a1", 8) + pose.get("a2", 10), loa)
    b1 = seg(shB, pose.get("b1", -8), upa); b2 = seg(b1, pose.get("b1", -8) + pose.get("b2", 10), loa)
    hs = U * 0.045
    hipA, hipB = (hip[0] + hs * f, hip[1]), (hip[0] - hs * f, hip[1])
    l1 = seg(hipA, pose.get("l1", 3), U * .15); l2 = seg(l1, pose.get("l1", 3) + pose.get("l2", 0), U * .15)
    m1 = seg(hipB, pose.get("m1", -3), U * .15); m2 = seg(m1, pose.get("m1", -3) + pose.get("m2", 0), U * .15)

    def foot(p):
        d.ellipse([p[0] - U * .03 * f - U * .015, p[1] - U * .018, p[0] + U * .075 * f + U * .015, p[1] + U * .022], fill=(70, 52, 44), outline=INK, width=lw) if f > 0 else \
            d.ellipse([p[0] + U * .03 - U * .075 - U * .015, p[1] - U * .018, p[0] + U * .03 + U * .015, p[1] + U * .022], fill=(70, 52, 44), outline=INK, width=lw)
    # shadow
    d.ellipse([X - U * .17, Y - U * .012, X + U * .17, Y + U * .02], fill=(0, 0, 0, 55))
    # far arm and leg
    _limb(d, shB, b1, b2, limb); draw_hand(d, b2[0], b2[1], pose.get("b1", -8) + pose.get("b2", 10), pose.get("hand_b", 0), U, -f, lw)
    _limb(d, hipB, m1, m2, limb); foot(m2)
    # tunic: tapered body with a ragged hem, in the role colour
    nx, ny = -up[1], up[0]
    wt, wb = U * .08, U * .115
    pts_l = [(sh[0] + nx * wt, sh[1] + ny * wt)]
    pts_r = [(sh[0] - nx * wt, sh[1] - ny * wt)]
    hemy = U * .04
    for i in range(1, 6):
        t_ = i / 5
        c = (sh[0] + (hip[0] - sh[0]) * t_ * 1.08, sh[1] + (hip[1] - sh[1]) * t_ * 1.08 + (hemy * t_ ** 2))
        w_ = wt * (1 - t_) + wb * t_
        pts_l.append((c[0] + nx * w_, c[1] + ny * w_)); pts_r.append((c[0] - nx * w_, c[1] - ny * w_))
    hem = []
    if look.fur:                                                            # jagged fur hem
        bl, br = pts_l[-1], pts_r[-1]
        n = 7
        for i in range(n + 1):
            t_ = i / n
            hem.append((bl[0] + (br[0] - bl[0]) * t_, bl[1] + (br[1] - bl[1]) * t_ + (U * .045 if i % 2 else U * .005)))
    else:
        hem = [pts_l[-1], pts_r[-1]]
    body = pts_l + hem[::-1] + pts_r[::-1]
    d.polygon(body, fill=look.tunic)
    d.line(body + [body[0]], fill=INK, width=lw, joint="curve")
    d.line([sh[0] + nx * wt * .55, sh[1] + ny * wt * .55, sh[0] + (hip[0] - sh[0]) * .12, sh[1] + (hip[1] - sh[1]) * .12 + U * .01, sh[0] - nx * wt * .55, sh[1] - ny * wt * .55], fill=INK, width=lw)    # V neckline
    belt = (sh[0] + (hip[0] - sh[0]) * .74, sh[1] + (hip[1] - sh[1]) * .74)
    d.line([belt[0] + nx * wb * .9, belt[1] + ny * wb * .9, belt[0] - nx * wb * .9, belt[1] - ny * wb * .9], fill=_dark(look.tunic), width=int(U * .022))
    # near leg and arm
    _limb(d, hipA, l1, l2, limb); foot(l2)
    _limb(d, shA, a1, a2, limb); draw_hand(d, a2[0], a2[1], pose.get("a1", 8) + pose.get("a2", 10), pose.get("hand_a", 0), U, f, lw)
    # head: big white circle with messy hair, the face shifts with the head turn
    d.line([neck, head], fill=INK, width=limb)
    d.ellipse([head[0] - R, head[1] - R, head[0] + R, head[1] + R], fill=SKIN, outline=INK, width=lw)
    turn = max(-1.0, min(1.0, pose.get("turn", 0.0))) * f
    hc = look.hair
    rng = np.random.default_rng(look.seed)
    tuft = []
    for i in range(11):
        a = math.radians(212 + 116 * i / 10 + turn * -6)
        rr = R * (1.26 + 0.16 * math.sin(i * 2.3 + look.seed) if i % 2 else 0.97)
        tuft.append((head[0] + math.cos(a) * rr + turn * R * .08, head[1] + math.sin(a) * rr))
    d.polygon(tuft, fill=hc); d.line(tuft, fill=INK, width=lw, joint="curve")
    d.arc([head[0] - R, head[1] - R, head[0] + R, head[1] + R], 205, 335, fill=INK, width=lw)
    hat = look.hat
    if hat == "crown":
        cw, ch = R * .75, R * .62
        base = head[1] - R * 0.92
        d.polygon([(head[0] - cw, base), (head[0] - cw, base - ch), (head[0] - cw / 2, base - ch * .45), (head[0], base - ch * 1.1), (head[0] + cw / 2, base - ch * .45), (head[0] + cw, base - ch), (head[0] + cw, base)], fill=(240, 196, 50), outline=INK)
    elif hat == "hardhat":
        d.pieslice([head[0] - R * 1.05, head[1] - R * 1.12, head[0] + R * 1.05, head[1] - R * .05], 180, 360, fill=(246, 200, 40), outline=INK, width=lw)
        d.rectangle([head[0] - R * 1.2, head[1] - R * .22, head[0] + R * 1.2, head[1] - R * .08], fill=(226, 176, 28), outline=INK, width=lw)
    elif hat == "helmet":
        d.pieslice([head[0] - R * 1.06, head[1] - R * 1.08, head[0] + R * 1.06, head[1] + R * .15], 180, 360, fill=(150, 156, 168), outline=INK, width=lw)
        d.line([head[0], head[1] - R * 1.05, head[0], head[1] - R * .1], fill=INK, width=lw)
    elif hat == "cap":
        d.pieslice([head[0] - R * 1.0, head[1] - R * 1.12, head[0] + R * 1.0, head[1] - R * .02], 180, 360, fill=(52, 70, 120), outline=INK, width=lw)
        d.rectangle([head[0] - R * .1, head[1] - R * .22, head[0] + R * 1.3 * f, head[1] - R * .06], fill=(52, 70, 120), outline=INK, width=lw)
    fcx = head[0] + turn * R * .38
    ey = head[1] - R * .02
    esp = R * (.38 - abs(turn) * .06)
    mood = pose.get("mood", "neutral")
    lid = {"sleepy": .55, "worried": .25, "angry": .3}.get(mood, pose.get("lid", 0.0))
    tilt = {"angry": -1.0, "worried": 1.0, "sad": 1.0}.get(mood, 0.0)
    for sgn in (-1, 1):
        ex = fcx + sgn * esp
        draw_eye(d, ex, ey, R, pose.get("look", (0.0, 0.0)), pose.get("blink", 0.0), lid, lw)
        draw_brow(d, ex, ey - R * (.2 if mood != "shock" else .28), R, tilt, pose.get("brow", 0.0) + (1.0 if mood == "shock" else 0), sgn, lw)
    draw_mouth(d, pose.get("mouth", "X"), fcx + turn * R * .1, head[1] + R * .52, R, mood, lw)
    if mood == "worried" or pose.get("sweat", 0):
        sx_, sy_ = head[0] + R * .85 * f, head[1] - R * .35 + (pose.get("t", 0) % 1.2) * R * .8
        d.ellipse([sx_ - R * .09, sy_ - R * .13, sx_ + R * .09, sy_ + R * .13], fill=(130, 200, 245), outline=INK, width=max(1, lw - 1))
    return img.resize((W, H), Image.LANCZOS)


# ----------------------------------------------------------------------------------- motion: spring-damper chasers ------------------------------------
CHANNELS = ["torso", "head", "turn", "a1", "a2", "b1", "b2", "l1", "l2", "m1", "m2", "x", "y", "squash", "brow"]
# (frequency rad/s, damping ratio): lower freq lags more = follow-through; low damping = overshoot
SPRING = {"torso": (11, .62), "head": (13, .5), "turn": (14, .8), "a1": (12, .48), "a2": (9, .42), "b1": (12, .48), "b2": (9, .42),
          "l1": (16, .8), "l2": (16, .8), "m1": (16, .8), "m2": (16, .8), "x": (9, .9), "y": (18, .55), "squash": (22, .35), "brow": (18, .7)}

POSES = {   # hand-designed key poses (degrees from straight down; positive = toward facing)
    "idle":    dict(torso=0, head=0, a1=15, a2=12, b1=-15, b2=12, l1=2, l2=0, m1=-2, m2=0),
    "explain": dict(torso=-2, head=3, a1=70, a2=40, b1=-10, b2=14, l1=3, l2=0, m1=-3, m2=0, hand_a=1),
    "explain2": dict(torso=2, head=-3, a1=34, a2=70, b1=-62, b2=38, l1=3, l2=0, m1=-3, m2=0, hand_a=1, hand_b=1),
    "point":   dict(torso=4, head=2, a1=92, a2=2, b1=-8, b2=10, l1=5, l2=0, m1=-5, m2=0, hand_a=2),
    "shrug":   dict(torso=0, head=7, a1=44, a2=82, b1=-44, b2=82, l1=2, l2=0, m1=-2, m2=0, hand_a=1, hand_b=1),
    "shock":   dict(torso=-8, head=-6, a1=128, a2=-34, b1=-122, b2=-34, l1=8, l2=-2, m1=-8, m2=-2, hand_a=1, hand_b=1),
    "facepalm": dict(torso=7, head=14, a1=116, a2=128, b1=-10, b2=12, l1=2, l2=0, m1=-2, m2=0),
    "cheer":   dict(torso=-6, head=-6, a1=152, a2=-10, b1=-152, b2=-10, l1=4, l2=0, m1=-4, m2=0, hand_a=1, hand_b=1),
    "think":   dict(torso=3, head=8, a1=62, a2=126, b1=-8, b2=12, l1=2, l2=0, m1=-2, m2=0),
    "cower":   dict(torso=12, head=14, a1=110, a2=-92, b1=-100, b2=-92, l1=14, l2=-16, m1=-4, m2=-14),
}


class Actor:
    """Plays a timeline [(t, pose_name, {extra})] through spring chasers, with blinks, eye darts, breathing and an audio-driven mouth."""

    def __init__(self, look, timeline, mouth_cues, x_frac=0.5, scale=0.78, facing=1, seed=0, mood_track=None):
        self.look, self.tl, self.cues, self.xf, self.sc, self.facing, self.seed = look, sorted(timeline, key=lambda k: k[0]), mouth_cues, x_frac, scale, facing, seed
        self.moods = mood_track or [(0.0, "neutral")]
        self._sim = None

    def _simulate(self, dur, dt=1 / 120):
        n = int(dur / dt) + 2
        state = {c: [POSES["idle"].get(c, 0.0), 0.0] for c in CHANNELS}
        out = []
        k = 0
        tgt = dict(POSES["idle"])
        for i in range(n):
            t = i * dt
            while k < len(self.tl) and self.tl[k][0] <= t:
                nm = self.tl[k][1]
                tgt = {**{c: 0.0 for c in CHANNELS}, **POSES.get(nm, POSES["idle"])}
                tgt.update(self.tl[k][2] if len(self.tl[k]) > 2 else {})
                k += 1
            row = {}
            for c in CHANNELS:
                w, z = SPRING[c]
                x, v = state[c]
                a = -2 * z * w * v - w * w * (x - tgt.get(c, 0.0))
                v += a * dt; x += v * dt
                state[c] = [x, v]
                row[c] = x
            row["hand_a"] = tgt.get("hand_a", 0); row["hand_b"] = tgt.get("hand_b", 0)
            out.append(row)
        self._sim = (out, dt)

    def pose(self, t, dur):
        if self._sim is None:
            self._simulate(dur)
        out, dt = self._sim
        row = dict(out[min(len(out) - 1, int(t / dt))])
        sd = self.seed
        per = 3.4 + (sd % 5) * .3
        ph = (t + sd * .7) % per
        row["blink"] = (1 - abs(2 * ((ph - (per - .16)) / .16) - 1)) if ph > per - .16 else 0.0
        row["look"] = (math.sin(t * .55 + sd) * .8, math.sin(t * .83 + sd * 2) * .4)
        row["y"] = row.get("y", 0) + math.sin(t * 2.0 + sd) * 1.6                                            # breathing
        row["torso"] = row["torso"] + math.sin(t * 1.1 + sd) * 0.9
        m = "X"
        for (a, b, s) in self.cues:
            if a <= t < b:
                m = s
                break
        row["mouth"] = m
        mood = "neutral"
        for (tm, nm) in self.moods:
            if tm <= t:
                mood = nm
        row["mood"] = mood
        row["facing"] = self.facing
        row["t"] = t
        return row

    def render(self, t, dur, W, H):
        """Returns (RGBA tile, x0, y0): a tight canvas around the character instead of a full-frame layer, so rendering stays fast."""
        p = self.pose(t, dur)
        u = H * self.sc
        tw, th = int(u * 1.9), int(u * 1.12)
        tile = draw_character(p, self.look, u, tw, th, tw / 2, th * 0.96)
        return tile, int(W * self.xf - tw / 2), int(H * 0.97 - th * 0.96)
