"""Build studio/motion_data.npz from the CMU Graphics Lab motion-capture database (BVH conversion by B. Hahne, free for all uses).

Each clip is projected onto the stickman's 2D skeleton: we keep only the DIRECTION of every bone (so the stickman keeps its own proportions) and
store the angles in the engine's convention (degrees from 'straight down', positive toward the way the figure faces):
    [torso, head, a1, a2, b1, b2, l1, l2, m1, m2, lift]
a = near arm, b = far arm, l = near leg, m = far leg; a2/b2/l2/m2 are relative to their parent bone, head is relative to the torso.

usage: python tools/build_motion_lib.py [--cache DIR]
Source: https://github.com/una-dinosauria/cmu-mocap (data/<subject>/<subject>_<trial>.bvh)
"""
import json
import math
import pathlib
import sys
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/{d}/{s}_{t}.bvh"
FPS_OUT = 30

# name: (subject, trial, camera azimuth in degrees (90 = pure profile, 0 = front), (start_s, end_s) or None, loops, description)
CLIPS = {
    # locomotion
    "walk": ("07", "01", 90, None, True, "normal walk"),
    "walk_slow": ("07", "04", 90, None, True, "slow walk"),
    "walk_brisk": ("07", "12", 90, None, True, "brisk walk"),
    "run": ("09", "01", 90, None, True, "run"),
    "walk_sad": ("105", "13", 90, None, True, "sad walk"),
    "walk_scared": ("105", "32", 85, None, True, "scared walk"),
    "walk_careful": ("105", "18", 85, None, True, "careful walk looking around"),
    "walk_happy": ("104", "11", 90, None, True, "happy go lucky walk"),
    "sneak": ("139", "29", 90, None, True, "sneaking"),
    "walk_wounded": ("139", "19", 90, None, True, "walk with a wounded leg"),
    # talking: story telling and explaining with the whole body
    "talk_1": ("138", "11", 62, None, True, "story telling"),
    "talk_2": ("138", "13", 62, None, True, "story telling"),
    "talk_3": ("138", "15", 62, None, True, "story telling"),
    "talk_4": ("138", "17", 62, None, True, "story telling"),
    "talk_5": ("138", "20", 62, None, True, "story telling"),
    "talk_6": ("138", "24", 62, None, True, "story telling"),
    "explain_a": ("18", "08", 62, None, True, "conversation, explain with hand gestures (A)"),
    "explain_b": ("19", "08", 62, None, True, "conversation, explain with hand gestures (B)"),
    "quarrel_a": ("18", "10", 62, None, True, "quarrel, angry hand gestures (A)"),
    "quarrel_b": ("19", "10", 62, None, True, "quarrel, angry hand gestures (B)"),
    "threat_a": ("22", "21", 62, None, True, "conversation, pounds stool and points (A)"),
    "threat_b": ("23", "21", 62, None, True, "conversation, pounds stool and points (B)"),
    # emotions and reactions
    "scared": ("142", "16", 62, None, False, "scared"),
    "surprised": ("120", "15", 62, None, False, "surprised"),
    "sad": ("142", "15", 62, None, True, "sad"),
    "cry": ("80", "45", 62, None, True, "crying"),
    "happy": ("142", "08", 62, None, True, "happy"),
    "shrug": ("141", "21", 62, None, False, "shrug"),
    "wave": ("141", "16", 62, None, False, "wave hello"),
    "point": ("13", "27", 75, None, True, "direct traffic, wave, point"),
    "think": ("74", "12", 75, None, True, "thinker"),
    "look_around": ("139", "01", 70, None, True, "looking around"),
    "duck": ("77", "09", 80, None, False, "duck to avoid a flying object"),
    "dodge": ("76", "03", 80, None, False, "avoid attacker"),
    # action
    "punch": ("143", "23", 80, None, True, "punching"),
    "kick": ("143", "24", 80, None, True, "kicking"),
    "sword_1": ("02", "07", 80, None, True, "swordplay"),
    "sword_2": ("02", "08", 80, None, True, "swordplay"),
    "boxing": ("14", "01", 80, None, True, "boxing"),
    "push": ("81", "05", 85, None, True, "push heavy object"),
    "pull": ("81", "07", 85, None, True, "pull heavy object"),
    "pick_up": ("115", "01", 85, None, False, "pick up a box"),
    "stumble": ("105", "59", 80, None, False, "small stumble"),
    "fall": ("90", "16", 85, None, False, "fall on face"),
    "get_up": ("139", "16", 85, None, False, "get up from the ground"),
    "sit_down": ("13", "01", 85, None, False, "sit on a stool and stand up"),
}

STATIONARY = {"talk_1", "talk_2", "talk_3", "talk_4", "talk_5", "talk_6", "explain_a", "explain_b", "quarrel_a", "quarrel_b", "threat_a", "threat_b", "sad", "cry", "happy",
              "point", "think", "look_around", "shrug", "wave"}
NEAR = {"Left": ["LeftArm", "LeftForeArm", "LeftHand", "LeftUpLeg", "LeftLeg", "LeftFoot"],
        "Right": ["RightArm", "RightForeArm", "RightHand", "RightUpLeg", "RightLeg", "RightFoot"]}


def parse_bvh(text):
    lines = [l.strip() for l in text.splitlines()]
    joints, stack, i = [], [], 0
    while lines[i] != "MOTION":
        l = lines[i]
        if l.startswith("ROOT") or l.startswith("JOINT"):
            joints.append(dict(name=l.split()[1], parent=stack[-1] if stack else -1, off=None, ch=[]))
            stack.append(len(joints) - 1)
        elif l.startswith("End Site"):
            joints.append(dict(name=joints[stack[-1]]["name"] + "End", parent=stack[-1], off=None, ch=[]))
            stack.append(len(joints) - 1)
        elif l.startswith("OFFSET"):
            joints[stack[-1]]["off"] = np.array([float(v) for v in l.split()[1:4]])
        elif l.startswith("CHANNELS"):
            joints[stack[-1]]["ch"] = l.split()[2:]
        elif l == "}":
            stack.pop()
        i += 1
    n = int(lines[i + 1].split()[1])
    ft = float(lines[i + 2].split(":")[1])
    data = np.array([[float(v) for v in lines[i + 3 + k].split()] for k in range(n)])
    return joints, data, ft


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def fk(joints, data):
    n = data.shape[0]
    pos = [None] * len(joints)
    R = [None] * len(joints)
    col = 0
    for j, jt in enumerate(joints):
        loc = np.zeros((n, 3)); rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for c in jt["ch"]:
            v = data[:, col]; col += 1
            if c.endswith("position"):
                loc[:, "XYZ".index(c[0])] = v
            else:
                rot = rot @ _rot(c[0], v)
        if jt["parent"] < 0:
            pos[j], R[j] = loc, rot
        else:
            p = jt["parent"]
            pos[j] = pos[p] + np.einsum("nij,j->ni", R[p], jt["off"])
            R[j] = R[p] @ rot
    return {jt["name"]: pos[j] for j, jt in enumerate(joints)}


def _ang(vx, vy):
    """Angle from straight down (image y points down), positive toward +x."""
    return np.degrees(np.arctan2(vx, vy))


def retarget(P, azimuth, ft, to_fps=FPS_OUT):
    """returns (pose angles at to_fps, extras)"""
    n = P["Hips"].shape[0]
    up = np.array([0, 1.0, 0])
    h = P["LeftUpLeg"] - P["RightUpLeg"]
    fwd = np.cross(h, up)[:, [0, 2]]
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    f0 = fwd.mean(axis=0); f0 /= np.linalg.norm(f0) + 1e-9
    f0 = np.array([f0[0], 0, f0[1]])
    left0 = np.cross(up, f0)
    phi = math.radians(azimuth)
    Rs = f0 * math.sin(phi) + left0 * math.cos(phi)
    zs = np.cross(Rs, up)                                               # toward the viewer
    X = lambda name: P[name] @ Rs
    Y = lambda name: -P[name][:, 1]
    depth = lambda name: P[name] @ zs
    near = "Left" if depth("LeftArm").mean() + depth("LeftUpLeg").mean() > depth("RightArm").mean() + depth("RightUpLeg").mean() else "Right"
    far = "Right" if near == "Left" else "Left"
    legl = np.linalg.norm(P["LeftUpLeg"] - P["LeftLeg"], axis=1).mean() + np.linalg.norm(P["LeftLeg"] - P["LeftFoot"], axis=1).mean()

    def d(a, b):                                                         # direction a -> b in screen space (x right, y down)
        return X(b) - X(a), Y(b) - Y(a)
    out = np.zeros((n, 11))
    vx, vy = d("Hips", "Neck")
    torso = _ang(vx, -vy)
    out[:, 0] = torso
    hx, hy = d("Neck", "HeadEnd")
    out[:, 1] = _ang(hx, -hy) - torso
    for k, side in ((2, near), (4, far)):
        s_, e_, w_ = side + "Arm", side + "ForeArm", side + "Hand"
        ax, ay = d(s_, e_); bx, by = d(e_, w_)
        out[:, k] = _ang(ax, ay)
        out[:, k + 1] = _ang(bx, by) - out[:, k]
    for k, side in ((6, near), (8, far)):
        h_, kn, an = side + "UpLeg", side + "Leg", side + "Foot"
        ax, ay = d(h_, kn); bx, by = d(kn, an)
        out[:, k] = _ang(ax, ay)
        out[:, k + 1] = _ang(bx, by) - out[:, k]
    # wrap relative angles into (-180, 180]
    for k in (1, 3, 5, 7, 9):
        out[:, k] = (out[:, k] + 180) % 360 - 180
    # airborne amount: how far the lower foot is above the clip's floor, in leg lengths
    low = np.minimum(P["LeftFoot"][:, 1], P["RightFoot"][:, 1])
    floor = np.percentile(low, 3)
    out[:, 10] = np.clip((low - floor) / legl, 0, 1.2) * 0.44
    # forward speed of the root in leg lengths per second (for locomotion foot-slide matching)
    step = max(1, int(round((1.0 / ft) / to_fps)))
    sel = np.arange(1, n, step)                                           # frame 0 is the T-pose the converter added
    hd = np.degrees(np.arctan2(fwd[:, 0] * f0[2] - fwd[:, 1] * f0[0], fwd[:, 0] * f0[0] + fwd[:, 1] * f0[2]))     # heading deviation from the clip's mean heading
    root = P["Hips"][:, [0, 2]]
    return out[sel], dict(hd=hd[sel], root=root[sel], legl=legl)


def _energy(arr, fps):
    d = np.abs(np.diff(arr[:, :10], axis=0)).sum(axis=1)
    d = np.concatenate([d, d[-1:]])
    k = max(3, int(fps * 0.4))
    return np.convolve(np.pad(d, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]


def _loop_window(arr, valid, fps, lo, hi):
    """Best seamless cycle: start s and period p (seconds in [lo, hi]) where the pose at s+p matches the pose at s, inside the valid frames."""
    n = len(arr)
    best, bs, bp = 1e18, 0, int(lo * fps)
    w = np.array([1, 1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=float)
    vel = np.vstack([np.zeros((1, 10)), np.diff(arr[:, :10], axis=0)]) * fps * 0.15
    for p in range(int(lo * fps), int(hi * fps) + 1):
        for s0 in range(0, n - p - 1):
            if not (valid[s0] and valid[s0 + p]):
                continue
            c = (np.abs(arr[s0, :10] - arr[s0 + p, :10]) * w).sum() + (np.abs(vel[s0] - vel[s0 + p]) * w).sum()
            if c < best:
                best, bs, bp = c, s0, p
    return bs, bp, best


def window(arr, ex, kind, fps, kind_name=""):
    """kind: 'cycle' (walk/run), 'loop' (looping gestures), 'once' (a single reaction). Returns (frames, speed in leg lengths per second)."""
    n = len(arr)
    valid = np.abs(ex["hd"]) < (18 if kind == "cycle" else 60)
    if kind == "cycle":
        if valid.sum() < fps * 1.5:
            valid = np.abs(ex["hd"]) < 35
        bs, bp, _ = _loop_window(arr, valid, fps, 0.5 if kind_name == "run" else 0.95, 0.75 if kind_name == "run" else 1.6)
        seg = arr[bs:bs + bp + 1].copy()
        # a stride pair reads more natural than a single stride, but one is enough for a cartoon walk; close the loop smoothly
        k = min(4, len(seg) // 4)
        for i in range(k):
            u = (i + 1) / (k + 1)
            seg[-k + i] = seg[-k + i] * (1 - u) + seg[i] * u
        seg = seg[:-1]
        d = ex["root"][bs + bp] - ex["root"][bs]
        speed = float(np.linalg.norm(d) / (bp / fps) / ex["legl"])
        if speed < 0.3:                                                  # the root did not travel in this clip: estimate from the leg swing instead
            l1 = np.radians(arr[bs:bs + bp, 6])
            step = float(np.sin(l1.max()) - np.sin(l1.min()))
            speed = max(1.0, 2 * step / (bp / fps) * 0.9)
        return seg, speed
    e = _energy(arr, fps)
    L = int((6.0 if kind == "loop" else 3.6) * fps)
    L = min(L, n)
    if n <= L:
        a0 = 0
    else:
        cs = np.concatenate([[0], np.cumsum(e)])
        scores = cs[L:] - cs[:-L]
        scores = np.where(np.convolve(valid.astype(float), np.ones(L) / L, mode="valid") > 0.5, scores, -1)
        a0 = int(np.argmax(scores))
        if kind == "once":
            a0 = max(0, a0 - int(0.3 * fps))
    seg = arr[a0:a0 + L].copy()
    if kind == "loop":                                                  # cross-fade the tail into the head so the gesture cycle has no seam
        k = min(int(0.6 * fps), len(seg) // 3)
        for i in range(k):
            u = (i + 1) / (k + 1)
            seg[-k + i] = seg[-k + i] * (1 - u) + seg[i] * u * 0 + (seg[i - k] if False else seg[i]) * u * 1.0 - seg[-k + i] * u * 0
        seg = seg[:-1]
    return seg, 0.0


def main():
    cache = pathlib.Path(sys.argv[sys.argv.index("--cache") + 1]) if "--cache" in sys.argv else ROOT / ".cache_bvh"
    cache.mkdir(exist_ok=True)
    arrays, meta = {}, {}
    for name, (s, t, az, win, loops, desc) in CLIPS.items():
        f = cache / f"{s}_{t}.bvh"
        if not f.exists():
            urllib.request.urlretrieve(RAW.format(d=s.zfill(3), s=s, t=t), f)
        joints, data, ft = parse_bvh(f.read_text())
        P = fk(joints, data)
        arr0, ex = retarget(P, az, ft)
        kind = "cycle" if name.startswith(("walk", "run", "sneak")) else ("loop" if loops else "once")
        arr, speed = window(arr0, ex, kind, FPS_OUT, name)
        if name in STATIONARY:                                           # standing performances: keep weight shifts but drop the foot shuffling (the stickman does not travel)
            k = int(FPS_OUT * 1.3) | 1
            pad = np.pad(arr[:, 6:10], ((k, k), (0, 0)), mode="edge")
            ker = np.ones(k) / k
            arr[:, 6:10] = np.stack([np.convolve(pad[:, c], ker, mode="same")[k:-k] for c in range(4)], axis=1)
        arrays[name] = arr.astype(np.float16)
        meta[name] = dict(desc=desc, kind=kind, loops=bool(loops), speed=round(speed, 3), frames=int(len(arr)), fps=FPS_OUT, src=f"CMU {s}_{t}")
        print(f"{name:14s} {kind:5s} {len(arr)/FPS_OUT:5.1f}s speed {speed:.2f}  ({desc})")
    np.savez_compressed(ROOT / "studio" / "motion_data.npz", **arrays)
    (ROOT / "studio" / "motion_index.json").write_text(json.dumps(meta, indent=1))
    print("saved", len(arrays), "clips")


if __name__ == "__main__":
    main()
