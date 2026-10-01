"""Scene acting director: turns an actor's action + emotion into a timeline of retargeted motion-capture clips.

Keys it returns are the engine's normal keys with pose "mc:<clip>" plus: ct0 (clip start), rate (playback speed), lin (constant-speed travel),
xf (cross-fade seconds into the next clip) and speak (this actor is talking, so the face lip-syncs).
"""
import random

from . import mocap

W, H = 1280, 720
TALK = ["talk_1", "talk_2", "talk_3", "talk_4", "talk_5", "talk_6"]
WALK_BY_EMOTION = {"sad": "walk_sad", "worried": "walk_careful", "shock": "walk_scared", "angry": "walk_brisk", "smile": "walk_happy", "neutral": "walk"}
# legacy action names -> clip (loop) ; "once" clips are followed by an idle so nobody freezes
ACTION_CLIP = {"think": ("think", True), "point": ("point", True), "shrug": ("shrug", False), "cheer": ("happy", True), "scared": ("scared", False),
               "slump": ("slump", False), "crouch": ("duck", False), "demand": ("quarrel_a", True), "proud": ("talk_3", True),
               "look_around": ("look_around", True), "duck": ("duck", False), "dodge": ("duck", False), "wave": ("wave", False), "punch": ("punch", True),
               "kick": ("kick", True), "push": ("talk_2", True), "pull": ("talk_2", True), "pick_up": ("talk_4", True), "stumble": ("stumble", False),
               "fall": ("fall", False), "get_up": ("talk_1", True), "cry": ("cry", True), "flinch": ("flinch", False), "sit": ("slump", False)}


LEGACY = {"think": "think", "look_around": "rx:look", "flinch": "rx:flinch", "scared": "rx:cower", "duck": "rx:duck", "dodge": "rx:duck", "fall": "rx:slump", "stumble": "rx:stumble", "slump": "rx:slump", "cry": "mc:cry", "surprised": "rx:flinch", "sit_down": "rx:slump", "crouch": "rx:duck"}       # these two captures are crouches; the hand-posed versions read better


def _pn(clip):
    return LEGACY.get(clip) or "mc:" + clip


BEND_ACTS = {"pull", "push", "pick_up", "duck", "dodge", "stumble", "fall", "sit", "crouch", "cry", "scared", "kick", "punch"}


RX_DUR = {"flinch": 2.0, "cower": 3.4, "duck": 1.7, "slump": None, "stumble": 1.3}


def _has(clip):
    return clip in LEGACY or mocap.has(clip)


def _dur(clip):
    c = LEGACY.get(clip)
    if c and c.startswith("rx:"):
        return RX_DUR.get(c[3:], 2.0)
    if c:
        return 2.0
    return mocap.duration(clip)


def available():
    return bool(mocap._load())


def _outward(x, n):
    return 0.0 if n < 2 else (-0.07 if x < 0.5 else 0.07)


def _leg_speed(dxr, dt, scale):
    """Walking speed of an actor in leg lengths per second, so the feet match the ground they cover."""
    unit = H * 0.52 * scale
    return abs(dxr) * W / max(dt, 1e-6) / (unit * mocap.LEG)


def _walk_rate(clip, dxr, dt, scale):
    sp = mocap.info(clip).get("speed", 1.2) or 1.2
    return max(0.55, min(1.8, _leg_speed(dxr, dt, scale) / sp))


def _talk_clip(em, rnd, last=None):
    pool = {"angry": ["quarrel_a", "quarrel_b", "talk_3", "talk_5"], "worried": ["talk_2", "talk_4", "talk_5"], "shock": ["talk_6", "talk_1", "talk_4"],
            "sad": ["talk_4", "sad", "talk_2"]}.get(em, TALK)
    pool = [c for c in pool if mocap.has(c) and c != last] or [c for c in TALK if mocap.has(c)]
    return rnd.choice(pool)


def _base_keys(a, dur, idx, n, pos_x, facing, other=None):
    """a: cleaned actor dict; pos_x: x position (0-1); facing: +1/-1 toward the way it looks; other: the other actor's cleaned dict (for pair scenes)."""
    if not available():
        return None
    em, act = a["emotion"], a["action"]
    sc = a.get("scale", 1.0)
    rnd = random.Random(idx * 97 + int(pos_x * 1000))
    K = lambda t, x, clip, **kw: {**dict(t=t, x=x, pose=_pn(clip), face=em, facing=facing), **kw}
    center = 0.5
    face_other = (1 if pos_x < center else -1) if n > 1 else facing

    if act == "talk":
        # two people who are both calm or both angry get the genuinely paired capture (the two halves of one real conversation)
        pair = n == 2 and other is not None and other["action"] == "talk"
        if pair and em in ("neutral", "smile", "worried") and mocap.has("explain_a"):
            clip = "explain_a" if idx == 0 else "explain_b"
            return [K(0, pos_x, clip, loop=True, facing=face_other, speak=(idx == 0)), K(dur + 2, pos_x, clip, loop=True, facing=face_other, speak=(idx == 0), ct0=0)]
        if pair and em == "angry" and mocap.has("quarrel_a"):
            clip = "quarrel_a" if idx == 0 else "quarrel_b"
            return [K(0, pos_x, clip, loop=True, facing=face_other, speak=True), K(dur + 2, pos_x, clip, loop=True, facing=face_other, speak=True, ct0=0)]
        ks, t, last, i = [], 0.0, None, 0
        while t < dur + 3.0:
            speaking = n == 1 or (i + idx) % 2 == 0
            seg = rnd.uniform(2.6, 4.2)
            if speaking:
                clip = _talk_clip(em, rnd, last)
            else:
                clip = rnd.choice(["think", "look_around", "shrug", "think"])
            last = clip
            ks.append(K(t, pos_x, clip, facing=face_other, speak=speaking, xf=0.35, co=rnd.uniform(0, 1.5)))
            t += seg
            i += 1
        return ks
    if act in ("walk", "enter_walk", "run", "enter_run", "exit_run"):
        run = "run" in act
        clip = "run" if run else WALK_BY_EMOTION.get(em, "walk")
        if not _has(clip):
            return None
        if act in ("walk", "run"):
            end = pos_x + (.2 if facing > 0 else -.2)
            r = _walk_rate(clip, end - pos_x, dur, sc)
            return [K(0, pos_x, clip, rate=r, lin=True), K(dur, end, clip, rate=r, lin=True, ct0=0)]
        if act in ("enter_walk", "enter_run"):
            start = -0.12 if pos_x < .5 else 1.12
            f0 = 1 if start < 0 else -1
            k1 = min(dur * 0.45, 2.6)
            r = _walk_rate(clip, pos_x - start, k1, sc)
            return [K(0, start, clip, rate=r, lin=True, facing=f0), K(k1, pos_x, clip, rate=r, lin=True, facing=f0, ct0=0, xf=0.4),
                    dict(t=k1 + .6, x=pos_x, pose="stand", face=em, facing=facing)]
        end = 1.15 if pos_x >= .5 else -0.15
        f1 = 1 if end > pos_x else -1
        t1 = dur * .4
        r = _walk_rate("run", end - pos_x, 0.9, sc)
        return [dict(t=0, x=pos_x, pose="stand", face=em, facing=f1), dict(t=t1, x=pos_x, pose="stand", face=em, facing=f1),
                K(t1 + .5, end, "run", rate=r, lin=True, facing=f1, xf=0.3)]
    if act == "fight":
        meet = .5 + (-.13 if pos_x < center else .13)
        armed = a["role"] in ("soldier", "warrior", "knight", "pirate", "general", "rebel")
        t1 = min(1.0 + .15 * idx, dur * .3)
        r = _walk_rate("run", meet - pos_x, t1, sc)
        f0 = 1 if meet > pos_x else -1
        first = ("sword_1" if idx % 2 == 0 else "sword_2") if armed else "punch"
        ks = [K(0, pos_x, "run", rate=r, lin=True, facing=f0), K(t1, meet, first, facing=f0, loop=True, ct0=t1 - 0.3, xf=0.25, co=idx * 1.7)]
        t, j = t1 + 2.6, 0
        while t < dur + 3:
            clip = rnd.choice(["sword_1", "sword_2"] if armed else ["punch", "kick", "boxing"])
            back = meet + (-.03 if pos_x < center else .03) * (j % 2)
            ks.append(K(t, back, clip, facing=f0, loop=True, xf=0.3, co=rnd.uniform(0, 3)))
            t += rnd.uniform(2.2, 3.4)
            j += 1
        return ks
    if act in ACTION_CLIP:
        clip, loop = ACTION_CLIP[act]
        if not _has(clip):
            return None
        px = pos_x + _outward(pos_x, n) if act in BEND_ACTS else pos_x                # a person bending over needs room that the other person is not standing in
        ks = [K(0, px, clip, loop=loop, facing=face_other, speak=False, ct0=0)]
        dn = _dur(clip)
        if not loop and dn is not None:
            ks.append(dict(t=dn + 0.2, x=px, pose="stand", face=em, facing=face_other, xf=0.5))
        elif not loop:
            ks.append(K(dur + 2, px, clip, loop=False, facing=face_other, ct0=0))
        else:
            ks.append(K(dur + 2, px, clip, loop=True, facing=face_other, ct0=0))
        return ks
    return None


BEAT_CLIP = {"flinch": "flinch", "scared": "scared", "cry": "cry", "cheer": "happy", "duck": "duck", "point": "point", "look_around": "look_around", "wave": "wave",
             "shrug": "shrug", "think": "think", "punch": "punch", "kick": "kick", "fight_burst": "sword_1", "stumble": "stumble", "fall": "fall", "sit": "slump"}
BEAT_LEN = {"flinch": 1.7, "scared": 2.6, "cry": 3.2, "cheer": 2.8, "duck": 2.2, "point": 2.2, "look_around": 2.6, "wave": 2.2, "shrug": 2.0, "think": 2.6, "punch": 2.0,
            "kick": 2.0, "fight_burst": 2.4, "push": 2.4, "pull": 2.4, "pick_up": 2.4, "stumble": 1.8, "sit": 2.8, "idle": 2.2, "talk_angry": 2.8}
STATIC = {"talk", "stand", "think", "point", "shrug", "cheer", "scared", "slump", "crouch", "proud", "demand", "look_around", "cry", "sit", "wave"}


def _static(k0, k1):
    return abs(k0["x"] - k1["x"]) < 1e-6 and not k0.get("lin")


def _face_schedule(em0, dur, evs_faces):
    """[(t0, t1, face)]: the scene's mood comes in waves rather than as a permanent frown; beats override it for their length."""
    sched = []
    if em0 != "neutral":
        t = 0.0
        while t < dur + 2:
            sched.append((t, t + 1.9, em0))
            t += 3.8
    for t0, t1, f in evs_faces:
        sched = [(a, b, f_) for a, b, f_ in sched if b <= t0 - 0.05 or a >= t1]
        sched.append((t0, t1, f))
    return sorted(sched)


def _apply_faces(keys, sched):
    """Give every key the face the schedule asks for at its time, and split static intervals where the face changes."""
    def face_at(t):
        for a, b, f in sched:
            if a <= t < b:
                return f
        return "neutral"
    cut = sorted({x for a, b, _ in sched for x in (a, b)})
    out = []
    for i, k in enumerate(keys):
        out.append(k)
        nxt = keys[i + 1]["t"] if i + 1 < len(keys) else k["t"] + 1.0
        if i + 1 < len(keys) and not _static(k, keys[i + 1]):
            continue
        for tb in cut:
            if k["t"] + 0.12 < tb < nxt - 0.2:
                dup = dict(k)
                dup.update(t=tb, xf=0.06, ct0=k.get("ct0", k["t"] - k.get("xf", 0.3)))
                out.append(dup)
    out.sort(key=lambda k: k["t"])
    for k in out:
        k["face"] = face_at(k["t"]) if k.get("face") is not None else "neutral"
        if k["pose"] in ("stand", "think") and False:
            pass
    return out


def keys_for(a, dur, idx, n, pos_x, facing, other=None, beats=None):
    """The actor's timeline: the scene's base acting plus beats timed to the words being spoken."""
    ks = _keys_for(a, dur, idx, n, pos_x, facing, other, beats)
    if not ks or a["action"] not in STATIC | {"talk"}:
        return ks
    ev_faces = [(k["t"], k["t"] + 2.4, k["face"]) for k in ks if k.get("face") not in (None, a["emotion"]) and k["t"] > 0.1] if beats else []
    # beat events carry their own emotion; collect them before the schedule rewrites faces
    return _apply_faces(ks, _face_schedule(a["emotion"], dur, ev_faces))


def _keys_for(a, dur, idx, n, pos_x, facing, other=None, beats=None):
    base = _base_keys(a, dur, idx, n, pos_x, facing, other)
    if not base or not beats or a["action"] not in STATIC:
        return base
    em0, sc = a["emotion"], a.get("scale", 1.0)
    rnd = random.Random(idx * 53 + 7)
    face_other = (1 if pos_x < 0.5 else -1) if n > 1 else facing
    evs, cur_x, last_end = [], pos_x, -1.0
    for b in sorted(beats, key=lambda b: b["t"]):
        who = b["who"]
        mine = who == "all" or (who == "agent" and (b["k"] + idx) % 2 == 0) or (who == "reactor" and (b["k"] + idx) % 2 == 1)
        if n == 1:
            mine = True
        if not mine:
            continue
        ts = b["t"] + (0.25 if who == "reactor" and n > 1 else 0.0) - 0.05
        if ts < last_end + 0.15 or ts > dur - 0.4:
            continue
        act = b["action"]
        emo = b["emotion"] or em0
        if act in ("run_to", "walk_to"):
            run = act == "run_to"
            clip = "run" if run else WALK_BY_EMOTION.get(emo, "walk")
            if not _has(clip):
                continue
            if other is not None and n > 1 and "x" in other:
                ox_ = other["x"]
                tx = ox_ - 0.17 if pos_x < ox_ else ox_ + 0.17                    # stop a comfortable distance short of the other person
                tx = tx if abs(tx - cur_x) < abs(ox_ - cur_x) else cur_x
            else:
                tx = cur_x + (0.2 if face_other > 0 else -0.2) * (1.1 if run else 0.8)
            tx = max(0.1, min(0.9, tx))
            dist = abs(tx - cur_x)
            if dist < 0.04:
                continue
            ln = max(0.8, min(3.0, dist / (0.21 if run else 0.1)))
            fx = 1 if tx > cur_x else -1
            r = _walk_rate(clip, tx - cur_x, ln, sc)
            evs.append((ts, ts + ln, dict(x=cur_x, pose=_pn(clip), face=emo, facing=fx, rate=r, lin=True, xf=0.3, ct0=ts), tx, fx))
            cur_x, last_end = tx, ts + ln
            continue
        if act == "talk_angry":
            clip = rnd.choice([c for c in ("quarrel_a", "quarrel_b") if mocap.has(c)] or [None])
            if clip is None:
                continue
            ln = BEAT_LEN["talk_angry"]
            evs.append((ts, ts + ln, dict(x=cur_x, pose=_pn(clip), face="angry", facing=face_other, speak=True, xf=0.3, ct0=ts, loop=True, co=rnd.uniform(0, 2)), cur_x, face_other))
            last_end = ts + ln
            continue
        if act == "idle":
            evs.append((ts, ts + BEAT_LEN["idle"], dict(x=cur_x, pose="stand", face=emo, facing=face_other, xf=0.4), cur_x, face_other))
            last_end = ts + BEAT_LEN["idle"]
            continue
        clip = BEAT_CLIP.get(act)
        if not clip or not _has(clip):
            continue
        ln = BEAT_LEN.get(act, 2.2)
        if act == "fall":
            ln = max(1.0, dur - ts + 1.0)                                   # a body that has fallen stays down for the rest of the scene
        ex_ = max(0.08, min(0.92, cur_x + (_outward(cur_x, n) if act in BEND_ACTS else 0.0)))
        evs.append((ts, ts + ln, dict(x=ex_, pose=_pn(clip), face=emo, facing=face_other, speak=False, xf=0.25, ct0=ts - 0.05, loop=False if act in ("flinch", "fall", "stumble", "duck", "wave", "shrug", "pick_up", "sit") else True), cur_x, face_other))
        last_end = ts + ln
    if not evs:
        return base

    def base_at(t):
        cur = base[0]
        for k in base:
            if k["t"] <= t:
                cur = k
        return cur
    times = sorted(set([k["t"] for k in base] + [e[0] for e in evs] + [e[1] for e in evs]))
    out, x_now, fx_now = [], pos_x, None
    for tm in times:
        ev = next((e for e in evs if e[0] <= tm < e[1]), None)
        moved = [e for e in evs if e[1] <= tm]
        x_now = moved[-1][3] if moved else pos_x
        if ev is not None:
            if abs(tm - ev[0]) < 1e-6:
                out.append({**ev[2], "t": tm})
            continue
        bk = dict(base_at(tm))
        bk["t"] = tm
        bk["x"] = x_now if moved else bk["x"]
        if moved:
            bk["facing"] = moved[-1][4] if moved[-1][1] > tm - 0.05 else bk.get("facing", 1)
            bk["ct0"] = tm - 0.3
        out.append(bk)
    return out
