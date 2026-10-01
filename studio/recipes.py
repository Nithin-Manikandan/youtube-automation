"""Turns the simple scene recipes an LLM can write into stickman scene data, and cleans up bad values.

The LLM never writes keyframes; it picks from small vocabularies, so output is reliable.
"""
import random

from .stick import BLUE, GOLD, GREY, PURPLE, RED

COLORS = {"red": RED, "blue": BLUE, "purple": PURPLE, "gold": GOLD, "green": (70, 130, 80), "grey": GREY,
          "white": (238, 234, 224), "brown": (140, 104, 72), "black": (52, 50, 56), "orange": (214, 120, 40)}
POS = {"far_left": .12, "left": .24, "center_left": .38, "center": .5, "center_right": .62, "right": .76, "far_right": .88}
ROLES = {  # role -> (props, default tunic colour name)
    "emperor": (["crown", "cape", "beard"], "purple"), "king": (["crown", "cape"], "red"), "queen": (["crown", "hair"], "purple"),
    "soldier": (["helmet", "shield", "spear"], "blue"), "warrior": (["helmet", "sword"], "red"), "knight": (["helmet", "shield", "sword"], "grey"),
    "citizen": (["hair"], "brown"), "peasant": (["hair"], "brown"), "merchant": (["hair", "beard"], "green"), "scholar": (["beard", "scroll"], "white"),
    "general": (["helmet", "cape", "sword"], "red"), "pirate": (["hair", "sword"], "black"), "explorer": (["hair", "flag"], "green"),
    "pharaoh": (["crown", "beard"], "gold"), "priest": (["beard", "scroll"], "white"), "rebel": (["hair", "spear"], "orange"),
    # modern (1800s onward): no swords, spears or shields
    "sailor": (["navycap"], "white"), "captain": (["navycap", "beard"], "blue"), "officer": (["navycap", "tie"], "grey"),
    "scientist": (["glasses", "hair"], "white"), "president": (["hair", "tie"], "black"), "worker": (["hardhat"], "orange"),
    "astronaut": (["spacehelmet"], "white"), "pilot": (["helmet"], "green"), "modern_soldier": (["helmet", "rifle"], "green"), "spy": (["hat", "tie"], "black"), "reporter": (["hat", "tie"], "brown"),
}
ACTIONS = {"think", "salute", "armscross", "facepalm", "demand", "stand", "talk", "cheer", "scared", "slump", "point", "proud", "shrug", "sword_up", "crouch", "fight",
           "enter_walk", "enter_run", "exit_run", "walk", "run"}
EMOTIONS = {"neutral", "smile", "sad", "angry", "shock", "worried"}
BACKGROUNDS = {
    "city_day": dict(sky=((150, 190, 222), (250, 228, 190)), sun=(0.80, 0.22, (252, 214, 120)), ground_color=(176, 158, 120),
                     hills=[dict(color=(178, 190, 200), base=.66, amp=.05, freq=3.0, seed=1, par=.10), dict(color=(150, 160, 150), base=.72, amp=.045, freq=4.2, seed=4, par=.22)]),
    "countryside": dict(sky=((140, 190, 235), (238, 240, 214)), sun=(0.2, 0.2, (252, 226, 140)), ground_color=(132, 160, 96),
                        hills=[dict(color=(150, 184, 160), base=.66, amp=.06, freq=2.6, seed=7, par=.10), dict(color=(108, 148, 92), base=.73, amp=.05, freq=3.8, seed=2, par=.22)]),
    "desert": dict(sky=((240, 196, 140), (252, 236, 196)), sun=(0.7, 0.2, (255, 240, 200)), ground_color=(222, 190, 130),
                   hills=[dict(color=(232, 200, 150), base=.68, amp=.04, freq=2.4, seed=3, par=.10), dict(color=(214, 176, 120), base=.74, amp=.035, freq=3.4, seed=5, par=.22)]),
    "storm": dict(sky=((70, 78, 96), (150, 146, 140)), ground_color=(120, 108, 88),
                  hills=[dict(color=(92, 98, 110), base=.64, amp=.06, freq=3.2, seed=2, par=.10), dict(color=(72, 76, 84), base=.72, amp=.05, freq=4.6, seed=6, par=.22)]),
    "night": dict(sky=((14, 20, 48), (52, 54, 90)), sun=(0.78, 0.2, (232, 232, 214)), ground_color=(60, 66, 70), dim=0.10,
                  hills=[dict(color=(30, 36, 62), base=.66, amp=.05, freq=3.0, seed=8, par=.10), dict(color=(22, 28, 48), base=.73, amp=.045, freq=4.0, seed=9, par=.22)]),
    "battlefield": dict(sky=((120, 70, 60), (226, 160, 110)), ground_color=(112, 92, 70),
                        hills=[dict(color=(96, 64, 60), base=.66, amp=.05, freq=3.0, seed=3, par=.10), dict(color=(74, 52, 48), base=.73, amp=.04, freq=4.4, seed=1, par=.22)]),
    "palace": dict(sky=((232, 214, 176), (246, 236, 214)), ground_color=(200, 184, 150),
                   hills=[dict(color=(220, 202, 164), base=.62, amp=.03, freq=3.0, seed=2, par=.05)]),
    "sea": dict(sky=((150, 196, 232), (240, 240, 226)), sun=(0.75, 0.22, (252, 232, 160)), ground_color=(70, 120, 168),
                hills=[dict(color=(110, 154, 196), base=.62, amp=.012, freq=6.0, seed=1, par=.05)]),
    "snow": dict(sky=((186, 204, 226), (240, 244, 250)), ground_color=(236, 240, 246),
                 hills=[dict(color=(214, 222, 236), base=.66, amp=.07, freq=2.8, seed=6, par=.10), dict(color=(228, 234, 244), base=.73, amp=.05, freq=3.6, seed=3, par=.22)]),
    "space": dict(sky=((1, 3, 12), (12, 16, 40)), ground_color=(6, 8, 18), hills=[]),
    "capsule": dict(sky=((40, 46, 48), (60, 66, 68)), ground_color=(40, 44, 46), hills=[]),
    "mission_control": dict(sky=((18, 24, 38), (30, 38, 56)), ground_color=(24, 30, 44), hills=[]),
    "harbor": dict(sky=((150, 196, 226), (236, 232, 214)), sun=(0.78, 0.2, (252, 214, 120)), ground_color=(190, 170, 126),
                   hills=[dict(color=(104, 150, 190), base=.64, amp=.004, freq=1.0, seed=3, par=.02), dict(color=(80, 128, 172), base=.70, amp=.004, freq=1.2, seed=5, par=.04)]),
    "underwater": dict(sky=((8, 52, 104), (40, 150, 176)), ground_color=(158, 146, 106),
                       hills=[dict(color=(22, 92, 132), base=.70, amp=.03, freq=3.4, seed=4, par=.10)]),
    "submarine_interior": dict(sky=((26, 36, 40), (60, 74, 76)), ground_color=(72, 80, 82), hills=[]),
    "city_modern": dict(sky=((70, 80, 120), (244, 176, 124)), sun=(0.3, 0.5, (255, 214, 150)), ground_color=(92, 92, 100),
                        hills=[dict(color=(60, 66, 92), base=.62, amp=.04, freq=9.0, seed=2, par=.10), dict(color=(44, 48, 68), base=.70, amp=.05, freq=12.0, seed=5, par=.22)]),
    "volcanic": dict(sky=((38, 14, 16), (206, 92, 52)), ground_color=(62, 50, 46),
                     hills=[dict(color=(64, 40, 40), base=.64, amp=.05, freq=3.0, seed=3, par=.10), dict(color=(44, 30, 30), base=.72, amp=.045, freq=4.4, seed=1, par=.22)]),
    "ashen": dict(sky=((70, 68, 72), (152, 148, 146)), ground_color=(96, 92, 90),
                  hills=[dict(color=(104, 100, 102), base=.66, amp=.05, freq=3.0, seed=7, par=.10), dict(color=(80, 76, 78), base=.73, amp=.04, freq=4.2, seed=2, par=.22)]),
    "forest": dict(sky=((120, 168, 160), (214, 226, 196)), ground_color=(96, 120, 76),
                   hills=[dict(color=(84, 122, 100), base=.66, amp=.05, freq=3.0, seed=5, par=.10), dict(color=(62, 98, 78), base=.73, amp=.045, freq=4.2, seed=2, par=.22)]),
}
OBJECTS = {"castle", "column", "pedestal", "cloud", "tree", "tent", "pyramid", "tower", "torch", "ship",
           "submarine", "warship", "missile", "plane", "building", "hatch", "pipes", "gauge",
           "volcano", "ash_cloud", "wave", "fire", "smoke", "house", "explosion", "depth_charge", "torpedo", "crowd", "liner", "iceberg", "burning_town", "airship", "spacecraft", "planet", "capsule_interior", "mission_control", "reactor"}
EFFECTS = {"rain", "flash", "sparks", "dust", "shake", "ashfall", "embers", "bubbles", "sonar"}
CAMERAS = {"push_in": ([1.0, 1.10], None), "pull_out": ([1.12, 1.0], None), "pan_right": ([1.06, 1.06], (-.04, .04)), "pan_left": ([1.06, 1.06], (.04, -.04)), "static": ([1.0, 1.0], None)}


def _pick(v, allowed, default):
    v = str(v or "").lower().strip().replace(" ", "_")
    return v if v in allowed else default


def clean_visual(v, rnd):
    """Coerce whatever the LLM wrote into values the engine understands."""
    if not isinstance(v, dict):
        v = {}
    t = _pick(v.get("type"), {"stage", "map", "card"}, "stage")
    out = {"type": t}
    if t == "card":
        c = v.get("card") or v
        out["big"] = str(c.get("big") or c.get("title") or "")[:40]
        out["small"] = str(c.get("small") or c.get("subtitle") or "")[:80]
        out["bullets"] = [str(b)[:60] for b in (c.get("bullets") or [])][:4]
        out["dark"] = bool(c.get("dark", False))
        return out
    if t == "map":
        m = v.get("map") or v
        out["template"] = _pick(m.get("template"), {"invasion", "route", "expanding"}, "invasion")
        out["center"] = str(m.get("center") or m.get("center_label") or "EMPIRE")[:24].upper()
        out["city"] = str(m.get("city") or "")[:20]
        out["labels"] = [str(x)[:16].upper() for x in (m.get("attackers") or m.get("labels") or m.get("points") or [])][:4]
        out["title"] = str(m.get("title") or v.get("title") or "")[:40].upper()
        return out
    out["background"] = _pick(v.get("background"), BACKGROUNDS, rnd.choice(["city_day", "countryside", "palace"]))
    actors = []
    for a in (v.get("actors") or [])[:5]:
        if not isinstance(a, dict):
            continue
        actors.append({"role": _pick(a.get("role"), ROLES, "citizen"), "color": _pick(a.get("color"), COLORS, ""),
                       "pos": _pick(a.get("pos") or a.get("position"), POS, "center"), "action": _pick(a.get("action"), ACTIONS, "talk"),
                       "emotion": _pick(a.get("emotion"), EMOTIONS, "neutral"), "facing": _pick(a.get("facing"), {"left", "right"}, ""),
                       "scale": max(0.6, min(1.2, float(a.get("scale", 1) or 1)))})
    out["actors"] = actors
    out["objects"] = [o for o in (str(x).lower() for x in (v.get("objects") or [])) if o in OBJECTS][:4]
    out["effects"] = [e for e in (str(x).lower() for x in (v.get("effects") or [])) if e in EFFECTS][:3]
    out["title"] = str(v.get("title") or "")[:34].upper()
    out["camera"] = _pick(v.get("camera"), CAMERAS, rnd.choice(["push_in", "pan_right", "pull_out", "pan_left"]))
    return out


def actor_keys(a, dur, idx, n):
    x = POS[a["pos"]]
    center = .5
    facing = 1 if (a["facing"] == "right" or (not a["facing"] and x < center - .05)) else -1
    if a["pos"] == "center" and not a["facing"]:
        facing = 1
    em = a["emotion"]
    act = a["action"]
    K = lambda t, xx, pose, **kw: dict(t=t, x=xx, pose=pose, face=em, facing=facing, **kw)
    if act == "talk":
        rng = random.Random(idx * 31 + int(x * 100))
        per = 1.6
        gestures = {"angry": ["demand", "point", "point", "proud"], "worried": ["think", "shrug", "facepalm", "point"], "shock": ["scared", "shrug", "point"],
                    "sad": ["slump", "think", "shrug"], "smile": ["proud", "point", "shrug", "salute"]}.get(em, ["point", "proud", "shrug", "think", "armscross"])
        listening = ["armscross", "think", "stand", "shrug", "facepalm"] if n > 1 else gestures
        ks, cur, t = [], x, 0.0
        i = 0
        while t < dur + per:
            speaking = n == 1 or (i + idx) % 2 == 0          # two people trade the floor; the other one reacts
            pose = rng.choice(gestures) if speaking else rng.choice(listening)
            fc = facing
            if n > 1:
                fc = 1 if (x < center) else -1               # face the other person
            if i % 3 == 2 and speaking:                      # take a few steps while making the point
                nx = max(.1, min(.9, cur + rng.choice([-1, 1]) * .07))
                ks.append(dict(t=t, x=cur, pose="walk", face=em, facing=1 if nx > cur else -1))
                ks.append(dict(t=t + .7, x=nx, pose="walk", face=em, facing=1 if nx > cur else -1))
                cur, t = nx, t + .7
            else:
                ks.append(dict(t=t, x=cur, pose=pose, face=em, facing=fc))
                t += per
            i += 1
        return ks
    if act in ("enter_walk", "enter_run"):
        start = -0.12 if x < .5 else 1.12
        f0 = 1 if start < 0 else -1
        pose = "walk" if act == "enter_walk" else "run"
        k1 = min(dur * 0.45, 2.6)
        return [dict(t=0, x=start, pose=pose, face=em, facing=f0), dict(t=k1, x=x, pose=pose, face=em, facing=f0), dict(t=k1 + .5, x=x, pose="stand", face=em, facing=facing)]
    if act == "exit_run":
        end = 1.15 if x >= .5 else -0.15
        f1 = 1 if end > x else -1
        return [dict(t=0, x=x, pose="stand", face=em, facing=f1), dict(t=dur * .4, x=x, pose="stand", face=em, facing=f1), dict(t=dur * .4 + .6, x=end, pose="run", face=em, facing=f1)]
    if act in ("walk", "run"):
        end = x + (.2 if facing > 0 else -.2)
        return [K(0, x, act), K(dur, end, act)]
    if act == "fight":
        meet = .5 + (-.09 if x < center else .09)            # both fighters charge in and trade blows at the middle
        t1 = min(1.0 + .15 * idx, dur * .3)
        ks = [K(0, x, "sword_up"), K(t1, meet, "run"), K(t1 + .25, meet, "fight_swing")]
        t, j = t1 + .25, 0
        while t < dur + 1:
            t += 1.0
            j += 1
            back = meet + (-.03 if x < center else .03) * (1 if j % 2 else 0)
            ks.append(K(t, back, "sword_up" if j % 2 else "fight_swing"))
        return ks
    pose = act if act not in ("stand",) else "stand"
    return [K(0, x, pose), K(dur, x, pose)]


def build_stage(v, dur, rnd, seed=0):
    bg = BACKGROUNDS[v["background"]]
    scene = {k: bg[k] for k in ("sky", "ground_color", "hills") if k in bg}
    if "sun" in bg:
        scene["sun"] = bg["sun"]
    scene["dim"] = bg.get("dim", 0)
    scene["duration"] = dur
    scene["bg_name"] = v["background"]
    actors = []
    for i, a in enumerate(v["actors"]):
        props, tun = ROLES[a["role"]]
        col = COLORS.get(a["color"] or tun, COLORS[tun])
        keys = actor_keys(a, dur, i, len(v["actors"]))
        for k in keys:
            if k["pose"] == "fight_swing":
                k["pose"] = "swing"
        crown_fall = dur * 0.3 if ("crown" in props and a["action"] in ("slump", "scared")) else None
        actors.append(dict(id=f"a{i}", crown_fall=crown_fall, color=col, tunic=col if a["role"] not in ("scholar", "priest") else (238, 234, 224),
                           props=props, scale=a["scale"] * (0.98 if a["role"] in ("soldier", "knight") else 1.0),
                           hair=(random.Random(i + seed).choice([(70, 48, 30), (40, 30, 24), (150, 110, 60), (200, 200, 196)])), keys=keys))
    seen = set()
    for a_ in actors:                                                   # two characters never share the same shirt colour
        if a_["tunic"] in seen:
            for alt in (RED, BLUE, GREY, PURPLE, (70, 130, 80), (214, 120, 40)):
                if alt not in seen:
                    a_["tunic"] = alt
                    break
        seen.add(a_["tunic"])
    scene["actors"] = actors
    xs = [.12, .88, .3, .7]
    objs = []
    BIG = {"airship": .5, "reactor": .72, "spacecraft": .5, "planet": .84, "liner": .5, "iceberg": .76, "submarine": .56, "warship": .56, "plane": .3, "missile": .86, "pyramid": .5, "ship": .6, "volcano": .62, "wave": .6, "ash_cloud": .05, "explosion": .5, "fire": .5}
    flags = v.get("flags") or {}
    for j, o in enumerate(v["objects"]):
        objs.append(dict(type=o, x=BIG.get(o, [.16, .84, .5, .3][j % 4]) if o not in ("torch",) else xs[j % 4], y=.18 + .05 * j, r=.05, scale=1.0,
                         harbor=v["background"] == "harbor", dur=dur, modern=bool(flags.get("modern")), afloat=v["background"] == "underwater", launch=bool(flags.get("launch")), burning=bool(flags.get("burning")), venting=bool(flags.get("venting")), kind=flags.get("kind", "earth"),
                         t0=dur * (.3 + .12 * j) if o in ("depth_charge", "explosion", "torpedo") else .4,
                         color=(255, 255, 255, 160) if v["background"] not in ("storm", "night", "battlefield", "volcanic", "ashen", "space") else (92, 94, 104)))
    dc = [o for o in objs if o["type"] == "depth_charge"]
    if v.get("anchor") and v["background"] not in ("submarine_interior", "underwater"):          # the story's landmark looms far behind every outdoor scene
        ax = .9 if not actors or sum(a["keys"][0]["x"] for a in actors) / len(actors) < .55 else .1
        objs.insert(0, dict(type=v["anchor"], x=ax, y=.2, r=.05, scale=.5, dur=dur, modern=False, afloat=False, launch=False, t0=.4, color=(92, 94, 104)))
    if v["background"] == "underwater":
        objs.insert(0, dict(type="seascape", x=.5, dur=dur))
    if v["background"] == "submarine_interior":
        objs.insert(0, dict(type="interior", x=.5, dur=dur))
    for bgname, otype in (("capsule", "capsule_interior"), ("mission_control", "mission_control")):
        if v["background"] == bgname:
            objs.insert(0, dict(type=otype, x=.5, dur=dur))
    for o in objs:
        if o["type"] == "explosion":
            o["x"] = .78 if not actors or actors[0]["keys"][0]["x"] < .6 else .22     # blast beside the people, not on top of them
    big = [o for o in objs if o["type"] in ("submarine", "warship", "ship", "liner", "iceberg", "volcano", "wave", "pyramid")]
    if big and actors:                                                           # keep people out from in front of the main subject
        for i, a in enumerate(actors):
            side = .13 if i % 2 == 0 else .87
            if abs(a["keys"][0]["x"] - big[0]["x"]) < .3:
                for k in a["keys"]:
                    k["x"] = side
    if v["background"] == "submarine_interior" and not any(o["type"] == "pipes" for o in objs):
        objs.append(dict(type="pipes", x=.5))
    if not objs and v["background"] == "city_modern":
        objs = [dict(type="building", x=.12), dict(type="building", x=.88, scale=1.15)]
    if not objs and v["background"] in ("city_day", "palace"):
        objs = [dict(type="column", x=.1), dict(type="column", x=.9)]
    if not objs and v["background"] in ("countryside", "forest"):
        objs = [dict(type="tree", x=.1), dict(type="tree", x=.9, scale=.9)]
    for o_ in objs:                                                     # no two spaceflight shots look alike
        if o_["type"] == "spacecraft":
            o_["scale"] = rnd.uniform(.8, 1.25)
            o_["x"] = rnd.uniform(.4, .6)
            o_["tilt"] = rnd.uniform(-.5, .5)
        elif o_["type"] == "planet":
            o_["side"] = rnd.choice([-1, 1])
            o_["scale"] = rnd.uniform(.8, 1.5)
    scene["objects"] = objs
    fx = []
    for e in v["effects"]:
        if e == "rain":
            fx.append(dict(type="rain", t0=0, t1=dur))
        elif e == "flash":
            fx.append(dict(type="flash", t=min(dur * .4, 2.0)))
        elif e == "sparks":
            fx.append(dict(type="sparks", t=dur * .55, x=.5, y=.52))
        elif e == "dust":
            fx += [dict(type="dust", actor=a["id"]) for a in actors]
        elif e == "shake":
            scene["shake"] = [dict(t=(dc[0]["t0"] + 1.3) if dc else dur * .5, dur=.6, amp=14)]
        elif e == "bubbles":
            fx.append(dict(type="bubbles", t0=0, t1=dur))
        elif e == "sonar":
            fx.append(dict(type="sonar", t0=0, t1=dur))
    if sum(a["action"] == "fight" for a in v["actors"]) >= 2:      # a spark burst on every exchange of blows
        tt = min(1.0, dur * .3) + .5
        while tt < dur - .3:
            fx.append(dict(type="sparks", t=tt, x=.5, y=.40))
            tt += 1.0
    fx.append(dict(type="ambient", bg=v["background"]))
    scene["fx"] = fx
    if v.get("title"):
        scene["text"] = [dict(t=.35, end=min(dur - .2, 3.6), text=v["title"], y=.085,
                              color=(255, 255, 255) if v["background"] in ("storm", "night", "battlefield", "underwater", "submarine_interior", "city_modern", "volcanic", "ashen", "space", "capsule", "mission_control") else (27, 27, 32))]
    z, pan = CAMERAS[v["camera"]]
    scene["zoom"] = z
    _late_shots = True
    scene["focus"] = (.5 + (pan[0] if pan else 0), .6)
    if pan:
        scene["focus_to"] = (.5 + pan[1], .6)
    scene["blur"] = "fight" in [a["action"] for a in v["actors"]] or "run" in " ".join(a["action"] for a in v["actors"])
    from . import stick
    stick.make_shots(scene, rnd)
    ms = [o for o in objs if o["type"] == "missile" and o.get("launch")]
    if ms:                                                                # the camera tilts up with the rocket
        m = ms[0]
        scene["shots"] = [dict(t0=0, t1=dur, z0=1.35, z1=1.35, x0=m["x"], x1=m["x"], y0=.62, y1=.12)]
        scene["hits"] = [m["t0"]]
    return scene


def build_map(v, dur):
    t = v["template"]
    lands = [dict(pts=[(.10, .06), (.92, .05), (.94, .22), (.72, .26), (.55, .23), (.36, .22), (.2, .26), (.08, .18)], color=(224, 214, 178)),
             dict(pts=[(.78, .30), (.96, .27), (.96, .72), (.80, .68), (.73, .50)], color=(224, 214, 178)),
             dict(pts=[(.12, .45), (.22, .30), (.40, .28), (.58, .34), (.70, .47), (.66, .65), (.52, .73), (.34, .71), (.20, .62)], color=(236, 204, 132),
                  label=v["center"], label_at=(.40, .36), label_color=(120, 70, 30))]
    city = dict(name=v["city"], x=.44, y=.56) if v["city"] else None
    labels = v["labels"] or ["", "", ""]
    scene = dict(kind="map", duration=dur, land=lands, cities=[city] if city else [], arrows=[],
                 text=[dict(t=.3, end=min(dur - .2, 3.6), text=v["title"], y=.03, color=(196, 57, 43))] if v["title"] else [])
    if t == "invasion":
        routes = [[(.70, .22), (.58, .38), (.47, .53)], [(.88, .52), (.70, .55), (.50, .57)], [(.16, .22), (.24, .36), (.41, .52)], [(.5, .95), (.47, .75), (.45, .6)]]
        lab_at = [(.69, .18), (.80, .44), (.10, .18), (.52, .88)]
        for i, lab in enumerate(labels[:4]):
            t0 = .5 + i * dur * .14
            scene["arrows"].append(dict(pts=routes[i], t0=t0, t1=min(t0 + dur * .35, dur - .3), label=lab, label_at=lab_at[i]))
    elif t == "route":
        pts = [(.14, .55), (.30, .40), (.50, .48), (.68, .36), (.86, .50)]
        scene["land"] = [dict(pts=[(.06, .30), (.30, .20), (.60, .24), (.92, .28), (.95, .70), (.66, .78), (.34, .74), (.08, .66)], color=(236, 222, 184))]
        scene["cities"] = [dict(name=n, x=p[0], y=p[1]) for n, p in zip(labels, pts)]
        scene["arrows"] = [dict(pts=pts[:max(2, len(labels))], t0=.6, t1=dur - .4, color=(196, 57, 43))]
    else:  # expanding
        ce = (.5, .55)
        scene["land"] = [dict(pts=[(.06, .16), (.94, .14), (.96, .86), (.05, .88)], color=(236, 222, 184))]
        scene["cities"] = [dict(name=v["city"] or v["center"], x=ce[0], y=ce[1])]
        outs = [((.20, .32), (.08, .24)), ((.80, .32), (.72, .24)), ((.80, .72), (.72, .80)), ((.20, .72), (.08, .80))]
        for i, (end, la) in enumerate(outs[:max(2, len(labels))]):
            t0 = .5 + i * dur * .12
            scene["arrows"].append(dict(pts=[ce, end], t0=t0, t1=min(t0 + dur * .4, dur - .3), color=(196, 57, 43),
                                        label=labels[i] if i < len(labels) else "", label_at=la))
    return scene


def build_card(v, dur):
    return dict(kind="card", duration=dur, card=dict(big=v["big"], small=v["small"], bullets=v["bullets"], dark=v["dark"]))
