"""Blender (bpy) renderer: builds a real 3D scene from the same scene dict the 2D engine uses.

Characters are outlined capsule rigs driven by the 2D engine's pose channels (Actor.key_state / extras), so
every acting beat, mocap clip and lip-sync cue carries over unchanged. Cycles on CPU, low samples + denoise.
Only some backgrounds are supported (see SUPPORTED); everything else stays on the 2D engine.
"""
import math
import random

import bpy
import numpy as np
from mathutils import Vector

from . import stick as K

SUPPORTED = {"capsule", "space"}
WORLD_W = 4.44                       # world units across a full-width frame at the base camera
CAM_DIST = 6.17
LB = dict(torso=0.30, upper=0.17, fore=0.17, thigh=0.22, shin=0.22, head=0.132, neck=0.035)


def _rgb(c, k=1.0):
    f = lambda v: ((v / 255.0) ** 2.2) * k
    return (f(c[0]), f(c[1]), f(c[2]), 1.0)


_MATS = {}


def mat(name, color, rough=0.55, emit=0.0, metal=0.0):
    key = (name, tuple(color), rough, emit, metal)
    if key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = _rgb(color)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = _rgb(color)
        b.inputs["Emission Strength"].default_value = emit
    _MATS[key] = m
    return m


def _link(o, collection=None):
    (collection or bpy.context.scene.collection).objects.link(o)
    return o


def _mesh_obj(name, bm_fn, material, smooth=True):
    me = bpy.data.meshes.new(name)
    bm_fn(me)
    o = bpy.data.objects.new(name, me)
    _link(o)
    o.data.materials.append(material)
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    return o


def _sphere(name, material, seg=24):
    import bmesh

    def f(me):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1.0)
        bm.to_mesh(me)
        bm.free()
    return _mesh_obj(name, f, material)


def _cyl(name, material, seg=20):
    import bmesh

    def f(me):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=1.0, radius2=1.0, depth=1.0)
        bm.to_mesh(me)
        bm.free()
    return _mesh_obj(name, f, material)


def _box(name, material):
    import bmesh

    def f(me):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(me)
        bm.free()
    return _mesh_obj(name, f, material, smooth=False)


def _place_capsule(o, p, q, r):
    """Cylinder of radius r from p to q (world vectors). Cylinder mesh is unit radius, unit depth along Z."""
    p, q = Vector(p), Vector(q)
    d = q - p
    L = max(d.length, 1e-5)
    o.location = (p + q) / 2
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.scale = (r, r, L)


def _place_ball(o, p, r):
    o.location = p
    o.scale = (r, r, r)


class Rig:
    """One character: capsules for limbs, spheres for joints/hands/head, a few face parts."""

    def __init__(self, actor):
        self.a = actor
        s = actor.s
        props = s.get("props", [])
        self.props = props
        skin = s.get("skin", (247, 222, 190))
        tunic = s.get("tunic")
        shirt = tunic if tunic else (52, 54, 66)
        pants = s.get("pants") or (shirt if "spacehelmet" in props else tuple(int(c * .5) for c in shirt) if tunic else (36, 38, 50))
        self.m_skin = mat("skin", skin, 0.5)
        self.m_shirt = mat("shirt" + str(shirt), shirt, 0.7)
        self.m_pants = mat("pants" + str(pants), pants, 0.7)
        self.m_shoe = mat("shoe", (66, 46, 36), 0.6)
        self.m_white = mat("eyew", (255, 255, 255), 0.3)
        self.m_ink = mat("ink", (20, 20, 24), 0.4)
        self.limbs = {}
        for n in ("torso", "ua", "fa", "ub", "fb", "tl", "sl", "tm", "sm"):
            col = {"torso": self.m_shirt, "ua": self.m_shirt, "fa": self.m_shirt, "ub": self.m_shirt, "fb": self.m_shirt}.get(n, self.m_pants)
            self.limbs[n] = _cyl(n, col)
        self.balls = {}
        for n in ("sh", "hip", "ha", "hb", "ea", "eb", "ka", "kb", "head", "foota", "footb", "nose"):
            m = self.m_skin if n in ("ha", "hb", "head") else self.m_pants if n in ("hip", "ka", "kb") else self.m_shirt if n in ("sh", "ea", "eb") else self.m_shoe
            self.balls[n] = _sphere(n, m)
        self.eyes = [_sphere("eye%d" % i, self.m_white, 16) for i in range(2)]
        self.pupils = [_sphere("pup%d" % i, self.m_ink, 12) for i in range(2)]
        self.brows = [_box("brow%d" % i, self.m_ink) for i in range(2)]
        self.mouth = _sphere("mouth", mat("mouth", (60, 20, 24), 0.5), 16)
        self.glass = None
        self.hair = None
        if "spacehelmet" in props:
            gm = bpy.data.materials.new("glass")
            gm.use_nodes = True
            bsdf = gm.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (0.8, 0.9, 1, 1)
            bsdf.inputs["Transmission Weight"].default_value = 0.92
            bsdf.inputs["Roughness"].default_value = 0.03
            bsdf.inputs["IOR"].default_value = 1.05
            self.glass = _sphere("glass", gm, 28)
            self.collar = _cyl("collar", mat("collar", (210, 214, 220), 0.35, metal=0.4))
        if "hair" in props:
            self.hair = _sphere("hair", mat("hair", s.get("hair", (70, 48, 30)), 0.8))

    def pose(self, t, scene, world_x0):
        a = self.a
        S = a.s.get("scale", 1.0)
        pose, xr, f, face = a.key_state(t)
        a._look_dx = 0.40 * f
        ex = a.extras(t, scene)
        seg = K.seg
        zlay = a.s.get("z", 0) * 0.0 + a.s.get("depth", 0.0)

        def W(p):                                                 # local (x right, y DOWN, ground at 1.38) -> world (x, y, z up)
            return (world_x0 + p[0] * S, zlay, (1.38 - p[1]) * S)
        add = lambda p, q: (p[0] + q[0], p[1] + q[1])
        l1 = seg((0, 0), pose["l1"], LB["thigh"], f); l2 = seg(l1, pose["l1"] + pose["l2"], LB["shin"], f)
        m1 = seg((0, 0), pose["m1"], LB["thigh"], f); m2 = seg(m1, pose["m1"] + pose["m2"], LB["shin"], f)
        drop = max(l2[1], m2[1])
        hip = (0.0, 1.38 - drop - pose.get("lift", 0))
        tr = math.radians(pose["torso"])
        up = (math.sin(tr) * f, -math.cos(tr))
        tl = LB["torso"]
        neck = add(hip, (up[0] * tl, up[1] * tl))
        sh = add(hip, (up[0] * tl * .9, up[1] * tl * .9))
        hr = math.radians(pose["torso"] + pose["head"])
        hup = (math.sin(hr) * f, -math.cos(hr))
        rad = LB["head"]
        head = add(neck, (hup[0] * (LB["neck"] + rad * .92), hup[1] * (LB["neck"] + rad * .92)))
        a1 = add(sh, seg((0, 0), pose["a1"], LB["upper"], f)); a2 = add(a1, seg((0, 0), pose["a1"] + pose["a2"], LB["fore"], f))
        b1 = add(sh, seg((0, 0), pose["b1"], LB["upper"], f)); b2 = add(b1, seg((0, 0), pose["b1"] + pose["b2"], LB["fore"], f))
        L, B = self.limbs, self.balls
        lw = 0.034 * S
        P = {k: W(v) for k, v in dict(hip=hip, neck=neck, sh=sh, a1=a1, a2=a2, b1=b1, b2=b2, l1=add(hip, l1), l2=add(hip, l2), m1=add(hip, m1), m2=add(hip, m2)).items()}
        _place_capsule(L["torso"], P["hip"], P["sh"], 0.062 * S)
        _place_capsule(L["ua"], P["sh"], P["a1"], lw); _place_capsule(L["fa"], P["a1"], P["a2"], lw * .92)
        _place_capsule(L["ub"], P["sh"], P["b1"], lw); _place_capsule(L["fb"], P["b1"], P["b2"], lw * .92)
        _place_capsule(L["tl"], P["hip"], P["l1"], 0.042 * S); _place_capsule(L["sl"], P["l1"], P["l2"], 0.036 * S)
        _place_capsule(L["tm"], P["hip"], P["m1"], 0.042 * S); _place_capsule(L["sm"], P["m1"], P["m2"], 0.036 * S)
        _place_ball(B["sh"], P["sh"], 0.068 * S); _place_ball(B["hip"], P["hip"], 0.06 * S)
        _place_ball(B["ea"], P["a1"], lw * 1.05); _place_ball(B["eb"], P["b1"], lw * 1.05)
        _place_ball(B["ka"], P["l1"], 0.042 * S); _place_ball(B["kb"], P["m1"], 0.042 * S)
        _place_ball(B["ha"], P["a2"], lw * 1.35); _place_ball(B["hb"], P["b2"], lw * 1.35)
        for n, p in (("foota", P["l2"]), ("footb", P["m2"])):
            B[n].location = (p[0] + f * 0.035 * S, p[1], p[2] + 0.012 * S)
            B[n].scale = (0.075 * S, 0.04 * S, 0.034 * S)
        hp = Vector(W(head))
        _place_ball(B["head"], hp, rad * S)
        B["nose"].scale = (0, 0, 0)
        # face: turn the head 3/4 toward the camera (-Y), looking the way the character faces
        fwd = Vector((f * 0.55, -0.83, 0)).normalized()
        ctr = hp + fwd * rad * S * 0.80
        sx = Vector((0.83, 0.55 * f, 0)).normalized()               # across the face, perpendicular to fwd in the XY plane
        er = rad * S * 0.30
        blink = ex.get("blink", 0.0)
        look = ex.get("look", (0.4 * f, 0.0))
        mouth = ex.get("mouth", 0.0)
        for i, sgn in enumerate((-1, 1)):
            ep = hp + fwd * rad * S * 0.74 + sx * sgn * rad * S * 0.36 + Vector((0, 0, rad * S * 0.10))
            self.eyes[i].location = ep
            self.eyes[i].scale = (er, er * 0.8, er * (1 - 0.88 * blink))
            self.eyes[i].rotation_euler = (0, 0, math.atan2(fwd.y, fwd.x) - math.pi / 2)
            pp = ep + fwd * er * 0.55 + sx * er * 0.25 * math.tanh(look[0] * f) + Vector((0, 0, -er * 0.05 + er * 0.2 * -look[1]))
            self.pupils[i].location = pp
            pr = er * (0.62 if face != "shock" else 0.42)
            self.pupils[i].scale = (pr, pr * .6, pr * (1 - 0.88 * blink))
            # brows
            tilt = {"angry": -0.5, "worried": 0.5, "sad": 0.5, "shock": 0.0, "smile": 0.0}.get(face, 0.0) * sgn
            lift = 0.0 + (0.5 * ex.get("brow", 0.0)) + (0.5 if face == "shock" else 0.0)
            bp = ep + Vector((0, 0, er * (1.35 + lift)))
            self.brows[i].location = bp + fwd * er * 0.2
            self.brows[i].scale = (er * 1.7, er * 0.28, er * 0.28)
            self.brows[i].rotation_euler = (0, tilt * 0.6 * -f, math.atan2(fwd.y, fwd.x) - math.pi / 2)
        mp = hp + fwd * rad * S * 0.86 + Vector((0, 0, -rad * S * 0.42))
        self.mouth.location = mp
        base = {"smile": 0.35, "sad": 0.1, "shock": 0.7, "angry": 0.12, "worried": 0.2}.get(face, 0.12)
        self.mouth.scale = (rad * S * (0.30 if face != "smile" else 0.38), rad * S * 0.1, rad * S * (0.04 + 0.22 * max(mouth, base * (1 if face == "shock" else .35))))
        self.mouth.rotation_euler = (0, 0, math.atan2(fwd.y, fwd.x) - math.pi / 2)
        if self.glass:
            _place_ball(self.glass, hp, rad * S * 1.28)
            _place_capsule(self.collar, W(neck), (W(neck)[0], W(neck)[1], W(neck)[2] + 0.02 * S), rad * S * 0.95)
        if self.hair:
            _place_ball(self.hair, hp + Vector((0, 0.01, rad * S * 0.22)), rad * S * 1.05)
            self.hair.scale = (rad * S * 1.04, rad * S * 1.04, rad * S * 0.85)
        return xr


# --------------------------------------------------------------------------------------- sets
def _light_setup(kind):
    for o in list(bpy.data.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o)
    sun = bpy.data.lights.new("key", "AREA")
    sun.energy = 900 if kind == "capsule" else 1400
    sun.size = 3.0
    ko = bpy.data.objects.new("key", sun)
    _link(ko)
    ko.location = (-2.2, -3.0, 3.4)
    ko.rotation_euler = (math.radians(60), 0, math.radians(-35))
    fill = bpy.data.lights.new("fill", "AREA")
    fill.energy = 250
    fill.size = 4
    fo = bpy.data.objects.new("fill", fill)
    _link(fo)
    fo.location = (3, -3, 1.5)
    fo.rotation_euler = (math.radians(75), 0, math.radians(40))
    rim = bpy.data.lights.new("rim", "AREA")
    rim.energy = 800
    rim.size = 2
    rim.color = (0.7, 0.85, 1.0)
    ro = bpy.data.objects.new("rim", rim)
    _link(ro)
    ro.location = (1.5, 2.5, 2.5)
    ro.rotation_euler = (math.radians(-110), 0, math.radians(160))
    return ko


class CapsuleSet:
    """Cramped command module: padded wall, two round windows onto Earth, dense blinking panel, red alarm light."""

    def __init__(self):
        wall = mat("wall", (82, 90, 94), 0.85)
        dark = mat("panel", (40, 44, 48), 0.6)
        floor = mat("floor", (44, 48, 50), 0.8)
        w = _box("wall", wall); w.location = (0, 1.6, 1.4); w.scale = (9, 0.1, 3.4)
        f = _box("floor", floor); f.location = (0, 0, -0.05); f.scale = (9, 6, 0.1)
        ce = _box("ceil", wall); ce.location = (0, 0.6, 3.2); ce.scale = (9, 4, 0.1)
        for sx in (-1, 1):
            sw = _box("sidewall", wall); sw.location = (sx * 4.4, 0.6, 1.4); sw.scale = (0.1, 4, 3.4)
        for i in range(1, 9):                                          # padded seams
            s = _box("seam", dark); s.location = (-4.4 + i, 1.53, 1.4); s.scale = (0.04, 0.05, 3.2)
        pn = _box("panel", dark); pn.location = (0, 1.35, 0.42); pn.scale = (8.8, 0.35, 0.84)
        rng = random.Random(7)
        self.buttons = []
        for r in range(3):
            for c in range(30):
                col = ((120, 230, 130), (255, 190, 60), (255, 80, 70))[(c + r) % 3]
                b = _sphere("btn", mat("b%d" % ((c + r) % 3), col, 0.3, emit=0.0), 8)
                b.location = (-4.0 + c * 0.28, 1.17, 0.72 - r * 0.18)
                b.scale = (0.035, 0.02, 0.035)
                self.buttons.append((b, rng.random() * 3 + 1, c + r, r))
        self.emat = [mat("btnE%d" % i, col, 0.3, emit=6.0) for i, col in enumerate(((120, 230, 130), (255, 190, 60), (255, 80, 70)))]
        self.off = [mat("btnO%d" % i, tuple(int(v * .18) for v in col), 0.5) for i, col in enumerate(((120, 230, 130), (255, 190, 60), (255, 80, 70)))]
        import bmesh
        self.earth_mat = mat("earthwin", (44, 110, 200), 0.4, emit=2.0)
        for wx in (-2.4, 2.4):
            ring = _cyl("ring", mat("rim", (150, 154, 158), 0.3, metal=0.8))
            ring.rotation_euler = (math.radians(90), 0, 0)
            ring.location = (wx, 1.5, 1.9); ring.scale = (0.62, 0.62, 0.12)
            gl = _cyl("win", mat("space", (6, 8, 22), 0.9, emit=0.4))
            gl.rotation_euler = (math.radians(90), 0, 0)
            gl.location = (wx, 1.44, 1.9); gl.scale = (0.5, 0.5, 0.12)
            for k in range(14):
                st = _sphere("star", mat("star", (255, 250, 230), 0.5, emit=8.0), 6)
                st.location = (wx + (rng.random() - .5) * .8, 1.36, 1.9 + (rng.random() - .5) * .8)
                st.scale = (0.012, 0.01, 0.012)
        ea = _sphere("earthw", self.earth_mat, 24)
        ea.location = (2.4 + 0.18, 1.5, 1.72); ea.scale = (0.3, 0.1, 0.3)
        self.alarm = _sphere("alarm", mat("alarm", (255, 60, 40), 0.3, emit=10.0), 12)
        self.alarm.location = (0, 1.38, 2.7); self.alarm.scale = (0.1, 0.06, 0.1)
        al = bpy.data.lights.new("alarmL", "POINT"); al.color = (1, 0.2, 0.1)
        self.alarm_light = bpy.data.objects.new("alarmL", al); _link(self.alarm_light)
        self.alarm_light.location = (0, 1.0, 2.6)
        self.world_color = (0.01, 0.012, 0.02)

    def at(self, t):
        for b, speed, ph, r in self.buttons:
            on = math.sin(t * speed + ph) > (-0.1 if r < 2 else 0.5)
            b.data.materials.clear()
            b.data.materials.append((self.emat if on else self.off)[(ph) % 3])
        al = 0.5 + 0.5 * math.sin(t * 4)
        self.alarm.scale = (0.1, 0.06, 0.1)
        self.alarm.data.materials.clear()
        self.alarm.data.materials.append(mat("alarm", (255, 60, 40), 0.3, emit=2 + 12 * al))
        self.alarm_light.data.energy = 60 * al + 5


def setup_render(W, H, samples=24, denoise=True):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = samples
    cy.use_denoising = denoise
    cy.max_bounces = 4
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 4
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    sc.render.use_freestyle = True
    sc.render.line_thickness_mode = "ABSOLUTE"
    sc.render.line_thickness = max(1.4, H / 360.0)
    vl = sc.view_layers[0]
    vl.use_freestyle = True
    ls = vl.freestyle_settings.linesets
    if not ls:
        vl.freestyle_settings.linesets.new("ink")
    lset = vl.freestyle_settings.linesets[0]
    if lset.linestyle is None:
        lset.linestyle = bpy.data.linestyles.new("ink")
    lset.select_silhouette = True
    lset.select_border = True
    lset.select_crease = False
    lset.linestyle.color = (0.06, 0.06, 0.08)
    lset.linestyle.thickness = max(1.4, H / 360.0)
    return sc


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _MATS.clear()


def build(scene, W, H, samples=24):
    kind = "capsule" if scene.get("background") == "capsule" else "space"
    reset()
    sc = setup_render(W, H, samples)
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = 35
    cam_d.sensor_width = 36
    cam = bpy.data.objects.new("cam", cam_d)
    _link(cam)
    sc.camera = cam
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    sc.world = world
    _light_setup(kind)
    stage = CapsuleSet()
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*stage.world_color, 1)
    rigs = [Rig(a) for a in scene["_actors"]]
    return dict(scene=sc, cam=cam, stage=stage, rigs=rigs)


def frame(ctx, scene, t, path):
    sc = ctx["scene"]
    for r in ctx["rigs"]:
        xr = r.a.key_state(t)[1]
        r.pose(t, scene, (xr - 0.5) * WORLD_W)
    ctx["stage"].at(t)
    z, cx, cy, sx, sy, punch = K._camera(scene, t, K.smooth(t / max(scene["duration"], 1e-6)), scene.get("focus", (0.5, 0.55)), scene.get("focus_to"))
    cam = ctx["cam"]
    dist = CAM_DIST / z
    wx = (cx - 0.5) * WORLD_W
    wz = 0.85 + (0.5 - cy) * 0.0
    gyr = scene.get("ground", 0.80)
    # keep the ground line at its 2D screen position at z=1; shift the lens vertically for the shot's y centre
    cam.location = (wx, -dist, 0.62 + 0.3 / z)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
