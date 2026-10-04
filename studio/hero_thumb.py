"""Hero-object thumbnails drawn in code: a clean shaded 3D-looking pyramid on a dark background, a red graphic overlay (ramp), tiny workers hauling a block, a hand-drawn arrow and a huge glowing number.
Same recipe as the high-CTR science/history channels: one recognisable subject, one bold colour accent, one specific claim, no faces."""
import math
import pathlib
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = str(pathlib.Path(__file__).resolve().parent.parent / "assets/fonts/BigShoulders-Bold.ttf")


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _shaded_face(img, pts, base, light, seed, courses=14, grain=0.10):
    """Fill a triangle with sandstone: vertical gradient, horizontal block courses and per-block brightness noise."""
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    apex, bl, br = pts
    rnd = random.Random(seed)
    face = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(face)
    for c in range(courses):
        t0, t1 = c / courses, (c + 1) / courses
        for (ta, tb) in ((t0, t1),):
            L0, R0 = _lerp(apex, bl, ta), _lerp(apex, br, ta)
            L1, R1 = _lerp(apex, bl, tb), _lerp(apex, br, tb)
            nb = max(1, int(2 + ta * 22))
            for b in range(nb):
                u0, u1 = b / nb, (b + 1) / nb
                q = [_lerp(L0, R0, u0), _lerp(L0, R0, u1), _lerp(L1, R1, u1), _lerp(L1, R1, u0)]
                k = 1.0 + (rnd.random() - .5) * grain * 2 - (ta * .06)
                col = tuple(max(0, min(255, int(v * light * k))) for v in base)
                d.polygon(q, fill=col, outline=tuple(int(v * light * .62) for v in base))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(face, (0, 0), mask)
    img.alpha_composite(out)


def pyramid_thumbnail(lines, out_path, accent=(255, 236, 40), seed=3, ramp=True, arrow=True, workers=True):
    S = 2
    W, H = 1280 * S, 720 * S
    # background: near-black with a cool glow behind the pyramid and a pale sand floor
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow = np.exp(-(((xx - W * 0.66) / (W * 0.42)) ** 2 + ((yy - H * 0.42) / (H * 0.55)) ** 2))
    bgc = np.zeros((H, W, 3), np.float32)
    bgc += np.array([4, 6, 12], np.float32)
    bgc += glow[..., None] * np.array([28, 36, 60], np.float32)
    img = Image.fromarray(np.clip(bgc, 0, 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(img)
    # floor: pale sand with soft gradient, wide ellipse
    floor_y = int(H * 0.70)
    fl = np.zeros((H, W, 4), np.uint8)
    for y in range(floor_y, H):
        t = (y - floor_y) / (H - floor_y)
        c = (int(222 - 40 * t), int(214 - 44 * t), int(196 - 52 * t))
        fl[y, :, :3] = c; fl[y, :, 3] = 255
    horizon = Image.fromarray(fl, "RGBA")
    img.alpha_composite(horizon)
    # pyramid geometry (three visible base corners + apex)
    apex = (W * 0.705, H * 0.085)
    bl = (W * 0.425, H * 0.800)
    bf = (W * 0.700, H * 0.930)
    br = (W * 0.990, H * 0.785)
    # ground shadow
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon([bf, br, (W * 1.04, H * 0.83), (W * 0.80, H * 0.99)], fill=(0, 0, 0, 90))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(24)))
    gold, shade = (216, 168, 92), (216, 168, 92)
    _shaded_face(img, [apex, bf, br], shade, 0.52, seed + 1)          # right face in shadow
    _shaded_face(img, [apex, bl, bf], gold, 1.08, seed)               # left face in the light
    d = ImageDraw.Draw(img)
    d.line([apex, bf], fill=(255, 226, 150, 255), width=5 * S)         # bright edge
    d.line([apex, bl], fill=(40, 28, 16, 255), width=3 * S)
    d.line([apex, br], fill=(30, 20, 12, 255), width=3 * S)
    # one long straight red ramp from the ground in front all the way up to near the top, like a slide leaning on the pyramid
    bx = by = bw = 0
    if ramp:
        top = _lerp(apex, _lerp(bl, bf, 0.40), 0.10)                         # the ramp ends just under the tip
        foot_l = (W * 0.405, H * 0.955)
        foot_r = (W * 0.585, H * 0.985)
        rp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(rp)
        side = [foot_r, (foot_r[0] + 20 * S, foot_r[1] - 16 * S), (top[0] + 8 * S, top[1] + 4 * S), top]
        rd.polygon(side, fill=(150, 14, 12, 255), outline=(30, 4, 4, 255))
        rd.polygon([foot_l, foot_r, top], fill=(236, 38, 32, 255), outline=(30, 4, 4, 255))
        rd.line([foot_l, top], fill=(255, 168, 150, 255), width=5 * S)
        img.alpha_composite(rp)
        d = ImageDraw.Draw(img)
        # a big clean stone block sitting low on the ramp so the slope is obvious
        q = 0.22
        cx_, cy_ = _lerp(_lerp(foot_l, top, q), _lerp(foot_r, top, q), 0.5)
        bx, by, bw = cx_, cy_, 84 * S
        d.polygon([(bx - bw / 2, by + 4 * S), (bx + bw / 2, by + 4 * S), (bx + bw / 2, by - bw * .5), (bx - bw / 2, by - bw * .5)], fill=(246, 246, 248), outline=(20, 20, 24), width=2 * S)
        d.polygon([(bx - bw / 2, by - bw * .5), (bx + bw / 2, by - bw * .5), (bx + bw / 2 + 18 * S, by - bw * .7), (bx - bw / 2 + 18 * S, by - bw * .7)], fill=(255, 255, 255), outline=(20, 20, 24), width=2 * S)
        d.polygon([(bx + bw / 2, by + 4 * S), (bx + bw / 2 + 18 * S, by - bw * .18), (bx + bw / 2 + 18 * S, by - bw * .7), (bx + bw / 2, by - bw * .5)], fill=(198, 198, 204), outline=(20, 20, 24), width=2 * S)
    # text: huge glowing number with an outline
    lines = lines.split("|")
    size = 430
    f = ImageFont.truetype(FONT, size * S // 2)
    while size > 150:
        f = ImageFont.truetype(FONT, size * S // 2)
        if max(d.textlength(l, font=f) for l in lines) <= W * 0.46 and (size * S // 2) * .88 * len(lines) <= H * 0.78:
            break
        size -= 10
    y = H * 0.07
    txt = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt)
    yy_ = y
    for i, l in enumerate(lines):
        col = accent if i == 0 else (255, 255, 255)
        td.text((40 * S, yy_), l, font=f, fill=col, stroke_width=10 * S, stroke_fill=(0, 0, 0))
        yy_ += (size * S // 2) * .88
    glow_l = txt.filter(ImageFilter.GaussianBlur(26 * S // 2))
    glow_c = Image.new("RGBA", (W, H), accent + (0,)); glow_c.putalpha(glow_l.split()[3].point(lambda v: int(v * .55)))
    img.alpha_composite(glow_c)
    img.alpha_composite(txt)
    d = ImageDraw.Draw(img)
    # arrow: a big white curved arrow pointing at the ramp
    if arrow:
        tipx, tipy = _lerp(_lerp(bl, bf, 0.38), apex, 0.32)
        sx, sy = W * 0.05, H * 0.86
        pts = []
        for i in range(0, 21):
            t = i / 20
            x = sx + (bx - bw * .5 - sx) * t
            y = sy + (by - bw * .3 - sy) * t + math.sin(t * math.pi) * (H * 0.10)
            pts.append((x, y))
        d.line(pts, fill=(10, 10, 12), width=24 * S, joint="curve")
        d.line(pts, fill=(255, 255, 255), width=14 * S, joint="curve")
        ex, ey = pts[-1]
        ang = math.atan2(pts[-1][1] - pts[-3][1], pts[-1][0] - pts[-3][0])
        hd = [(ex + math.cos(ang) * 40 * S, ey + math.sin(ang) * 40 * S), (ex + math.cos(ang + 2.4) * 44 * S, ey + math.sin(ang + 2.4) * 44 * S), (ex + math.cos(ang - 2.4) * 44 * S, ey + math.sin(ang - 2.4) * 44 * S)]
        d.polygon(hd, fill=(10, 10, 12)); 
        hd2 = [(ex + math.cos(ang) * 30 * S, ey + math.sin(ang) * 30 * S), (ex + math.cos(ang + 2.4) * 32 * S, ey + math.sin(ang + 2.4) * 32 * S), (ex + math.cos(ang - 2.4) * 32 * S, ey + math.sin(ang - 2.4) * 32 * S)]
        d.polygon(hd2, fill=(255, 255, 255))
    img.convert("RGB").resize((1280, 720), Image.LANCZOS).save(out_path, "JPEG", quality=95)
    return out_path
