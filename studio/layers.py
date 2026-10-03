"""Layered 2.5D scene animation: a wide background plus cut-out characters that move independently.

Camera: slow dolly with parallax (the background moves less than the character). Character: breathing, weight shift, speech bounce keyed to the narration envelope,
an entrance pop that overshoots and settles, plus a soft light bloom and drifting dust. Everything is a pure function of time, so a scene renders the same every time."""
import math

import cv2
import numpy as np
from PIL import Image

W, H = 1280, 720


def _spring(t, wn=16.0, z=0.45):
    """Underdamped spring step response, 0 -> 1 with overshoot."""
    if t <= 0:
        return 0.0
    wd = wn * math.sqrt(1 - z * z)
    return 1 - math.exp(-z * wn * t) * (math.cos(wd * t) + z * wn / wd * math.sin(wd * t))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Scene:
    def __init__(self, bg_path, chars, dur, seed=0, speech=None):
        """chars: [(png_path, x_frac, height_frac, flip)]; speech: function t -> 0..1 mouth/voice energy (or None)."""
        self.bg = np.asarray(Image.open(bg_path).convert("RGB").resize((int(W * 1.18), int(H * 1.18)), Image.LANCZOS))
        self.chars = []
        for path, xf, hf, flip in chars:
            im = Image.open(path).convert("RGBA")
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            k = H * hf / im.height
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
            self.chars.append((np.asarray(im), xf))
        self.dur, self.seed, self.speech = dur, seed, speech
        rng = np.random.default_rng(seed)
        self.dust = [(rng.random(), rng.random(), 0.4 + rng.random() * 1.2, 1 + rng.random() * 2.5) for _ in range(36)]

    def frame(self, t):
        u = ease(t / max(self.dur, 1e-6))
        # background layer: slow push-in with parallax drift
        zb = 1.0 + 0.07 * u
        bh, bw = self.bg.shape[:2]
        sx = W / (bw / 1.18) * 1.0
        vw, vh = W / zb * (bw / (W * 1.18)) * 1.18 / 1.18, H / zb
        cx = bw / 2 + (u - 0.5) * bw * 0.03
        cy = bh / 2
        x0, y0 = cx - W / zb / 1.0 * (bw / (W * 1.18)) / 2 * 1.18 / 1.18, cy - H / zb * (bw / (W * 1.18)) / 2 * 1.18 / 1.18
        sc = zb * W * 1.0 / (W / (bw / (W * 1.18)) * 1.0) if False else None
        M = cv2.getRotationMatrix2D((cx, cy), 0, 1.0)
        scale = zb * (W / bw) * 1.18 * 0.5 + 0
        # direct, simple crop-and-scale: window of size (W,H)/zb' inside the big background
        win_w = bw / 1.18 / zb
        win_h = bh / 1.18 / zb
        x0 = max(0, min(bw - win_w, bw / 2 - win_w / 2 + (u - 0.5) * bw * 0.035))
        y0 = max(0, min(bh - win_h, bh / 2 - win_h / 2))
        Mx = np.float32([[W / win_w, 0, -x0 * W / win_w], [0, H / win_h, -y0 * H / win_h]])
        out = cv2.warpAffine(self.bg, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        # soft warm light bloom from the top corner, slowly breathing
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        glow = np.exp(-(((xx - W * 0.85) / (W * 0.5)) ** 2 + ((yy - H * 0.05) / (H * 0.6)) ** 2)) * (0.16 + 0.03 * math.sin(t * 0.7))
        out += glow[..., None] * np.array([255, 214, 140], np.float32)
        # characters
        for ci, (arr, xf) in enumerate(self.chars):
            h0, w0 = arr.shape[:2]
            pop = _spring(t - 0.05 * ci, 15, 0.42)                              # entrance pop with overshoot
            ent = 0.86 + 0.14 * pop
            breathe = 1 + 0.012 * math.sin(2 * math.pi * 0.28 * t + ci)
            voice = self.speech(t) if self.speech else 0.0
            squash = 1 + 0.018 * voice
            sy = ent * breathe * squash * (1 + 0.06 * u)                         # parallax: the character pushes in faster than the background
            sxs = ent * (1 / math.sqrt(breathe * squash)) * (1 + 0.06 * u)
            sway = math.sin(2 * math.pi * 0.17 * t + ci * 1.7) * 0.012 * H
            tilt = math.sin(2 * math.pi * 0.21 * t + ci) * 1.6 + (voice - 0.3) * 1.2
            hop = -abs(math.sin(2 * math.pi * 1.6 * t + ci)) * 0.012 * H * voice
            gx = W * xf + sway * 2
            gy = H * 1.0 + hop                                                  # feet line at the bottom edge
            M = cv2.getRotationMatrix2D((w0 / 2, h0), tilt, 1.0)
            M[0, 0] *= sxs; M[0, 1] *= sxs; M[1, 0] *= sy; M[1, 1] *= sy
            M[0, 2] += gx - w0 / 2 * 1.0 - (sxs - 1) * 0 ; M[1, 2] += gy - h0
            M[0, 2] = gx - (M[0, 0] * (w0 / 2) + M[0, 1] * h0) + (w0 / 2) * 0
            M[1, 2] = gy - (M[1, 0] * (w0 / 2) + M[1, 1] * h0)
            warped = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            a = warped[..., 3:4].astype(np.float32) / 255.0
            # soft contact shadow under the feet
            sh = np.exp(-(((xx - gx) / (w0 * 0.42 * sxs)) ** 2 + ((yy - (gy - 6)) / (H * 0.02)) ** 2)) * 0.35
            out *= (1 - sh[..., None])
            out = out * (1 - a) + warped[..., :3].astype(np.float32) * a
        # dust motes drifting through the light
        for (px, py, sp, r) in self.dust:
            x = (px * W + t * 14 * sp) % W
            y = (py * H - t * 9 * sp) % H
            al = 0.25 * (0.5 + 0.5 * math.sin(t * sp * 2 + px * 9))
            cv2.circle(out, (int(x), int(y)), int(r), (255, 244, 220), -1)
            out_alpha = 1
        out = out * (1 - 0.0)
        # gentle vignette
        vig = 1 - 0.18 * (((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        out *= np.clip(vig, 0.7, 1)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
