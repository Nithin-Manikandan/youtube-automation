"""Documentary-style captions: phrase by phrase, soft dark bar, clean font, gold word highlight."""
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pipeline.config import ROOT

FONT = str(ROOT / "assets/fonts/InstrumentSans-Bold.ttf")
TEXT, HI = (245, 243, 236), (255, 206, 84)


STOP = {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but", "with", "by", "from", "as", "that", "than", "his", "her",
        "its", "their", "our", "my", "was", "were", "is", "are", "be", "had", "has", "have", "into", "onto", "over", "under", "who", "which"}


def _bare(w):
    return re.sub(r"[^\w']", "", w).lower()


def _split_sentence(ws, max_words):
    """Split one sentence into balanced phrases, preferring commas, never ending a phrase on a function word."""
    if len(ws) <= max_words + 1:
        return [ws]
    n = -(-len(ws) // max_words)
    out, i = [], 0
    for k in range(n - 1):
        ideal = i + round((len(ws) - i) / (n - k))
        best = ideal
        for cut in range(max(i + 3, ideal - 3), min(len(ws) - 2, ideal + 3) + 1):   # look a few words either side of the ideal cut
            w = ws[cut - 1][0]
            if re.search(r"[,;:\u2014]$", w):
                best = cut
                break
            if _bare(w) in STOP:
                continue
            if abs(cut - ideal) < abs(best - ideal) or _bare(ws[best - 1][0]) in STOP:
                best = cut
        while best > i + 2 and _bare(ws[best - 1][0]) in STOP:
            best -= 1
        out.append(ws[i:best])
        i = best
    out.append(ws[i:])
    return out


def phrases(words, max_words=9, min_words=3, pause=0.4):
    sentences, cur = [], []
    for w in words:
        if not w[0].strip():
            continue
        if cur and (w[1] - cur[-1][2] > pause or re.search(r"[.!?]$", cur[-1][0]) or (len(w) > 3 and w[3])):
            sentences.append(cur)
            cur = []
        cur.append(w)
    if cur:
        sentences.append(cur)
    out = []
    for sent in sentences:
        out += _split_sentence(sent, max_words)
    return out


class DocCaptions:
    def __init__(self, words, W, H):
        self.W, self.H = W, H
        self.size = int(H * 0.047)
        self.font = ImageFont.truetype(FONT, self.size)
        self.ph = phrases(words)
        self.cache = {}
        self.starts = [p[0][1] for p in self.ph]

    def _layout(self, words):
        maxw = self.W * 0.66
        lines, cur, cw = [], [], 0
        sp = self.font.getlength(" ")
        for i, w in enumerate(words):
            ww = self.font.getlength(w[0])
            if cur and cw + sp + ww > maxw:
                lines.append(cur)
                cur, cw = [], 0
            cur.append(i)
            cw += (sp if len(cur) > 1 else 0) + ww
        lines.append(cur)
        return lines

    def _draw(self, pi, active):
        words = self.ph[pi]
        lines = self._layout(words)
        lh = int(self.size * 1.32)
        widths = [sum(self.font.getlength(words[i][0]) for i in ln) + self.font.getlength(" ") * (len(ln) - 1) for ln in lines]
        padx, pady = int(self.size * 0.75), int(self.size * 0.42)
        bw, bh = int(max(widths) + padx * 2), int(lh * len(lines) + pady * 2 - (lh - self.size) * 0.4)
        margin = 24
        img = Image.new("RGBA", (bw + margin * 2, bh + margin * 2), (0, 0, 0, 0))
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.rounded_rectangle([margin, margin + 4, margin + bw, margin + bh + 4], radius=int(self.size * .5), fill=(0, 0, 0, 120))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(10)))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([margin, margin, margin + bw, margin + bh], radius=int(self.size * .5), fill=(14, 16, 22, 196))
        y = margin + pady - int((lh - self.size) * 0.15)
        for ln, lw_ in zip(lines, widths):
            x = margin + (bw - lw_) / 2
            for i in ln:
                col = HI if i == active else TEXT
                d.text((x + 1.5, y + 2), words[i][0], font=self.font, fill=(0, 0, 0, 150))
                d.text((x, y), words[i][0], font=self.font, fill=col + (255,))
                x += self.font.getlength(words[i][0]) + self.font.getlength(" ")
            y += lh
        return np.asarray(img)

    def overlay(self, frame, t):
        pi = None
        for i, p in enumerate(self.ph):
            end = self.ph[i + 1][0][1] - 0.04 if i + 1 < len(self.ph) and self.ph[i + 1][0][1] - p[-1][2] < 0.4 else p[-1][2] + 0.25
            if p[0][1] - 0.06 <= t < end:
                pi, tend = i, end
                break
        if pi is None:
            return
        p = self.ph[pi]
        active = max([k for k, w in enumerate(p) if w[1] <= t] or [0])
        key = (pi, active)
        if key not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[key] = self._draw(pi, active)
        base = self.cache[key]
        fade = min(1.0, (t - (p[0][1] - 0.06)) / 0.14, (tend - t) / 0.12)
        h, w = base.shape[:2]
        x0 = (self.W - w) // 2
        y0 = int(self.H * 0.965 - h)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(self.W, x0 + w), min(self.H, y0 + h)
        if xa >= xb or ya >= yb:
            return
        sub = base[ya - y0:yb - y0, xa - x0:xb - x0]
        a = sub[..., 3:4].astype(np.float32) / 255 * max(0.0, fade)
        reg = frame[ya:yb, xa:xb].astype(np.float32)
        frame[ya:yb, xa:xb] = (reg * (1 - a) + sub[..., :3] * a).astype(np.uint8)
