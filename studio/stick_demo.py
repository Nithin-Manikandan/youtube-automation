"""Render a sample stickman history scene:  python -m studio.stick_demo out.mp4"""
import subprocess
import sys
import wave

import cv2
import numpy as np

from pipeline import audio as audiolib, ffmpeg
from pipeline.render import Captions
from pipeline.tts import SR
from . import stick
from .stick import BLUE, GOLD, PURPLE, RED

W, H, FPS = 1280, 720, 30


def K(t, x, pose="stand", **kw):
    return dict(t=t, x=x, pose=pose, **kw)


SCENES = [
    dict(duration=7.0, narration="Rome, four seventy six. The mighty empire looked unbeatable.",
         objects=[dict(type="sun", x=0.82, y=0.16), dict(type="column", x=0.12), dict(type="column", x=0.88), dict(type="pedestal", x=0.5)],
         text=[dict(t=0.3, end=3.2, text="ROME, 476 AD", y=0.10)],
         actors=[
             dict(id="emperor", color=PURPLE, props=["crown", "cape"], keys=[K(0, .5, "proud", face="smile"), K(4.6, .5, "proud", face="smile"), K(5.6, .5, "shrug", face="worried")]),
             dict(id="c1", x=0, scale=.8, keys=[K(0, .26, "cheer", face="smile"), K(7, .26, "cheer", face="smile")]),
             dict(id="c2", scale=.8, keys=[K(0, .74, "cheer", facing=-1, face="smile"), K(7, .74, "cheer", facing=-1, face="smile")]),
         ], zoom=[1.0, 1.08], focus=(.5, .6)),
    dict(duration=8.0, narration="But invaders were pouring over the border, and the legions were stretched thin.",
         objects=[dict(type="castle", x=0.16), dict(type="cloud", x=0.7, y=.15, r=.06), dict(type="cloud", x=0.4, y=.22, r=.05)],
         text=[dict(t=0.4, end=4, text="THE INVADERS", y=0.10, color=RED)], dim=0.12,
         actors=[
             dict(id="s1", color=BLUE, props=["helmet", "shield", "spear"], scale=.95, keys=[K(0, .30, "stand", face="worried"), K(6, .30, "tremble", face="shock")]),
             dict(id="s2", color=BLUE, props=["helmet", "shield", "sword"], scale=.95, keys=[K(0, .38, "stand", face="worried"), K(6, .38, "swing", face="angry")]),
             dict(id="i1", color=RED, props=["helmet", "sword"], keys=[K(0, 1.15, "run", facing=-1, face="angry"), K(5.5, .52, "run", facing=-1, face="angry"), K(6.5, .52, "swing", facing=-1, face="angry")]),
             dict(id="i2", color=RED, props=["helmet", "spear"], scale=.9, keys=[K(0, 1.35, "run", facing=-1, face="angry"), K(5.5, .66, "run", facing=-1, face="angry"), K(6.5, .66, "sword_up", facing=-1, face="angry")]),
             dict(id="i3", color=RED, props=["helmet", "sword"], scale=.85, keys=[K(0, 1.55, "run", facing=-1, face="angry"), K(5.5, .80, "run", facing=-1, face="angry"), K(6.5, .80, "swing", facing=-1, face="angry")]),
         ], zoom=[1.0, 1.05], focus=(.5, .6)),
    dict(duration=7.0, narration="In four seventy six, the last emperor lost his crown, and the West fell.",
         objects=[dict(type="pedestal", x=0.5), dict(type="cloud", x=0.2, y=.18, r=.07), dict(type="cloud", x=0.8, y=.14, r=.06)],
         text=[dict(t=3.2, end=7.0, text="THE WEST FALLS", y=0.10, color=RED, size=0.095)], dim=0.22,
         actors=[
             dict(id="emperor", color=PURPLE, props=["crown", "cape"], crown_fall=2.3,
                  keys=[K(0, .5, "point", face="worried"), K(2.2, .5, "scared", face="shock"), K(3.6, .5, "slump", face="sad"), K(7, .5, "slump", face="sad")]),
         ], zoom=[1.0, 1.12], focus=(.5, .62)),
]


def main(out):
    frames = []
    total = sum(s["duration"] for s in SCENES)
    words, c = [], 0.0
    for s in SCENES:
        toks = s["narration"].split()
        d = s["duration"] - 1.0
        words += [[w, c + .4 + d * i / len(toks), c + .4 + d * (i + 1) / len(toks)] for i, w in enumerate(toks)]
        c += s["duration"]
    cfg = {"captions": {"font": "assets/fonts/BricolageGrotesque-Bold.ttf", "size": 64, "words_per_chunk": 4,
                        "color": "#FFFFFF", "highlight": "#FFD23F", "y_position": 0.88}}
    caps = Captions(words, cfg, W, H)
    music = audiolib.drone(total) * 0.35
    wav = out + ".wav"
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(music, -1, 1) * 32767).astype(np.int16).tobytes())
    proc = subprocess.Popen([ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "pipe:0", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out], stdin=subprocess.PIPE)
    cursor = 0.0
    for s in SCENES:
        stick.prepare(s, W, H)
        n = int(s["duration"] * FPS)
        for i in range(n):
            t = i / FPS
            fr = stick.render_frame(s, t, W, H).copy()
            caps.overlay(fr, cursor + t)
            fade = min(1.0, t / 0.25, (s["duration"] - t) / 0.25)
            if fade < 1:
                fr = (fr * fade + np.array(stick.PAPER) * (1 - fade)).astype(np.uint8)
            proc.stdin.write(fr.tobytes())
        cursor += s["duration"]
    proc.stdin.close(); proc.wait()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stick_demo.mp4")
