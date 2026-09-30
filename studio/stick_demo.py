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


WARM = dict(sky=((150, 190, 222), (250, 228, 190)), sun=(0.80, 0.22, (252, 214, 120)),
            hills=[dict(color=(178, 190, 200), base=.66, amp=.05, freq=3.0, seed=1, par=.10),
                   dict(color=(150, 160, 150), base=.72, amp=.045, freq=4.2, seed=4, par=.22)])
STORM = dict(sky=((70, 78, 96), (150, 146, 140)), ground_color=(120, 108, 88),
             hills=[dict(color=(92, 98, 110), base=.64, amp=.06, freq=3.2, seed=2, par=.10),
                    dict(color=(72, 76, 84), base=.72, amp=.05, freq=4.6, seed=6, par=.22)])
TOGA, CLOTH = (236, 232, 222), (150, 120, 88)

SCENES = [
    dict(WARM, duration=7.0, narration="Rome, four seventy six. The mighty empire looked unbeatable.",
         objects=[dict(type="column", x=0.12), dict(type="column", x=0.88), dict(type="pedestal", x=0.5),
                  dict(type="cloud", x=0.3, y=.14, r=.045, color=(255, 255, 255, 170)), dict(type="cloud", x=0.66, y=.24, r=.04, color=(255, 255, 255, 150))],
         text=[dict(t=0.3, end=3.4, text="ROME, 476 AD", y=0.09)],
         actors=[
             dict(id="emperor", color=PURPLE, tunic=PURPLE, props=["crown", "cape", "beard"], keys=[K(0, .5, "proud", face="smile"), K(4.6, .5, "proud", face="smile"), K(5.6, .5, "shrug", face="worried")]),
             dict(id="c1", scale=.78, tunic=TOGA, props=["hair"], keys=[K(0, .27, "cheer", face="smile"), K(7, .27, "cheer", face="smile")]),
             dict(id="c2", scale=.78, tunic=CLOTH, props=["hair"], hair=(40, 30, 24), keys=[K(0, .73, "cheer", facing=-1, face="smile"), K(7, .73, "cheer", facing=-1, face="smile")]),
         ], zoom=[1.0, 1.08], focus=(.5, .6), focus_to=(.52, .6)),
    dict(kind="map", duration=7.0, narration="But invaders were pouring over the borders from every side.",
         text=[dict(t=0.3, end=3.4, text="THE INVADERS", y=0.03, color=RED)],
         land=[dict(pts=[(.10, .06), (.92, .05), (.94, .22), (.72, .26), (.55, .23), (.36, .22), (.2, .26), (.08, .18)], color=(224, 214, 178), label="BARBARIAN LANDS", label_at=(.5, .075), label_color=(110, 86, 50)),
               dict(pts=[(.78, .30), (.96, .27), (.96, .72), (.80, .68), (.73, .50)], color=(224, 214, 178), label="STEPPE", label_at=(.875, .62), label_color=(110, 86, 50)),
               dict(pts=[(.12, .45), (.22, .30), (.40, .28), (.58, .34), (.70, .47), (.66, .65), (.52, .73), (.34, .71), (.20, .62)], color=(236, 204, 132), label="ROMAN EMPIRE", label_at=(.40, .36), label_color=(120, 70, 30))],
         cities=[dict(name="Rome", x=.44, y=.56)],
         arrows=[dict(pts=[(.70, .15), (.58, .38), (.47, .53)], t0=.8, t1=3.2, label="GOTHS", label_at=(.69, .10)),
                 dict(pts=[(.88, .52), (.70, .55), (.50, .57)], t0=1.6, t1=4.0, label="HUNS", label_at=(.80, .44)),
                 dict(pts=[(.16, .14), (.24, .36), (.41, .52)], t0=2.4, t1=4.8, label="VANDALS", label_at=(.10, .10))]),
    dict(STORM, duration=8.0, narration="The legions were stretched thin, and the storm finally broke at the gates.", blur=True,
         objects=[dict(type="castle", x=0.14), dict(type="cloud", x=0.7, y=.13, r=.06, color=(92, 94, 104)), dict(type="cloud", x=0.36, y=.2, r=.05, color=(100, 102, 112))],
         text=[dict(t=0.4, end=4, text="AT THE GATES", y=0.09, color=(255, 255, 255))],
         fx=[dict(type="rain", t0=0, t1=8), dict(type="flash", t=3.0), dict(type="dust", actor="i1"), dict(type="dust", actor="i2"), dict(type="dust", actor="i3"),
             dict(type="sparks", t=5.75, x=.43, y=.52), dict(type="sparks", t=6.1, x=.47, y=.50)],
         shake=[dict(t=5.75, dur=.5, amp=16)],
         actors=[
             dict(id="s1", color=BLUE, tunic=BLUE, props=["helmet", "shield", "spear"], scale=.95, keys=[K(0, .28, "stand", face="worried"), K(6, .28, "tremble", face="shock")]),
             dict(id="s2", color=BLUE, tunic=BLUE, props=["helmet", "shield", "sword"], scale=.95, keys=[K(0, .36, "stand", face="worried"), K(5.2, .36, "sword_up", face="angry"), K(6.5, .36, "swing", face="angry")]),
             dict(id="i1", color=RED, tunic=(120, 40, 34), props=["helmet", "sword"], keys=[K(0, 1.15, "run", facing=-1, face="angry"), K(5.6, .50, "run", facing=-1, face="angry"), K(6.4, .50, "swing", facing=-1, face="angry")]),
             dict(id="i2", color=RED, tunic=(120, 40, 34), props=["helmet", "spear"], scale=.9, keys=[K(0, 1.35, "run", facing=-1, face="angry"), K(5.6, .64, "run", facing=-1, face="angry"), K(6.4, .64, "sword_up", facing=-1, face="angry")]),
             dict(id="i3", color=RED, tunic=(120, 40, 34), props=["helmet", "sword"], scale=.85, keys=[K(0, 1.55, "run", facing=-1, face="angry"), K(5.6, .78, "run", facing=-1, face="angry"), K(6.4, .78, "swing", facing=-1, face="angry")]),
         ], zoom=[1.0, 1.07], focus=(.42, .6), focus_to=(.50, .6)),
    dict(STORM, duration=7.0, narration="In four seventy six, the last emperor lost his crown, and the West fell.", dim=0.12,
         objects=[dict(type="pedestal", x=0.5), dict(type="cloud", x=0.2, y=.16, r=.07, color=(92, 94, 104)), dict(type="cloud", x=0.8, y=.13, r=.06, color=(100, 102, 112))],
         text=[dict(t=3.2, end=7.0, text="THE WEST FALLS", y=0.09, color=(232, 88, 70), size=0.095)],
         fx=[dict(type="rain", t0=0, t1=7), dict(type="flash", t=1.4)], shake=[dict(t=2.3, dur=.45, amp=9)],
         actors=[
             dict(id="emperor", color=PURPLE, tunic=PURPLE, props=["crown", "cape", "beard"], crown_fall=2.3,
                  keys=[K(0, .5, "point", face="worried"), K(2.2, .5, "scared", face="shock"), K(3.6, .5, "slump", face="sad"), K(7, .5, "slump", face="sad")]),
         ], zoom=[1.0, 1.14], focus=(.5, .62)),
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
            if s.get("blur"):
                fr = np.mean([stick.render_frame(s, max(0, t + dt), W, H).astype(np.float32) for dt in (-0.012, 0, 0.012)], axis=0).astype(np.uint8)
            else:
                fr = stick.render_frame(s, t, W, H).copy()
            caps.overlay(fr, cursor + t)
            fade = min(1.0, t / 0.25, (s["duration"] - t) / 0.25)
            if fade < 1:
                fr = (fr * fade).astype(np.uint8)
            proc.stdin.write(fr.tobytes())
        cursor += s["duration"]
    proc.stdin.close(); proc.wait()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "stick_demo.mp4")
