"""Script + scene clips -> voiceover, fitted clips, captions, music -> one finished MP4."""
import os
import pathlib
import re
import subprocess
import wave

import cv2
import numpy as np

from pipeline import audio as audiolib, ffmpeg, tts
from pipeline.render import Captions, _blend, make_hook_image

FPS = 30
GAP = 0.35  # silence after each line of narration


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr[-600:])


def clip_seconds(path):
    cap = cv2.VideoCapture(str(path))
    n, f = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 24
    cap.release()
    return n / f if n > 0 else 0


def fit_clip(src, dst, dur, W, H, tmp, fade=0.12):
    """Make a silent segment of exactly `dur` seconds from a clip: trim, gently slow, or ping-pong loop."""
    have = clip_seconds(src)
    if have <= 0:
        raise RuntimeError(f"Cannot read clip {src}")
    vf_fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},setsar=1"
    tail = f"fade=t=in:d={fade},fade=t=out:st={max(dur - fade, 0):.3f}:d={fade}"
    enc = ["-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    base = [ffmpeg.exe(), "-y", "-loglevel", "error"]
    if have >= dur - 0.05:
        _run(base + ["-i", str(src), "-vf", f"{vf_fill},{tail}", "-t", f"{dur:.3f}"] + enc + [str(dst)])
    elif dur / have <= 1.3:  # a little short: slow it down instead of looping
        _run(base + ["-i", str(src), "-vf", f"{vf_fill},setpts=PTS*{dur / have:.4f},{tail}", "-t", f"{dur:.3f}"]
             + enc + [str(dst)])
    else:  # much shorter: forward+reverse, looped
        pp = pathlib.Path(tmp) / (pathlib.Path(dst).stem + "_pp.mp4")
        _run(base + ["-i", str(src), "-filter_complex",
                     f"[0:v]{vf_fill},split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[o]", "-map", "[o]"] + enc + [str(pp)])
        _run(base + ["-stream_loop", "-1", "-i", str(pp), "-vf", tail, "-t", f"{dur:.3f}"] + enc + [str(dst)])


def build(proj, log=print):
    """proj: dict with scenes[{voiceover, clip_path}], settings, out_dir, music_path. Returns output path."""
    st = proj["settings"]
    vertical = st.get("aspect", "16:9") == "9:16"
    W, H = (1080, 1920) if vertical else (1920, 1080)
    out_dir = pathlib.Path(proj["out_dir"])
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    scenes = proj["scenes"]
    missing = [s["n"] for s in scenes if not s.get("clip_path")]
    if missing:
        raise RuntimeError(f"These scenes have no clip yet: {missing}")

    cfg = {"voice": {"name": st.get("voice", "en-US-AndrewNeural"), "rate": st.get("rate", "+0%"),
                     "pitch": st.get("pitch", "+0Hz")},
           "captions": {"font": None, "size": 118 if vertical else 76, "words_per_chunk": 3 if vertical else 4,
                        "color": "#FFFFFF", "highlight": "#FFD23F", "y_position": 0.66 if vertical else 0.92}}

    log("voiceover")
    voices, words, starts, durs, cursor = [], [], [], [], 0.0
    for i, s in enumerate(scenes):
        text = s["voiceover"].strip()
        if os.environ.get("STUDIO_FAKE_TTS"):
            v, w = tts.silent(text)
        else:
            v, w = tts.narrate(text, cfg)
        lead = 0.15 if i == 0 else 0.05
        d = lead + len(v) / tts.SR + GAP
        voices.append(v)
        starts.append(cursor + lead)
        durs.append(d)
        words += [[x[0], cursor + lead + x[1], cursor + lead + x[2]] for x in w]
        cursor += d
        log(f"  scene {s['n']}: {d:.1f}s of narration")
    total = cursor

    log("fitting clips to the narration")
    segs = []
    for i, s in enumerate(scenes):
        seg = tmp / f"seg{i:03d}.mp4"
        fit_clip(s["clip_path"], seg, durs[i], W, H, tmp)
        segs.append(seg)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    joined = tmp / "joined.mp4"
    _run([ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])

    log("mixing audio")
    n = int(total * tts.SR)
    voice = np.zeros(n, dtype=np.float32)
    for v, s0 in zip(voices, starts):
        i0 = int(s0 * tts.SR)
        seg = v[:max(0, n - i0)]
        voice[i0:i0 + len(seg)] += seg
    mix = voice
    if np.abs(voice).max() > 0:
        mix = np.tanh(1.6 * voice / np.abs(voice).max()) * 0.9
    mp = proj.get("music_path")
    if mp:
        raw = subprocess.run([ffmpeg.exe(), "-loglevel", "error", "-stream_loop", "-1", "-i", str(mp), "-t", f"{total:.2f}",
                              "-f", "f32le", "-ac", "1", "-ar", str(tts.SR), "pipe:1"], capture_output=True, check=True).stdout
        music = np.frombuffer(raw, dtype=np.float32)[:n]
        music = np.pad(music, (0, n - len(music))) * st.get("music_volume", 0.12)
        mix = mix + music * (1 - 0.6 * audiolib._envelope(voice))
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix = mix / peak * 0.9
    wav_path = tmp / "mix.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())

    log("captions + final render")
    caps = Captions(words, cfg, W, H) if st.get("captions", True) else None
    hook = None
    if st.get("hook_text"):
        hook = make_hook_image(st["hook_text"], W, caps.font_path if caps else Captions([], cfg, W, H).font_path)
    final = out_dir / "final.mp4"
    cmd = [ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "pipe:0", "-i", str(wav_path), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-maxrate", "12M", "-bufsize", "24M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-movflags", "+faststart", "-shortest", str(final)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(joined))
    k = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = k / FPS
        if caps:
            caps.overlay(fr, t)
        if hook is not None and t < 2.4:
            a = 1.0 if t < 1.8 else max(0.0, 1 - (t - 1.8) / 0.6)
            ov = hook.copy()
            ov[..., 3] = (ov[..., 3] * a).astype(np.uint8)
            _blend(fr, ov, 0, int(H * 0.14))
        proc.stdin.write(fr.tobytes())
        k += 1
        if k % (FPS * 10) == 0:
            log(f"  {k / FPS:.0f}s of {total:.0f}s")
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("final render failed")
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    log(f"done: {total:.0f}s video")
    return final


# ---- parsing helpers -------------------------------------------------------------------------

def parse_table(text):
    """Parse a markdown scene table from NotebookLM: Scene # | Voiceover | Image Prompt | Animation Prompt."""
    rows = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    rows = [r for r in rows if not re.match(r"^\|?[\s:\-|]+\|?$", r)]
    if len(rows) < 2:
        return []
    cells = lambda r: [re.sub(r"<br\s*/?>", " ", c).strip() for c in r.strip().strip("|").split("|")]
    head = [h.lower() for h in cells(rows[0])]

    def col(*names):
        for i, h in enumerate(head):
            if any(nm in h for nm in names):
                return i
        return None
    iv, ii, ia = col("voice", "narrat"), col("image"), col("anim", "video")
    if iv is None:
        return []
    out = []
    for r in rows[1:]:
        c = cells(r)
        if len(c) <= iv or not c[iv]:
            continue
        out.append({"n": len(out) + 1, "voiceover": c[iv], "image_prompt": c[ii] if ii is not None and ii < len(c) else "",
                    "anim_prompt": c[ia] if ia is not None and ia < len(c) else ""})
    return out


def split_script(text, target_words=22):
    """Plain voiceover prose -> scenes of roughly one 8-second clip each."""
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    scenes, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= target_words:
            scenes.append(" ".join(cur))
            cur = []
    if cur:
        if scenes and sum(len(x.split()) for x in cur) < 8:
            scenes[-1] += " " + " ".join(cur)
        else:
            scenes.append(" ".join(cur))
    return [{"n": i + 1, "voiceover": t, "image_prompt": "", "anim_prompt": ""} for i, t in enumerate(scenes)]
