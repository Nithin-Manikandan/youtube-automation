"""Turn a long video you have rights to into ready-to-review vertical clips.

    python -m pipeline.clipper --source <file-or-direct-url> [--count 5]

transcribe (faster-whisper) -> LLM picks the moments people will not skip -> 9:16 reframe that
follows the speaker's face -> word-by-word captions + hook line -> MP4s in output/clips/.
Only use footage you own, that is licensed for reuse (CC-BY, credited), or that a creator's
clipping programme supplied to you. This tool does not download from YouTube.
"""
import argparse
import json
import pathlib
import re
import subprocess
import time

import cv2
import numpy as np
import requests

from . import ffmpeg, script
from .config import ROOT, load
from .render import Captions, _blend, make_hook_image

FPS = 30
OW, OH = 1080, 1920


def log(m):
    print(m, flush=True)


def fetch(source, cache):
    cache.mkdir(parents=True, exist_ok=True)
    if not re.match(r"https?://", source):
        return pathlib.Path(source)
    dest = cache / "source.mp4"
    with requests.get(source, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            raise RuntimeError("That link is a web page, not a direct video file. Use a direct download link.")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def transcribe(src, cache, model_size="base.en"):
    wav = cache / "audio.wav"
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    str(wav)], check=True)
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
    words, segments = [], []
    for s in segs:
        segments.append((s.start, s.end, s.text.strip()))
        for w in s.words or []:
            words.append([w.word.strip(), w.start, w.end])
    return words, segments


def pick_clips(cfg, segments, count, duration):
    lines = "\n".join(f"[{int(s // 60):02d}:{int(s % 60):02d} = {s:.1f}s -> {e:.1f}s] {t}" for s, e, t in segments)
    prompt = f"""You are an expert short-form video editor for the channel "{cfg['channel_name']}".
Below is a timestamped transcript of a long video ({duration / 60:.0f} minutes). Pick the {count} best moments to
cut into vertical Shorts that viewers will not scroll past.

Rules:
- Each clip is 20 to 55 seconds and starts exactly at the beginning of a segment and ends at the end of one.
- The first sentence must hook (surprising claim, strong opinion, story peak, conflict or a question).
- The clip must make sense on its own and end on a payoff, not mid-thought.
- No overlapping clips. Prefer emotional, funny, controversial or surprising moments over filler.
- Use the exact start/end seconds shown in the transcript.

Return JSON only:
{{"clips": [{{"start": 12.3, "end": 48.0, "title": "<=70 chars", "hook_text": "<=6 words on screen",
"description": "one sentence plus #shorts", "why": "one sentence"}}]}}

TRANSCRIPT:
{lines}"""
    clips = script.llm_json(prompt, 0.4).get("clips", [])
    good = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 12 <= e - s <= 75 and 0 <= s < e <= duration + 1:
            c["start"], c["end"] = s, e
            good.append(c)
    if not good:
        raise RuntimeError("The LLM returned no usable clip ranges")
    return good[:count]


class FaceFollower:
    """Smoothly steers the 9:16 crop window toward the largest face; holds still otherwise."""

    def __init__(self, src_w, src_h):
        self.cw = int(src_h * 9 / 16)
        self.cx = src_w / 2
        self.target = src_w / 2
        self.w = src_w
        self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def update(self, frame, detect):
        if detect:
            small = cv2.resize(frame, None, fx=0.4, fy=0.4)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            faces = self.det.detectMultiScale(gray, 1.15, 5, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                new = (x + w / 2) / 0.4
                if abs(new - self.target) > self.cw * 0.18:  # ignore jitter, follow real speaker changes
                    self.target = new
        self.cx += (self.target - self.cx) * 0.08
        half = self.cw / 2
        self.cx = min(max(self.cx, half), self.w - half)
        return int(self.cx - half)


def render_clip(src, clip, words, out, cfg):
    cap = cv2.VideoCapture(str(src))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30
    start, end = clip["start"], clip["end"]
    dur = end - start
    local = [[w[0], w[1] - start, w[2] - start] for w in words if w[1] >= start - 0.05 and w[2] <= end + 0.3]
    caps = Captions(local, cfg, OW, OH)
    hook = make_hook_image(clip.get("hook_text", ""), OW, caps.font_path) if clip.get("hook_text") else None
    follow = FaceFollower(sw, sh)
    portrait = sh > sw

    cmd = [ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
           "-r", str(FPS), "-i", "pipe:0", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "9M",
           "-bufsize", "18M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-movflags", "+faststart", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    n_out = int(dur * FPS)
    src_idx, frame = -1, None
    for n in range(n_out):
        want = int(round(n / FPS * sfps))
        while src_idx < want:
            ok, bgr = cap.read()
            if not ok:
                break
            src_idx += 1
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if frame is None:
            break
        if portrait:
            fr = cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_CUBIC)
        else:
            x0 = follow.update(frame, n % 8 == 0)
            crop = frame[:, x0:x0 + follow.cw]
            fr = cv2.resize(crop, (OW, OH), interpolation=cv2.INTER_CUBIC)
        t = n / FPS
        caps.overlay(fr, t)
        if hook is not None and t < 2.4:
            a = 1.0 if t < 1.8 else max(0.0, 1 - (t - 1.8) / 0.6)
            ov = hook.copy()
            ov[..., 3] = (ov[..., 3] * a).astype(np.uint8)
            _blend(fr, ov, 0, int(OH * 0.14))
        proc.stdin.write(fr.tobytes())
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file path or DIRECT download URL")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--model", default="base.en", help="whisper model: tiny.en, base.en, small.en")
    ap.add_argument("--out", default=str(ROOT / "output" / "clips"))
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load(args.config)
    cache = ROOT / "cache" / "clipper"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("[1/4] getting source")
    src = fetch(args.source, cache)
    cap = cv2.VideoCapture(str(src))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    log(f"      {duration / 60:.1f} minutes")

    log("[2/4] transcribing")
    words, segments = transcribe(src, cache, args.model)
    log(f"      {len(words)} words, {len(segments)} segments")

    log("[3/4] choosing moments")
    clips = pick_clips(cfg, segments, args.count, duration)

    log(f"[4/4] rendering {len(clips)} clips")
    meta = []
    for i, c in enumerate(clips, 1):
        path = out / f"clip{i:02d}.mp4"
        render_clip(src, c, words, path, cfg)
        meta.append({"file": path.name, **c})
        log(f"      {path.name}: {c['end'] - c['start']:.0f}s  {c['title']}")
    (out / "clips.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
