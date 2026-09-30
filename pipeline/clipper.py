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

from . import audio as audiolib, ffmpeg, script, tts
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
        try:  # needs OpenCV 4.x; without it we simply keep a centred crop
            self.det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except AttributeError:
            self.det = None

    def update(self, frame, detect):
        if detect and self.det is not None:
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


def pick_stories(cfg, segments, count, duration, commentary):
    """LLM builds short story arcs out of several moments, each joined by a tiny bridge line."""
    words_cap = {"minimal": 8, "light": 14, "medium": 24}.get(commentary, 14)
    lines = "\n".join(f"[{s:.1f}s -> {e:.1f}s] {t}" for s, e, t in segments)
    prompt = f"""You edit viral vertical Shorts for "{cfg['channel_name']}" from a long video ({duration / 60:.0f} min).
Build {count} different Shorts. Each Short is a mini story made of 3 or 4 moments from the transcript, in the order
that builds the most tension, surprise or humour. The footage does the talking; you only add tiny bridge lines.

Rules:
- Each moment is 8 to 20 seconds, starts and ends on segment boundaries (use the exact seconds shown), no overlap
  between moments inside a Short, total length 35 to 58 seconds.
- The FIRST moment must be the strongest hook. Each Short must make sense without the rest of the video.
- "bridge" is a voice-over line spoken over the start of that moment, max {words_cap} words, punchy, sets up what
  the viewer is about to see. Use "" for moments that need no setup. Never retell what is said in the footage.
- Never invent facts about real people; bridges may only reflect what the transcript shows.

Return JSON only:
{{"stories": [{{"title": "<=70 chars", "hook_text": "<=6 words shown on screen", "description": "one sentence plus #shorts",
"parts": [{{"start": 10.0, "end": 24.5, "bridge": "He had no idea what was coming."}}]}}]}}

TRANSCRIPT:
{lines}"""
    out = []
    for st in script.llm_json(prompt, 0.5).get("stories", []):
        parts = []
        for p in st.get("parts", []):
            try:
                a, b = float(p["start"]), float(p["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if 5 <= b - a <= 25 and 0 <= a < b <= duration + 1:
                parts.append({"start": a, "end": b, "bridge": str(p.get("bridge", ""))[:160]})
        total = sum(p["end"] - p["start"] for p in parts)
        if len(parts) >= 2 and 20 <= total <= 70:
            st["parts"] = parts
            out.append(st)
    if not out:
        raise RuntimeError("The LLM returned no usable stories")
    return out[:count]


def _decode_audio(src, start, dur):
    p = subprocess.run([ffmpeg.exe(), "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
                        "-vn", "-f", "f32le", "-ac", "1", "-ar", str(tts.SR), "pipe:1"],
                       capture_output=True, check=True)
    a = np.frombuffer(p.stdout, dtype=np.float32).copy()
    need = int(dur * tts.SR)
    return np.pad(a, (0, max(0, need - len(a))))[:need]


def render_story(src, story, words, out, cfg, narrate=None):
    """Concatenate moments from the source with short bridge voice-overs. narrate(text)->(samples, words)."""
    narrate = narrate or (lambda t: tts.narrate(t, cfg))
    cap = cv2.VideoCapture(str(src))
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30
    follow, portrait = FaceFollower(sw, sh), sh > sw

    track, cap_words, plan, cursor = [], [], [], 0.0
    for part in story["parts"]:
        a, b = part["start"], part["end"]
        dur = b - a
        seg = _decode_audio(src, a, dur)
        bridge = part.get("bridge", "").strip()
        bwin = None
        if bridge:
            voice, bwords = narrate(bridge)
            blen = len(voice) / tts.SR
            if blen < dur - 1.5:
                off = 0.25
                bwin = (off, off + blen)
                i0, i1 = int(off * tts.SR), int(off * tts.SR) + len(voice)
                duck = np.ones(len(seg), dtype=np.float32)
                duck[max(0, i0 - 2000):i1 + 2000] = 0.18     # footage drops under the voice-over
                seg = seg * duck
                seg[i0:i1] += voice[:max(0, len(seg) - i0)][:i1 - i0]
                cap_words += [[w[0], cursor + off + w[1], cursor + off + w[2]] for w in bwords]
        for w in words:  # the footage's own speech becomes captions, except under the bridge
            if w[1] >= a and w[2] <= b + 0.1:
                rel = w[1] - a
                if bwin and bwin[0] - 0.1 < rel < bwin[1] + 0.1:
                    continue
                cap_words.append([w[0], cursor + rel, cursor + w[2] - a])
        track.append(seg)
        plan.append((a, cursor, dur))
        cursor += dur
    cap_words.sort(key=lambda w: w[1])
    mix = np.concatenate(track)
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix = mix / peak * 0.9

    import wave
    wav_path = str(out) + ".wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.SR)
        wf.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())

    caps = Captions(cap_words, cfg, OW, OH)
    hook = make_hook_image(story.get("hook_text", ""), OW, caps.font_path) if story.get("hook_text") else None
    cmd = [ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
           "-r", str(FPS), "-i", "pipe:0", "-i", wav_path, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-maxrate", "9M", "-bufsize", "18M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-movflags", "+faststart", "-shortest", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_total = int(cursor * FPS)
    pi, frame, src_idx = 0, None, -1
    for n in range(n_total):
        t = n / FPS
        while pi + 1 < len(plan) and t >= plan[pi + 1][1]:
            pi += 1
            src_idx, frame = -1, None
        a, c0, d = plan[pi]
        local = t - c0
        if frame is None or src_idx < 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, (a + local) * 1000)
            src_idx = int(round((a + local) * sfps)) - 1
        want = int(round((a + local) * sfps))
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
            fr = cv2.resize(frame[:, x0:x0 + follow.cw], (OW, OH), interpolation=cv2.INTER_CUBIC)
        if pi > 0 and local < 0.1:  # quick flash on each cut
            fr = np.clip(fr.astype(np.int16) + int(30 * (1 - local / 0.1)), 0, 255).astype(np.uint8)
        caps.overlay(fr, t)
        if hook is not None and t < 2.4:
            al = 1.0 if t < 1.8 else max(0.0, 1 - (t - 1.8) / 0.6)
            ov = hook.copy()
            ov[..., 3] = (ov[..., 3] * al).astype(np.uint8)
            _blend(fr, ov, 0, int(OH * 0.14))
        proc.stdin.write(fr.tobytes())
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    pathlib.Path(wav_path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file path or DIRECT download URL")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--mode", choices=["story", "clips"], default="story",
                    help="story: moments stitched with tiny bridge lines (default); clips: single moments")
    ap.add_argument("--commentary", choices=["minimal", "light", "medium"], default="light",
                    help="how long the bridge lines may be. Raise it if you get claims or demonetisation")
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
    meta = []
    if args.mode == "story":
        stories = pick_stories(cfg, segments, args.count, duration, args.commentary)
        log(f"[4/4] rendering {len(stories)} stories")
        for i, st in enumerate(stories, 1):
            path = out / f"story{i:02d}.mp4"
            render_story(src, st, words, path, cfg)
            meta.append({"file": path.name, **st})
            log(f"      {path.name}: {st['title']}")
    else:
        clips = pick_clips(cfg, segments, args.count, duration)
        log(f"[4/4] rendering {len(clips)} clips")
        for i, c in enumerate(clips, 1):
            path = out / f"clip{i:02d}.mp4"
            render_clip(src, c, words, path, cfg)
            meta.append({"file": path.name, **c})
            log(f"      {path.name}: {c['end'] - c['start']:.0f}s  {c['title']}")
    (out / "clips.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
