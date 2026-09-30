"""One button: pick an engaging history topic -> script -> fact-check -> stickman scenes -> voice, music, sound effects
-> video -> thumbnails -> metadata, then stop and wait for you to approve."""
import concurrent.futures as cf
import multiprocessing as mp
import json
import os
import pathlib
import random
import re
import subprocess
import time
import wave

import cv2
import numpy as np

from pipeline import audio as audiolib, ffmpeg, tts
from . import ai, music, recipes, sfx, stick, thumb
from .doccap import DocCaptions

W, H, FPS = 1280, 720, 30
GAP, LEAD = 0.55, 0.12
MIN_S, MAX_S = 8 * 60, 15 * 60
FAKE = lambda: bool(os.environ.get("STUDIO_FAKE"))
HERE = pathlib.Path(__file__).parent
STAGES = ["Picking a topic", "Writing the script", "Fact-checking", "Planning visuals", "Recording the voiceover",
          "Drawing the scenes", "Mixing music and sound effects", "Making thumbnails", "Packaging"]

VOCAB = f"""VISUAL RECIPE VOCABULARY (use only these words):
- type: "stage" (stick figures), "map", or "card" (big text).
- stage.background: {sorted(recipes.BACKGROUNDS)}
- stage.actors (0-4): role {sorted(recipes.ROLES)}; color {sorted(recipes.COLORS)}; pos {list(recipes.POS)}; action {sorted(recipes.ACTIONS)}; emotion {sorted(recipes.EMOTIONS)}; facing "left"/"right" (optional)
- stage.objects (0-3): {sorted(recipes.OBJECTS)}
- stage.effects (0-2): {sorted(recipes.EFFECTS)}  (rain, flash=lightning, sparks=sword clash, dust=running dust, shake=impact)
- stage.title: optional on-screen label up to 4 words; stage.camera: {sorted(recipes.CAMERAS)}
- map: {{"template":"invasion"|"route"|"expanding","center":"EMPIRE NAME","city":"Capital","labels":["Attacker 1","Attacker 2"],"title":"SHORT TITLE"}} (invasion: labels are who attacks; route: labels are places along a journey in order; expanding: labels are regions reached)
- card: {{"big":"476 AD","small":"one line","bullets":["up to 3 short points"],"dark":true}} for dates, numbers, key facts."""


def _log(job, msg):
    job["log"].append(msg)
    print(msg, flush=True)


def _stage(job, i):
    job["stage"] = i
    _log(job, f"== {STAGES[i]}")


# ---------------------------------------------------------------- LLM steps ----------------------------------------
def _llm(prompt, temp=0.8):
    if FAKE():
        raise RuntimeError("fake mode has no LLM")
    time.sleep(1.5)  # stay polite to the free-tier per-minute limits
    for attempt in range(3):
        try:
            return ai._json(ai.gemini_text(prompt, json_out=True, temperature=temp))
        except (ValueError, KeyError) as e:
            if attempt == 2:
                raise RuntimeError(f"The AI returned something unreadable ({e}). Try again.")
    return None


def used_topics():
    p = HERE / "topics_used.json"
    return json.loads(p.read_text()) if p.exists() else []


def remember_topic(topic):
    p = HERE / "topics_used.json"
    p.write_text(json.dumps((used_topics() + [topic])[-300:], indent=1))


def pick_topic(hint):
    if FAKE():
        return {"topic": "The Fall of Rome: how an empire died in 476 AD", "hook": "The day the ancient world ended"}
    prompt = f"""You run a faceless YouTube history channel with clear, dramatic, accurate storytelling. Choose the single best next video.

Pick a topic that MANY people would click and watch to the end:
- a well-documented story with high stakes, a surprising twist, and a clear protagonist or event (think: how a small decision changed everything,
  the worst day in history, the real story behind a legend, why a mighty empire collapsed, a daring escape, a famous mystery)
- broad appeal (not niche academic); curiosity gap in the title; evergreen
- facts must be solid and widely documented, so the script can be accurate. Do not choose topics needing obscure or disputed claims.
- do NOT reuse any of these already-made topics: {used_topics()[-60:]}
{f'Viewer hint to honour if it fits: {hint}' if hint else ''}

Brainstorm 8 candidates, score each 1-10 for: broad_appeal, story_drama, factual_solidity. Return JSON:
{{"candidates": [{{"topic": "working title as a sentence", "hook": "the one-line hook", "broad_appeal": 0, "story_drama": 0, "factual_solidity": 0}}]}}"""
    c = _llm(prompt, 1.0).get("candidates", [])
    if not c:
        raise RuntimeError("No topic candidates came back.")
    best = max(c, key=lambda x: (x.get("broad_appeal", 0) + x.get("story_drama", 0) + 1.5 * x.get("factual_solidity", 0)))
    return best


def write_outline(topic, hook):
    if FAKE():
        fx = _fixture()
        titles = [c["title"] for c in fx["chapters"]] if fx else ("A Mighty Empire", "Trouble at the Borders", "The Last Emperor")
        return {"working_title": topic, "chapters": [{"title": t, "purpose": "", "key_facts": [], "target_words": 160} for t in titles]}
    prompt = f"""Plan a 12-minute YouTube history video (about 2000 spoken words).
TOPIC: {topic}
HOOK: {hook}

Structure for maximum retention: a cold open that starts inside the most dramatic moment (no greetings, no "welcome back"), then 7-9 chapters
that each end with an open loop or twist that pulls the viewer into the next one, and a final payoff that answers the opening question and
lands one memorable takeaway. Chronology must be correct. Use only well-documented facts.
Return JSON: {{"working_title": "", "chapters": [{{"title": "", "purpose": "", "key_facts": ["specific documented facts/dates/names this chapter must use"], "target_words": 240}}]}}
Chapter target_words must sum to about 2000 (between 1800 and 2200)."""
    o = _llm(prompt, 0.7)
    if len(o.get("chapters", [])) < 4:
        raise RuntimeError("The outline came back too short.")
    return o


def write_chapter(topic, outline, idx, prev_tail, words):
    ch = outline["chapters"][idx]
    if FAKE():
        return _fake_chapter(idx, ch["title"])
    n_scenes = max(5, round(words / 24))
    prompt = f"""You are writing chapter {idx + 1} of {len(outline['chapters'])} of the voiceover for a YouTube history video.
VIDEO TOPIC: {topic}
FULL OUTLINE: {json.dumps([{'title': c['title'], 'purpose': c.get('purpose', '')} for c in outline['chapters']])}
THIS CHAPTER: "{ch['title']}" - {ch.get('purpose', '')}
FACTS TO USE (accurate only): {json.dumps(ch.get('key_facts', []))}
{f'The previous chapter ended with: "{prev_tail}"' if prev_tail else 'This is the cold open: start inside the most dramatic moment, no greetings.'}

Write about {words} words of spoken narration split into {n_scenes} scenes (each scene 18-30 words, one idea, short punchy sentences, vivid but factual).
Rules: never invent quotes, numbers or dates; where historians disagree say so; keep it gripping (tension, stakes, contrast); end the chapter with an
open loop leading to the next one (except the final chapter, which gives the payoff and one memorable closing line).
For each scene choose a visual recipe that ILLUSTRATES what is said. Vary backgrounds and compositions; use a map when geography matters and a card
for key dates and numbers (about 1 scene in 5); keep characters consistent (same role and color for the same person across scenes).

{VOCAB}

Return JSON only: {{"scenes": [{{"narration": "...", "visual": {{"type": "stage", "background": "city_day", "actors": [{{"role": "emperor", "color": "purple", "pos": "center", "action": "talk", "emotion": "worried"}}], "objects": ["column"], "effects": [], "title": "", "camera": "push_in"}}}}]}}"""
    o = _llm(prompt, 0.8)
    sc = [s for s in o.get("scenes", []) if isinstance(s, dict) and str(s.get("narration", "")).strip()]
    if len(sc) < 3:
        raise RuntimeError(f"Chapter {idx + 1} came back with too few scenes.")
    return sc


def factcheck(topic, text):
    if FAKE():
        return [{"claim": "(fake mode) example flag", "issue": "none", "fix": "", "severity": "low"}]
    prompt = f"""You are a strict history fact-checker. Below is the narration of a YouTube video about: {topic}.
List every statement that is likely WRONG, exaggerated, disputed among historians, or not verifiable (dates, numbers, names, causes, quotes).
Be specific and honest; if a claim is fine do not list it. Return JSON: {{"flags": [{{"claim": "quote the sentence", "issue": "what is wrong or uncertain", "fix": "safer wording", "severity": "high|medium|low"}}]}}

NARRATION:
{text}"""
    return _llm(prompt, 0.2).get("flags", [])


def make_metadata(topic, outline, chapters, text):
    if FAKE():
        return {"title": "How Rome Really Fell in 476 AD", "title_options": ["How Rome Really Fell in 476 AD", "Rome Didn't Fall in a Day. It Fell in 5 Steps", "The Empire That Ended Without a Fight"],
                "description_intro": "The real story of how the Western Roman Empire ended.",
                "tags": ["rome", "history", "fall of rome", "roman empire", "476 ad"], "hashtags": ["#history", "#rome", "#ancienthistory"],
                "pinned_comment": "What surprised you most? Tell me below.",
                "thumbs": [{"text": "ROME FELL", "mood": "fire", "role": "emperor", "action": "scared", "emotion": "shock", "badge": "476 AD", "objects": ["castle", "column"]},
                           {"text": "NO ONE NOTICED", "mood": "ice", "role": "warrior", "action": "sword_up", "emotion": "angry", "objects": ["tent"]},
                           {"text": "WHY IT COLLAPSED", "mood": "night", "role": "king", "action": "shrug", "emotion": "worried", "badge": "476", "mark": "?", "objects": ["tower"]}]}
    prompt = f"""Create the YouTube click package for a history video. It must earn the click, honestly.
TOPIC: {topic}
CHAPTERS: {json.dumps([c['title'] for c in chapters])}
NARRATION (for accuracy and keywords):
{text[:6000]}

TITLE RULES (from CTR research): 40-60 characters; front-load the main keyword; create a curiosity gap (name the problem or mystery, withhold the answer);
use one of: a specific number or date, a contrarian angle ("Everything you know about X is wrong" only if true), a time marker ("in one day"), or a strong
emotional word. No ALL CAPS, no lies, nothing the video does not deliver. Write 5 different titles using different patterns and mark the best as "title".

THUMBNAIL RULES: 2-3 words maximum (never more than 4), huge and readable on a phone; the text must ADD to the title, not repeat it (tease the twist);
one giant expressive face showing the emotion the video promises; high contrast; optional date badge (like "476 AD") and ?/! mark. Make 3 distinctly
different concepts (different text, mood, character and emotion) so the best can be chosen.

Return JSON:
{{"title": "best title",
 "title_options": ["5 titles"],
 "description_intro": "2-3 sentences: a hook then what the viewer will learn, with natural search keywords (this is the part shown before 'show more')",
 "tags": ["12-15 search tags, most important first, mix of broad and specific"],
 "hashtags": ["#three", "#relevant", "#hashtags"],
 "pinned_comment": "a question that sparks comments",
 "thumbs": [{{"text": "2-3 WORDS", "mood": "fire|ice|gold|storm|blood|night", "role": "one of {sorted(recipes.ROLES)}", "color": "one of {sorted(recipes.COLORS)}", "action": "scared|point|sword_up|proud|shrug|cheer|slump", "emotion": "shock|angry|worried|sad|smile", "badge": "optional short date like 476 AD or empty", "mark": "! or ? or empty", "objects": ["1-2 of {sorted(recipes.OBJECTS - {'cloud', 'torch'})} matching the story setting"]}}]}}"""
    return _llm(prompt, 0.7)


def _fixture():
    p = os.environ.get("STUDIO_FAKE_SCRIPT")
    return json.loads(pathlib.Path(p).read_text()) if p else None


def _fake_chapter(idx, title):
    fx = _fixture()
    if fx:
        return fx["chapters"][idx]["scenes"]
    rnd = random.Random(idx)
    lines = [f"{title}, scene {i + 1}. The story keeps moving, and every turn raises the stakes a little higher." for i in range(4)]
    vis = [
        {"type": "stage", "background": "city_day", "actors": [{"role": "emperor", "pos": "center", "action": "proud", "emotion": "smile"}, {"role": "citizen", "pos": "left", "action": "cheer", "emotion": "smile"}], "objects": ["column"], "title": title},
        {"type": "map", "template": "invasion", "center": "ROMAN EMPIRE", "city": "Rome", "labels": ["GOTHS", "HUNS"], "title": "BORDERS UNDER PRESSURE"},
        {"type": "stage", "background": "storm", "actors": [{"role": "soldier", "color": "blue", "pos": "left", "action": "scared", "emotion": "shock"}, {"role": "warrior", "color": "red", "pos": "right", "action": "enter_run", "emotion": "angry", "facing": "left"}], "objects": ["castle"], "effects": ["rain", "dust"]},
        {"type": "card", "big": "476 AD", "small": "The last emperor is deposed", "bullets": ["Odoacer takes Italy", "The Senate sends the regalia east"]},
    ]
    return [{"narration": lines[i], "visual": vis[(i + idx) % 4]} for i in range(4)]


# ---------------------------------------------------------------- rendering ---------------------------------------
def render_segment(args):
    sc, t0, words, path, first, last = args
    stick.prepare(sc, W, H)
    caps = DocCaptions(words, W, H)
    cmd = [ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-maxrate", "3500k", "-bufsize", "7000k", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(round(sc["duration"] * FPS))
    for i in range(n):
        t = i / FPS
        if sc.get("blur"):
            fr = np.mean([stick.render_frame(sc, max(0, t + d), W, H).astype(np.float32) for d in (-0.012, 0, 0.012)], axis=0).astype(np.uint8)
        else:
            fr = stick.render_frame(sc, t, W, H).copy()
        caps.overlay(fr, t0 + t)
        f = 1.0
        if first:
            f = min(f, t / 0.6)
        if last:
            f = min(f, (sc["duration"] - t) / 0.8)
        if f < 1:
            fr = (fr * max(f, 0)).astype(np.uint8)
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("segment render failed")
    return str(path)


def _wav(path, x):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def run(job, pdir, settings, hint=None):
    """Runs the whole chain. Saves everything needed for review into pdir/auto.json."""
    pdir = pathlib.Path(pdir)
    (pdir / "out").mkdir(parents=True, exist_ok=True)
    seg_dir = pdir / "seg"
    seg_dir.mkdir(exist_ok=True)
    t_start = time.time()

    _stage(job, 0)
    pick = pick_topic(hint)
    topic, hook = pick["topic"], pick.get("hook", "")
    _log(job, f"topic: {topic}")

    _stage(job, 1)
    outline = write_outline(topic, hook)
    chapters = outline["chapters"]
    total_target = sum(int(c.get("target_words", 220)) for c in chapters) or 1
    scripts = []  # list of (chapter_idx, scene)
    prev_tail = ""
    for i, ch in enumerate(chapters):
        words = int(ch.get("target_words", 220))
        _log(job, f"  chapter {i + 1}/{len(chapters)}: {ch['title']}")
        scenes = write_chapter(topic, outline, i, prev_tail, words)
        prev_tail = " ".join(scenes[-1]["narration"].split()[-25:])
        scripts += [(i, s) for s in scenes]
    wc = sum(len(s["narration"].split()) for _, s in scripts)
    _log(job, f"script: {wc} words, {len(scripts)} scenes")
    if not FAKE() and wc < 1450:
        _log(job, "  script is short for 8 minutes; extending the shortest chapters")
        for _ in range(2):
            counts = {}
            for ci, s in scripts:
                counts[ci] = counts.get(ci, 0) + len(s["narration"].split())
            ci = min(counts, key=counts.get)
            extra = write_chapter(topic, outline, ci, "", int(counts[ci] * 1.7))
            scripts = [(c, s) for c, s in scripts if c != ci]
            scripts += [(ci, s) for s in extra]
            scripts.sort(key=lambda cs: cs[0])
            wc = sum(len(s["narration"].split()) for _, s in scripts)
            if wc >= 1450:
                break
    if wc > 2450:
        _log(job, "  script is long for 15 minutes; trimming")
        while wc > 2400:
            mid = [k for k, (c, _) in enumerate(scripts) if 0 < c < len(chapters) - 1]
            k = random.Random(1).choice(mid)
            wc -= len(scripts[k][1]["narration"].split())
            scripts.pop(k)
    full_text = "\n".join(s["narration"] for _, s in scripts)

    _stage(job, 2)
    flags = factcheck(topic, full_text)
    _log(job, f"  {len(flags)} statements flagged for your review")

    _stage(job, 3)
    rnd = random.Random(7)
    visuals = [recipes.clean_visual(s.get("visual"), rnd) for _, s in scripts]

    _stage(job, 4)
    voices, words_all, scenes, cursor, starts = [], [], [], 0.0, []
    v = settings
    cfg = {"voice": {"name": v.get("voice", "en-US-AndrewNeural"), "rate": v.get("rate", "+0%"), "pitch": v.get("pitch", "+0Hz")}}
    for i, (ci, s) in enumerate(scripts):
        vis = visuals[i]
        text = s["narration"].strip()
        voice, w = tts.silent(text) if FAKE() else tts.narrate(text, cfg)
        dur = LEAD + len(voice) / tts.SR + GAP
        if vis["type"] == "card":
            dur = max(dur, 3.2)
        if vis["type"] == "map":
            dur = max(dur, 4.5)
        if vis["type"] == "stage":
            sc = recipes.build_stage(vis, dur, rnd, seed=i)
        elif vis["type"] == "map":
            sc = recipes.build_map(vis, dur)
        else:
            sc = recipes.build_card(vis, dur)
        sc["chapter"] = ci
        scenes.append(sc)
        voices.append(voice)
        starts.append(cursor + LEAD)
        words_all += [[x[0], cursor + LEAD + x[1], cursor + LEAD + x[2]] for x in w]
        sc["_t0"] = cursor
        cursor += dur
        if i % 10 == 0:
            _log(job, f"  voiced {i + 1}/{len(scripts)} scenes")
    total = cursor
    _log(job, f"total length {total / 60:.1f} minutes")
    if not FAKE() and not (MIN_S <= total <= MAX_S):
        _log(job, f"  WARNING: outside the 8-15 minute target ({total / 60:.1f} min)")

    _stage(job, 5)
    jobs = []
    for i, sc in enumerate(scenes):
        plain = {k: v_ for k, v_ in sc.items() if not k.startswith("_actors")}
        jobs.append((plain, sc["_t0"], words_all, seg_dir / f"s{i:03d}.mp4", i == 0, i == len(scenes) - 1))
    workers = max(1, min(os.cpu_count() or 2, int(os.environ.get("STUDIO_WORKERS", 4))))
    segs = [None] * len(jobs)
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(render_segment, j): k for k, j in enumerate(jobs)}
        done = 0
        for f in cf.as_completed(futs):
            segs[futs[f]] = f.result()
            done += 1
            if done % 5 == 0 or done == len(jobs):
                el = time.time() - t_start
                _log(job, f"  drew {done}/{len(jobs)} scenes")
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{pathlib.Path(p).resolve()}'\n" for p in segs))
    joined = seg_dir / "joined.mp4"
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)], check=True)

    _stage(job, 6)
    n = int(total * tts.SR)
    voice = np.zeros(n, dtype=np.float32)
    for vv, s0 in zip(voices, starts):
        i0 = int(s0 * tts.SR)
        seg = vv[:max(0, n - i0)]
        voice[i0:i0 + len(seg)] += seg
    if np.abs(voice).max() > 0:
        voice = np.tanh(1.5 * voice / np.abs(voice).max()) * 0.9
    bed = music.bed(total)
    music_tr = bed * 0.17 * (1 - 0.7 * audiolib._envelope(voice)) if np.abs(voice).max() > 0 else bed * 0.3
    events = []
    for sc in scenes:
        events += sfx.scene_events({k: v_ for k, v_ in sc.items() if k != "_actors"}, sc["_t0"])
    fx_tr = sfx.render_sfx(events, total) * 0.55
    mix = voice + music_tr + fx_tr
    mix = mix / max(1e-6, float(np.abs(mix).max())) * 0.92
    wav = seg_dir / "mix.wav"
    _wav(wav, mix)
    final = pdir / "out" / "final.mp4"
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-i", str(joined), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-movflags", "+faststart", "-shortest", str(final)], check=True)
    _log(job, f"  sound effects: {len(events)} cues")

    _stage(job, 7)
    meta = make_metadata(topic, outline, chapters, full_text)
    thumbs = []
    for k, tr in enumerate((meta.get("thumbs") or [])[:3]):
        pth = pdir / "out" / f"thumb{k + 1}.jpg"
        thumb.render(str(tr.get("text", topic))[:40], tr, pth, k)
        thumbs.append(pth.name)
    if not thumbs:
        thumb.render(meta["title"][:30], {}, pdir / "out" / "thumb1.jpg", 0)
        thumbs = ["thumb1.jpg"]

    _stage(job, 8)
    starts_ch = {}
    for sc in scenes:
        starts_ch.setdefault(sc["chapter"], sc["_t0"])

    def stamp(sec):
        sec = int(sec)
        return f"{sec // 60:02d}:{sec % 60:02d}" if sec < 3600 else f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    chap_lines = [f"{stamp(starts_ch.get(i, 0) if i else 0)} {c['title']}" for i, c in enumerate(chapters) if i in starts_ch or i == 0]
    desc = (meta.get("description_intro", "").strip() + "\n\nChapters\n" + "\n".join(chap_lines) +
            "\n\nNew history stories every week: subscribe so you don't miss the next one.\n\n"
            "Made with AI-assisted stick-figure illustrations and a synthetic narrator.\n\n" + " ".join(meta.get("hashtags", [])[:3]))
    package = {"topic": topic, "hook": hook, "title": meta["title"][:100], "title_options": [t for t in (meta.get("title_options") or [meta["title"]])][:6], "description": desc, "tags": meta.get("tags", [])[:15],
               "pinned_comment": meta.get("pinned_comment", ""), "thumbs": thumbs, "chosen_thumb": thumbs[0], "flags": flags,
               "chapters": [{"title": c["title"], "start": stamp(starts_ch.get(i, 0))} for i, c in enumerate(chapters)],
               "minutes": round(total / 60, 1), "words": wc, "script": full_text, "length_ok": MIN_S <= total <= MAX_S,
               "made_in_min": round((time.time() - t_start) / 60, 1)}
    (pdir / "auto.json").write_text(json.dumps(package, indent=1))
    for f in seg_dir.glob("*"):
        f.unlink()
    seg_dir.rmdir()
    remember_topic(topic)
    _log(job, f"ready for review ({package['minutes']} min video, built in {package['made_in_min']} min)")
    return package
