"""One button: pick an engaging history topic -> script -> fact-check -> stickman scenes -> voice, music, sound effects
-> video -> thumbnails -> metadata, then stop and wait for you to approve."""
import concurrent.futures as cf
import pickle
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
from . import ai, moods, music, recipes, score, sfx, stick, thumb
from . import voice as narrator
from .doccap import DocCaptions

W, H, FPS = 1280, 720, 30
GAP, LEAD = 0.4, 0.12
MIN_S, MAX_S = 8 * 60, 15 * 60
FAKE = lambda: bool(os.environ.get("STUDIO_FAKE"))
HERE = pathlib.Path(__file__).parent
STAGES = ["Picking a topic", "Writing the script", "Fact-checking", "Planning visuals", "Recording the voiceover",
          "Mixing music and sound effects", "Thumbnails and publishing package", "Drawing the scenes", "Putting it together"]

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


def _unwrap(o, key):
    """Gemini sometimes returns a bare list, or wraps the list under another key. Normalise to {key: list}."""
    if isinstance(o, list):
        return {key: o}
    if isinstance(o, dict):
        if key in o:
            return o
        for v in o.values():
            if isinstance(v, list) and v and isinstance(v[0], (dict, str)):
                return {key: v}
    return {key: []}


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
    c = [x for x in _unwrap(_llm(prompt, 1.0), "candidates")["candidates"] if isinstance(x, dict) and x.get("topic")]
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
    if isinstance(o, dict) and "working_title" not in o and "chapters" not in o:
        o = _unwrap(o, "chapters")
    o = o if isinstance(o, dict) else {"chapters": o}
    o["chapters"] = [c for c in _unwrap(o, "chapters")["chapters"] if isinstance(c, dict) and c.get("title")]
    if len(o["chapters"]) < 4:
        raise RuntimeError("The outline came back too short.")
    return o


def write_chapter(topic, outline, idx, prev_tail, words):
    ch = outline["chapters"][idx]
    if FAKE():
        return _fake_chapter(idx, ch["title"])
    words = int(words * 1.4)                      # the model reliably delivers about 70% of the length it is asked for
    n_scenes = max(5, round(words / 26))
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

ERA RULE: match the drawing to the period. For stories after about 1800 use the modern roles (sailor, captain, officer, scientist, president, worker,
pilot, modern_soldier, spy, reporter) and modern objects and backgrounds (submarine, warship, missile, plane, building, hatch, pipes, gauge, underwater,
submarine_interior, city_modern, sea). Never give modern characters swords, spears, shields or castles; use those only for ancient and medieval stories.
For natural disasters and wars use the matching scenery: volcano, ash_cloud, wave, fire, smoke, house, explosion with the volcanic or ashen background and the ashfall, embers, flash, shake and rain effects.
Use 2-3 actors in most stage scenes so the screen feels alive; show the actual subject of each line (the volcano erupting, the wave hitting, the ship sinking) rather than a generic scene.

{VOCAB}

Also direct the narrator like a great documentary reader. For every scene give a "mood" for the music ("calm", "tense", "epic", "sad", "triumph" or "mystery";
change it only when the story's feeling really changes, at most once every 15-20 seconds) and a "delivery": the narrator's "emotion" ("neutral", "warm",
"tense", "somber", "dramatic", "excited", "urgent" or "awed"), 1-3 "emphasis" words to lean on (names, numbers, the turning-point word), and an optional
"pause_before_ms" (0-700) for dramatic beats (a pause before a reveal). Vary the delivery across scenes so the reading never sounds flat.

Return JSON only: {{"scenes": [{{"narration": "...", "mood": "calm", "delivery": {{"emotion": "warm", "emphasis": ["thousand"], "pause_before_ms": 0}}, "visual": {{"type": "stage", "background": "city_day", "actors": [{{"role": "emperor", "color": "purple", "pos": "center", "action": "talk", "emotion": "worried"}}], "objects": ["column"], "effects": [], "title": "", "camera": "push_in"}}}}]}}"""
    for attempt in range(3):
        o = _unwrap(_llm(prompt, 0.8), "scenes")
        sc = [s for s in o["scenes"] if isinstance(s, dict) and str(s.get("narration", "")).strip()]
        if len(sc) >= 3:
            return sc
    raise RuntimeError(f"Chapter {idx + 1} came back with too few scenes after 3 tries.")


def factcheck(topic, text):
    if FAKE():
        return [{"claim": "(fake mode) example flag", "issue": "none", "fix": "", "severity": "low"}]
    prompt = f"""You are a strict history fact-checker. Below is the narration of a YouTube video about: {topic}.
List every statement that is likely WRONG, exaggerated, disputed among historians, or not verifiable (dates, numbers, names, causes, quotes).
Be specific and honest; if a claim is fine do not list it. Return JSON: {{"flags": [{{"claim": "quote the sentence", "issue": "what is wrong or uncertain", "fix": "safer wording", "severity": "high|medium|low"}}]}}

NARRATION:
{text}"""
    return [f for f in _unwrap(_llm(prompt, 0.2), "flags")["flags"] if isinstance(f, dict)]


def make_metadata(topic, outline, chapters, text):
    if FAKE():
        return {"title": "How Rome Really Fell in 476 AD", "title_options": ["How Rome Really Fell in 476 AD", "Rome Didn't Fall in a Day. It Fell in 5 Steps", "The Empire That Ended Without a Fight"],
                "description_intro": "The real story of how the Western Roman Empire ended.",
                "tags": ["rome", "history", "fall of rome", "roman empire", "476 ad"], "hashtags": ["#history", "#rome", "#ancienthistory"],
                "pinned_comment": "What surprised you most? Tell me below.",
                "thumbs": [{"text": "ROME FELL", "mood": "fire", "role": "emperor", "color": "purple", "action": "scared", "emotion": "shock", "badge": "476 AD", "concept": "looming", "enemy_role": "warrior"},
                           {"text": "WHY IT COLLAPSED", "mood": "blood", "role": "king", "color": "purple", "action": "shrug", "emotion": "worried", "badge": "476", "concept": "ruin", "objects": ["castle", "tower", "column"]},
                           {"text": "NO ONE NOTICED", "mood": "ice", "role": "soldier", "color": "blue", "action": "sword_up", "emotion": "angry", "concept": "versus", "enemy_role": "warrior", "enemy_color": "red"}]}
    prompt = f"""Create the YouTube click package for a history video. It must earn the click, honestly.
TOPIC: {topic}
CHAPTERS: {json.dumps([c['title'] for c in chapters])}
NARRATION (for accuracy and keywords):
{text[:6000]}

TITLE RULES (from CTR research): 40-60 characters; front-load the main keyword; create a curiosity gap (name the problem or mystery, withhold the answer);
use one of: a specific number or date, a contrarian angle ("Everything you know about X is wrong" only if true), a time marker ("in one day"), or a strong
emotional word. No ALL CAPS, no lies, nothing the video does not deliver. Write 5 different titles using different patterns and mark the best as "title".

THUMBNAIL RULES: 2-3 words maximum, huge and readable on a phone; the text must ADD to the title, not repeat it (tease the twist or stakes);
one clear focal character with a strong emotion; scale contrast (a small hero against something huge) is proven to work; high contrast; an optional
date badge (like "476 AD"). Make 3 different concepts, one per layout: "looming" (hero vs a giant menacing silhouette and an army), "ruin" (a burning
skyline of the story's setting), "versus" (two sides clashing, ONLY if the story really has two opposing sides such as a war, duel or rivalry; for disasters, mysteries and discoveries use "looming" or "ruin" instead). Different text, mood, character and emotion in each.

Return JSON:
{{"title": "best title",
 "title_options": ["5 titles"],
 "description_intro": "2-3 sentences: a hook then what the viewer will learn, with natural search keywords (this is the part shown before 'show more')",
 "tags": ["12-15 search tags, most important first, mix of broad and specific"],
 "hashtags": ["#three", "#relevant", "#hashtags"],
 "pinned_comment": "a question that sparks comments",
 "thumbs": [{{"text": "2-3 WORDS", "mood": "fire|ice|gold|storm|blood|night", "role": "one of {sorted(recipes.ROLES)}", "color": "one of {sorted(recipes.COLORS)}", "action": "scared|point|sword_up|proud|shrug|cheer|slump", "emotion": "shock|angry|worried|sad|smile", "concept": "looming|ruin|versus", "backdrop": "for looming: the giant silhouette behind the hero, one of {sorted(recipes.OBJECTS - {'cloud', 'torch'})} that matches the story, or figure", "army": "true only for ancient or medieval stories, false for modern ones", "enemy_role": "role of the opposing side", "enemy_color": "colour name", "badge": "optional short date like 476 AD or empty", "objects": ["1-2 of {sorted(recipes.OBJECTS - {'cloud', 'torch'})} matching the story setting"]}}]}}"""
    m = _llm(prompt, 0.7)
    if isinstance(m, list):
        m = next((x for x in m if isinstance(x, dict)), {})
    if not m.get("title"):
        raise RuntimeError("The package came back without a title.")
    return m


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
    """x: (channels, n) float array."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(x.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(tts.SR)
        wf.writeframes((np.clip(x.T, -1, 1) * 32767).astype(np.int16).tobytes())


def _clean_title(t):
    """YouTube chapter names should read like titles: drop 'Chapter 3:' and 'Cold Open:' style prefixes."""
    t = re.sub(r"^\s*(cold open|prologue|chapter\s*\d+|part\s*\d+)\s*[:\-\u2013\u2014.]\s*", "", str(t), flags=re.I).strip()
    return t[:1].upper() + t[1:] if t else "Chapter"


def _stamp(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}" if sec < 3600 else f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def plan(job, pdir, settings, hint=None):
    """Phases 1-6: topic, script, fact-check, voice, music/sfx mix, thumbnails, package. Saves plan.pkl + mix.flac."""
    pdir = pathlib.Path(pdir)
    (pdir / "out").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    _stage(job, 0)
    pick = pick_topic(hint)
    topic, hook = pick["topic"], pick.get("hook", "")
    _log(job, f"topic: {topic}")

    _stage(job, 1)
    outline = write_outline(topic, hook)
    chapters = outline["chapters"]
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
    if not FAKE() and wc < 1750:
        _log(job, "  script is short for 8 minutes; extending the shortest chapters")
        for _ in range(3):
            counts = {}
            for ci, s in scripts:
                counts[ci] = counts.get(ci, 0) + len(s["narration"].split())
            ci = min(counts, key=counts.get)
            extra = write_chapter(topic, outline, ci, "", int(counts[ci] * 1.7))
            scripts = [(c, s) for c, s in scripts if c != ci]
            scripts += [(ci, s) for s in extra]
            scripts.sort(key=lambda cs: cs[0])
            wc = sum(len(s["narration"].split()) for _, s in scripts)
            if wc >= 1750:
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
    scene_moods = []
    cur = "calm"
    for vis, (_, s) in zip(visuals, scripts):
        m = moods.scene_mood(vis, s.get("mood"))
        cur = m or cur
        scene_moods.append(cur)              # maps and date cards keep the previous mood

    _stage(job, 4)
    v = settings
    cfg = {"voice": {"name": v.get("voice", "en-US-AndrewMultilingualNeural"), "rate": v.get("rate", "+5%"), "pitch": v.get("pitch", "+0Hz")}}

    def _speak(i):
        text = scripts[i][1]["narration"].strip()
        if FAKE():
            return tts.silent(text)
        return narrator.speak(text, cfg, scripts[i][1].get("delivery"), scene_moods[i])
    workers = int(os.environ.get("STUDIO_TTS_WORKERS", 6))
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:       # sentences are independent: record them in parallel
        spoken = list(ex.map(_speak, range(len(scripts))))
    _log(job, f"  voiced {len(scripts)} scenes")
    voices, words_all, scenes, cursor, starts = [], [], [], 0.0, []
    for i, (ci, s) in enumerate(scripts):
        vis = visuals[i]
        vx, w = spoken[i]
        dur = LEAD + len(vx) / tts.SR + GAP
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
        sc["_mood"] = scene_moods[i]
        scenes.append(sc)
        voices.append(vx)
        starts.append(cursor + LEAD)
        words_all += [[x[0], cursor + LEAD + x[1], cursor + LEAD + x[2], k == 0] for k, x in enumerate(w)]
        sc["_t0"] = cursor
        cursor += dur
    total = cursor
    _log(job, f"total length {total / 60:.1f} minutes")
    if not FAKE() and not (MIN_S <= total <= MAX_S):
        _log(job, f"  WARNING: outside the 8-15 minute target ({total / 60:.1f} min)")

    _stage(job, 5)
    n = int(total * tts.SR)
    voice = np.zeros(n, dtype=np.float32)
    for vv, s0 in zip(voices, starts):
        i0 = int(s0 * tts.SR)
        seg = vv[:max(0, n - i0)]
        voice[i0:i0 + len(seg)] += seg
    if np.abs(voice).max() > 0:
        voice = np.tanh(1.5 * voice / np.abs(voice).max()) * 0.9
    env = audiolib._envelope(voice) if np.abs(voice).max() > 0 else np.zeros(n, dtype=np.float32)
    mood_runs = moods.runs([sc["_mood"] for sc in scenes], [sc["duration"] for sc in scenes])
    _log(job, "  music moods: " + " > ".join(f"{m} {e - s0:.0f}s" for s0, e, m in mood_runs))
    bed = score.build(mood_runs, total)
    music_l = music_r = bed * 1.0                               # centred: delays on bass cancel on phone speakers
    duck = (1 - 0.7 * env) if np.abs(voice).max() > 0 else 1.0
    events, prev = [], None
    for sc in scenes:
        plain = {k: v_ for k, v_ in sc.items() if k != "_actors"}
        events += sfx.scene_events(plain, sc["_t0"], prev)
        prev = plain
    fx_tr = sfx.render_sfx(events, total)
    fx_duck = 1 - 0.3 * env
    mix = np.stack([voice + music_l * duck + fx_tr[0] * 0.75 * fx_duck, voice + music_r * duck + fx_tr[1] * 0.75 * fx_duck])
    mix = mix / max(1e-6, float(np.abs(mix).max())) * 0.92
    wav = pdir / "mix.wav"
    _wav(wav, mix)
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "flac", str(pdir / "mix.flac")], check=True)
    wav.unlink()
    _log(job, f"  sound effects: {len(events)} cues")

    _stage(job, 6)
    meta = make_metadata(topic, outline, chapters, full_text)
    thumbs = []
    for k, tr in enumerate((meta.get("thumbs") or [])[:3]):
        pth = pdir / "out" / f"thumb{k + 1}.jpg"
        thumb.render(str(tr.get("text", topic))[:40], tr, pth, k)
        thumbs.append(pth.name)
    if not thumbs:
        thumb.render(meta["title"][:30], {}, pdir / "out" / "thumb1.jpg", 0)
        thumbs = ["thumb1.jpg"]
    starts_ch = {}
    for sc in scenes:
        starts_ch.setdefault(sc["chapter"], sc["_t0"])
    chap_lines = [f"{_stamp(starts_ch.get(i, 0) if i else 0)} {_clean_title(c['title'])}" for i, c in enumerate(chapters) if i in starts_ch or i == 0]
    desc = (meta.get("description_intro", "").strip() + "\n\nChapters\n" + "\n".join(chap_lines) +
            "\n\nNew history stories every week: subscribe so you don't miss the next one.\n\n"
            "Made with AI-assisted stick-figure illustrations and a synthetic narrator.\n\n" + " ".join(meta.get("hashtags", [])[:3]))
    package = {"topic": topic, "hook": hook, "title": meta["title"][:100], "title_options": [t for t in (meta.get("title_options") or [meta["title"]])][:6],
               "description": desc, "tags": meta.get("tags", [])[:15], "pinned_comment": meta.get("pinned_comment", ""), "thumbs": thumbs,
               "chosen_thumb": thumbs[0], "flags": flags,
               "chapters": [{"title": _clean_title(c["title"]), "start": _stamp(starts_ch.get(i, 0))} for i, c in enumerate(chapters)],
               "minutes": round(total / 60, 1), "words": wc, "script": full_text, "length_ok": MIN_S <= total <= MAX_S}
    plain_scenes = [{k: v_ for k, v_ in sc.items() if k != "_actors"} for sc in scenes]
    with open(pdir / "plan.pkl", "wb") as f:
        pickle.dump({"scenes": plain_scenes, "words": words_all, "total": total, "package": package, "t_start": t_start}, f)
    _log(job, f"plan ready: {len(scenes)} scenes, {package['minutes']} min")
    return package


def _load_plan(pdir):
    with open(pathlib.Path(pdir) / "plan.pkl", "rb") as f:
        return pickle.load(f)


def render_shard(job, pdir, shard=0, shards=1):
    """Phase 7: draw the scenes whose index % shards == shard (other machines draw the rest)."""
    pdir = pathlib.Path(pdir)
    P = _load_plan(pdir)
    seg_dir = pdir / "seg"
    seg_dir.mkdir(exist_ok=True)
    scenes, words_all = P["scenes"], P["words"]
    _stage(job, 7)
    mine = [i for i in range(len(scenes)) if i % shards == shard]
    jobs = [(scenes[i], scenes[i]["_t0"], words_all, seg_dir / f"s{i:03d}.mp4", i == 0, i == len(scenes) - 1) for i in mine]
    workers = max(1, min(os.cpu_count() or 2, int(os.environ.get("STUDIO_WORKERS", 4))))
    _log(job, f"  drawing {len(jobs)} of {len(scenes)} scenes (shard {shard + 1}/{shards}) with {workers} workers")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(render_segment, j) for j in jobs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            if k % 5 == 0 or k == len(jobs):
                _log(job, f"  drew {k}/{len(jobs)} scenes")


def assemble(job, pdir):
    """Phase 8: join the drawn scenes, add the mixed audio, finish the package."""
    pdir = pathlib.Path(pdir)
    P = _load_plan(pdir)
    seg_dir = pdir / "seg"
    _stage(job, 8)
    n = len(P["scenes"])
    segs = [seg_dir / f"s{i:03d}.mp4" for i in range(n)]
    missing = [p.name for p in segs if not p.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} scene files are missing, e.g. {missing[:3]}")
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    joined = seg_dir / "joined.mp4"
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)], check=True)
    final = pdir / "out" / "final.mp4"
    subprocess.run([ffmpeg.exe(), "-y", "-loglevel", "error", "-i", str(joined), "-i", str(pdir / "mix.flac"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-af", "loudnorm=I=-16:TP=-2.0:LRA=11", "-movflags", "+faststart", "-shortest", str(final)], check=True)
    package = P["package"]
    package["made_in_min"] = round((time.time() - P["t_start"]) / 60, 1)
    (pdir / "auto.json").write_text(json.dumps(package, indent=1))
    for f in seg_dir.glob("*"):
        f.unlink()
    seg_dir.rmdir()
    (pdir / "plan.pkl").unlink()
    (pdir / "mix.flac").unlink()
    remember_topic(package["topic"])
    _log(job, f"ready for review ({package['minutes']} min video, built in {package['made_in_min']} min)")
    return package


def run(job, pdir, settings, hint=None):
    """Everything on this machine, start to finish (used by the local app). Saves the review package to pdir/auto.json."""
    plan(job, pdir, settings, hint)
    render_shard(job, pdir, 0, 1)
    return assemble(job, pdir)
