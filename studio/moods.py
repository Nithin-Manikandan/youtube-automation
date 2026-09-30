"""Decides each scene's mood (drives the score and the narrator's delivery) and keeps changes smooth."""
MOODS = ("calm", "tense", "epic", "sad", "triumph", "mystery")
MIN_RUN = 14.0   # seconds; shorter mood blips are absorbed so the music never flutters


def scene_mood(visual, llm_mood=None):
    m = str(llm_mood or "").lower().strip()
    if m in MOODS:
        return m
    if visual.get("type") in ("map", "card"):
        return None                               # inherit: maps and date cards don't change the feeling
    acts = {a.get("action") for a in visual.get("actors", [])}
    emos = {a.get("emotion") for a in visual.get("actors", [])}
    fx = set(visual.get("effects", []))
    bg = visual.get("background")
    if "fight" in acts or "sparks" in fx:
        return "epic"
    if acts & {"enter_run", "run", "exit_run", "scared"} or "flash" in fx:
        return "tense"
    if acts & {"slump"} or emos & {"sad"}:
        return "sad"
    if acts & {"cheer", "proud"} or emos & {"smile"}:
        return "triumph"
    if bg in ("night", "desert", "snow"):
        return "mystery"
    if bg in ("storm", "battlefield"):
        return "tense"
    return "calm"


def runs(moods, durations):
    """moods: per-scene mood or None; durations: per-scene seconds. Returns contiguous [(start, end, mood)]."""
    cur = "calm"
    per = []
    for m in moods:
        cur = m or cur
        per.append(cur)
    out = []
    t = 0.0
    for m, d in zip(per, durations):
        if out and out[-1][2] == m:
            out[-1][1] = t + d
        else:
            out.append([t, t + d, m])
        t += d
    merged = []                                   # absorb blips shorter than MIN_RUN into the previous mood
    for r in out:
        if merged and (r[1] - r[0]) < MIN_RUN:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < MIN_RUN:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    clean = []
    for r in merged:
        if clean and clean[-1][2] == r[2]:
            clean[-1][1] = r[1]
        else:
            clean.append(r)
    return [tuple(r) for r in clean]
