"""The planning steps (your four NotebookLM prompts) run through the free Gemini API instead."""
import json
import os
import re
import time

import requests

from pipeline import script

MODELS = script.GEMINI_MODELS


def gemini_text(prompt, urls=None, json_out=False, temperature=0.7):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Add your free Gemini API key in Settings first (aistudio.google.com/apikey).")
    parts = [{"file_data": {"file_uri": u}} for u in (urls or [])] + [{"text": prompt}]
    cfg = {"temperature": temperature}
    if json_out:
        cfg["responseMimeType"] = "application/json"
    last = None
    for rnd in range(3):
        for model in MODELS:
            try:
                r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
                                  json={"contents": [{"parts": parts}], "generationConfig": cfg}, timeout=240)
            except requests.RequestException as e:
                last = str(e)
                continue
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            last = f"{r.status_code} {r.text[:300]}"
            if r.status_code == 400 and urls:
                raise RuntimeError("Gemini could not read those YouTube links (" + last[:200] + "). "
                                   "Paste the formula text yourself instead.")
        time.sleep(10 * (rnd + 1))
    raise RuntimeError(f"Gemini failed: {last}")


def _json(text):
    raw = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    try:                                                    # models sometimes emit unescaped quotes, trailing commas or cut-off output
        import json_repair
        out = json_repair.loads(raw)
        if out not in ("", None):
            return out
    except ImportError:
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", raw.replace("\u201c", '"').replace("\u201d", '"'))
    return json.loads(fixed)


# The four prompts from the video's workflow, verbatim. The only additions are the niche, the source
# videos (which NotebookLM would have held as sources) and a request for JSON so the app can use the output.
FORMULA_PROMPT = ("Analyse all the videos in this notebook and extract the complete formula behind this channel. Break it down "
                  "into: the exact target audience and what they want; the topics that repeat across the top performers; the "
                  "tone and energy; how the hooks open and what makes them work; and the structural shape every script follows "
                  "from opening to close. Be specific and concrete \u2014 name the actual patterns, not general observations. "
                  "Also describe the visual style of the channel (palette, mood, illustration or footage style, pacing) so it "
                  "can be reused later.")

NAMES_PROMPT = ("Based on this formula, give me 10 channel name ideas that would fit this niche and this exact style. For each "
                "one, add a short line explaining why it fits.")

SCRIPT_PROMPT = ("Using the formula you extracted from this channel, write a complete voiceover script for this video idea. "
                 "Match the hook style, the tone, the pacing, and the script structure exactly as you decoded them. Output "
                 "continuous voiceover prose only \u2014 no chapter labels, no scene directions, no production notes. Do not copy "
                 "any wording or specific examples from the source channel; the structure is what we're using, the content must "
                 "be original.")

VISUAL_PROMPT = ("You are now a visual director for this channel. Using the script and the visual style you decoded from the "
                 "source channel, first output character and subject consistency prompts for every recurring figure, object, or "
                 "setting \u2014 tag each one in capitals. Then output a scene table with these columns: Scene #, Voiceover (that "
                 "matches the tone and length of each scene), Image Prompt, Animation Prompt. Keep the palette, mood, and "
                 "illustration style consistent across every prompt, matching the style cues from the source channel. No text, "
                 "no logos, no watermark, and never a real identifiable person.")


def decode_formula(urls, niche=""):
    return gemini_text(FORMULA_PROMPT + (f" The channel's niche is: {niche}." if niche else ""), urls=urls)


def ideas(formula, niche=""):
    p = f"""{NAMES_PROMPT}
Also brainstorm 10 different video ideas for the channel, built on the same formula (niche: {niche or 'as in the formula'}).
Return JSON: {{"channel_names": [{{"name": "", "why": ""}}], "ideas": ["..."]}}

FORMULA:
{formula}"""
    return _json(gemini_text(p, json_out=True, temperature=0.95))


def plan_video(formula, idea, minutes=3, vertical=False, niche=""):
    scenes = max(6, int(minutes * 60 / 8))  # one 8-second Flow clip per scene
    prompt = f"""FORMULA (extracted from the source channel):
{formula}

NICHE: {niche or 'as in the formula'}
VIDEO IDEA: {idea}
TARGET: about {minutes} minutes, {scenes} scenes; each scene is one 8-second AI video clip so each scene's voiceover is 15 to 25 words.
ORIENTATION: {'vertical 9:16 short' if vertical else 'horizontal 16:9 long form'}
Historical content must be accurate; do not invent facts, quotes or dates.

PART 1. {SCRIPT_PROMPT}

PART 2. {VISUAL_PROMPT}
The scene table's Voiceover column must split the PART 1 script, in order, without changing it.

Return JSON only (this carries both parts):
{{"title": "<=70 chars, curiosity-driven, honest", "hook_text": "<=6 words", "description": "2-3 sentences plus hashtags", "tags": ["..."],
"characters": [{{"tag": "NAME", "prompt": "full consistent description"}}],
"scenes": [{{"voiceover": "...", "image_prompt": "...", "anim_prompt": "..."}}]}}"""
    data = _json(gemini_text(prompt, json_out=True, temperature=0.8))
    if len(data.get("scenes", [])) < 3:
        raise RuntimeError("The plan came back without enough scenes; try again.")
    return data
