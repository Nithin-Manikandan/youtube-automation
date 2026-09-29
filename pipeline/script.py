"""Topic + script writing through free LLM tiers (Gemini -> Groq -> OpenRouter)."""
import json
import os
import re
import time

import requests

WORDS_PER_SECOND = 2.7


def _post(url, headers, payload, retries=3):
    err = None
    for i in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()
        err = f"{r.status_code} {r.text[:300]}"
        if r.status_code in (429, 500, 502, 503):
            time.sleep(4 * (i + 1))
            continue
        break
    raise RuntimeError(err)


GEMINI_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]


def _gemini(prompt, temperature):
    key = os.environ["GEMINI_API_KEY"]
    models = [os.environ["GEMINI_MODEL"]] if os.environ.get("GEMINI_MODEL") else GEMINI_MODELS
    last = None
    for rnd in range(4):  # models get retired (404) or overloaded (503): rotate, then wait and go again
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                data = _post(url, {}, {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
                }, retries=1)
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except RuntimeError as e:
                last = e
                if e.args[0][:3] not in ("404", "429", "500", "502", "503"):
                    raise
        time.sleep(15 * (rnd + 1))
    raise last


def _openai_compat(url, key, model, prompt, temperature):
    data = _post(url, {"Authorization": f"Bearer {key}"}, {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    })
    return data["choices"][0]["message"]["content"]


def _groq(prompt, temperature):
    return _openai_compat("https://api.groq.com/openai/v1/chat/completions",
                          os.environ["GROQ_API_KEY"],
                          os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), prompt, temperature)


def _openrouter(prompt, temperature):
    return _openai_compat("https://openrouter.ai/api/v1/chat/completions",
                          os.environ["OPENROUTER_API_KEY"],
                          os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
                          prompt, temperature)


PROVIDERS = [("GEMINI_API_KEY", _gemini), ("GROQ_API_KEY", _groq), ("OPENROUTER_API_KEY", _openrouter)]


def llm_json(prompt, temperature=0.9):
    errors = []
    for env, fn in PROVIDERS:
        if not os.environ.get(env):
            continue
        try:
            text = fn(prompt, temperature)
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            return json.loads(text)
        except Exception as e:  # try the next provider
            errors.append(f"{env}: {e}")
    raise RuntimeError("No LLM provider worked. Set GEMINI_API_KEY (free). Errors: " + " | ".join(errors))


def pick_topic(cfg, history):
    prompt = f"""You run the YouTube Shorts channel "{cfg['channel_name']}" about {cfg['niche']}.
Audience: {cfg['audience']}.
Suggest ONE fresh video idea with a killer hook. Avoid these already-used ideas: {history[-40:]}.
Return JSON: {{"topic": "one sentence premise", "why_it_hooks": "one sentence"}}"""
    return llm_json(prompt, 1.0)["topic"]


def write_script(cfg, topic):
    v = cfg["video"]
    words = int(v["target_seconds"] * WORDS_PER_SECOND)
    n = v["scenes"]
    prompt = f"""You write viral YouTube Shorts for "{cfg['channel_name']}" ({cfg['niche']}). Audience: {cfg['audience']}.

Topic: {topic}

Write a voiceover of about {words} words split into exactly {n} scenes.
Retention rules:
- Scene 1 is a HOOK: a shocking or intriguing first sentence (max 12 words) that makes a viewer stop scrolling. No greetings, no "have you ever".
- Short punchy sentences. Spoken language. Every scene must raise a question or add a twist so nobody leaves.
- Last scene pays off the story and its final sentence loops back to the first line so the video replays.
- No hashtags or stage directions inside narration; plain speakable text only.
{cfg['script'].get('extra_rules', '')}

For every scene also write "image_prompt": one vivid visual description (subject, setting, lighting, camera angle) for an image generator.
Rules for image prompts: describe concrete visible things, never text or words in the image, keep any recurring character/location identical in wording across scenes for consistency, vertical 9:16 framing. Do not include style words, they are added automatically.

Return JSON only:
{{"title": "<=70 chars, curiosity-driven, no clickbait lies",
 "hook_text": "<=6 words shown on screen in the first 2 seconds",
 "description": "2 sentences plus 3 hashtags including #shorts",
 "tags": ["8 short tags"],
 "scenes": [{{"narration": "...", "image_prompt": "..."}}]}}"""
    data = llm_json(prompt, cfg["script"].get("temperature", 0.95))
    scenes = data.get("scenes", [])
    if len(scenes) < 3 or any("narration" not in s or "image_prompt" not in s for s in scenes):
        raise RuntimeError("LLM returned an unusable script")
    return data
