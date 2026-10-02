"""Try Gemini image models on the repo's existing GEMINI_API_KEY and save whatever works."""
import base64, json, os, pathlib, sys, urllib.request, urllib.error
key = os.environ["GEMINI_API_KEY"]
out = pathlib.Path("imgtest"); out.mkdir(exist_ok=True)
PROMPT = ("Hand-drawn 2D webcomic illustration in the style of the Ink Explainer YouTube channel: simple white round-headed stick figure characters with messy brown hair, "
          "tiny dot eyes, thin black ink outlines, wearing rough brown tunics, warm muted peach and sepia palette, soft paper grain. Scene: a medieval London street at dawn, "
          "a worker standing knee-deep in a muddy cesspit holding a wooden bucket, a second man pinching his nose with a speech bubble containing a clothespin icon. Wide 16:9 composition, no text.")
models = ["gemini-2.5-flash-image", "gemini-2.0-flash-preview-image-generation", "gemini-2.5-flash-image-preview", "imagen-4.0-fast-generate-001", "imagen-3.0-generate-002"]
for m in models:
    try:
        if m.startswith("imagen"):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:predict?key={key}"
            body = {"instances": [{"prompt": PROMPT}], "parameters": {"sampleCount": 1, "aspectRatio": "16:9"}}
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
            body = {"contents": [{"parts": [{"text": PROMPT}]}], "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=120))
        b64 = None
        if m.startswith("imagen"):
            b64 = d["predictions"][0]["bytesBase64Encoded"]
        else:
            for p in d["candidates"][0]["content"]["parts"]:
                if "inlineData" in p:
                    b64 = p["inlineData"]["data"]
        if b64:
            (out / f"{m}.png").write_bytes(base64.b64decode(b64)); print("OK", m)
        else:
            print("NOIMG", m, str(d)[:200])
    except urllib.error.HTTPError as e:
        print("ERR", m, e.code, e.read()[:200].decode(errors="replace"))
    except Exception as e:
        print("ERR", m, repr(e)[:200])
