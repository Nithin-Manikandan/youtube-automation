"""Cloudflare Workers AI (free tier) image test: FLUX schnell, a few style variants of one scene."""
import base64, json, os, pathlib, time, urllib.request, urllib.error
acct, tok = os.environ["CF_ACCOUNT_ID"], os.environ["CF_API_TOKEN"]
out = pathlib.Path("cftest"); out.mkdir(exist_ok=True)
STYLES = {
 "A": "hand-drawn 2D webcomic illustration, simple white round-headed stick figure characters with messy brown hair and tiny dot eyes, thin black ink outlines, rough brown tunics, warm muted peach and sepia colour palette, soft paper grain, flat colours, wide shot, no text",
 "B": "simple cartoon drawing in the style of the Ink Explainer YouTube channel, stick figure people with round white heads, black dot eyes, messy brown hair, brown fur tunics, thin sketchy ink lines, flat pastel colors, hand-drawn grass and hills, cozy warm light, wide shot, no text",
}
SCENE = "a medieval London street at dawn, a worker standing knee-deep in a muddy pit holding a wooden bucket, a second man pinching his nose and a speech bubble with a clothespin icon"
for k, st in STYLES.items():
    t = time.time()
    body = {"prompt": f"{st}. {SCENE}", "steps": 6}
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/black-forest-labs/flux-1-schnell", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
        (out / f"cf_{k}.jpg").write_bytes(base64.b64decode(d["result"]["image"])); print("OK", k, round(time.time() - t, 1), "s")
    except urllib.error.HTTPError as e:
        print("ERR", k, e.code, e.read()[:300].decode(errors="replace"))
