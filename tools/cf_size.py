import base64, io, json, os, urllib.request, urllib.error
from PIL import Image
acct, tok = os.environ["CF_ACCOUNT_ID_2"], os.environ["CF_API_TOKEN_2"]
P = ("hand-drawn webcomic cartoon, stick figure people with round white heads, thin black ink lines, flat warm colours, wide landscape scene: a knight in full shiny plate armour sitting on a wooden toilet "
     "in the middle of a medieval castle courtyard with a shocked face, sweat drops, a tiny squire next to him, detailed stone castle walls and blue sky, the top fifth of the picture is empty sky")
def run(model, body, name):
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{model}", json.dumps(body).encode(), {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=150); raw = r.read()
        try:
            d = json.loads(raw); raw = base64.b64decode(d["result"]["image"])
        except Exception:
            pass
        im = Image.open(io.BytesIO(raw)); print("OK", model, im.size); im.convert("RGB").save(name)
    except urllib.error.HTTPError as e:
        print("ERR", model, e.code, e.read()[:260])
run("@cf/bytedance/stable-diffusion-xl-lightning", {"prompt": P, "width": 1280, "height": 720, "num_steps": 8, "guidance": 2}, "sz_lightning.jpg")
run("@cf/stabilityai/stable-diffusion-xl-base-1.0", {"prompt": P, "width": 1280, "height": 720, "num_steps": 20}, "sz_sdxl.jpg")
run("@cf/black-forest-labs/flux-2-klein-4b", {"prompt": P, "width": 1280, "height": 720}, "sz_klein.jpg")
run("@cf/black-forest-labs/flux-2-dev", {"prompt": P, "width": 1280, "height": 720}, "sz_flux2.jpg")
