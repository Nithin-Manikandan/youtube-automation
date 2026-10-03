import base64, io, json, os, urllib.request, urllib.error
from PIL import Image
acct, tok = os.environ["CF_ACCOUNT_ID_2"], os.environ["CF_API_TOKEN_2"]
P = ("hand-drawn webcomic cartoon, stick figure people with round white heads, thin black ink lines, flat warm colours, wide landscape scene: a knight in full shiny plate armour sitting on a wooden toilet "
     "in the middle of a medieval castle courtyard with a shocked face, sweat drops, a tiny squire next to him, detailed stone castle walls and blue sky, the top fifth of the picture is empty sky")
for w, h in ((1280, 720), (1024, 576)):
    body = {"prompt": P, "steps": 6, "width": w, "height": h}
    req = urllib.request.Request(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/@cf/black-forest-labs/flux-1-schnell", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
        im = Image.open(io.BytesIO(base64.b64decode(d["result"]["image"]))); print("size", w, h, "->", im.size)
        im.save(f"sz_{w}.jpg")
    except urllib.error.HTTPError as e:
        print("ERR", w, e.code, e.read()[:200])
