"""Redraw a few chosen scene pictures and crop the signature strip off every picture; rewrites the plan's stills folder in place."""
import json, os, pathlib, sys
sys.path.insert(0, ".")
from PIL import Image
from studio import stills
d = pathlib.Path("output/auto/stills")
MODE = open("tools/mode.txt").read().strip()
FIX = {} if MODE == "thumbs" else json.load(open("tools/fix_stills.json"))
for k, prompt in FIX.items():
    img = stills.generate(f"{stills.STYLE}. {prompt}")
    if img is None:
        print("FAILED", k); continue
    w, h = img.size
    img = img.crop((0, 0, w, int(h * 0.93))).resize((w, w), Image.LANCZOS)
    img.save(d / f"s{int(k):03d}.jpg", quality=93); print("redrawn", k)
done = set(f"s{int(k):03d}.jpg" for k in FIX)
for f in (sorted(d.glob("s*.jpg")) if MODE == "all" else []):
    if f.name in done:
        continue
    im = Image.open(f).convert("RGB"); w, h = im.size
    im.crop((0, 0, w, int(h * 0.89))).resize((w, w), Image.LANCZOS).save(f, quality=93)
print("cropped", len(list(d.glob('s*.jpg'))))

if MODE != "thumbs":
    sys.exit(0)
# thumbnails in the same illustrated style as the video
out = pathlib.Path("output/auto/out"); out.mkdir(parents=True, exist_ok=True)
T = json.load(open("tools/thumbs.json"))
for k, t in enumerate(T):
    r = stills.thumbnail(t["text"], t["art"], out / f"thumb{k + 1}.jpg", k)
    print("thumb", k + 1, "ok" if r else "FAILED")
