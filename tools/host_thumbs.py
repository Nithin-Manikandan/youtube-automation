import json, sys
sys.path.insert(0, ".")
from studio import stills
T = json.load(open("tools/host_thumbs.json"))
import pathlib
pathlib.Path("thumbs_out").mkdir(exist_ok=True)
for i, t in enumerate(T):
    r = stills.host_thumbnail(t["text"], t["bg"], f"thumbs_out/t{i + 1}.jpg", mood=t.get("mood", "shock"), mouth=t.get("mouth", "D"), pose=t.get("pose", "shock"), hat=t.get("hat"), side=t.get("side", "left"))
    print("thumb", i + 1, "ok" if r else "FAILED")
