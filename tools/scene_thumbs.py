import json, sys, pathlib
sys.path.insert(0, ".")
from studio import stills
T = json.load(open("tools/scene_thumbs.json"))
pathlib.Path("thumbs_out").mkdir(exist_ok=True)
for i, t in enumerate(T):
    r = stills.scene_thumbnail(t["text"], t["scene"], f"thumbs_out/t{i + 1}.jpg", side=t.get("side", "left"))
    print("thumb", i + 1, "ok" if r else "FAILED")
