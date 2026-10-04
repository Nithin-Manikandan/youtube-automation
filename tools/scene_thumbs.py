import json, sys, pathlib
sys.path.insert(0, ".")
from studio import stills
T = json.load(open("tools/scene_thumbs.json"))
pathlib.Path("thumbs_out").mkdir(exist_ok=True)
for i, t in enumerate(T):
    for v in range(t.get("n", 1)):
        r = stills.epic_thumbnail(t["text"], t["scene"], f"thumbs_out/t{i + 1}_{v + 1}.jpg", arrow_to=t.get("arrow"))
        print("thumb", i + 1, v + 1, "ok" if r else "FAILED")
