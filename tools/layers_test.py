import json, sys
sys.path.insert(0, ".")
from studio import stills
S = json.load(open("tools/layers_scenes.json"))
for i, s in enumerate(S):
    r = stills.make_layers(s["bg"], s["char"], "layers_out", f"s{i}")
    print("layers", i, "ok" if r else "FAILED")
