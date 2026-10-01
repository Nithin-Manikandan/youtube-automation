"""Subject guard: whatever the narration is about must be on screen, whatever the AI picked.

The AI chooses visuals loosely, so a line about depth charges can end up as a generic ocean with one standing person.
Here we read the narration, add the objects/effects/background it names, and drop anachronisms."""
import re
import statistics

MODERN_ROLES = {"sailor", "captain", "officer", "scientist", "president", "worker", "pilot", "modern_soldier", "spy", "reporter"}
ANCIENT_PROPS_ROLES = {"soldier", "warrior", "knight", "general", "king", "queen", "emperor", "pharaoh", "priest", "rebel", "pirate"}

# (regex, objects to add, effects to add, background hint, extra flags)
RULES = [
    (r"depth[- ]?charge", ["submarine", "depth_charge"], ["bubbles", "shake"], "underwater", {}),
    (r"\b(submarine|u-boat|periscope|b-59|submerged|submariners?)\b", ["submarine"], ["bubbles"], "underwater", {}),
    (r"\btorpedo", ["submarine", "torpedo"], ["bubbles"], "underwater", {}),
    (r"\bsonar|ping\b", ["submarine"], ["sonar", "bubbles"], "underwater", {}),
    (r"\b(warships?|destroyers?|cruiser|battleship|fleet|navy|naval|carrier|flotilla)\b", ["warship"], [], "sea", {}),
    (r"\b(missiles?|rockets?|icbm|launch(?:ed|es|ing)?)\b", ["missile"], [], None, {"launch": True}),
    (r"\b(planes?|aircraft|bombers?|airplane|jets?)\b", ["plane"], [], None, {}),
    (r"\b(volcano|volcanic|eruption|erupt\w*|lava|magma|crater|caldera)\b", ["volcano"], ["embers"], "volcanic", {}),
    (r"\b(ash|ashfall|pumice|soot)\b", ["ash_cloud"], ["ashfall"], "ashen", {}),
    (r"\b(tsunami|tidal wave|waves?|flood\w*|surge)\b", ["wave"], [], "sea", {}),
    (r"\b(fires?|burn\w*|blaze|flames?|inferno|ablaze)\b", ["fire", "smoke"], ["embers"], None, {}),
    (r"\b(explo\w+|blast\w*|detonat\w+|bomb\w*|shell\w*)\b", ["explosion"], ["shake", "flash"], None, {}),
    (r"\b(earthquakes?|tremors?|quakes?|shaking)\b", [], ["shake"], None, {}),
    (r"\b(storms?|thunder|lightning|hurricane|typhoon)\b", [], ["rain", "flash"], "storm", {}),
    (r"\b(castle|fortress|citadel|siege|ramparts?)\b", ["castle"], [], None, {}),
    (r"\b(pyramids?|pharaoh|nile)\b", ["pyramid"], [], "desert", {}),
    (r"\b(towns?|villages?|settlements?|houses?|homes)\b", ["house"], [], None, {}),
    (r"\b(cities|city|skyscraper|metropolis)\b", ["building"], [], None, {}),
]
UNDERWATER_OBJECTS = {"submarine", "depth_charge", "torpedo"}
ANCIENT_OBJECTS = {"castle", "pyramid", "column", "tent", "torch", "pedestal"}
ANCIENT_ROLE_SWAP = {"soldier": "modern_soldier", "warrior": "modern_soldier", "knight": "officer", "general": "officer", "king": "president",
                     "queen": "president", "emperor": "president", "pharaoh": "president", "priest": "scientist", "rebel": "worker", "pirate": "sailor"}


def era_modern(full_text, default=False):
    years = [int(y) for y in re.findall(r"\b([3-9][0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b", full_text)]
    if not years:
        return default
    return statistics.median(years) >= 1800


def enrich(visual, narration, modern):
    """Return the visual with the narration's subjects added. Only stage scenes are touched."""
    if visual.get("type") != "stage":
        return visual
    v = dict(visual)
    text = narration.lower()
    objs = list(v.get("objects", []))
    fx = list(v.get("effects", []))
    bg = v.get("background")
    flags = {}
    hinted_bg = None
    for pat, add_o, add_f, bghint, extra in RULES:
        if re.search(pat, text):
            for o in add_o:
                if o not in objs:
                    objs.append(o)
            for f in add_f:
                if f not in fx:
                    fx.append(f)
            hinted_bg = hinted_bg or bghint
            flags.update(extra)
    inside = re.search(r"\b(inside|aboard|hull|control room|crew|compartment|cramped|bunk)\b", text)
    if any(o in UNDERWATER_OBJECTS for o in objs):
        hinted_bg = "submarine_interior" if inside and not re.search(r"depth[- ]?charge|torpedo|sonar", text) else "underwater"
    if hinted_bg and bg in (None, "city_day", "countryside", "palace", "desert", "forest", "snow", "city_modern", "sea", "underwater", "submarine_interior", "storm", "night", "ashen", "volcanic"):
        # take the narration's setting unless the AI already chose something equally specific
        if bg != hinted_bg and not (bg in ("underwater", "submarine_interior") and hinted_bg == "sea"):
            bg = hinted_bg
    if bg == "submarine_interior":
        objs = [o for o in objs if o not in UNDERWATER_OBJECTS]       # inside the boat you don't see the boat
        fx = [f for f in fx if f != "bubbles"]
    if modern:
        objs = [o for o in objs if o not in ANCIENT_OBJECTS or v.get("background") in ("palace",)]
        for a in v.get("actors", []):
            if a["role"] in ANCIENT_ROLE_SWAP:
                a["role"] = ANCIENT_ROLE_SWAP[a["role"]]
    # sailboats don't belong in modern naval scenes
    if modern and "ship" in objs and ("warship" in objs or "submarine" in objs):
        objs.remove("ship")
    if modern and "ship" in objs:
        objs[objs.index("ship")] = "warship"
    # keep what the narration named first, then the AI's own picks; cap to three objects
    named = [o for o in objs if any(o in r[1] for r in RULES if re.search(r[0], text))]
    rest = [o for o in objs if o not in named]
    objs = named + rest
    if bg == "underwater":                       # no surface ships or people on the seabed
        objs = [o for o in objs if o not in ("warship", "ship")]
        v["actors"] = []
    v["objects"] = objs[:3]
    v["effects"] = fx[:3]
    v["background"] = bg or v.get("background")
    v["flags"] = flags
    # a sunk-in-thought lone character is boring: if the subject is a vehicle/disaster, let it dominate
    if any(o in v["objects"] for o in ("submarine", "volcano", "wave", "missile", "warship", "plane", "explosion")) and len(v.get("actors", [])) > 1:
        v["actors"] = v["actors"][:1]
    return v
