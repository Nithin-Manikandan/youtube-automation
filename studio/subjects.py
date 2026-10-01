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
    (r"\b(?:missiles?|rockets?|icbms?)\b.{0,40}\b(?:launch\w*|fired|fire|lift\w* off)\b|\blaunch\w* (?:the |a |its |their )?(?:nuclear )?(?:missiles?|rockets?)|\blift\w* off\b|\bsilos?\b|\brockets?\b", ["missile"], [], None, {"launch": True}),
    (r"\b(planes?|aircraft|bombers?|airplane|jets?)\b", ["plane"], [], None, {}),
    (r"\b(volcano|volcanic|eruption|erupt\w*|lava|magma|crater|caldera)\b", ["volcano"], ["embers"], "volcanic", {}),
    (r"\b(ash|ashfall|pumice|soot)\b", ["ash_cloud"], ["ashfall"], "ashen", {}),
    (r"\b(tsunami|tidal wave|flood\w*|(?:ocean|sea|giant|huge|massive|towering|enormous) waves?|waves? (?:crash\w*|hit|struck|rose|rolled|swept|slammed))\b", ["wave"], [], "sea", {}),
    (r"\b(fires?|burn\w*|blaze|flames?|inferno|ablaze)\b", ["fire", "smoke"], ["embers"], None, {}),
    (r"\b(explo\w+|blast\w*|detonat\w+|bomb\w*|shell\w*)\b", ["explosion"], ["shake", "flash"], None, {}),
    (r"\b(earthquakes?|tremors?|quakes?|shaking)\b", [], ["shake"], None, {}),
    (r"\b(storms?|thunder|lightning|hurricane|typhoon)\b", [], ["rain", "flash"], "storm", {}),
    (r"\b(armies|army|legions?|troops|hordes?|marched|marching|invad\w+|advanced on|stormed)\b|(?<!depth )(?<!depth-)\bcharg(?:ed|ing) (?:at|into|across|forward|toward|towards|the|down|up|through|over)\b|\bthe charge\b(?! of)", ["crowd"], ["dust"], None, {}),
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
    # objects that depict an event or vehicle are only drawn when the narration is literally about them
    bound = {"missile", "torpedo", "depth_charge", "explosion", "volcano", "wave", "fire", "plane", "warship", "submarine", "pyramid", "castle", "ash_cloud", "crowd"}
    said = {o for pat, add_o, _, _, _ in RULES if re.search(pat, text) for o in add_o}
    objs = [o for o in objs if o not in bound or o in said]
    inside = re.search(r"\b(inside|aboard|hull|control room|crew|compartment|cramped|bunk|captain|commander|officers?|shouted|declared|declaration|consent|vot(?:e|ed|es)|refus\w+|argued|orders?|ordered|authoriz\w+|veto|beside him|protocol|sailors|men)\b", text)
    if any(o in UNDERWATER_OBJECTS for o in objs):
        hinted_bg = "submarine_interior" if inside and not re.search(r"depth[- ]?charge|sonar|hunted|surfaced|dove|dived|diving|surface ships|destroyers? (?:closed|dropped|hunted)", text) else "underwater"
    if hinted_bg in ("underwater", "submarine_interior"):
        bg = hinted_bg                                          # a submarine scene is never on a battlefield
    elif hinted_bg and bg in (None, "city_day", "countryside", "palace", "desert", "forest", "snow", "city_modern", "sea", "underwater", "submarine_interior", "storm", "night", "ashen", "volcanic"):
        # take the narration's setting unless the AI already chose something equally specific
        if bg != hinted_bg and not (bg in ("underwater", "submarine_interior") and hinted_bg == "sea"):
            bg = hinted_bg
    if bg == "underwater" and inside and not re.search(r"depth[- ]?charge|sonar|torpedo|hunted|surfaced|dove|dived|diving|destroyers?|warships?", text):
        bg = "submarine_interior"                                    # people talking inside the boat, not the boat from outside
    if bg == "submarine_interior":
        objs = [o for o in objs if o not in UNDERWATER_OBJECTS and o not in ("explosion", "fire", "smoke")]       # inside the boat you don't see the boat
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
    if bg == "underwater":
        objs = ["depth_charge" if o == "explosion" else o for o in objs]          # a blast under water is a blue-white burst, not a fireball
        objs = [o for o in objs if o != "fire"]
        if "submarine" not in objs:
            objs.insert(0, "submarine")                                          # never an empty seabed
    if bg != "submarine_interior":                            # interior fittings only make sense inside the boat
        objs = [o for o in objs if o not in ("pipes", "gauge", "hatch")]
    if bg in ("underwater", "submarine_interior"):            # nothing from the surface world belongs down here
        objs = [o for o in objs if o not in ("warship", "ship", "crowd", "plane", "building", "house", "wave", "castle", "pyramid", "volcano")]
    if str(v.get("title", "")).strip().lower().replace(" ", "_") in {"submarine_interior", "underwater", "city_day", "battlefield", "palace", "sea"}:
        v["title"] = ""                                       # a place name is not a headline
    if bg == "underwater":                                    # nobody is standing on the seabed
        v["actors"] = []
    v["objects"] = objs[:3]
    if bg == "submarine_interior" and not v.get("actors"):                      # an interior scene is about the people in it
        v["actors"] = [dict(role="captain", color="", pos="left", action="talk", emotion="angry", facing="", scale=1.0),
                       dict(role="officer", color="", pos="right", action="talk", emotion="worried", facing="", scale=1.0)]
    v["effects"] = fx[:3]
    v["background"] = bg or v.get("background")
    flags["modern"] = bool(modern)
    v["flags"] = flags
    # a sunk-in-thought lone character is boring: if the subject is a vehicle/disaster, let it dominate
    if any(o in v["objects"] for o in ("submarine", "volcano", "wave", "missile", "warship", "plane", "explosion")) and len(v.get("actors", [])) > 1:
        v["actors"] = v["actors"][:1]
    return v
