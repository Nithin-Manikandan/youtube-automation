"""Subject guard: whatever the narration is about must be on screen, whatever the AI picked.

The AI chooses visuals loosely, so a line about depth charges can end up as a generic ocean with one standing person.
Here we read the narration, add the objects/effects/background it names, and drop anachronisms."""
import re
import statistics

EVERYDAY = {"on": False}      # entertainment-history mode: ordinary people, not kings and soldiers, unless the line is about them
MODERN_ROLES = {"sailor", "captain", "officer", "scientist", "president", "worker", "pilot", "modern_soldier", "spy", "reporter"}
ANCIENT_PROPS_ROLES = {"soldier", "warrior", "knight", "general", "king", "queen", "emperor", "pharaoh", "priest", "rebel", "pirate"}

# (regex, objects to add, effects to add, background hint, extra flags)
RULES = [
    (r"\b(lunar surface|surface of the moon|moon'?s surface|moonwalk\w*|tranquility base|walked on the moon|lunar landscape|stepped (?:out )?onto the moon)\b", ["planet"], [], "moon", {}),
    (r"\b(?:oxygen|fuel|hydrogen|cryogenic|propellant|pressure) (?:tanks?|vessels?|cylinders?)|\bcryogenic\b|\b(?:tank|cylinder) (?:blew|ruptured|exploded|burst)\b", ["tank"], [], None, {}),
    (r"\b(parachutes?|chutes?|splashdown|splashed down|descended under)\b", ["parachute"], [], None, {}),
    (r"\b(flags?|banners?|standards?|colou?rs raised)\b", ["flag"], [], None, {}),
    (r"\b(cannons?|artillery|cannonballs?|bombard\w*|broadsides?)\b", ["cannon"], ["shake"], None, {}),
    (r"\b(countdown|ticking|clock|the final (?:seconds|minutes|hour)|minutes (?:left|remaining)|deadline|running out of time)\b", ["clock"], [], None, {}),
    (r"\b(zeppelins?|airships?|dirigibles?|hindenburg|blimps?|graf zeppelin)\b", ["airship"], [], "countryside", {}),
    (r"depth[- ]?charge", ["submarine", "depth_charge"], ["bubbles", "shake"], "underwater", {}),
    (r"\b(submarine(?! (?:caldera|volcano|volcanic|eruption|landslide|earthquake|canyon|crater|vent|shock|blast|collapse))|u-boat|periscope|b-59|submerged (?:in|beneath) the (?:boat|sub)|submariners?)\b", ["submarine"], ["bubbles"], "underwater", {}),
    (r"\btorpedo", ["submarine", "torpedo"], ["bubbles"], "underwater", {}),
    (r"\b(sonar|pings?)\b", ["submarine"], ["sonar", "bubbles"], "underwater", {}),
    (r"\b(titanic|liner|steamships?|steamer|ocean liner|passenger ship|rms|lusitania|carpathia|californian)\b", ["liner"], [], "sea", {}),
    (r"\b(icebergs?|pack ice|ice fields?|bergs?|ice floes?)\b", ["iceberg"], [], "sea", {}),
    (r"\b(warships?|destroyers?|cruiser|battleship|fleet|navy|naval|carrier|flotilla)\b", ["warship"], [], "sea", {}),
    (r"\b(?:missiles?|rockets?|icbms?)\b.{0,40}\b(?:launch\w*|fired|fire|lift\w* off)\b|\blaunch\w* (?:the |a |its |their )?(?:nuclear )?(?:missiles?|rockets?)|\blift\w* off\b|\bsilos?\b|\brockets?\b", ["missile"], [], None, {"launch": True}),
    (r"\b(planes?|aircraft|bombers?|airplane|jets?)\b", ["plane"], [], None, {}),
    (r"\b(volcano|volcanic|eruption|lava|magma|crater|caldera)\b", ["volcano"], ["embers"], "volcanic", {}),
    (r"\b(ash|ashfall|pumice|soot)\b", ["ash_cloud"], ["ashfall"], "ashen", {}),
    (r"\b(tsunami|tidal wave|flood\w*|(?:ocean|sea|giant|huge|massive|towering|enormous) waves?|waves? (?:crash\w*|hit|struck|rose|rolled|swept|slammed))\b", ["wave"], [], "sea", {}),
    (r"\b(fires?|burn\w*|blaze|flames?|inferno|ablaze)\b", ["fire", "smoke"], ["embers"], None, {}),
    (r"\b(explo\w+|blast\w*|detonat\w+|bomb\w*|shell\w*)\b", ["explosion"], ["shake", "flash"], None, {}),
    (r"\b(earthquakes?|tremors?|quakes?|shaking)\b", [], ["shake"], None, {}),
    (r"\b(storms?|thunder|lightning|hurricane|typhoon)\b", [], ["rain", "flash"], "storm", {}),
    (r"\b(armies|army|legions?|troops|hordes?|marched|marching|invad\w+|advanced on|stormed)\b|(?<!depth )(?<!depth-)\bcharg(?:ed|ing) (?:at|into|across|forward|toward|towards|the|down|up|through|over)\b|\bthe charge\b(?! of)", ["crowd"], ["dust"], None, {}),
    # everyday props: what people are standing next to, working with and carrying
    (r"\b(cesspits?|cesspools?|latrines?|sewers?|drains?|pits?|trench\w*|vaults?)\b", ["pit"], [], None, {}),
    (r"\b(swamps?|marsh\w*|ponds?|bogs?|leech\w*)\b", ["swamp"], [], "countryside", {}),
    (r"\b(barrels?|casks?|kegs?|tubs?|weans?|jars?|vats?|urns?|jugs?|amphorae?|vessels? of)\b", ["barrel"], [], None, {}),
    (r"\b(buckets?|pails?|ladles?)\b", ["bucket"], [], None, {}),
    (r"\b(baskets?|wicker|hampers?)\b", ["basket"], [], None, {}),
    (r"\b(carts?|wagons?|carted|hauled|hauling)\b", ["cart"], [], None, {}),
    (r"\b(coins?|gold|wages?|pay(?:check|ment)?s?|paid|salary|fortune|pennies|silver|money|purses?|rich(?:es)?|wealth\w*)\b", ["sack"], [], None, {}),
    (r"\b(chests?|treasure\w*|loot\w*|hoards?)\b", ["chest"], [], None, {}),
    (r"\b(beds?|bedrooms?|sleep\w*|slept|asleep|bedtime|pillows?|alarm|wake\w*|woke|snor\w+)\b", ["bed"], [], None, {}),
    (r"\b(bells?|ringing|rang|criers?|heralds?|announce\w*|proclaim\w*)\b", ["bell"], [], None, {}),
    (r"\b(tables?|taverns?|inns?|dinner|supper|feasts?|meals?|breakfast|ale|mugs?|eating|ate)\b", ["table"], [], None, {}),
    (r"\b(stools?|toilets?|chamber ?pots?|privy|privies|groom of the stool|royal bottom)\b", ["stool"], [], None, {}),
    (r"\b(ladders?|descend\w*|climb\w*|rungs?)\b", ["ladder"], [], None, {}),
    (r"\b(markets?|stalls?|vendors?|merchants?|shops?|bazaar|hawkers?)\b", ["stall"], [], None, {}),
    (r"\b(poles?|tapp\w*|knock\w*|rattl\w*)\b", ["pole"], [], None, {}),
    (r"\b(castle|fortress|citadel|siege|ramparts?)\b", ["castle"], [], None, {}),
    (r"\b(pyramids?|pharaoh|nile)\b", ["pyramid"], [], "desert", {}),
    (r"\b(towns?|villages?|settlements?|houses?|homes)\b", ["house"], [], None, {}),
    (r"\b(cities|city|skyscraper|metropolis)\b", ["building"], [], None, {}),
]
UNDERWATER_OBJECTS = {"submarine", "depth_charge", "torpedo"}
MODERN_TO_ANCIENT = {"sailor": "citizen", "captain": "knight", "officer": "soldier", "scientist": "scholar", "president": "king", "worker": "citizen", "pilot": "citizen",
                     "modern_soldier": "soldier", "spy": "citizen", "reporter": "scholar"}
ANCIENT_OBJECTS = {"castle", "pyramid", "column", "tent", "torch", "pedestal"}
ANCIENT_ROLE_SWAP = {"soldier": "modern_soldier", "warrior": "modern_soldier", "knight": "officer", "general": "officer", "king": "president",
                     "queen": "president", "emperor": "president", "pharaoh": "president", "priest": "scientist", "rebel": "worker", "pirate": "sailor"}


def era_modern(full_text, default=False):
    years = [int(y) for y in re.findall(r"\b([3-9][0-9]{2}|1[0-9]{3}|20[0-2][0-9])\b", full_text)]
    if not years:
        return default
    return statistics.median(years) >= 1800


def anchor_for(topic_text):
    """A place-defining subject of the whole video (the volcano, the pyramids) that should loom in the background of its outdoor scenes."""
    t = topic_text.lower()
    if re.search(r"\b(vesuvius|volcano|volcanic|eruption|krakatoa|etna|pompeii|st\.? helens|pinatubo)\b", t):
        return "volcano"
    if re.search(r"\b(pyramids?|giza|pharaoh)\b", t):
        return "pyramid"
    if re.search(r"\b(chernobyl|nuclear (?:power|plant|reactor|accident|disaster)|reactor|fukushima|three mile island|pripyat|meltdown)\b", t):
        return "reactor"
    if re.search(r"\b(great fire|fire of|burn(?:ed|ing|t)?|blaze|inferno|fire)\b", t):
        return "burning_town"
    return None


GEO_WORDS = re.compile(r"\b(march\w*|invad\w*|invasion|route|sail\w*|voyage|spread|spreading|border\w*|empire|territor\w*|advanc\w*|retreat\w*|expand\w*|conquer\w*|kingdom|coast|island|continent|across the|crossed|trade route|armies|fleet)\b", re.I)


def domain_for(topic_text):
    if re.search(r"\b(hindenburg|zeppelins?|airships?|dirigibles?)\b", topic_text.lower()):
        return "airship"
    return "space" if re.search(r"\b(apollo|nasa|astronauts?|spacecraft|spaceflight|space shuttle|challenger|saturn v|moon landing|lunar|orbit\w*|cosmonaut|sputnik|gemini|mercury seven|voyager|hubble)\b", topic_text.lower()) else None


SPACE_MC = re.compile(r"\b(mission control|houston|flight director|flight controllers?|controllers?|consoles?|ground team|kranz|capcom|engineers? (?:on|at)|on the ground|simulator)\b", re.I)
SPACE_IN = re.compile(r"\b(inside|aboard|cabin|cockpit|panel|alarm|warning|switch\w*|gauge\w*|dials?|hatch|suits?|cold|freez\w*|breath\w*|carbon dioxide|crew|astronauts?|lovell|swigert|haise|exhausted|shiver\w*|cramped)\b", re.I)
SPACE_OUT = re.compile(r"\b(moon|lunar|earth|reentry|re-entry|splashdown|orbit\w*|trajectory|launch\w*|lifted|rocket|saturn|engines?|burn|vent\w*|explo\w+|bang|spacecraft|space|drift\w*|tumbl\w*|miles|stage|separat\w*|window|stars?|home)\b", re.I)
SPACE_VENT = re.compile(r"\b(vent\w*|leak\w*|explo\w+|rupture\w*|damag\w*|blew|blast\w*|burst|tank|crippled|debris)\b", re.I)


def _space(v, text):
    v = dict(v)
    flags = {"modern": True, "venting": bool(SPACE_VENT.search(text))}
    objs, fx = [], []
    if re.search(r"\b(parachutes?|chutes?|splashdown|splashed down|recovery ship|recovered)\b", text):          # the return to Earth happens under a sky, on the sea
        v.update(background="sea", objects=["parachute"] if re.search(r"parachute|chute|splash", text) else [], actors=[], effects=[], flags={"modern": True})
        return v
    if re.search(r"\b(lunar surface|surface of the moon|moon'?s surface|moonwalk\w*|tranquility base|walked on the moon|lunar landscape|stepped (?:out )?onto the moon)\b", text):
        acts = list(v.get("actors") or [])[:2] or [dict(role="astronaut", color="", pos="center", action="walk", emotion="smile", facing="", scale=1.0)]
        for a in acts:
            a["role"] = "astronaut"
        v.update(background="moon", actors=acts, objects=["planet"], effects=[], flags={"modern": True, "kind": "earth"})
        return v
    if SPACE_MC.search(text):
        bg = "mission_control"
        roles = ("scientist", "officer")
    elif len(SPACE_IN.findall(text)) > len(SPACE_OUT.findall(text)) or (SPACE_IN.search(text) and len(SPACE_IN.findall(text)) == len(SPACE_OUT.findall(text)) and len(text) % 2):
        bg = "capsule"
        roles = ("astronaut", "astronaut")
    else:
        bg = "space"
        roles = ()
        objs = ["spacecraft"]
    if bg == "space":
        if re.search(r"\b(earth|home|reentry|re-entry|splashdown|atmosphere|pacific)\b", text):
            objs.append("planet")
            flags["kind"] = "earth"
        elif re.search(r"\b(moon|lunar|crater)\b", text):
            objs.append("planet")
            flags["kind"] = "moon"
        v["actors"] = []
    else:
        acts = list(v.get("actors") or [])[:2]
        while len(acts) < 2:
            acts.append(dict(role=roles[len(acts) % 2], color="", pos="left" if not acts else "right", action="talk", emotion="worried", facing="", scale=1.0))
        for i, a in enumerate(acts):
            a["role"] = roles[i % 2]
            a["pos"] = ("center_left", "center_right")[i % 2]
        v["actors"] = acts
    if SPACE_VENT.search(text) and re.search(r"\b(explo\w+|blast|blew|boom|bang)\b", text) and bg == "space":
        objs.append("explosion")
        fx += ["shake", "flash"]
    v.update(background=bg, objects=objs[:3], effects=fx, flags=flags)
    if str(v.get("title", "")).strip().lower().replace(" ", "_") in {"space", "capsule", "mission_control"}:
        v["title"] = ""
    return v


def enrich(visual, narration, modern, anchor=None, story_has_sub=True, domain=None):
    """Return the visual with the narration's subjects added. Only stage scenes are touched."""
    if visual.get("type") == "map" and not GEO_WORDS.search(narration):
        # a map with no geography in the line is decoration: draw the scene instead
        visual = dict(type="stage", background="countryside" if not modern else "city_modern", actors=[], objects=[], effects=[], title="", camera="push_in")
    if visual.get("type") != "stage":
        return visual
    v = dict(visual)
    text = re.sub(r"rapid[- ]fire|open[- ]fire|fire[- ]?(?:and|or) brimstone|under fire|friendly fire|fired (?:up|from|him|her|them|the (?:cook|worker|servant))|set on fire", " ", narration.lower())     # figures of speech are not flames
    if domain == "space":
        return _space(v, text)
    if v.get("background") in ("mission_control", "capsule", "space", "moon"):            # spacecraft sets only exist in spaceflight stories
        v["background"] = "night" if re.search(r"\b(night|dark|torch|candle|underground|cellar|pit|sewer|vault|midnight)\b", text) else ("palace" if re.search(r"\b(council|court|official|authorit|magistrate|mayor|parliament)\b", text) else "city_day")
        if v["background"] != "space":
            v["objects"] = [o for o in v.get("objects", []) if o not in ("spacecraft", "planet", "capsule_interior", "mission_control")]
    if EVERYDAY["on"]:                                                         # no crowns, spears or shields on a leech collector: royalty and soldiers only when the line names them
        royal = re.search(r"\b(king|queen|emperor|empress|royal|monarch|crown|palace|henry|vespasian|pharaoh|prince|court)\b", text)
        fight = re.search(r"\b(soldiers?|army|battle|knights?|war|guards?|sword|spear|shield|warriors?)\b", text)
        for a_ in v.get("actors", []):
            r_ = a_.get("role")
            if r_ in ("king", "queen", "emperor", "pharaoh") and not royal:
                a_["role"] = "merchant"
            elif r_ in ("soldier", "warrior", "knight", "general", "rebel", "pirate") and not fight:
                a_["role"] = "worker"
            elif r_ == "scholar" and not re.search(r"\b(scholar|doctors?|physicians?|books?|scrolls?|read|reading|wrote|writing|law|laws|history|historians?|professors?|teach\w*|school|medical)\b", text):
                a_["role"] = "citizen"
            elif r_ == "priest" and not re.search(r"\b(priest|church|vicar|clergy|funeral|prayer|sin|sins|heaven)\b", text):
                a_["role"] = "scholar"
    emo = None                                                                 # faces follow what is being said, whatever the script writer picked
    if re.search(r"\b(gross|disgust\w*|stench|stink\w*|smell\w*|reek\w*|vomit\w*|sewage|waste|filth\w*|poop|pee|urine|excrement|rotten|rot|slime|muck|corpses?|puke)\b", text):
        emo = "worried"
    if re.search(r"\b(died|dead|death|dying|suffocat\w*|poison\w*|deadly|danger\w*|collaps\w*|explod\w*|fire|trapped|scream\w*|shock\w*|sudden\w*|horrif\w*)\b", text):
        emo = "shock"
    if re.search(r"\b(paid|pay|wages?|rich|wealth\w*|fortune|gold|silver|coins?|salary|pension|promotion|power|won|win|winner|survived|celebrat\w*|funny|hilarious|laugh\w*|lucky|best)\b", text) and emo is None:
        emo = "smile"
    if re.search(r"\b(angry|furious|rage|banned|outcasts?|shunned|hated|punish\w*|fined|arrest\w*|robbery|stole|steal\w*)\b", text) and emo is None:
        emo = "angry"
    if emo:
        for a_ in v.get("actors", []):
            a_["emotion"] = emo
    objs = list(v.get("objects", []))
    fx = list(v.get("effects", []))
    bg = v.get("background")
    if bg == "underwater" and not story_has_sub and not re.search(r"submarine|u-boat|torpedo|sonar|depth[- ]?charge|periscope|diver|diving|scuba", text):
        bg = "sea"                                                       # nothing in this story goes under water
    if bg == "submarine_interior" and not story_has_sub:
        bg = "mission_control"                                           # any other industrial interior is a control room, not a submarine
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
    bound = {"missile", "torpedo", "depth_charge", "explosion", "volcano", "wave", "fire", "plane", "warship", "submarine", "pyramid", "castle", "ash_cloud", "crowd", "liner", "iceberg", "airship", "tank", "parachute", "flag", "cannon", "clock"}
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
    if bg == "sea" and v.get("actors"):
        bg = "harbor"                                                # people need a shore to stand on
    if bg == "underwater" and not story_has_sub and not any(o in objs for o in ("submarine", "depth_charge", "torpedo")):
        bg = "sea"                                                   # no submarine in this story: stay on the surface
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
    naval = re.search(r"\b(navy|naval|warships?|destroyers?|cruiser|battleship|fleet|gunboat|frigate)\b", text)
    if modern and "ship" in objs and "airship" not in objs:
        objs[objs.index("ship")] = "warship" if naval else "liner"
    if modern and "warship" in objs and not naval and re.search(r"\b(liner|steamship|steamer|passenger|cargo|vessel|ship)\b", text) and not re.search(r"\b(navy|naval|warships?|destroyers?)\b", text):
        objs[objs.index("warship")] = "liner"                     # a civilian ship is not a gunboat
    if domain == "airship":
        objs = ["airship" if o in ("liner", "ship", "warship") else o for o in objs if o not in ("wave", "building")]
        if "airship" not in objs and re.search(r"\b(ships?|vessel|craft|hull|envelope|giant|titan|passengers?|helium|hydrogen|gas|cells?|tail|nose|mooring|descent|landing|flight|flew|flying|sky|skies)\b", text):
            objs.insert(0, "airship")
        if bg in ("sea", "harbor", "underwater", "city_modern", None):
            bg = "countryside"
        if bg in ("capsule", "mission_control", "submarine_interior", "space"):
            bg = "night" if re.search(r"\b(night|dark|storm|fire|flames?|inferno|spark|burn\w*)\b", text) else "countryside"   # an airship story is told under the open sky
        objs = [o for o in objs if o not in ("tower", "capsule_interior", "mission_control", "spacecraft", "planet", "pipes", "gauge", "missile", "column", "pedestal", "ash_cloud")]
    if "airship" in objs:
        objs = [o for o in objs if o not in ("liner", "ship", "warship")]
        if re.search(r"\b(fire|flames?|burn\w*|inferno|blaze|ignit\w*|spark|explo\w+|crash\w*|collaps\w*|tilt\w*)\b", text):
            flags["burning"] = True
    # keep what the narration named first, then the AI's own picks; cap to three objects
    named = [o for o in objs if any(o in r[1] for r in RULES if re.search(r[0], text))]
    rest = [o for o in objs if o not in named]
    objs = named + rest
    if not modern:
        if bg == "city_modern":
            bg = "city_day"                                            # no modern skyline before 1800
        for a_ in v.get("actors", []):
            a_["role"] = MODERN_TO_ANCIENT.get(a_.get("role"), a_.get("role"))
        mapped = {"warship": "ship", "liner": "ship", "submarine": "ship", "plane": "cloud", "missile": "tower", "depth_charge": "ship", "torpedo": "ship", "building": "house", "tank": "cloud", "parachute": "cloud", "clock": "cloud"}
        objs = [mapped.get(o, o) for o in objs]
        objs = [o for o in objs if o != "cloud"]
        objs = list(dict.fromkeys(objs))
    if not modern:
        objs = ["house" if o == "building" else o for o in objs]                 # no skyscrapers in the ancient world
        objs = list(dict.fromkeys(objs))
    if bg == "underwater":
        objs = ["depth_charge" if o == "explosion" else o for o in objs]          # a blast under water is a blue-white burst, not a fireball
        objs = [o for o in objs if o != "fire"]
        if "submarine" not in objs and story_has_sub:
            objs.insert(0, "submarine")                                          # a submarine story never shows an empty seabed                                          # never an empty seabed
    if bg != "submarine_interior":                            # interior fittings only make sense inside the boat
        objs = [o for o in objs if o not in ("pipes", "gauge", "hatch")]
    if bg in ("underwater", "submarine_interior"):            # nothing from the surface world belongs down here
        objs = [o for o in objs if o not in ("warship", "ship", "crowd", "plane", "building", "house", "wave", "castle", "pyramid", "volcano")]
    if str(v.get("title", "")).strip().lower().replace(" ", "_") in {"submarine_interior", "underwater", "city_day", "battlefield", "palace", "sea"}:
        v["title"] = ""                                       # a place name is not a headline
    if bg == "underwater":                                    # nobody is standing on the seabed
        v["actors"] = []
    v["objects"] = objs[:3]
    scenery = {"seascape", "interior", "pipes", "column", "tree", "cloud", "torch", "smoke", "house", "ash_cloud", "fire", "building"}
    if EVERYDAY["on"] and not v.get("actors") and bg != "underwater":                # entertainment scenes are always about a person doing something
        v["actors"] = [dict(role="citizen", color="", pos="center", action="talk", emotion="neutral", facing="", scale=1.0)]
    if not v.get("actors") and not [o for o in v["objects"] if o not in scenery] and bg != "underwater":
        v["actors"] = [dict(role="president" if modern else "citizen", color="", pos="center", action="talk", emotion="worried", facing="", scale=1.0)]   # never an empty stage
    if bg == "submarine_interior" and len(v.get("actors") or []) < 2:           # an interior scene is about the people in it
        keep = list(v.get("actors") or [])[:1]
        v["actors"] = keep + [dict(role="captain", color="red", pos="left", action="talk", emotion="angry", facing="", scale=1.0),
                       dict(role="officer", color="grey", pos="right", action="talk", emotion="worried", facing="", scale=1.0)][:2 - len(keep)]
        for a_, p_ in zip(v["actors"], ("center_left", "center_right")):
            a_["pos"] = p_
    v["effects"] = fx[:3]
    v["background"] = bg or v.get("background")
    flags["modern"] = bool(modern)
    v["flags"] = flags
    gate = {"burning_town": r"\b(fire|fires|flames?|burn\w*|blaze|inferno|ember|embers|smoke|ablaze|ash|ashes|scorch\w*)\b"}.get(anchor)
    if anchor and bg not in ("submarine_interior", "underwater") and anchor not in (v.get("objects") or []) and (gate is None or re.search(gate, text)):
        v["anchor"] = anchor
    # a sunk-in-thought lone character is boring: if the subject is a vehicle/disaster, let it dominate
    if any(o in v["objects"] for o in ("submarine", "volcano", "wave", "missile", "warship", "plane", "explosion")) and len(v.get("actors", [])) > 1:
        v["actors"] = v["actors"][:1]
    return v


FAMILY = {"city_day": ["palace", "countryside"], "palace": ["city_day", "desert"], "countryside": ["forest", "city_day"], "desert": ["palace", "countryside"],
          "battlefield": ["storm", "desert"], "storm": ["battlefield", "night"], "night": ["storm", "city_modern"], "city_modern": ["night", "sea"],
          "sea": ["storm", "city_modern"], "forest": ["countryside", "snow"], "snow": ["forest", "night"]}


BYSTANDER = {True: ("reporter", "worker", "officer"), False: ("citizen", "scholar", "soldier")}


def variety_pass(visuals, modern=False, domain=None):
    """Never three scenes in a row with the same setting: cut away to a related one (outside the boat, another location, a close two-shot)."""
    out = list(visuals)
    if domain == "space":                                                   # two exterior spacecraft shots never run back to back: cut inside to the crew or to Houston
        flip = 0
        for i in range(1, len(out)):
            if out[i].get("type") == "stage" and out[i - 1].get("type") == "stage" and out[i].get("background") == out[i - 1].get("background") == "space":
                c = dict(out[i])
                bg, roles = (("capsule", ("astronaut", "astronaut")), ("mission_control", ("scientist", "officer")))[flip % 2]
                flip += 1
                c.update(background=bg, objects=[], effects=[], flags={"modern": True, "kind": "earth"},
                         actors=[dict(role=roles[k], color="", pos=("center_left", "center_right")[k], action="talk", emotion="worried", facing="", scale=1.0) for k in range(2)])
                out[i] = c
    for i in range(2, len(out)):
        a, b, c = out[i - 2], out[i - 1], out[i]
        if not (a.get("type") == b.get("type") == c.get("type") == "stage"):
            continue
        if not (a.get("background") == b.get("background") == c.get("background")):
            continue
        c = dict(c)
        if c["background"] in ("capsule", "mission_control") and domain != "space":
            c["background"] = "night"                                       # an industrial control room is never a spacecraft
        elif c["background"] in ("capsule", "mission_control"):             # cutaway to the spacecraft from outside
            c.update(background="space", actors=[], objects=["spacecraft", "planet"], effects=[], flags={"modern": True, "kind": "earth"}, title=c.get("title", ""))
        elif c["background"] == "submarine_interior":                       # cutaway to the boat seen from outside
            c.update(background="underwater", actors=[], objects=["submarine"], effects=["bubbles"], title=c.get("title", ""))
        elif c["background"] in FAMILY:
            c["background"] = FAMILY[c["background"]][i % 2]
            if not modern and c["background"] == "city_modern":
                c["background"] = "city_day"                            # no modern skyline in an old story
        out[i] = c
    return out


def company_pass(visuals, modern=False):
    """Never three lone figures in a row: the third scene gets a second person reacting."""
    out = list(visuals)
    lone = 0
    for i, v in enumerate(out):
        if v.get("type") == "stage" and len(v.get("actors") or []) == 1 and v.get("background") not in ("underwater",):
            lone += 1
            if lone >= 3:
                v = dict(v)
                a0 = v["actors"][0]
                role = BYSTANDER[bool(modern)][i % 3]
                v["actors"] = list(v["actors"]) + [dict(role=role, color="", pos="right" if str(a0.get("pos", "center")).endswith(("left", "center")) else "left", action="talk",
                                                       emotion="worried", facing="", scale=.95)]
                out[i] = v
                lone = 0
        else:
            lone = 0
    return out


def era_fix_thumb(tr, modern, domain=None, anchor=None):
    """Thumbnails use the same era rules as the scenes: no modern roles or vessels in an old story."""
    tr = dict(tr)
    if anchor in ("reactor", "volcano", "pyramid", "burning_town") and not domain:
        tr["backdrop"] = anchor                                      # the story's landmark is the subject of every thumbnail
        tr["objects"] = []
    if domain == "airship":
        tr["backdrop"] = "airship"
        tr["objects"] = []
        tr["scene"] = "countryside"
        return tr
    if domain == "space":
        tr["role"] = "astronaut"
        tr["enemy_role"] = "scientist"
        if tr.get("backdrop") not in ("spacecraft", "planet", "explosion"):
            tr["backdrop"] = "spacecraft"
        tr["objects"] = [o for o in (tr.get("objects") or []) if o in ("spacecraft", "planet", "explosion")]
        return tr
    if not modern:
        for k in ("role", "enemy_role"):
            if tr.get(k) in MODERN_TO_ANCIENT:
                tr[k] = MODERN_TO_ANCIENT[tr[k]]
        mapped = {"warship": "ship", "liner": "ship", "submarine": "ship", "plane": "tower", "missile": "tower", "building": "house", "depth_charge": "ship", "torpedo": "ship"}
        if tr.get("backdrop") in mapped:
            tr["backdrop"] = mapped[tr["backdrop"]]
        tr["objects"] = [mapped.get(o, o) for o in (tr.get("objects") or [])]
    return tr
