"""Beat director: reads what the narrator is saying, word by word, and decides what the picture should do at that exact moment.

A scene's narration comes with word timings. Trigger words ("screamed", "ducked", "collapsed", "exploded", "wept", "pointed") become beats:
an acting beat (which motion-capture clip plays, with which expression) and, for violent words, a camera event (shake, punch-in, flash).
Characters react to each other: the agent of a beat acts, the other character reacts a moment later.
"""
import re

# (regex on a lowercase word, action, emotion or None, event or None)   action: acting vocabulary of studio.acting.ACTION_CLIP, "talk_angry", or "idle"
def _st(*stems):
    """A stem matches as a prefix ("shout" -> shouted, shouting); a stem ending in $ must match the whole word."""
    parts = [re.escape(x[:-1]) if x.endswith("$") else re.escape(x) + r"\w*" for x in stems]
    return r"^(" + "|".join(parts) + r")$"


LEX = [
    (_st("dig", "dug$", "shovel", "scoop", "scrap", "scrub", "sweep", "swept$", "mop", "stir", "rak", "chop", "hammer", "scrap"), "dig", None, None),
    (_st("eat", "ate$", "eaten$", "swallow", "devour", "chew", "feast", "drank$", "drink", "gulp", "sip", "bite", "bit$", "dine", "dining$", "munch"), "eat", None, None),
    (_st("knock", "tap$", "taps$", "tapped$", "tapping$", "poke", "poking$", "rattl", "pound", "bang", "shoot", "shot$", "blast"), "knock", None, None),
    (_st("stomp", "trampl", "tread", "trod$", "wad", "slog", "trudg", "march", "paddl", "splash", "crush"), "stomp", None, None),
    (_st("cough", "chok", "suffocat", "asphyx", "breath", "wheez", "gagg", "gasping$", "lungs$", "oxygen$", "carbon$", "co2$", "fumes$", "smoke$"), "cough", "worried", None),
    (_st("cold$", "chill", "frigid", "icy$", "frost", "shiver", "bitter$", "arctic", "hypotherm"), "shiver", "worried", None),
    (_st("sweat", "sweltering$", "humid", "scorching$", "stifling$", "overheat", "feverish"), "wipe", "worried", None),
    (_st("switch", "button", "lever", "flip$", "flipped$", "flips$", "toggle", "dial$", "knob", "throttle", "controls$", "keypad", "pressed$", "pressing$"), "pick_up", None, None),
    (_st("scream", "shout", "yell", "roar", "bellow", "demand", "ordered$", "ordering$", "commanded$", "threaten", "furious", "rage$", "raged$", "snapped$", "bark", "argu", "blamed$", "accus", "betray", "refus", "insist", "defian", "protest"), "talk_angry", "angry", None),
    (_st("gasp", "froze$", "frozen$", "startl", "horrif", "stun", "shock", "astonish", "flinch", "recoil", "sudden", "abrupt", "alarm", "jolt", "unexpect", "ripped$", "tore$", "warning$"), "flinch", "shock", None),
    (_st("terrif", "tremble", "trembl", "panic", "terror", "afraid", "fear", "dread", "cower", "shiver", "freezing$", "frighten", "scared$", "nightmar", "danger", "deadl", "peril", "desperat", "trapped$", "stranded$", "helpless", "dying$", "lethal", "toxic", "poison"), "scared", "worried", None),
    (_st("wept$", "weep", "cried$", "crying$", "sob", "mourn", "grief", "despair", "devastat", "heartbr", "tears$", "tragic", "tragedy", "lament", "lost$", "loss$"), "cry", "sad", None),
    (_st("cheer", "celebrat", "triumph", "victor", "rejoic", "smile", "laugh", "joy", "relief", "thrill", "proud", "success", "saved$", "rescu", "surviv", "safely$", "hero", "won$", "winning$", "glory", "reunit"), "cheer", "smile", None),
    (_st("duck", "dive$", "dove$", "crouch", "shelter", "hid$", "hiding$", "dodg", "evad", "escap", "avoid", "brac"), "duck", "worried", None),
    (_st("pointed$", "pointing$", "gestur", "indicat"), "point", None, None),
    (_st("looked$", "looking$", "stare", "staring$", "stared$", "watch", "scan", "search", "glanc", "gaz", "peer", "noticed$", "spotted$", "observ", "inspect", "examin", "listen", "heard$", "hearing$", "studied$", "studying$", "check"), "look_around", "worried", None),
    (_st("wave$", "waved$", "waving$", "greet", "hello$", "farewell", "goodbye", "salut"), "wave", "smile", None),
    (_st("shrug", "uncertain", "unsure", "doubt", "hesitat", "perhaps$", "maybe$", "wondered$", "wondering$", "confus", "puzzl", "mysteri", "unknown$", "unclear"), "shrug", "worried", None),
    (_st("think", "thought", "consider", "realiz", "realis", "decided$", "deciding$", "ponder", "imagin", "calculat", "comput", "solv", "figured$", "planning$", "planned$", "reasoned$", "analy", "understood$", "remember", "recall", "equation", "math"), "think", "worried", None),
    (_st("punch", "strike$", "struck$", "slam", "smash", "attack", "beat$", "beaten$", "hit$", "hits$", "hitting$", "hammer", "pound", "lunged$", "assault", "bash", "knocked$"), "punch", "angry", "hit"),
    (_st("kick", "stomp", "trampl"), "kick", "angry", "hit"),
    (_st("fought$", "fight", "battl", "clash", "duel", "charged$", "charging$", "stabbed$", "slashed$", "swung$", "war$", "wars$", "warfare", "conquer", "invad", "siege", "stormed$", "raid"), "fight_burst", "angry", "hit"),
    (_st("push", "shov", "forced$", "forcing$", "rammed$", "heav", "pressed$", "pressing$", "propel", "thrust"), "push", None, None),
    (_st("pull", "haul", "drag", "tug", "yank", "towed$", "reeled$"), "pull", None, None),
    (_st("lift", "picked$", "picking$", "carried$", "carrying$", "grab", "collect", "gather", "loaded$", "loading$", "clutch", "grasp", "seiz", "assembl", "repair", "tape$", "taped$", "build$", "built$", "construct", "fix$", "fixed$", "mend", "weld", "connect", "install", "strap", "fasten", "improvis", "adapt", "jury"), "pick_up", None, None),
    (_st("stumbl", "tripped$", "lurch", "slipped$", "stagger", "sway", "wobbl", "tumbl"), "stumble", "shock", "shake"),
    (_st("died$", "dies$", "die$", "kill", "perish", "dead$", "collaps", "fell$", "fall$", "falling$", "fallen$", "plung", "sank$", "sunk$", "drown", "buried$"), "fall", "shock", "hit"),
    (_st("sat$", "seated$", "sitting$", "rested$", "resting$", "settled$", "slept$", "sleep", "exhaust", "weary"), "sit", None, None),
    (_st("ran$", "run$", "runs$", "running$", "raced$", "racing$", "rush", "sprint", "bolted$", "fled$", "flee", "hurri", "dashed$", "chased$", "scrambl"), "run_to", "worried", None),
    (_st("walk", "march", "strode$", "approach", "entered$", "entering$", "advanc", "stepped$", "crossed$", "climb", "arrived$", "arriving$", "wander", "travel", "journey", "sailed$", "sailing$"), "walk_to", None, None),
    (_st("silen", "quiet", "hush", "calm$", "alone$", "paused$", "ceased$"), "idle", "worried", None),
]
EVENTS = [
    (r"^(explo\w*|blast\w*|blew|boom\w*|detonat\w*|erupt\w*|bang\w*|burst\w*|shatter\w*|blown|ripped|ruptur\w*)$", "boom"),
    (r"^(crash\w*|collid\w*|collision|impact\w*|slam\w*|smash\w*|thud\w*|struck|knock\w*)$", "hit"),
    (r"^(earthquake\w*|tremor\w*|shook|shak\w*|quake\w*|rumbl\w*|thunder\w*|roar\w*|vibrat\w*|trembl\w*)$", "shake"),
    (r"^(lightning|flash\w*|flare\w*|spark\w*|ignit\w*|ignition|fireball|blinding|aflame|inferno)$", "flash"),
    (r"^(shot|gunshot\w*|fired|firing|torpedo\w*|launch\w*|missile\w*|bomb\w*|shell\w*|cannon\w*)$", "hit"),
]
MIN_GAP = 1.5          # seconds between acting beats of one actor
MAX_BEATS = 6
_L = [(re.compile(p), a, e, ev) for p, a, e, ev in LEX]
_E = [(re.compile(p), ev) for p, ev in EVENTS]


def _norm(w):
    return re.sub(r"[^a-z']", "", w.lower())


def extract(wins, n_actors=1):
    """wins: [(word, t0, t1)] relative to the scene start. Returns {"acts": [beat], "events": [(t, kind)]}.
    beat = dict(t, action, emotion, who) with who = "agent" | "reactor" | "all"."""
    acts, events, last_t = [], [], -9.0
    for w, a, b in wins or []:
        n = _norm(w)
        if not n:
            continue
        for rx, ev in _E:
            if rx.match(n):
                events.append((a, ev))
                break
        for rx, act, emo, ev in _L:
            if rx.match(n):
                if a - last_t < MIN_GAP or len(acts) >= MAX_BEATS:
                    break
                last_t = a
                k = len(acts)
                if act in ("flinch", "scared", "cry", "duck", "cheer", "idle", "sit", "look_around", "think", "shrug", "shiver", "eat"):
                    who = "all" if n_actors == 1 else ("reactor" if act in ("flinch", "scared", "cry") else "all")
                else:
                    who = "agent"
                acts.append(dict(t=a, action=act, emotion=emo, who=who, k=k, word=n))
                if ev:
                    events.append((a, ev))
                break
    for t, kind in events:                                                  # a loud event makes everyone flinch, if nothing else is happening just then
        if kind in ("boom", "hit", "shake") and len(acts) < MAX_BEATS and all(abs(b["t"] - t) > 1.2 for b in acts):
            acts.append(dict(t=t, action="flinch", emotion="shock", who="all", k=len(acts), word=kind))
    acts.sort(key=lambda b: b["t"])
    return {"acts": acts, "events": events}


def event_time(wins, kinds):
    """First time a word of one of the given event kinds is spoken (for timing objects like explosions to the word)."""
    for t, k in extract(wins)["events"] if wins else []:
        if k in kinds:
            return t
    return None
