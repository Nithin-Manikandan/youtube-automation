"""Make one video.  python -m pipeline.make [--topic "..."] [--offline] [--upload]"""
import argparse
import json
import pathlib
import re
import time

import cv2
import numpy as np

from . import audio, images, render, script, tts
from .config import ROOT, load

GAP = 0.22  # breathing room after each scene's narration


def log(msg):
    print(msg, flush=True)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def contact_sheet(stills, path):
    tiles = [cv2.resize(stills[k], (270, 480)) for k in sorted(stills)]
    if not tiles:
        return
    cols = min(5, len(tiles))
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * 480, cols * 270, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 480:(r + 1) * 480, c * 270:(c + 1) * 270] = t
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--config")
    ap.add_argument("--offline", action="store_true",
                    help="engine test: bundled script, placeholder art, silent voice (no network)")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    cfg = load(args.config)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hist_path = ROOT / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []

    t0 = time.time()
    if args.offline:
        data = json.loads((pathlib.Path(__file__).parent / "sample_script.json").read_text())
        topic = "offline engine test"
    else:
        topic = args.topic or script.pick_topic(cfg, history)
        log(f"[1/5] topic: {topic}")
        data = script.write_script(cfg, topic)
    log(f"[1/5] script: {data['title']}  ({len(data['scenes'])} scenes)")

    vis = cfg["visuals"]
    specs, voices, starts = [], [], []
    cursor = 0.0
    order = ["placeholder"] if args.offline else None
    log("[2/5] images + narration")
    for i, sc in enumerate(data["scenes"]):
        prompt = f"{sc['image_prompt']}. {vis['style']}"
        img = images.generate(prompt, vis.get("negative", ""), 1000 + i, ROOT / "cache", order, log)
        voice, words = tts.silent(sc["narration"]) if args.offline else tts.narrate(sc["narration"], cfg)
        lead = 0.15 if i == 0 else 0.08
        dur = lead + len(voice) / tts.SR + GAP
        specs.append({"path": img, "start": cursor, "dur": dur,
                      "words": [[w[0], cursor + lead + w[1], cursor + lead + w[2]] for w in words]})
        voices.append(voice)
        starts.append(cursor + lead)
        cursor += dur
        log(f"  scene {i + 1}/{len(data['scenes'])}: {dur:.1f}s")
    total = cursor + 0.5
    specs[-1]["dur"] += 0.5

    log(f"[3/5] audio bed + mix ({total:.1f}s)")
    mixed = audio.mix(voices, starts, total, cfg)

    slugname = slug(data["title"]) or "video"
    mp4 = out / f"{time.strftime('%Y%m%d-%H%M')}-{slugname}.mp4"
    log("[4/5] rendering video")
    stills = render.render(specs, data.get("hook_text", ""), mixed, mp4, cfg, log)
    pathlib.Path(str(mp4) + ".wav").unlink(missing_ok=True)
    contact_sheet(stills, mp4.with_suffix(".jpg"))
    (mp4.with_suffix(".json")).write_text(json.dumps({"topic": topic, **data}, indent=2))
    log(f"[5/5] done in {time.time() - t0:.0f}s -> {mp4}")

    if not args.offline:
        history.append(topic)
        hist_path.write_text(json.dumps(history[-200:], indent=1))
    if args.upload:
        from . import upload
        upload.upload(mp4, data, cfg, log)


if __name__ == "__main__":
    main()
