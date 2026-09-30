"""Headless auto mode (used by the GitHub workflow).

    python -m studio.run_auto --phase all                      everything on this machine
    python -m studio.run_auto --phase plan                     topic, script, voice, sound, thumbnails  -> plan.pkl, mix.flac
    python -m studio.run_auto --phase render --shard 2 --shards 4   draw every 4th scene (run on several machines)
    python -m studio.run_auto --phase assemble                 join the scenes, add the audio -> final.mp4
"""
import argparse
import os
import pathlib
import sys
import traceback

from . import auto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint", default="")
    ap.add_argument("--out", default="output/auto")
    ap.add_argument("--phase", choices=["all", "plan", "render", "assemble"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    job = {"log": [], "stage": 0}
    settings = {"voice": os.environ.get("NARRATOR_VOICE", "en-US-AndrewMultilingualNeural")}
    try:
        if a.phase == "all":
            pkg = auto.run(job, out, settings, a.hint.strip() or None)
        elif a.phase == "plan":
            pkg = auto.plan(job, out, settings, a.hint.strip() or None)
        elif a.phase == "render":
            auto.render_shard(job, out, a.shard, a.shards)
            return
        else:
            pkg = auto.assemble(job, out)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("\nTITLE:", pkg["title"])
    print("LENGTH:", pkg["minutes"], "min,", pkg["words"], "words")
    print("FLAGS:", len(pkg["flags"]))


if __name__ == "__main__":
    main()
