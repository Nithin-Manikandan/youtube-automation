"""Headless auto mode (used by the GitHub workflow):  python -m studio.run_auto [--hint "..."] [--out output/auto]"""
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
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    job = {"log": [], "stage": 0}
    try:
        pkg = auto.run(job, out, {"voice": os.environ.get("NARRATOR_VOICE", "en-US-AndrewMultilingualNeural")}, a.hint.strip() or None)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("\nTITLE:", pkg["title"])
    print("LENGTH:", pkg["minutes"], "min,", pkg["words"], "words")
    print("FLAGS:", len(pkg["flags"]))


if __name__ == "__main__":
    main()
