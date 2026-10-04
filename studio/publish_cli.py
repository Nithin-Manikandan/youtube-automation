"""Quality gate + YouTube upload for a finished Auto video (used by the Publish workflow).

    python -m studio.publish_cli DIR --thumb 1 --privacy private [--upload]

DIR holds out/final.mp4, out/thumb*.jpg and auto.json (the 'auto-video' artifact). Without --upload it only runs the checks.
The checks are the ones a machine can do honestly: length, loudness, clipping, black frames, hook shape, metadata, thumbnails,
and high-severity fact-check flags. They cannot judge whether the video is good or whether the narrator sounds right."""
import argparse
import json
import pathlib
import re
import subprocess
import sys

from pipeline import ffmpeg, upload as yt_upload


def _ff(*args):
    return subprocess.run([ffmpeg.exe(), "-hide_banner", *args], capture_output=True, text=True).stderr


def checks(d, pkg):
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    mp4 = d / "out" / "final.mp4"
    add("final.mp4 exists", mp4.exists())
    if not mp4.exists():
        return out
    log = _ff("-i", str(mp4))
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", log)
    secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0
    add("length 7-15 minutes", 420 <= secs <= 900, f"{secs / 60:.1f} min")
    loud = _ff("-i", str(mp4), "-af", "ebur128=peak=true", "-f", "null", "-")
    i = re.findall(r"I:\s+(-?\d+\.\d+) LUFS", loud)
    pk = re.findall(r"Peak:\s+(-?\d+\.\d+) dBFS", loud)
    add("loudness about -16 LUFS", i and -19 <= float(i[-1]) <= -13, f"{i[-1] if i else '?'} LUFS")
    add("no clipping (true peak below -1 dBFS)", pk and float(pk[-1]) < -1.0, f"{pk[-1] if pk else '?'} dBFS")
    black = _ff("-i", str(mp4), "-vf", "blackdetect=d=2:pix_th=0.05", "-an", "-f", "null", "-")
    bl = re.findall(r"black_duration:(\d+\.?\d*)", black)
    add("no long black frames", not bl, f"{len(bl)} black stretches" if bl else "")
    silence = _ff("-i", str(mp4), "-af", "silencedetect=n=-50dB:d=4", "-vn", "-f", "null", "-")
    add("no dead audio (4 s+)", "silence_start" not in silence)
    title = pkg.get("title", "")
    add("title 25-100 characters", 25 <= len(title) <= 100, f"{len(title)}")
    desc = pkg.get("description", "")
    add("description has chapters", len(re.findall(r"\b\d{1,2}:\d{2}\b", desc)) >= 4)
    add("at least 8 tags", len(pkg.get("tags", [])) >= 8, f"{len(pkg.get('tags', []))}")
    lines = [l for l in str(pkg.get("script", "")).split("\n") if l.strip()]
    first = lines[0] if lines else ""
    from studio.auto import hook_is_concrete
    add("hook: concrete first line", hook_is_concrete(first), first[:80])
    add("hook: asks a question", any(l.strip().endswith("?") for l in lines[:12]))
    add("three thumbnails", all((d / "out" / f"thumb{k}.jpg").exists() for k in (1, 2, 3)))
    high = [f for f in pkg.get("flags", []) if isinstance(f, dict) and str(f.get("severity", "")).lower() == "high"]
    add("no high-severity fact-check flags", not high, "; ".join(str(f.get("claim", ""))[:70] for f in high))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--thumb", type=int, default=1)
    ap.add_argument("--thumb-file", default="", help="use this image (a path in the repo) instead of thumb1-3")
    ap.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])
    ap.add_argument("--publish-at", default="", help="schedule the public release, e.g. 2026-10-01T19:00:00Z")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--force", action="store_true", help="upload even if a check fails")
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    pkg = json.loads((d / "auto.json").read_text())
    ov = pathlib.Path("tools/meta_override.json")                       # hand-written title / description / tags / hashtags take priority over the auto-generated ones
    if ov.exists():
        o = json.loads(ov.read_text())
        pkg.update({k: v for k, v in o.items() if k in ("title", "description", "tags") and v})
        print("Using the hand-written metadata from tools/meta_override.json")
    res = checks(d, pkg)
    for name, ok, detail in res:
        print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    failed = [n for n, ok, _ in res if not ok]
    print(f"\n{len(res) - len(failed)}/{len(res)} checks passed.")
    if failed and not a.force:
        print("Not uploading: fix the failing checks (or rerun with force).")
        return 1
    if not a.upload:
        print("Dry run: nothing uploaded.")
        return 0
    meta = {"title": pkg["title"], "description": pkg["description"], "tags": pkg.get("tags", [])}
    cfg = {"upload": {"privacy": a.privacy, "category_id": "27", "synthetic_media_disclosure": False, "publish_at": a.publish_at}}
    vid = yt_upload.upload(d / "out" / "final.mp4", meta, cfg, print, shorts=False, thumb=pathlib.Path(a.thumb_file) if a.thumb_file else d / "out" / f"thumb{a.thumb}.jpg")
    print(f"\nPUBLISHED ({a.privacy}): https://www.youtube.com/watch?v={vid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
