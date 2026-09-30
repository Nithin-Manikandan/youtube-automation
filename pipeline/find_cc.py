"""List popular YouTube videos whose uploader marked them Creative Commons (CC BY).

    YT_API_KEY=... python -m pipeline.find_cc "podcast interview" [--days 365] [--min-minutes 10]

Free: YT_API_KEY is a YouTube Data API key from the same Google Cloud project you use for uploads.
"CC BY" lets you reuse and edit the video if you credit the creator, but uploaders sometimes label
videos wrongly, so check the channel actually owns the footage. This only lists videos; you get the
file yourself (for example from the creator, or their own download option).
"""
import argparse
import datetime as dt
import os

import requests

API = "https://www.googleapis.com/youtube/v3"


def find(query, days, min_minutes, limit, key):
    after = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get(f"{API}/search", params={
        "part": "snippet", "q": query, "type": "video", "videoLicense": "creativeCommon",
        "order": "viewCount", "publishedAfter": after, "maxResults": 50, "key": key}, timeout=30)
    r.raise_for_status()
    ids = [i["id"]["videoId"] for i in r.json().get("items", [])]
    if not ids:
        return []
    v = requests.get(f"{API}/videos", params={"part": "snippet,statistics,contentDetails,status",
                                              "id": ",".join(ids), "key": key}, timeout=30)
    v.raise_for_status()
    out = []
    for it in v.json().get("items", []):
        if it["status"].get("license") != "creativeCommon":
            continue
        d = it["contentDetails"]["duration"]  # ISO 8601, e.g. PT1H2M3S
        h = int(d.split("T")[1].split("H")[0]) if "H" in d else 0
        m = int(d.split("H")[-1].split("M")[0]) if "M" in d else 0
        if h * 60 + m < min_minutes:
            continue
        out.append({"title": it["snippet"]["title"], "channel": it["snippet"]["channelTitle"],
                    "views": int(it["statistics"].get("viewCount", 0)), "minutes": h * 60 + m,
                    "url": f"https://www.youtube.com/watch?v={it['id']}"})
    return sorted(out, key=lambda x: -x["views"])[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--min-minutes", type=int, default=10)
    ap.add_argument("--limit", type=int, default=15)
    a = ap.parse_args()
    key = os.environ["YT_API_KEY"]
    rows = find(a.query, a.days, a.min_minutes, a.limit, key)
    lines = [f"### Creative Commons (CC BY) results for: {a.query}", ""]
    for r in rows:
        lines.append(f"- **{r['views']:,} views**, {r['minutes']} min, {r['channel']}: [{r['title']}]({r['url']})")
    if not rows:
        lines.append("Nothing found. Try a broader query or a longer --days window.")
    lines += ["", "Credit the creator in the description when you reuse a CC BY video."]
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
