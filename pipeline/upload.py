"""YouTube upload via the Data API (resumable). Needs YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN."""
import json
import os

import requests


def _access_token():
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": os.environ["YT_CLIENT_ID"], "client_secret": os.environ["YT_CLIENT_SECRET"],
        "refresh_token": os.environ["YT_REFRESH_TOKEN"], "grant_type": "refresh_token"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def upload(mp4, data, cfg, log=print):
    up = cfg.get("upload", {})
    title = data["title"][:100]
    desc = data.get("description", "")
    if "#shorts" not in desc.lower():
        desc += "\n#shorts"
    body = {
        "snippet": {"title": title, "description": desc, "tags": data.get("tags", [])[:15],
                    "categoryId": up.get("category_id", "24")},
        "status": {"privacyStatus": up.get("privacy", "private"), "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": bool(up.get("synthetic_media_disclosure", True))},
    }
    token = _access_token()
    size = os.path.getsize(mp4)
    init = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
                 "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(size)},
        data=json.dumps(body), timeout=60)
    init.raise_for_status()
    with open(mp4, "rb") as f:
        r = requests.put(init.headers["Location"], data=f, headers={"Content-Type": "video/mp4"}, timeout=600)
    r.raise_for_status()
    vid = r.json()["id"]
    log(f"uploaded: https://youtube.com/shorts/{vid}  (privacy: {body['status']['privacyStatus']})")
    return vid
