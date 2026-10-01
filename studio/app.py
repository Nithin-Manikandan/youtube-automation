"""Local studio: python -m studio  ->  http://localhost:5055"""
import json
import os
import pathlib
import re
import threading
import time
import traceback

from flask import Flask, jsonify, request, send_file, send_from_directory

from pipeline import upload as yt_upload
import shutil

from . import ai, auto, build

HERE = pathlib.Path(__file__).parent
PROJECTS = HERE / "projects"
SETTINGS = HERE / "settings.json"
PROJECTS.mkdir(exist_ok=True)
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3
JOBS = {}

DEFAULTS = {"aspect": "16:9", "voice": "en-US-AndrewMultilingualNeural", "rate": "+0%", "pitch": "+0Hz", "captions": True,
            "music_volume": 0.12, "hook_text": "", "privacy": "private", "minutes": 3, "niche": "History"}


def load_settings():
    s = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    for k in ("GEMINI_API_KEY", "YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"):
        if s.get(k):
            os.environ[k] = s[k]
    return s


def pdir(pid):
    d = PROJECTS / re.sub(r"[^a-z0-9\-]", "", pid)
    if not d.exists():
        raise FileNotFoundError(pid)
    return d


def load(pid):
    return json.loads((pdir(pid) / "project.json").read_text())


def save(pid, proj):
    (pdir(pid) / "project.json").write_text(json.dumps(proj, indent=1))


def ok(**kw):
    return jsonify(ok=True, **kw)


@app.errorhandler(Exception)
def err(e):
    traceback.print_exc()
    return jsonify(ok=False, error=str(e)), (404 if isinstance(e, FileNotFoundError) else 500)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/settings")
def get_settings():
    s = load_settings()
    return ok(settings={k: ("set" if v else "") for k, v in s.items()})


@app.post("/api/settings")
def set_settings():
    s = load_settings()
    for k, v in request.json.items():
        if v:
            s[k] = v.strip()
    SETTINGS.write_text(json.dumps(s))
    load_settings()
    return get_settings()


@app.get("/api/projects")
def projects():
    out = []
    for d in sorted(PROJECTS.iterdir()):
        if (d / "project.json").exists():
            out.append({"id": d.name, "name": json.loads((d / "project.json").read_text()).get("name", d.name)})
    return ok(projects=out)


@app.post("/api/projects")
def new_project():
    name = (request.json.get("name") or "Untitled").strip()
    pid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
    pid = f"{pid}-{int(time.time()) % 100000}"
    d = PROJECTS / pid
    (d / "clips").mkdir(parents=True)
    (d / "out").mkdir()
    save_path = d / "project.json"
    save_path.write_text(json.dumps({"name": name, "formula": "", "ideas": None, "idea": "", "scenes": [],
                                     "characters": [], "meta": {"title": "", "description": "", "tags": []},
                                     "settings": DEFAULTS, "music": None}))
    return ok(id=pid)


@app.get("/api/p/<pid>")
def get_project(pid):
    p = load(pid)
    p["has_final"] = (pdir(pid) / "out" / "final.mp4").exists()
    p["clip_files"] = sorted(f.name for f in (pdir(pid) / "clips").iterdir())
    return ok(project=p, job=JOBS.get(pid))


@app.put("/api/p/<pid>")
def put_project(pid):
    p = load(pid)
    for k in ("scenes", "settings", "meta", "characters", "formula", "idea"):
        if k in request.json:
            p[k] = request.json[k]
    save(pid, p)
    return ok()


@app.post("/api/p/<pid>/formula")
def formula(pid):
    p = load(pid)
    urls = [u for u in re.split(r"\s+", request.json.get("urls", "")) if u.startswith("http")]
    text = (request.json.get("text") or "").strip()
    if urls:
        text = ai.decode_formula(urls, p["settings"].get("niche", ""))
    if not text:
        raise RuntimeError("Paste the top video links, or paste a formula.")
    p["formula"] = text
    save(pid, p)
    return ok(formula=text)


@app.post("/api/p/<pid>/ideas")
def ideas(pid):
    p = load(pid)
    if not p["formula"]:
        raise RuntimeError("Do the Formula step first.")
    p["ideas"] = ai.ideas(p["formula"], p["settings"].get("niche", ""))
    save(pid, p)
    return ok(ideas=p["ideas"])


@app.post("/api/p/<pid>/plan")
def plan(pid):
    p = load(pid)
    idea = (request.json.get("idea") or "").strip()
    if not idea or not p["formula"]:
        raise RuntimeError("Need a formula and a video idea.")
    st = {**p["settings"], **request.json.get("settings", {})}
    data = ai.plan_video(p["formula"], idea, float(st.get("minutes", 3)), st.get("aspect") == "9:16", st.get("niche", ""))
    p["idea"], p["settings"] = idea, st
    p["characters"] = data.get("characters", [])
    p["meta"] = {k: data.get(k, p["meta"].get(k)) for k in ("title", "description", "tags")}
    p["settings"]["hook_text"] = data.get("hook_text", "")
    p["scenes"] = [{"n": i + 1, "voiceover": s["voiceover"], "image_prompt": s.get("image_prompt", ""),
                    "anim_prompt": s.get("anim_prompt", ""), "clip": None} for i, s in enumerate(data["scenes"])]
    save(pid, p)
    return ok(project=p)


@app.post("/api/p/<pid>/import")
def import_scenes(pid):
    p = load(pid)
    text = request.json.get("text", "")
    scenes = build.parse_table(text) or build.split_script(text)
    if not scenes:
        raise RuntimeError("Nothing to import.")
    p["scenes"] = [{**s, "clip": None} for s in scenes]
    save(pid, p)
    return ok(project=p)


@app.post("/api/p/<pid>/clips")
def upload_clips(pid):
    p, d = load(pid), pdir(pid)
    saved = []
    for f in request.files.getlist("files"):
        name = re.sub(r"[^A-Za-z0-9._\- ]", "_", f.filename or "clip.mp4")
        f.save(d / "clips" / name)
        saved.append(name)
    free = [s for s in p["scenes"] if not s.get("clip")]
    numbered = {}
    for name in saved:
        m = re.search(r"scene[\s_\-]*(\d+)", name, re.I) or re.search(r"(\d+)", name)
        if m and int(m.group(1)) <= len(p["scenes"]):
            numbered[name] = int(m.group(1))
    rest = []
    for name in sorted(saved, key=lambda n: (n not in numbered, numbered.get(n, 0), n.lower())):
        if name in numbered and not p["scenes"][numbered[name] - 1].get("clip"):
            p["scenes"][numbered[name] - 1]["clip"] = name
        else:
            rest.append(name)
    for name, s in zip(rest, [s for s in p["scenes"] if not s.get("clip")]):
        s["clip"] = name
    save(pid, p)
    return ok(project=p)


@app.post("/api/p/<pid>/music")
def upload_music(pid):
    p, d = load(pid), pdir(pid)
    f = request.files["file"]
    name = "music_" + re.sub(r"[^A-Za-z0-9._\-]", "_", f.filename)
    f.save(d / name)
    p["music"] = name
    save(pid, p)
    return ok(music=name)


@app.get("/api/p/<pid>/file/<kind>/<path:name>")
def file(pid, kind, name):
    base = pdir(pid) / ("clips" if kind == "clips" else "out")
    return send_file((base / name).resolve())


def _run_build(pid):
    job = JOBS[pid]
    log = lambda m: job["log"].append(m)
    try:
        p, d = load(pid), pdir(pid)
        scenes = [{**s, "clip_path": str(d / "clips" / s["clip"]) if s.get("clip") else None} for s in p["scenes"]]
        proj = {"scenes": scenes, "settings": p["settings"], "out_dir": str(d / "out"),
                "music_path": str(d / p["music"]) if p.get("music") else None}
        build.build(proj, log)
        job["state"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["log"].append("ERROR: " + str(e))
        job["state"] = "error"


@app.post("/api/p/<pid>/build")
def start_build(pid):
    load(pid)
    if JOBS.get(pid, {}).get("state") == "running":
        raise RuntimeError("Already building.")
    JOBS[pid] = {"state": "running", "log": []}
    threading.Thread(target=_run_build, args=(pid,), daemon=True).start()
    return ok()


@app.get("/api/p/<pid>/job")
def job(pid):
    return ok(job=JOBS.get(pid))


@app.post("/api/p/<pid>/publish")
def publish(pid):
    p, d = load(pid), pdir(pid)
    load_settings()
    if not all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")):
        raise RuntimeError("Add your YouTube keys in Settings first.")
    logs = []
    vertical = p["settings"].get("aspect") == "9:16"
    cfg = {"upload": {"privacy": p["settings"].get("privacy", "private"), "category_id": "27",
                      "synthetic_media_disclosure": True}}
    vid = yt_upload.upload(d / "out" / "final.mp4", p["meta"], cfg, logs.append, shorts=vertical)
    return ok(video_id=vid, url=logs[-1] if logs else "")


AUTO = {"pid": None, "job": None}


def _auto_thread(pid, hint):
    job = AUTO["job"]
    try:
        load_settings()
        settings = {**DEFAULTS, **json.loads((PROJECTS / pid / "project.json").read_text())["settings"]}
        auto.run(job, PROJECTS / pid, settings, hint)
        job["state"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["log"].append("ERROR: " + str(e))
        job["state"] = "error"


@app.post("/api/auto/start")
def auto_start():
    if AUTO["job"] and AUTO["job"]["state"] == "running":
        raise RuntimeError("A video is already being made.")
    pid = f"auto-{int(time.time())}"
    d = PROJECTS / pid
    (d / "out").mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"name": "Auto " + time.strftime("%b %d %H:%M"), "auto": True, "settings": DEFAULTS}))
    AUTO["pid"], AUTO["job"] = pid, {"state": "running", "stage": 0, "log": [], "stages": auto.STAGES, "started": time.time()}
    threading.Thread(target=_auto_thread, args=(pid, (request.json or {}).get("hint")), daemon=True).start()
    return ok(pid=pid)


@app.get("/api/auto/status")
def auto_status():
    return ok(pid=AUTO["pid"], job=AUTO["job"])


@app.get("/api/auto/pending")
def auto_pending():
    out = []
    for d in sorted(PROJECTS.glob("auto-*"), reverse=True):
        if (d / "auto.json").exists() and not (d / "published.json").exists():
            pkg = json.loads((d / "auto.json").read_text())
            pkg.pop("script", None)
            out.append({"pid": d.name, **pkg})
    return ok(pending=out)


@app.get("/api/auto/<pid>/script")
def auto_script(pid):
    return ok(script=json.loads((pdir(pid) / "auto.json").read_text()).get("script", ""))


@app.post("/api/auto/<pid>/publish")
def auto_publish(pid):
    d = pdir(pid)
    load_settings()
    if not all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")):
        raise RuntimeError("Add your YouTube keys in Settings first.")
    b = request.json
    meta = {"title": b["title"], "description": b["description"], "tags": b.get("tags", [])}
    cfg = {"upload": {"privacy": b.get("privacy", "public"), "category_id": "27", "synthetic_media_disclosure": False}}
    logs = []
    vid = yt_upload.upload(d / "out" / "final.mp4", meta, cfg, logs.append, shorts=False, thumb=d / "out" / b.get("thumb", "thumb1.jpg"))
    (d / "published.json").write_text(json.dumps({"video_id": vid, "log": logs, "at": time.time()}))
    return ok(video_id=vid, url=f"https://www.youtube.com/watch?v={vid}", log=logs)


@app.post("/api/auto/<pid>/discard")
def auto_discard(pid):
    shutil.rmtree(pdir(pid))
    return ok()


def main():
    load_settings()
    port = int(os.environ.get("PORT", 5055))
    print(f"Studio running at http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
