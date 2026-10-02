# Handoff: animation upgrade (read this first in a new Claude Code session)

Repo: Nithin-Manikandan/youtube-automation. **Work branch: `claude/youtube-automation-review-k9f3cu`** (main already has everything through PR 42; the animation work below is NOT merged to main).
Start with: `git fetch origin && git checkout claude/youtube-automation-review-k9f3cu`.

## The user's goal and rules
- Free, fully automated YouTube history-video pipeline (script, fact-check, stick-figure scenes, voice, music/SFX, thumbnails, metadata, approve/publish).
- Current mission: make the animation "Disney level", matching the voiceover at every moment, and stop looking "goofy". The user gets angry if told it is not possible, and wants problems fixed without being told to. Report honestly what is still weak.
- **Do NOT upload to YouTube and do NOT merge the animation branch to main without the user's explicit OK.**
- Never ask the user to paste credentials in chat. The "altered/synthetic content" label must stay OFF (pipeline/upload.py defaults it to False).
- Apollo 13 video was already published public (https://www.youtube.com/watch?v=AY9VlfxRhfM); leave it alone. Do not schedule or upload Krakatoa again. A new channel "Stickman Chronicles" is being set up.
- Pending housekeeping for the user: rotate the Google OAuth client secret (it appeared in screenshots), post/pin the first comment on Apollo, confirm the Apollo thumbnail shows, confirm which channel the Apollo upload landed on.

## What is built on the branch (animation engine v2)
- `studio/char.py`: shaded outlined characters (`draw_v2`), big faces, brows, lip-sync mouth, blink, mitten hands, cast shadows, rim light. Eyes: lids calmed, pupils bigger and centred (last commit).
- `studio/acting.py`, `studio/beats.py`, `studio/reactions.py`, `studio/mocap.py`, `tools/build_motion_lib.py`, `studio/motion_data.npz`, `studio/motion_index.json`: retargeted CMU mocap clips (free for all uses) plus hand-keyed spring reactions (flinch, cower, duck, slump, stumble, look). Acting beats are timed to narration word timings. Crouchy mocap clips were dropped because they caused the squatting/head-butting the user hated.
- `studio/stick.py`: Actor, camera, parallax near/far layers, bloom + split-tone grade, `make_shots` (cuts at narration pauses, titles keep heads below them, spacecraft cut-ins on different parts). `_warp` now takes a `border` argument; near-layer warps use BORDER_CONSTANT (fixed the smeared "flask" Earth artifact).
- `studio/space.py` (new): lit spherical Earth/Moon with procedural textures, atmosphere glow, slow rotation; gradient-shaded Apollo spacecraft sprite with rolling panels. Sprites are cached (about 0.3 s/frame extra on space scenes). `studio/art.py` `spacecraft`/`planet` call it.
- `studio/recipes.py`, `studio/subjects.py`, `studio/auto.py`, `studio/sfx.py`, `studio/thumb.py`: new props (tank, parachute, flag, cannon, clock), moon scenes, footsteps SFX, thumbnails with the new characters. Titles are now at y=.04, size .072.
- `studio/build.py`: captions at y_position 0.92 (landscape).
- Publishing: `.github/workflows/publish.yml` (inputs run_id, thumb, thumb_file, privacy, publish_at, upload), `studio/publish_cli.py`, `pipeline/upload.py`, `tools/youtube_auth.py`.

## Test builds (GitHub Actions workflow `auto-video.yml`, input name is `hint`, ref = the branch)
- Run 36 (commit 004ade1), 37 (6ed8b88), 38 (1aaea28): Apollo 13 test builds, reviewed. Findings: no squatting or head collisions; spacecraft shots repetitive; Earth/spacecraft looked flat; title overlapped heads in wide shots; captions overlapped legs in close two-shots.
- A build started around 21:08 UTC on 2026-10-01 (commit 4f2ccdb, shaded Earth/spacecraft) was NOT reviewed. Commit 0677a97 (eye fix) has NOT been built at all. **Next step: trigger a fresh Auto video run on the branch with hint "Apollo 13", download the `auto-video` artifact (out/final.mp4), review a contact sheet of frames across the video, cut a 90 s excerpt (ffmpeg via imageio-ffmpeg), and show the user.**
- The weakest remaining points: captions still cover legs in close cabin two-shots; fight weapons are plain sticks; arms/hands look blobby; scene variety (cabin and spacecraft repeat); audio never checked by ear.
- Local quick-render harness (in the old session scratchpad, not in the repo): `anim/test.py` rendered fixed stage scenes with `recipes.clean_visual` -> `subjects.enrich` -> `recipes.build_stage` -> `stick.prepare` -> `stick.render_frame`. Recreate it if needed.

## Blender experiment (user said: "Try blender and be as good as possible")
Context: the user shared a Notion tutorial "The Documentary Skill" (PYNK Society). Verified: it is only a prompt planner for paid AI image/video tools (Seedance etc.); it generates nothing, so it is not usable in the free automated pipeline. Borrowable story ideas: in-story clock-time overlays ("09:14"), a curiosity-loop question at the end of each scene, cold clinical narrator, locked style/character descriptions.

Findings so far (nothing of this is committed yet):
- `pip install bpy` (Blender 5.0.1 as a Python module, wheel about 374 MB, `bpy-5.0.1-cp311-cp311-manylinux_2_28_x86_64.whl`) works headless on Python 3.11 here.
- Cycles CPU works with no GPU: a trivial scene at 1280x720, 24 samples took about 8.5 s including startup (per-frame time for a real scene is unmeasured).
- EEVEE needs `apt-get update && apt-get install -y libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri ...`; it then renders through software GL but the first frame took about 43 s and printed EGL warnings. Cycles is the safer choice.
- Plan (not started): `studio/b3d.py` builds a Blender scene from the existing scene dict: characters as outlined capsule/sphere rigs driven by the same 11 pose channels (`torso, head, a1, a2, b1, b2, l1, l2, m1, m2, lift`) from `Actor.key_state(t)` and `Actor.extras(t, scene)` (blink, mouth, brow, look); inverted-hull outlines; helmet glass; capsule cabin set (panel with blinking emissive buttons, round windows, red alarm light), space set (textured Earth, spacecraft from primitives, star world); camera from the scene's shots; Cycles at low samples with denoise; render at 15 fps ("on twos") and/or lower resolution to keep time down; use 3D only for supported backgrounds (capsule, space) and fall back to the 2D engine otherwise; then run the existing `_grade`, title and caption overlays on the 3D frame. Key numbers: actor unit scale S=1 means character height about 1.0, ground at gy=1.38 in `draw_v2` local space; world x = (xr-0.5)*4.44; camera distance about 6.17 for a full-width wide shot (35 mm-equivalent, 36 mm sensor).
- Render cost is the risk: about 9000 frames for a 5-minute video, so budget seconds per frame and shard across more Actions jobs. Install `bpy` plus the apt libs inside the workflow, and measure before committing to it. If it is too slow, keep improving the 2D engine instead.

## Housekeeping notes
- GitHub MCP tools disconnect intermittently; reload them with ToolSearch (`select:mcp__github__actions_list,...`). In a normal Claude Code session use `gh` instead if available.
- Scheduled check-in triggers from the cloud session (e.g. `trig_01UqkmTRMtECvEiaYrafxvru`, `trig_018JJmP8hoEKB4jP1Dk55TGo`, `trig_013SiTV2a2SzfxZSBqKixdVt`) may still fire into the old session; they are harmless and can be ignored.
