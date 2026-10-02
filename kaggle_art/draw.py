"""Runs on a free Kaggle GPU: draws scene illustrations with FLUX schnell and SDXL-Turbo and writes PNGs to /kaggle/working."""
import json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate", "sentencepiece", "protobuf"], check=False)
import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline
STYLE = ("hand-drawn 2D webcomic illustration, simple white round-headed stick figure characters with messy brown hair and tiny dot eyes, thin black ink outlines, "
         "rough brown tunics, warm muted peach and sepia colour palette, soft paper grain, flat colours, wide shot, no text")
SCENES = ["a medieval London street at dawn, a worker standing knee-deep in a muddy pit holding a wooden bucket, a second man pinching his nose and a speech bubble with a clothespin icon"]
out = "/kaggle/working"
log = {}
try:
    t = time.time()
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    log["sdxl_load_s"] = round(time.time() - t, 1)
    for i, sc in enumerate(SCENES):
        t = time.time()
        img = pipe(prompt=f"{STYLE}, {sc}", negative_prompt="photo, 3d render, realistic, text, watermark, blurry", num_inference_steps=4, guidance_scale=0.0, width=1024, height=576).images[0]
        img.save(f"{out}/sdxl_{i}.png"); log[f"sdxl_img{i}_s"] = round(time.time() - t, 1)
    del pipe; torch.cuda.empty_cache()
except Exception as e:
    log["sdxl_error"] = repr(e)[:300]
try:
    t = time.time()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    log["flux_load_s"] = round(time.time() - t, 1)
    for i, sc in enumerate(SCENES):
        t = time.time()
        img = pipe(f"{STYLE}, {sc}", num_inference_steps=4, guidance_scale=0.0, width=1024, height=576, max_sequence_length=256).images[0]
        img.save(f"{out}/flux_{i}.png"); log[f"flux_img{i}_s"] = round(time.time() - t, 1)
except Exception as e:
    log["flux_error"] = repr(e)[:300]
json.dump(log, open(f"{out}/log.json", "w"), indent=1)
print(log)
