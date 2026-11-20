"""Compare image preprocessing: transformers AutoProcessor (training stack) vs mlx-vlm's numpy Qwen3VL processor.

usage: python -I tests/compare_processors.py <elephant-1 Hub snapshot> <work dir>
  draws four test images into <work dir>/images (verify_vision.py reuses them)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

D, IMG = Path(sys.argv[1]), Path(sys.argv[2]) / "images"
IMG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
sizes = {"tiny": (120, 90), "mid": (640, 480), "big": (2000, 1500), "tall": (300, 1400)}
paths = {}
for name, (w, h) in sizes.items():
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([(x * 255 / w), (y * 255 / h), ((x + y) % 64) * 4], -1)
    arr = np.clip(base + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
    p = IMG / f"{name}.png"
    Image.fromarray(arr).save(p)
    paths[name] = p

from transformers import AutoProcessor  # noqa: E402

hf = AutoProcessor.from_pretrained(D)
print("transformers image processor:", type(hf.image_processor).__name__, hf.image_processor.size)

from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor  # noqa: E402

for fix in (False, True):
    kw = dict(patch_size=16, temporal_patch_size=2, merge_size=2)
    if fix:
        kw.update(min_pixels=hf.image_processor.size["shortest_edge"], max_pixels=hf.image_processor.size["longest_edge"])
    mp = Qwen3VLImageProcessor(**kw)
    print(f"\nmlx-vlm processor ({'checkpoint sizes' if fix else 'defaults'}): min {mp.min_pixels} max {mp.max_pixels}")
    for name, p in paths.items():
        im = Image.open(p).convert("RGB")
        a = hf.image_processor(images=[im], return_tensors="np")
        b = mp(images=[im])
        ga, gb = a["image_grid_thw"].tolist(), np.asarray(b["image_grid_thw"]).tolist()
        msg = f"  {name:5s} {im.size}: grid hf {ga} mlx {gb}"
        if ga == gb:
            pa, pb = np.asarray(a["pixel_values"], np.float32), np.asarray(b["pixel_values"], np.float32)
            msg += f"  pixels max|d| {np.abs(pa - pb).max():.4f} mean|d| {np.abs(pa - pb).mean():.5f}"
        print(msg)
