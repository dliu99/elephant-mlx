"""Run elephant-1 (real weights) through elephant_mlx.py on text and image requests; dump answers + slot logits.

usage: python -I tests/real_test.py <model path or repo> <out.json> [--causal]
  --causal  full-attention layers run causal (what a stock mlx-lm / mlx-vlm forward does), for comparison only
"""
import json
import sys
import time
from pathlib import Path

MODEL, OUT = sys.argv[1], Path(sys.argv[2])
CAUSAL = "--causal" in sys.argv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IMG = ROOT / "examples" / "images"


def make_images():
    """Redraw examples/images (the receipt font depends on the machine, so this only runs when they are missing)."""
    from PIL import Image, ImageDraw, ImageFont

    IMG.mkdir(parents=True, exist_ok=True)

    def shape(name, kind, color, size=512):
        im = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(im)
        m = size // 5
        box = (m, m, size - m, size - m)
        if kind == "circle":
            d.ellipse(box, fill=color)
        elif kind == "square":
            d.rectangle(box, fill=color)
        else:
            d.polygon([(size // 2, m), (size - m, size - m), (m, size - m)], fill=color)
        im.save(IMG / name)

    shape("red_circle.png", "circle", (220, 30, 30))
    shape("blue_square.png", "square", (30, 60, 220))
    shape("green_triangle.png", "triangle", (30, 160, 60))
    shape("small_yellow_circle.png", "circle", (240, 200, 0), size=96)  # upscaled by the 65,536-pixel minimum

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = ImageFont.load_default(size=34)
    im = Image.new("RGB", (720, 560), "white")
    d = ImageDraw.Draw(im)
    lines = ["ACME HOME GOODS", "Order #4471   2026-11-02", "", "Desk lamp            $42.10",
             "Card ****1234        $42.10", "Card ****1234        $42.10", "", "TOTAL CHARGED        $84.20"]
    for i, line in enumerate(lines):
        d.text((40, 30 + i * 62), line, fill="black", font=font)
    im.save(IMG / "receipt.png")


COLOR = {"type": "choice", "instructions": "What color is the main shape in the image?",
         "criteria": {"red": "red", "green": "green", "blue": "blue", "yellow": "yellow"}}
SHAPE = {"type": "choice", "instructions": "What shape is shown in the image?",
         "criteria": {"circle": "a circle", "square": "a square", "triangle": "a triangle"}}
REQUESTS = {
    "text_card_example": {
        "state": "I was charged twice this month, please refund one of them.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "Today", "Blocking the customer now"]}}},
    **{f"shape_{n.split('.')[0]}": {"state": "<image:1>", "media": [{"type": "image", "path": n}],
                                    "questions": {"color": COLOR, "shape": SHAPE}}
       for n in ("red_circle.png", "blue_square.png", "green_triangle.png", "small_yellow_circle.png")},
    "two_images": {
        "state": {"first": "<image:1>", "second": "<image:2>"},
        "media": [{"type": "image", "path": "blue_square.png"}, {"type": "image", "path": "red_circle.png"}],
        "questions": {"which": {"type": "choice", "instructions": "Which image shows a circle?",
                                "criteria": {"first": "the first image", "second": "the second image"}},
                      "both_blue": {"type": "noul", "instructions": "Are both shapes blue?"}}},
    "receipt": {
        "state": {"ticket": "I think I was charged twice for order 4471. Receipt attached.", "receipt": "<image:1>"},
        "media": [{"type": "image", "path": "receipt.png"}],
        "questions": {
            "double": {"type": "noul", "instructions": "Does the receipt show the same card charge listed twice?"},
            "total": {"type": "choice", "instructions": "What total does the receipt show as charged?",
                      "criteria": {"42.10": "$42.10", "84.20": "$84.20", "126.30": "$126.30"}},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}}}},
}

if __name__ == "__main__":
    if not (IMG / "receipt.png").exists():
        make_images()
    import mlx.core as mx

    from elephant_mlx import Elephant

    t = time.perf_counter()
    eng = Elephant(MODEL)
    load_s = time.perf_counter() - t
    if CAUSAL:
        original = Elephant._forward

        def causal_forward(self, h, positions, cache):
            tm = self.lm.model
            pe = None
            for layer in tm.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        pe = layer.self_attn.rotary_emb(h, positions)
                    break
            for layer, c in zip(tm.layers, cache):
                mask = None if layer.is_linear or h.shape[1] == 1 else "causal"
                h = layer(h, mask=mask, cache=c, position_ids=positions, position_embeddings=pe)
            return tm.norm(h)

        Elephant._forward = causal_forward
    results = {"model": MODEL, "causal": CAUSAL, "load_s": load_s, "requests": {}}
    for name, req in REQUESTS.items():
        eng.decide(req["state"], req["questions"], req.get("media"), media_root=IMG)  # warm-up
        out = eng.decide(req["state"], req["questions"], req.get("media"), debug=True, media_root=IMG)
        results["requests"][name] = out
        short = {q: (a.get("choice"), a.get("noul"), a.get("score"), a.get("probabilities")) for q, a in out["answers"].items()}
        print(f"{name:28s} {out['qev']['timing']['total_s']*1000:7.0f} ms  prefix {out['qev']['prefix_tokens']:4d}  {short}")
    results["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    print(f"load {load_s:.1f}s  peak memory {results['peak_memory_gb']:.1f} GB")
    OUT.write_text(json.dumps(results, indent=1))
