"""Equivalence test for elephant_mlx.py (mlx-vlm backend, images) on a tiny random Qwen3.5 VLM that carries
elephant-1's tokenizer, processor config, config overrides and readout.json.

References (torch CPU, fp32):
  text   qev.engine.DecisionEngine (OneJev @81ce62f + serve.patch), sequential
  images the patched training scorer (train.common.batch_scores, multimodal, noncausal, slot_lm) for a batch of
         one: full prompt through Qwen3_5ForConditionalGeneration with qev.readout's block mask

usage: python -I tests/verify_vision.py <elephant-1 dir> <work dir> <OneJev dir>
  <elephant-1 dir>  the Hub snapshot of baselabs/elephant-1 (config, tokenizer, chat template, readout.json,
                    processor_config.json); the weights are not read
  <work dir>        where the tiny model is written; run compare_processors.py with the same work dir first,
                    it draws the test images into <work dir>/images
  <OneJev dir>      OneJev @81ce62f with serve.patch applied (see setup_reference.sh)
"""
import contextlib
import json
import shutil
import sys
from pathlib import Path

SRC, WORK, ONEJEV = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [ONEJEV, str(Path(__file__).resolve().parents[1])]
IMG = WORK / "images"

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration  # noqa: E402

torch.manual_seed(0)
ref_cfg = json.loads((SRC / "config.json").read_text())
drop = ("model_type", "dtype")
tc = {k: v for k, v in ref_cfg["text_config"].items() if k not in drop}
L = 8
tc.update(hidden_size=128, intermediate_size=256, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=2,
          head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
          linear_value_head_dim=32, mtp_num_hidden_layers=0,
          layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(L)])
tc["rope_parameters"] = dict(tc["rope_parameters"], mrope_section=[2, 1, 1])
vc = {k: v for k, v in ref_cfg["vision_config"].items() if k not in drop}
vc.update(depth=2, hidden_size=64, num_heads=2, intermediate_size=128, out_hidden_size=128)
top = {k: ref_cfg[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
                               "tie_word_embeddings")}
model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=tc, vision_config=vc, **top)).float().eval()
with torch.no_grad():
    for p in model.parameters():
        p.add_(torch.randn_like(p) * 0.05)

D = WORK / "vlm"
shutil.rmtree(D, ignore_errors=True)
model.save_pretrained(D)
cfg = json.loads((D / "config.json").read_text())
cfg["modeling_overrides"] = ref_cfg["modeling_overrides"]
(D / "config.json").write_text(json.dumps(cfg, indent=2))
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "readout.json", "processor_config.json"):
    shutil.copy(SRC / f, D / f)
del model

TEXT = {
    "state": {"ticket": "My card was charged twice for the same order and support has not replied in 3 days.",
              "customer": {"tenure_months": 14, "vip": False}},
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "charges and refunds", "technical": "bugs", "sales": "pricing"}},
        "many": {"type": "choice", "instructions": "Pick the product line.",
                 "criteria": {f"line_{i}": f"product line number {i}" for i in range(30)}},
    },
}
IMAGES = [
    {"state": {"photo": "<image:1>", "screenshot": "<image:2>", "note": "Customer says the item arrived damaged."},
     "media": [{"type": "image", "path": "tiny.png"}, {"type": "image", "path": "tall.png"}]},
    {"state": "Receipt photo: <image:1>\nThe customer asks whether this was charged twice.",
     "media": [{"type": "image", "path": "mid.png"}]},
]
IMG_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is a refund warranted?"},
    "kind": {"type": "choice", "instructions": "What does the image show?",
             "criteria": {"receipt": "a receipt", "product": "a product", "screenshot": "an app screenshot"}},
    "severity": {"type": "score", "instructions": "How severe is the problem?",
                 "criteria": ["none", "minor", "major", "critical"]},
}
DEBIAS = 2

from qev import media as qmedia  # noqa: E402
from qev import readout as ro  # noqa: E402
from qev.engine import DecisionEngine  # noqa: E402
from qev.prompt import build_messages, render_question  # noqa: E402
from qev.schema import SystemOneRequest  # noqa: E402

eng = DecisionEngine(str(D), device="cpu", dtype="float32", fork_mode="sequential")
assert eng.info.extra["attention"] == "noncausal", eng.info.extra
_, ref_text_meta = eng.decide(SystemOneRequest(**TEXT), debias=DEBIAS, debug=True)
slot_ids = eng.slot_ids
del eng

proc = AutoProcessor.from_pretrained(D)
vlm = Qwen3_5ForConditionalGeneration.from_pretrained(D, dtype=torch.float32).eval()
ro.install_noncausal(vlm)


def ref_image(req, noncausal=True):
    request = SystemOneRequest(state=req["state"], questions=IMG_QUESTIONS)
    images, _, _ = qmedia.load(req["media"], root=IMG)
    out, prefix_lens = {}, set()
    for qid, q in request.questions.items():
        k = render_question(qid, q).n_slots
        rows = []
        for s in range(min(DEBIAS, k)):
            r = render_question(qid, q, order=[(i + s) % k for i in range(k)] if s else None)
            prompt = qmedia.render_prompt(proc, {"enable_thinking": False}, build_messages(request.state, r.suffix),
                                          req["media"])
            inputs = proc(text=[prompt], images=images, return_tensors="pt")
            ids, prefix, _ = ro.token_layout(proc.tokenizer, prompt, prompt.rindex(r.suffix), None)
            n = inputs["input_ids"].shape[1]
            P = prefix + (n - len(ids))
            prefix_lens.add(P)
            ctx = ro.blocks(ro.block_ids([P], [n], n, "cpu")) if noncausal else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                logits = vlm(**inputs, use_cache=False, return_dict=True, logits_to_keep=1).logits[0, -1]
            rows.append(logits[slot_ids[:k]].tolist())
        out[qid] = rows
    return out, prefix_lens


from elephant_mlx import Elephant  # noqa: E402


def maxdiff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


print("== text request (reference: qev DecisionEngine)")
for name, kw in (("mlx-vlm backend", {}), ("mlx-lm --text-only", {"text_only": True})):
    out = Elephant(str(D), **kw).decide(TEXT["state"], TEXT["questions"], debias=DEBIAS, debug=True)
    worst = max(maxdiff(out["qev"]["diagnostics"][q]["slot_logits"], ref_text_meta["diagnostics"][q]["slot_logits"])
                for q in TEXT["questions"])
    print(f"  {name:20s} prefix {out['qev']['prefix_tokens']} (ref {ref_text_meta['prefix_tokens']})  "
          f"max |logit diff| {worst:.2e}")

engine = Elephant(str(D))


class HFPixels:
    """The transformers image processor (training stack), to isolate preprocessing from the model port."""

    def __init__(self, ip):
        self.ip = ip

    def __call__(self, images):
        o = self.ip(images=images, return_tensors="np")
        return {"pixel_values": o["pixel_values"], "image_grid_thw": o["image_grid_thw"]}


own_ip = engine.image_processor
print("== image requests (reference: patched training scorer, noncausal)")
for i, req in enumerate(IMAGES):
    ref, plens = ref_image(req)
    causal, _ = ref_image(req, noncausal=False)
    sens = max(maxdiff(ref[q], causal[q]) for q in IMG_QUESTIONS)
    for label, ip in (("same pixels", HFPixels(proc.image_processor)), ("end to end", own_ip)):
        engine.image_processor = ip
        out = engine.decide(req["state"], IMG_QUESTIONS, req["media"], debias=DEBIAS, debug=True, media_root=IMG)
        diff = max(maxdiff(out["qev"]["diagnostics"][q]["slot_logits"], ref[q]) for q in IMG_QUESTIONS)
        print(f"  request {i} [{label:11s}] prefix {out['qev']['prefix_tokens']} (ref {sorted(plens)})  "
              f"max |logit diff| {diff:.2e}   (causal-vs-noncausal in ref: {sens:.2e})")
