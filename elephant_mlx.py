"""baselabs/elephant-1 on Apple silicon: an MLX port of the checkpoint's decision path, text and images.

The reference is OneJev qev @81ce62f + the checkpoint's serve.patch (CUDA only), and for images the patched training
scorer (train.common.batch_scores), which is how the model was trained and evaluated on images. This file
reproduces it on mlx-vlm's qwen3_5 model (vision tower + language model), or on mlx-lm's text model with
--text-only:

  * prompt: qev-labels-v2 (system prompt, <state> block, question, lettered options, answer line), Qwen chat
    template with enable_thinking=False and the generation prompt. Images are request `media` referenced from the
    state by <image:1>, <image:2>, ... (once each, in order); each becomes a vision block at that spot.
  * one prefill of the shared prefix (system turn + state block, images included; the vision tower runs once),
    then one forward per question suffix from a copy of that cache. No token is sampled.
  * attention (config.json `modeling_overrides`): full-attention layers are block-bidirectional. With two blocks,
    prefix and suffix, that is no mask at all on both forwards; the GatedDeltaNet layers stay causal.
  * positions: multimodal RoPE from the model's get_rope_index on the prefix; suffix token i sits at i + delta.
  * images are resized with the checkpoint's own limits (processor_config.json: 65,536 to 16,777,216 pixels);
    mlx-vlm's processor would otherwise fall back to Qwen2-VL defaults and change the vision token count.
  * readout `slot_lm`: softmax over the LM-head logits of the answer labels (A..Z, then two-letter labels) at the
    last token, with the head rows in float32 as the reference computes them. No calibration.json ships, so T = 1.

    python elephant_mlx.py --model devinliu420/elephant-1-mlx-8bit --request examples/receipt.json   # 8-bit, 10.4 GB
    python elephant_mlx.py --model baselabs/elephant-1 --debias 2            # original bf16 weights, 18.8 GB
    python elephant_mlx.py --model ./elephant-1-mlx-lm-8bit --text-only      # mlx_lm.convert output (no vision)

A request is {"state": ..., "questions": {...}, "media": [{"type": "image", "path" | "url" | "data": ...}]};
relative paths resolve against the request file's folder. Videos are not supported.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------- prompt (qev/prompt.py, qev-labels-v2)

PROMPT_VERSION = "qev-labels-v2"
SYSTEM_PROMPT = ("Apply the question to the state. Choose exactly one of the listed options. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS = tuple(list(LETTERS) + [a + b for a in LETTERS for b in LETTERS
                                     if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)])
NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"
MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS = 255, 10
MARK = "QEV_PREFIX_MARK"
PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")


def render_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def render_question(q: dict, order: Sequence[int] | None = None) -> dict:
    kind = q.get("type")
    if kind == "noul":
        instructions = render_entry(q.get("instructions")) or NOUL_DEFAULT_INSTRUCTIONS
        crit = q.get("criteria") or {}
        t = render_entry(crit["true"]) if crit.get("true") is not None else NOUL_DEFAULT_TRUE
        f = render_entry(crit["false"]) if crit.get("false") is not None else NOUL_DEFAULT_FALSE
        labels, names, descs = ["yes", "no"], ["yes", "no"], [t, f]
    elif kind == "choice":
        crit = q.get("criteria") or {}
        if not crit or len(crit) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 1..{MAX_CHOICE_OPTIONS} options in criteria")
        instructions = render_entry(q.get("instructions")) or "Which option applies to the state?"
        labels = list(crit.keys())
        names, descs = list(labels), [crit[k] for k in labels]
    elif kind == "score":
        crit = q.get("criteria") or []
        if not crit or len(crit) > MAX_SCORE_LEVELS:
            raise ValueError(f"score needs 1..{MAX_SCORE_LEVELS} levels in criteria")
        instructions = render_entry(q.get("instructions")) or "Which level describes the state?"
        labels = [str(i) for i in range(len(crit))]
        names, descs = [f"level {i}" for i in range(len(crit))], list(crit)
    else:
        raise ValueError(f"unknown question type {kind!r}")
    if order is not None:
        labels, names, descs = ([x[i] for i in order] for x in (labels, names, descs))
    letters = SLOT_LABELS[: len(labels)]
    lines = []
    for letter, name, desc in zip(letters, names, descs):
        text = render_entry(desc)
        lines.append(f"{letter}. {name}: {text}" if text else f"{letter}. {name}")
    header = "Question:" if kind != "score" else "Rate the state:"
    answer = "Answer with one letter: {letters}." if len(letters) <= 26 else "Answer with one label: {letters}."
    suffix = f"{header} {instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\n" + answer.format(letters=", ".join(letters))
    return {"kind": kind, "labels": labels, "descriptions": descs, "suffix": suffix}


def content_parts(text: str, media: Sequence[dict]) -> list[dict]:
    """The user turn split at <image:N> placeholders into chat content parts (qev/media.py)."""
    parts, pos, seen = [], 0, []
    for m in PLACEHOLDER.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        kind, n = m.group(1), int(m.group(2))
        if n < 1 or n > len(media) or media[n - 1]["type"] != kind:
            raise ValueError(f"placeholder {m.group(0)} does not match media {n} of {len(media)}")
        parts.append({"type": kind})
        seen.append(n)
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != list(range(1, len(media) + 1)):
        raise ValueError(f"media must be referenced once each, in order, by <image:N>; saw {seen} for {len(media)} items")
    return parts


def build_messages(state: Any, suffix: str, media: Sequence[dict]) -> list[dict]:
    user = f"<state>\n{render_state(state)}\n</state>\n\n" + suffix
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_parts(user, media) if media else user}]


# ---------------------------------------------------------------- media (qev/media.py, images only)

def open_image(item: dict, root: Path | None = None):
    from PIL import Image

    if "data" in item:
        s = str(item["data"])
        raw = base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)
    elif "url" in item:
        req = urllib.request.Request(item["url"], headers={"User-Agent": "qev"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
    elif "path" in item:
        p = Path(item["path"])
        with Image.open(p if p.is_absolute() or root is None else root / p) as im:
            return im.convert("RGB")
    else:
        raise ValueError("an image needs one of path, url or data")
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")


# ---------------------------------------------------------------- answers (qev/answers.py)

def _normalize(p):
    s = float(sum(p))
    return [1.0 / len(p)] * len(p) if s <= 0 else [float(x) / s for x in p]


def _r(x):
    return float(round(x, 6))


def softmax(z):
    m = max(z)
    w = [math.exp(x - m) for x in z]
    s = sum(w)
    return [x / s for x in w]


def make_noul(p_yes):
    return {"type": "noul", "noul": _r(min(1.0, max(0.0, p_yes)))}


def make_choice(options, probs):
    p = _normalize(probs)
    best = max(range(len(p)), key=p.__getitem__)
    conf = 1.0 if len(p) == 1 else (max(p) - 1 / len(p)) / (1 - 1 / len(p))
    return {"type": "choice", "choice": options[best], "probabilities": {o: _r(x) for o, x in zip(options, p)},
            "confidence": _r(conf)}


def make_score(levels, probs):
    p = _normalize(probs)
    k = len(p)
    if k == 1:
        conf = 1.0
    else:
        mode = max(range(k), key=p.__getitem__)
        dist = sum(x * abs(i - mode) for i, x in enumerate(p))
        uniform_mad = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
        conf = max(0.0, 1.0 - dist / uniform_mad)
    return {"type": "score", "score": _r(sum(i * x for i, x in enumerate(p))),
            "legend": {str(i): lvl for i, lvl in enumerate(levels)},
            "probabilities": {str(i): _r(x) for i, x in enumerate(p)}, "confidence": _r(conf)}


# ---------------------------------------------------------------- engine

class Elephant:
    def __init__(self, model: str, text_only: bool = False, max_branch_tokens: int = 32768,
                 max_request_tokens: int = 65536):
        self.max_branch_tokens, self.max_request_tokens = max_branch_tokens, max_request_tokens
        self.vision = not text_only
        if text_only:
            from mlx_lm import load

            self.model, self.tokenizer, config = load(model, return_config=True)
        else:
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
            from mlx_vlm.utils import get_model_path, load_model
            from transformers import AutoTokenizer

            path = get_model_path(model)
            self.model = load_model(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            config = json.loads((path / "config.json").read_text())
            ip = json.loads((path / "processor_config.json").read_text())["image_processor"]
            size = ip.get("size") or {}  # HF layout: size{shortest,longest}_edge; mlx_vlm.convert: min/max_pixels
            self.image_processor = Qwen3VLImageProcessor(
                patch_size=ip["patch_size"], temporal_patch_size=ip["temporal_patch_size"], merge_size=ip["merge_size"],
                min_pixels=size.get("shortest_edge", ip.get("min_pixels")),
                max_pixels=size.get("longest_edge", ip.get("max_pixels")),
                image_mean=ip["image_mean"], image_std=ip["image_std"], rescale_factor=ip["rescale_factor"])
            self.merge = ip["merge_size"]
            self.image_id = config["image_token_id"]
            self.vision_ids = {config[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                                   "vision_end_token_id")}
        ov = config.get("modeling_overrides")
        if ov and (ov["attention"]["full_attention"] != "block_bidirectional" or ov.get("readout") != "slot_lm"):
            raise ValueError(f"this port implements block_bidirectional + slot_lm, checkpoint says {ov}")
        self.lm = self.model.language_model
        if not hasattr(self.lm, "lm_head"):
            raise ValueError("expected an untied LM head")
        self.slot_ids = self._verify_slots()
        self.slot_rows = self._head_rows(self.slot_ids)  # [256, H] float32

    def _verify_slots(self) -> list[int]:
        def one(label):
            enc = self.tokenizer.encode(label, add_special_tokens=False)
            return enc[0] if len(enc) == 1 and self.tokenizer.decode(enc) == label else None

        ids = [one(l) for l in SLOT_LABELS]
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("answer labels are not distinct single tokens in this tokenizer")
        return ids

    def _head_rows(self, ids: list[int]) -> mx.array:
        head, idx = self.lm.lm_head, mx.array(ids)
        if isinstance(head, nn.QuantizedLinear):
            biases = getattr(head, "biases", None)
            w = mx.dequantize(head.weight[idx], head.scales[idx], None if biases is None else biases[idx],
                              group_size=head.group_size, bits=head.bits, mode=getattr(head, "mode", "affine"))
        else:
            w = head.weight[idx]
        return w.astype(mx.float32)

    def _forward(self, h: mx.array, positions: mx.array | None, cache: list) -> mx.array:
        """Final normed hidden states. Full-attention layers run unmasked (block-bidirectional over a prefix block
        and a suffix block that sees all of the prefix); GatedDeltaNet layers are recurrent, hence causal."""
        tm = self.lm.model
        kw = {}
        if positions is not None:  # mlx-vlm layers take explicit M-RoPE positions [3, 1, L]
            pe = None
            for layer in tm.layers:
                if not layer.is_linear:
                    if not layer.self_attn.rotary_emb.fused_apply:
                        pe = layer.self_attn.rotary_emb(h, positions)
                    break
            kw = {"position_ids": positions, "position_embeddings": pe}
        for layer, c in zip(tm.layers, cache):
            h = layer(h, mask=None, cache=c, **kw)
        return tm.norm(h)

    @staticmethod
    def _fork(base: list) -> list:
        out = []
        for c in base:
            if hasattr(c, "keys"):  # KVCache: a sliced copy, so a branch's writes never touch the prefix buffers
                n = type(c)()
                n.keys, n.values, n.offset = c.keys[..., : c.offset, :], c.values[..., : c.offset, :], c.offset
            else:  # ArraysCache: conv and recurrent state; branches replace the list entries
                n = type(c)(len(c.cache))
                n.cache = list(c.cache)
            out.append(n)
        return out

    def _prompt(self, state, suffix: str, media) -> str:
        return self.tokenizer.apply_chat_template(build_messages(state, suffix, media), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _split(self, state, rendered: list[dict], media):
        """Longest common token prefix ending at or before the end of the state block (qev engine), with one
        unexpanded vision block per image."""
        full = [self._encode(self._prompt(state, r["suffix"], media)) for r in rendered]
        text = self._prompt(state, MARK, media)
        prefix = self._encode(text[: text.index(MARK)])
        n = len(prefix)
        while n > 0 and not all(ids[:n] == prefix[:n] for ids in full):
            n -= 1
        suffixes = [ids[n:] for ids in full]
        if any(not s for s in suffixes):
            raise ValueError("a branch has an empty suffix")
        if media and any(t in self.vision_ids for s in suffixes for t in s):
            raise ValueError("images belong in the state, not in a question")
        return prefix[:n], suffixes

    def _prefill_inputs(self, prefix: list[int], images: list):
        """(embeddings [1, P, H], positions [3, 1, P] or None, delta) for the prefix, vision blocks expanded."""
        tm = self.lm.model
        if not self.vision:
            return tm.embed_tokens(mx.array([prefix])), None, 0
        if not images:
            pos = mx.broadcast_to(mx.arange(len(prefix))[None, None], (3, 1, len(prefix)))
            return tm.embed_tokens(mx.array([prefix])), pos, 0
        if sum(t == self.image_id for t in prefix) != len(images):
            raise ValueError("vision blocks in the prompt do not match the images")
        out = self.image_processor(images=images)
        grid = [list(map(int, g)) for g in out["image_grid_thw"]]
        counts = iter(t * h * w // self.merge ** 2 for t, h, w in grid)
        ids = []
        for t in prefix:  # one image pad per vision block -> one per merged patch, as the processor expands it
            ids += [t] * next(counts) if t == self.image_id else [t]
        emb = self.model.get_input_embeddings(mx.array([ids]), mx.array(out["pixel_values"]),
                                              image_grid_thw=mx.array(grid))
        return emb.inputs_embeds, emb.position_ids, int(emb.rope_deltas.reshape(-1)[0].item())

    def decide(self, state: Any, questions: dict, media: Sequence[dict] | None = None, debias: int = 1,
               debug: bool = False, media_root: Path | None = None) -> dict:
        t0 = time.perf_counter()
        media = list(media or [])
        if any(m.get("type") != "image" for m in media):
            raise ValueError("only image media are supported")
        if media and not self.vision:
            raise ValueError("images need the vision model; load without --text-only")
        images = [open_image(m, media_root) for m in media]
        rendered, owners = [], []
        for qid, q in questions.items():
            k = len(render_question(q)["labels"])
            for s in (range(min(max(debias, 1), k)) if k > 1 else range(1)):
                order = [(i + s) % k for i in range(k)]
                rendered.append(render_question(q, order if s else None))
                owners.append((qid, order))
        prefix, suffixes = self._split(state, rendered, media)

        t1 = time.perf_counter()
        h, pos, delta = self._prefill_inputs(prefix, images)
        P = h.shape[1]
        longest, total = P + max(map(len, suffixes)), P + sum(map(len, suffixes))
        if longest > self.max_branch_tokens:
            raise ValueError(f"state plus longest question is {longest} tokens, limit is {self.max_branch_tokens}")
        if total > self.max_request_tokens:
            raise ValueError(f"request is {total} tokens, limit is {self.max_request_tokens}")
        base = self.lm.make_cache()
        self._forward(h, pos, base)
        mx.eval([c.state for c in base])
        branches = []
        for r, suf in zip(rendered, suffixes):
            spos = None
            if pos is not None:
                spos = mx.broadcast_to((mx.arange(P, P + len(suf)) + delta)[None, None], (3, 1, len(suf)))
            hs = self._forward(self.lm.model.embed_tokens(mx.array([suf])), spos, self._fork(base))
            hl = hs[0, -1].astype(mx.float32)
            logits = self.slot_rows[: len(r["labels"])] @ hl
            mass = None
            if debug:
                vocab = self.lm.lm_head(hl[None])[0].astype(mx.float32)
                mass = mx.exp(mx.logsumexp(logits) - mx.logsumexp(vocab)).item()
            branches.append((logits.tolist(), mass))
        t2 = time.perf_counter()

        acc: dict[str, dict] = {}
        for r, (qid, order), (logits, mass) in zip(rendered, owners, branches):
            probs = softmax(logits)
            a = acc.setdefault(qid, {"kind": r["kind"], "probs": [0.0] * len(probs), "labels": [None] * len(probs),
                                     "descs": [None] * len(probs), "n": 0, "logits": [], "mass": []})
            for pos_i, orig_i in enumerate(order):
                a["probs"][orig_i] += probs[pos_i]
                a["labels"][orig_i] = r["labels"][pos_i]
                a["descs"][orig_i] = r["descriptions"][pos_i]
            a["n"] += 1
            a["logits"].append(logits)
            a["mass"].append(mass)
        answers, diag = {}, {}
        for qid, a in acc.items():
            p = [x / a["n"] for x in a["probs"]]
            answers[qid] = (make_noul(p[0]) if a["kind"] == "noul" else
                            make_choice(a["labels"], p) if a["kind"] == "choice" else make_score(a["descs"], p))
            diag[qid] = {"slot_logits": a["logits"], "allowed_token_mass": a["mass"]}
        out = {"model": "elephant-1", "answers": answers,
               "usage": {"input_tokens": total, "output_tokens": len(questions)},
               "qev": {"prompt_version": PROMPT_VERSION, "prefix_tokens": P, "branches": len(rendered),
                       "images": len(images), "fork_mode": "sequential",
                       "timing": {"render_s": t1 - t0, "forward_s": t2 - t1, "total_s": time.perf_counter() - t0}}}
        if debug:
            out["qev"]["diagnostics"] = diag
        return out


EXAMPLE = {
    "state": "I was charged twice this month, please refund one of them.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                              "fraud": "Suspected unauthorised use"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Can wait", "Today", "Blocking the customer now"]},
    },
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="baselabs/elephant-1", help="Hub repo or local path")
    ap.add_argument("--request", help="JSON file with {state, questions, media}; defaults to the model card example")
    ap.add_argument("--text-only", action="store_true", help="load the text model with mlx-lm (no vision tower)")
    ap.add_argument("--debias", type=int, default=1, help="average over N cyclic shifts of the option order")
    ap.add_argument("--debug", action="store_true", help="include slot logits and slot probability mass")
    args = ap.parse_args()
    req, root = EXAMPLE, None
    if args.request:
        req, root = json.loads(Path(args.request).read_text()), Path(args.request).resolve().parent
    engine = Elephant(args.model, text_only=args.text_only)
    print(json.dumps(engine.decide(req.get("state"), req["questions"], req.get("media"), args.debias, args.debug,
                                   media_root=root), indent=2))
