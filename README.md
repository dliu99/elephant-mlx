# elephant-mlx

Run [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) on Apple silicon with [MLX](https://github.com/ml-explore/mlx), with text and images.

elephant-1 is a 9B **decision model** post-trained from Qwen3.5-9B. It does not write text. You give it a state (text, optionally with images) and a set of questions, and it returns a calibrated probability for every option. All questions are answered from one shared prefill.

Stock mlx-lm and mlx-vlm can load the weights, but they do not run the model the way it was trained. This port reproduces elephant-1's own decision path:

- **Bidirectional attention.** The full-attention layers use the block-bidirectional mask the model was trained with. Running them causal, as a stock forward does, moves probabilities by up to 0.10 on real requests.
- **The slot readout**, the prompt format and the answer formulas of the reference runtime.
- **Images** with the checkpoint's own resize limits. mlx-vlm would otherwise fall back to Qwen2-VL defaults and change the number of vision tokens.

Its output matches the reference PyTorch implementation within 4e-6 in logits on a tiny test model (see [Accuracy](#accuracy)).

## Quick start

```bash
pip install -r requirements.txt
python elephant_mlx.py --model devinliu420/elephant-1-mlx-8bit --request examples/receipt.json
```

The first run downloads the 10.4 GB 8-bit model from Hugging Face. Without `--request`, it runs the support-ticket example from the elephant-1 model card.

[`examples/receipt.json`](examples/receipt.json) asks three questions about a support ticket with a receipt photo attached:

```json
{
  "state": {"ticket": "I think I was charged twice for order 4471. Receipt attached.", "receipt": "<image:1>"},
  "media": [{"type": "image", "path": "images/receipt.png"}],
  "questions": {
    "double": {"type": "noul", "instructions": "Does the receipt show the same card charge listed twice?"},
    "total": {"type": "choice", "instructions": "What total does the receipt show as charged?",
              "criteria": {"42.10": "$42.10", "84.20": "$84.20", "126.30": "$126.30"}},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                          "fraud": "Suspected unauthorised use"}}
  }
}
```

The answers (abridged):

```json
{
  "double": {"type": "noul", "noul": 0.999128},
  "total": {"type": "choice", "choice": "84.20",
            "probabilities": {"42.10": 0.000091, "84.20": 0.994652, "126.30": 0.005257}},
  "team": {"type": "choice", "choice": "billing",
           "probabilities": {"billing": 0.984839, "technical": 0.001656, "fraud": 0.013506}}
}
```

## Requests

A request is a JSON object with three fields:

- **`state`**: a string or any JSON value. It is what the questions are about.
- **`questions`**: a map from your own question ids to questions. Each question has one of three types:
  - `noul`: a yes/no question. The answer is the probability of yes.
  - `choice`: pick one of up to 255 options, given as `criteria: {name: description}`.
  - `score`: rate on an ordered scale of up to 10 levels, given as `criteria: [level 0, level 1, ...]`. The answer is the expected level.
- **`media`** (optional): images, each as `{"type": "image", "path" | "url" | "data": ...}`. The state refers to them as `<image:1>`, `<image:2>`, ..., each exactly once and in order. Relative paths resolve against the request file's folder. `data` takes base64 or a data URL.

Options:

| Flag | Effect |
|---|---|
| `--model` | A Hugging Face repo or a local folder. The default is `baselabs/elephant-1` (the original bf16 weights). |
| `--debias N` | Average each question over N cyclic shifts of its option order, to cancel position bias. Costs N forwards per question. |
| `--text-only` | Load the language model alone with mlx-lm, from an `mlx_lm.convert` output. No images. |
| `--debug` | Add the raw slot logits and the share of probability mass on the answer labels. |

From Python:

```python
from elephant_mlx import Elephant

engine = Elephant("devinliu420/elephant-1-mlx-8bit")
out = engine.decide(
    state={"ticket": "Screen is cracked, see photo", "photo": "<image:1>"},
    questions={"damaged": {"type": "noul", "instructions": "Does the photo show damage?"}},
    media=[{"type": "image", "path": "photo.jpg"}],
)
print(out["answers"]["damaged"]["noul"])
```

## Models

| `--model` | Download | Peak memory | Notes |
|---|---|---|---|
| [`devinliu420/elephant-1-mlx-8bit`](https://huggingface.co/devinliu420/elephant-1-mlx-8bit) | 10.4 GB | 11.5 GB | 8-bit weights (group size 64), vision tower included. |
| [`baselabs/elephant-1`](https://huggingface.co/baselabs/elephant-1) | 18.8 GB | 23 GB | The original bf16 weights; mlx-vlm loads them directly. |

To make your own conversion:

```bash
mlx_vlm.convert --hf-path baselabs/elephant-1 --mlx-path elephant-1-mlx-8bit -q --q-bits 8 --q-group-size 64 --dtype bfloat16
```

Pass `--dtype bfloat16`. Without it, the converter stores every unquantized weight in float32, which makes the model about 1.5 GB larger and uses about 2 GB more memory.

## Accuracy

**Parity with the reference.** The reference is OneJev's qev runtime at the commit elephant-1 ships for, with the checkpoint's `serve.patch` applied. For images, it is the patched training scorer, because that is how the model was trained and evaluated on images. Both run in PyTorch on CPU in float32, on a tiny random model that carries elephant-1's tokenizer, config and readout.

| Comparison | Max slot-logit difference |
|---|---|
| Text, mlx-vlm backend | 3.7e-6 |
| Text, `--text-only` (mlx-lm) | 3.5e-6 |
| Images, same pixels | 2.0e-6 |
| Images, end to end (MLX image processor) | 2.2e-6 to 2.1e-4 |
| For scale: causal against bidirectional attention, same model | 2e-2 to 3e-2 |

The end-to-end figure includes small resize differences between the transformers and mlx-vlm image processors.

**Real weights**, 8-bit model, M1 Max (64 GB):

| Request | Answers | Time |
|---|---|---|
| Support ticket (text) | refund 0.993; team billing 0.998; urgency mostly "Today" (0.64) | 1.1 s |
| One shape image: color and shape | correct, ≥ 0.9995 each (red circle, blue square, green triangle) | 2.9–3.2 s |
| 96×96 image (upscaled to the 65,536-pixel minimum) | yellow 0.9999, circle 0.9998 | 1.7 s |
| Two images: which shows a circle? are both blue? | second 0.99997; both blue 0.00006 | 5.4 s |
| Receipt: charged twice? total? team? | yes 0.999; $84.20 0.995; billing 0.985 | 5.1 s |

Times include rendering, the vision tower and all question forwards, after a warm-up request. Loading takes about 6 s.

**8-bit against the original bf16 weights:** the largest probability difference is 0.0097, on the support ticket's urgency score. It is 0.0011 on the receipt and under 0.0001 on every shape question.

**What the bidirectional attention changes:** with causal attention, the urgency probabilities shift by 0.10 ("Can wait" goes from 0.26 to 0.36) and the receipt's team probabilities by 0.011. The shape questions barely move.

## How it works

- **Prompt:** the qev-labels-v2 format. A system prompt, a `<state>` block, the question, lettered options and an answer line, in the Qwen chat template with thinking off.
- **One prefill per request.** The shared prefix (the system turn and the state block, with any images) runs once, so the vision tower runs once. Each question then runs as a suffix forward from a copy of that cache. No token is sampled.
- **Attention:** the full-attention layers are block-bidirectional. With two blocks, prefix and suffix, that means no mask on either forward. The GatedDeltaNet layers are recurrent and stay causal.
- **Positions:** multimodal RoPE positions come from the model's `get_rope_index` on the prefix. Suffix token i sits at position i + delta.
- **Readout:** a softmax over the LM-head logits of the answer labels (A to Z, then two-letter labels) at the last token. The head rows are in float32, as the reference computes them. The checkpoint ships no `calibration.json`, so the temperature is 1.

## Limits

- Images only. Video is not supported.
- The model was trained on sequences up to 16,384 tokens. The engine refuses a request whose state plus longest question exceeds 32,768 tokens, or whose total exceeds 65,536, matching the reference runtime.
- Questions are answered one after another from the shared cache, not batched.

## Tests

The parity tests need PyTorch and the reference runtime:

```bash
pip install -r tests/requirements.txt
sh tests/setup_reference.sh   # clones OneJev at the pinned commit and applies serve.patch
hf download baselabs/elephant-1 --exclude "*.safetensors" --local-dir ref
python -I tests/compare_processors.py ref work
python -I tests/verify_vision.py ref work OneJev
```

To run the real-weight requests and compare two runs:

```bash
python -I tests/real_test.py devinliu420/elephant-1-mlx-8bit out_8bit.json
python -I tests/real_test.py devinliu420/elephant-1-mlx-8bit out_causal.json --causal
python -I tests/compare.py out_8bit.json out_causal.json
```

## License

The code is Apache-2.0. Its prompt format and answer logic are ported from [OneJev](https://github.com/OmniJev/OneJev) (Apache-2.0). The elephant-1 weights are MIT-licensed by Base Labs. This project is not affiliated with Base Labs or Baseten.
