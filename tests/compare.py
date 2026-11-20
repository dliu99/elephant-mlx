"""Max probability difference per request between two real_test.py result files.

usage: python -I compare.py <base.json> <other.json>
"""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])


def probs(ans):
    if ans["type"] == "noul":
        return [ans["noul"]]
    return list(ans["probabilities"].values())


print(f"{a['model']} (causal={a['causal']})  vs  {b['model']} (causal={b['causal']})")
worst = 0.0
for name, ra in a["requests"].items():
    rb = b["requests"][name]
    d = max(abs(x - y) for q in ra["answers"] for x, y in zip(probs(ra["answers"][q]), probs(rb["answers"][q])))
    flips = [q for q in ra["answers"] if ra["answers"][q].get("choice") != rb["answers"][q].get("choice")]
    worst = max(worst, d)
    print(f"  {name:28s} max |prob diff| {d:.4f}" + (f"   choice changed: {flips}" if flips else ""))
print(f"  overall max |prob diff| {worst:.4f}")
