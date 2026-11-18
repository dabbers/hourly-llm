"""Stage L quality sample: send a fixed set of prompts straight to one engine, greedy (temperature 0, seed 0),
and save the replies, so two engine builds (fp16 and fp8 KV) can be compared on identical inputs.

  build the prompt set (once, from a finished run's showcase samples):
    python scripts/quality_sample.py build runs/<run>/report/samples.csv --out config/phase3/quality-prompts-stage-l.json
  sample an engine (on the pod, engines idle):
    python scripts/quality_sample.py run --url http://127.0.0.1:8000 --out runs/stage-l/quality/fp16-a.json
  compare two samples:
    python scripts/quality_sample.py compare a.json b.json
"""
import argparse
import csv
import difflib
import json
import sys
import time

PROMPTS = "config/phase3/quality-prompts-stage-l.json"
WANT = {"chat": 20, "agent": 10}


def build(a) -> None:
    csv.field_size_limit(10**9)
    rows = [r for r in csv.DictReader(open(a.samples, encoding="utf-8")) if r["status"] == "ok" and r["tenant"] in WANT]
    out = []
    for tenant, n in WANT.items():
        mine = sorted((r for r in rows if r["tenant"] == tenant), key=lambda r: r["showcase"])
        step = len(mine) / n
        assert len(mine) >= n, (tenant, len(mine))
        for i in range(n):  # evenly through the sorted list: every prompt family and early and late turns
            r = mine[int(i * step)]
            out.append({"id": r["showcase"], "tenant": tenant, "turn": int(r["turn"]), "messages": json.loads(r["prompt_json"]),
                        "max_tokens": 400})
    json.dump({"source": a.samples, "prompts": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} prompts -> {a.out}")


def run(a) -> None:
    import httpx

    prompts = json.load(open(a.prompts, encoding="utf-8"))["prompts"]
    model = httpx.get(f"{a.url}/v1/models", timeout=10).json()["data"][0]["id"]
    out = []
    with httpx.Client(timeout=120) as c:
        for p in prompts:  # one at a time: batching with other requests can change greedy output at a near-tie
            t = time.time()
            r = c.post(f"{a.url}/v1/chat/completions", json={"model": model, "messages": p["messages"], "max_tokens": p["max_tokens"],
                                                             "temperature": 0, "seed": 0})
            r.raise_for_status()
            j = r.json()
            out.append({**{k: p[k] for k in ("id", "tenant", "turn")}, "output": j["choices"][0]["message"]["content"],
                        "finish": j["choices"][0]["finish_reason"], "prompt_tokens": j["usage"]["prompt_tokens"],
                        "output_tokens": j["usage"]["completion_tokens"], "s": round(time.time() - t, 2)})
            print(p["id"], out[-1]["output_tokens"], out[-1]["finish"], flush=True)
    json.dump({"url": a.url, "model": model, "label": a.label, "replies": out}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def compare(a) -> None:
    x, y = (json.load(open(f, encoding="utf-8"))["replies"] for f in (a.a, a.b))
    same, ratios, first = 0, [], []
    for p, q in zip(x, y, strict=True):
        assert p["id"] == q["id"]
        wa, wb = p["output"].split(), q["output"].split()
        r = difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
        d = next((i for i, (u, v) in enumerate(zip(wa, wb)) if u != v), min(len(wa), len(wb)))
        same += p["output"] == q["output"]
        ratios.append(r)
        first.append(d)
        print(f"{p['id']:34s} {p['tenant']:6s} same={p['output'] == q['output']!s:5s} word-match={r:.2f} first-diff-word={d:4d} tokens {p['output_tokens']}/{q['output_tokens']} finish {p['finish']}/{q['finish']}")
    print(f"identical {same}/{len(x)}; word-match mean {sum(ratios) / len(ratios):.3f}, min {min(ratios):.2f}")


def degenerate(text: str) -> str:
    """A reason the reply is unusable on its face, or "" (a mechanical screen; the verdicts were also read by hand)."""
    words = text.split()
    if len(words) < 3:
        return "empty"
    tail = words[-60:]
    if len(tail) >= 30 and len(set(tail)) / len(tail) < 0.35:
        return "repeats in a loop"
    return ""


def clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + f" [... {len(text) - n} more characters]"


def markdown(a) -> None:
    """Side-by-side replies for docs/SAMPLES.md: the reference sample against one other sample, prompt by prompt."""
    prompts = {p["id"]: p for p in json.load(open(a.prompts, encoding="utf-8"))["prompts"]}
    x, y = (json.load(open(f, encoding="utf-8"))["replies"] for f in (a.a, a.b))
    out, bad = [], 0
    for p, q in zip(x, y, strict=True):
        assert p["id"] == q["id"]
        fam, turn = p["id"].rsplit("#", 1)
        user = next(m["content"] for m in reversed(prompts[p["id"]]["messages"]) if m["role"] == "user")
        why = degenerate(q["output"])
        bad += bool(why)
        verdict = f"**clearly worse: {why}**" if why else ("identical" if p["output"] == q["output"] else "differs, readable")
        out += [f"#### {p['tenant']} · {fam} · turn {turn}", "",
                f"prompt {p['prompt_tokens']} tokens · {a.name_a} {p['output_tokens']} tokens ({p['finish']}) · "
                f"{a.name_b} {q['output_tokens']} tokens ({q['finish']}) · {verdict}", "",
                "**last user message**", "", "```text", clip(user, 300), "```", "",
                f"**{a.name_a}**", "", "```text", clip(p["output"], a.chars), "```", "",
                f"**{a.name_b}**", "", "```text", clip(q["output"], a.chars), "```", ""]
    open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"{len(x)} prompts, {bad} flagged clearly worse -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("samples"); b.add_argument("--out", default=PROMPTS)
    r = sub.add_parser("run"); r.add_argument("--url", default="http://127.0.0.1:8000"); r.add_argument("--prompts", default=PROMPTS)
    r.add_argument("--out", required=True); r.add_argument("--label", default="")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    m = sub.add_parser("markdown"); m.add_argument("a"); m.add_argument("b"); m.add_argument("--out", required=True)
    m.add_argument("--prompts", default=PROMPTS); m.add_argument("--name-a", dest="name_a", default="fp16")
    m.add_argument("--name-b", dest="name_b", default="fp8"); m.add_argument("--chars", type=int, default=600)
    a = ap.parse_args()
    {"build": build, "run": run, "compare": compare, "markdown": markdown}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
