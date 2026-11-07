"""Market price references: snapshot per-token prices from OpenRouter's public model list into
config/price_references.yaml (docs/DESIGN.md section 8).

    uv run slotgw-fetch-prices [--out config/price_references.yaml] [--model name=provider/id ...]
                               [--basket name=member,member ...] [--from-file models.json]

No API key is needed. The default references are the 3B-class open models comparable to the
Qwen2.5-3B served here, gpt-4o-mini (the placeholder the first shape prices were derived from), and
a basket that is the median of the 3B-class members. pricing.yaml's `reference:` picks the active one;
the report prices every run's traffic under all of them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import urllib.request
from pathlib import Path
from typing import Any

import yaml

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

DEFAULT_MODELS: dict[str, str] = {
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "ministral-3b": "mistralai/ministral-3b-2512",
    "qwen-2.5-7b": "qwen/qwen-2.5-7b-instruct",
    "gpt-4o-mini": "openai/gpt-4o-mini",
}
DEFAULT_BASKETS: dict[str, list[str]] = {
    "basket-3b": ["llama-3.2-3b", "gemma-3-4b", "ministral-3b", "qwen-2.5-7b"],
}


def fetch_models(url: str = OPENROUTER_MODELS_URL, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "slotgw-fetch-prices"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.load(resp)["data"]


def _per_million(value: Any) -> float:
    price = float(value)
    if price < 0:
        raise ValueError(f"negative (dynamic) price {value!r}")
    return round(price * 1e6, 4)


def build_references(
    models: list[dict[str, Any]],
    wanted: dict[str, str] | None = None,
    baskets: dict[str, list[str]] | None = None,
    *,
    source: str = OPENROUTER_MODELS_URL,
    fetched: str | None = None,
) -> dict[str, Any]:
    """The price_references.yaml document for the wanted {name: provider model id} entries plus the
    baskets (median of their members' prices). Raises if a wanted model is missing from the list."""
    wanted = dict(DEFAULT_MODELS if wanted is None else wanted)
    baskets = dict(DEFAULT_BASKETS if baskets is None else baskets)
    by_id = {m["id"]: m for m in models}
    refs: list[dict[str, Any]] = []
    for name, model_id in wanted.items():
        m = by_id.get(model_id)
        if m is None:
            raise KeyError(f"model {model_id!r} (reference {name!r}) not in the price list")
        pricing = m.get("pricing") or {}
        refs.append({
            "name": name,
            "model": model_id,
            "input_usd_per_million": _per_million(pricing.get("prompt", 0)),
            "output_usd_per_million": _per_million(pricing.get("completion", 0)),
            "context_length": m.get("context_length"),
        })
    by_name = {r["name"]: r for r in refs}
    for name, members in baskets.items():
        missing = [x for x in members if x not in by_name]
        if missing:
            raise KeyError(f"basket {name!r}: unknown members {missing}")
        refs.append({
            "name": name,
            "basket": list(members),
            "input_usd_per_million": round(statistics.median(by_name[x]["input_usd_per_million"] for x in members), 4),
            "output_usd_per_million": round(statistics.median(by_name[x]["output_usd_per_million"] for x in members), 4),
        })
    return {
        "source": source,
        "fetched": fetched or dt.date.today().isoformat(),
        "references": refs,
    }


def write_references(doc: dict[str, Any], out: Path) -> None:
    header = (
        "# Per-token market prices (USD per million tokens), snapshotted by `slotgw-fetch-prices` from\n"
        f"# {doc['source']} on {doc['fetched']}. Rerun the command to refresh; do not edit by hand.\n"
        "# A basket is the median of its members. pricing.yaml `reference:` names the active one.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _kv(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    k, v = text.split("=", 1)
    return k.strip(), v.strip()


def main() -> None:
    p = argparse.ArgumentParser(description="snapshot per-token market prices into price_references.yaml")
    p.add_argument("--out", default="config/price_references.yaml")
    p.add_argument("--url", default=OPENROUTER_MODELS_URL)
    p.add_argument("--from-file", default=None, help="read a saved models JSON instead of fetching")
    p.add_argument("--model", action="append", type=_kv, default=[], metavar="NAME=PROVIDER/ID",
                   help="add or replace a reference (repeatable)")
    p.add_argument("--basket", action="append", type=_kv, default=[], metavar="NAME=MEMBER,MEMBER",
                   help="add or replace a basket (repeatable)")
    args = p.parse_args()
    if args.from_file:
        models = json.loads(Path(args.from_file).read_text(encoding="utf-8"))["data"]
        source = args.from_file
    else:
        models = fetch_models(args.url)
        source = args.url
    wanted = {**DEFAULT_MODELS, **dict(args.model)}
    baskets = {**DEFAULT_BASKETS, **{k: [x.strip() for x in v.split(",") if x.strip()] for k, v in args.basket}}
    doc = build_references(models, wanted, baskets, source=source)
    out = Path(args.out)
    write_references(doc, out)
    for r in doc["references"]:
        what = r.get("model") or f"median of {', '.join(r['basket'])}"
        print(f"{r['name']:<14} in ${r['input_usd_per_million']:.4f}  out ${r['output_usd_per_million']:.4f}  /1M  ({what})")
    print(f"written {out}")


if __name__ == "__main__":
    main()
