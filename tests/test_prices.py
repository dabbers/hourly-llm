"""Market price references: snapshot building, the active reference, derived shape prices."""

from __future__ import annotations

from pathlib import Path

import pytest

from slotgw.config import load_config
from slotgw.prices import build_references, write_references

REPO = Path(__file__).resolve().parents[1]

MODELS = [
    {"id": "a/one", "context_length": 1000, "pricing": {"prompt": "0.00000005", "completion": "0.0000003"}},
    {"id": "b/two", "context_length": 2000, "pricing": {"prompt": "0.0000001", "completion": "0.0000001"}},
    {"id": "c/three", "context_length": 3000, "pricing": {"prompt": "0.0000002", "completion": "0.0000006"}},
]


def _config_dir(tmp_path: Path, pricing: str, shapes: str | None = None, refs: dict | None = None) -> Path:
    d = tmp_path / "config"
    d.mkdir(parents=True)
    (d / "shapes.yaml").write_text(shapes or """
shapes:
  - {name: small.8k.30tps, model: small, max_context: 8192, tps: 30}
  - {name: small.16k.30tps, model: small, max_context: 16384, tps: 30, price_premium: 1.25}
  - {name: small.8k.15tps.lazy, model: small, max_context: 8192, tps: 15, lazy: true}
  - {name: small.8k.60tps, model: small, max_context: 8192, tps: 60, price_per_second: 0.001}
""", encoding="utf-8")
    (d / "tenants.yaml").write_text("tenants:\n  - {name: t, api_key: sk-test-t, default_shape: small.8k.30tps}\n", encoding="utf-8")
    (d / "replicas.yaml").write_text(
        "models: {small: Qwen/Qwen2.5-3B-Instruct-AWQ}\nreplicas:\n  - {id: f, kind: fake, url: http://x, model: small, curve: '1:100,8:50'}\n",
        encoding="utf-8",
    )
    (d / "pricing.yaml").write_text(pricing, encoding="utf-8")
    if refs is not None:
        write_references(refs, d / "price_references.yaml")
    return d


def test_build_references_converts_per_million_and_takes_basket_median():
    doc = build_references(MODELS, {"one": "a/one", "two": "b/two", "three": "c/three"},
                           {"basket": ["one", "two", "three"]}, source="test", fetched="2026-11-06")
    by = {r["name"]: r for r in doc["references"]}
    assert by["one"] == {"name": "one", "model": "a/one", "input_usd_per_million": 0.05,
                         "output_usd_per_million": 0.3, "context_length": 1000}
    assert by["basket"]["basket"] == ["one", "two", "three"]
    assert by["basket"]["input_usd_per_million"] == 0.1  # median of 0.05, 0.1, 0.2
    assert by["basket"]["output_usd_per_million"] == 0.3  # median of 0.3, 0.1, 0.6
    assert doc["source"] == "test" and doc["fetched"] == "2026-11-06"


def test_build_references_rejects_missing_models_and_members():
    with pytest.raises(KeyError):
        build_references(MODELS, {"one": "a/one", "nope": "z/nope"}, {})
    with pytest.raises(KeyError):
        build_references(MODELS, {"one": "a/one"}, {"basket": ["one", "two"]})


def test_active_reference_sets_token_prices_prompt_charge_and_derived_shape_prices(tmp_path):
    refs = build_references(MODELS, {"one": "a/one", "two": "b/two"}, {"pair": ["one", "two"]}, fetched="2026-11-06")
    d = _config_dir(tmp_path, "reference: one\ntoken: {batch_discount: 0.5}\nprovider: {gpu_usd_per_hour: 0.3}\n", refs=refs)
    c = load_config(d)
    assert c.pricing.token.reference == "one"
    assert (c.pricing.token.input_usd_per_million, c.pricing.token.output_usd_per_million) == (0.05, 0.3)
    assert c.pricing.shape.prompt_usd_per_million == 0.05  # auto = the reference's input price
    assert c.pricing.references_fetched == "2026-11-06"
    assert c.shape("small.8k.30tps").price_per_second == pytest.approx(0.3e-6 * 30)
    assert c.shape("small.16k.30tps").price_per_second == pytest.approx(0.3e-6 * 30 * 1.25)
    assert c.shape("small.8k.15tps.lazy").price_per_second == pytest.approx(0.3e-6 * 15 * 0.5)
    assert c.shape("small.8k.60tps").price_per_second == 0.001  # explicit price kept


def test_custom_reference_uses_token_block_and_numeric_prompt_price(tmp_path):
    d = _config_dir(tmp_path, """
reference: custom
token: {input_usd_per_million: 0.15, output_usd_per_million: 0.6, batch_discount: 0.5}
provider: {gpu_usd_per_hour: 0.3, target_utilization: 0.8, margin_targets: [0.1, 0.4]}
shape: {prompt_usd_per_million: 0.1}
""")
    c = load_config(d)
    assert c.pricing.token.reference == "custom"
    assert c.pricing.shape.prompt_usd_per_million == 0.1
    assert c.pricing.provider.margin_targets == [0.1, 0.4]
    assert c.shape("small.8k.30tps").price_per_second == pytest.approx(0.6e-6 * 30)
    assert not c.pricing.references


def test_old_snapshot_without_reference_key_still_loads(tmp_path):
    d = _config_dir(tmp_path, "token: {input_usd_per_million: 0.15, output_usd_per_million: 0.6}\nprovider: {gpu_usd_per_hour: 0.3}\n")
    c = load_config(d)
    assert c.pricing.token.output_usd_per_million == 0.6
    assert c.pricing.shape.prompt_usd_per_million == 0.15


def test_unknown_reference_and_missing_prices_are_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown price reference"):
        load_config(_config_dir(tmp_path, "reference: nope\nprovider: {gpu_usd_per_hour: 0.3}\n"))
    with pytest.raises(ValueError, match="token prices"):
        load_config(_config_dir(tmp_path / "b", "provider: {gpu_usd_per_hour: 0.3}\n"))


def test_repo_config_reproduces_the_placeholder_prices_under_gpt_4o_mini(monkeypatch):
    monkeypatch.setenv("SLOTGW_PRICE_REFERENCE", "gpt-4o-mini")
    c = load_config(REPO / "config")
    assert c.pricing.token.reference == "gpt-4o-mini"
    assert c.shape("small.8k.30tps").price_per_second == pytest.approx(1.8e-5)
    assert c.shape("small.16k.30tps").price_per_second == pytest.approx(2.25e-5)
    assert c.shape("small.8k.15tps.lazy").price_per_second == pytest.approx(4.5e-6)
    assert c.shape("small.16k.30tps.lazy").price_per_second == pytest.approx(1.125e-5)
    monkeypatch.delenv("SLOTGW_PRICE_REFERENCE")
    c = load_config(REPO / "config")
    assert c.pricing.token.reference == "basket-3b"
    assert c.pricing.find_reference("basket-3b").basket
