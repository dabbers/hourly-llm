from __future__ import annotations

import csv
import time
from pathlib import Path

import pytest

from slotgw.config import Config, load_config
from slotgw.meter import Meter, RequestRecord
from slotgw.prom import first_labels, first_value, parse_prometheus

REPO = Path(__file__).resolve().parents[1]


def test_config_loads_and_validates():
    cfg = load_config(REPO / "config")
    assert cfg.shape("small.8k.30tps").tps == 30
    assert cfg.tenant_by_key("sk-local-chat").name == "chat"
    assert cfg.hf_model("small").startswith("Qwen/")
    assert cfg.shape("small.8k.15tps.lazy").lazy is True


def test_config_rejects_dangling_references():
    cfg = load_config(REPO / "config")
    bad = cfg.model_dump()
    bad["tenants"][0]["default_shape"] = "nope"
    with pytest.raises(ValueError):
        Config(**bad)


def test_env_expansion_in_replicas(monkeypatch, tmp_path):
    src = REPO / "config"
    for name in ("shapes.yaml", "tenants.yaml", "replicas.yaml", "pricing.yaml"):
        (tmp_path / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("VLLM_URL", "http://replica-x:9999")
    cfg = load_config(tmp_path)
    assert cfg.replicas[0].url == "http://replica-x:9999"
    monkeypatch.delenv("VLLM_URL")
    cfg = load_config(tmp_path)
    assert cfg.replicas[0].url == "http://localhost:8000"


def test_meter_record_and_export(tmp_path):
    m = Meter(tmp_path / "m.sqlite", run_id="r1")
    now = time.time()
    rec = RequestRecord(
        request_id="a", tenant="chat", shape="small.8k.30tps", lazy=False, delivery="stream",
        t_arrival=now, t_admitted=now + 0.01, t_first_token=now + 0.2, t_engine_done=now + 1.0,
        t_last_token=now + 1.1, prompt_tokens=100, output_tokens=30, output_tokens_guaranteed=30,
    )
    m.record(rec)
    rec.output_tokens = 31  # same request id replaces
    m.record(rec)
    assert m.count() == 1
    out = tmp_path / "m.csv"
    assert m.export_csv(out) == 1
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["output_tokens"] == "31"
    assert abs(float(rows[0]["ttft_ms"]) - 190.0) < 1e-3
    assert abs(float(rows[0]["e2e_ms"]) - 1100.0) < 1e-3
    m.close()


def test_prometheus_parser():
    text = """
# HELP vllm:kv_cache_usage_perc KV cache usage
vllm:kv_cache_usage_perc{model_name="m"} 0.25
vllm:num_requests_running 3
vllm:cache_config_info{block_size="16",num_gpu_blocks="9870",cache_dtype="auto"} 1.0
"""
    s = parse_prometheus(text)
    assert first_value(s, "vllm:kv_cache_usage_perc") == 0.25
    assert first_value(s, "vllm:num_requests_running") == 3
    assert first_labels(s, "vllm:cache_config_info")["num_gpu_blocks"] == "9870"
    assert first_value(s, "missing") is None
