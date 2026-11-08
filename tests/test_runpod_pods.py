"""scripts/runpod_pods.py against an httpx.MockTransport: request bodies, pods.json, readiness
polling, the env file, and that the API key never reaches stdout or error text."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
KEY = "rpa_SECRETKEY1234567890abcdef"


def _load():
    spec = importlib.util.spec_from_file_location("runpod_pods", REPO / "scripts" / "runpod_pods.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rpods = _load()


class FakeApi:
    """Records requests and plays a RunPod account: pods get an ip and ports after `ready_after` GETs."""

    def __init__(self, ready_after: int = 0):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.pods: dict[str, dict] = {}
        self.gets: dict[str, int] = {}
        self.ready_after = ready_after
        self.counter = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if request.headers.get("Authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"error": "unauthorized"})
        parts = request.url.path.strip("/").split("/")  # v1/pods[/id[/action]]
        if request.method == "POST" and parts == ["v1", "pods"]:
            self.counter += 1
            pod_id = f"pod{self.counter}"
            self.pods[pod_id] = {"id": pod_id, "name": body["name"], "desiredStatus": "RUNNING", "costPerHr": 0.89,
                                 "publicIp": "", "portMappings": {}}
            return httpx.Response(201, json=self.pods[pod_id])
        if request.method == "GET" and parts == ["v1", "pods"]:
            return httpx.Response(200, json=list(self.pods.values()))
        if request.method == "GET" and len(parts) == 3:
            pod = self.pods[parts[2]]
            n = self.gets[parts[2]] = self.gets.get(parts[2], 0) + 1
            if n > self.ready_after:
                pod["publicIp"] = "203.0.113.7"
                pod["portMappings"] = {"22": 10000 + self.counter, "8000": 20000 + self.counter}
            return httpx.Response(200, json=pod)
        if request.method == "POST" and len(parts) == 4 and parts[3] in ("stop", "start"):
            self.pods[parts[2]]["desiredStatus"] = "EXITED" if parts[3] == "stop" else "RUNNING"
            return httpx.Response(200, json=self.pods[parts[2]])
        if request.method == "DELETE" and len(parts) == 3:
            self.pods.pop(parts[2])
            return httpx.Response(204)
        return httpx.Response(404, json={"error": f"no route {request.method} {request.url.path}"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url=rpods.API_URL)


@pytest.fixture
def phase2(tmp_path, monkeypatch):
    d = tmp_path / "phase2"
    monkeypatch.setenv("SLOTGW_PHASE2_DIR", str(d))
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    return d


def _env_with_key() -> dict[str, str]:
    return {"RUNPOD_API_KEY": KEY}


def test_missing_key_is_a_clear_error(phase2, tmp_path, capsys):
    no_dotenv = tmp_path / "nope.env"
    with pytest.raises(rpods.RunPodError, match="RUNPOD_API_KEY: missing"):
        rpods.load_api_key({}, no_dotenv)
    rc = rpods.main(["status"], client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))),
                    env={}, dotenv=no_dotenv)
    assert rc == 2
    assert "RUNPOD_API_KEY: missing" in capsys.readouterr().err


def test_key_from_dotenv_with_quotes_and_export(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f'# comment\nexport OTHER=1\nRUNPOD_API_KEY="{KEY}"\n', encoding="utf-8")
    assert rpods.load_api_key({}, dotenv) == KEY
    assert rpods.load_api_key({"RUNPOD_API_KEY": "fromenv"}, dotenv) == "fromenv"


def test_create_builds_the_body_and_writes_pods_json(phase2, capsys):
    api = FakeApi()
    rc = rpods.main(["create", "--count", "2", "--env", "FOO=bar", "--datacenter", "US-TX-3"],
                    client=api.client(), env=_env_with_key())
    assert rc == 0
    posts = [b for m, p, b in api.requests if m == "POST" and p == "/v1/pods"]
    assert [b["name"] for b in posts] == ["slotgw-a", "slotgw-b"]
    body = posts[0]
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"] and body["gpuCount"] == 1
    assert body["cloudType"] == "SECURE" and body["globalNetworking"] is True
    assert body["ports"] == ["8000/tcp", "22/tcp"] and body["supportPublicIp"] is True
    assert body["imageName"] == rpods.TEMPLATE_IMAGE  # the template path: the image's own sshd, vLLM installed over SSH
    assert body["containerDiskInGb"] == 30 and body["volumeInGb"] == 20 and body["volumeMountPath"] == "/workspace"
    assert body["dataCenterIds"] == ["US-TX-3"]
    assert "dockerEntrypoint" not in body and "dockerStartCmd" not in body
    assert body["env"] == {"MODEL": "Qwen/Qwen2.5-7B-Instruct-AWQ", "VLLM_MAX_NUM_BATCHED_TOKENS": "512", "FOO": "bar"}
    saved = json.loads((phase2 / "pods.json").read_text(encoding="utf-8"))["pods"]
    assert [(p["name"], p["id"], p["gpu"]) for p in saved] == [
        ("slotgw-a", "pod1", "NVIDIA GeForce RTX 4090"), ("slotgw-b", "pod2", "NVIDIA GeForce RTX 4090")]
    assert all(p["created_at"] for p in saved)
    out = capsys.readouterr().out
    assert KEY not in out and "created slotgw-a id=pod1" in out

    # a second create continues the lettering instead of reusing a or b
    rpods.main(["create", "--count", "1", "--no-global-networking"], client=api.client(), env=_env_with_key())
    posts = [b for m, p, b in api.requests if m == "POST" and p == "/v1/pods"]
    assert posts[-1]["name"] == "slotgw-c" and posts[-1]["globalNetworking"] is False
    assert [p["name"] for p in rpods.load_pods(phase2 / "pods.json")] == ["slotgw-a", "slotgw-b", "slotgw-c"]


def test_dry_run_prints_redacted_bodies_and_calls_nothing(phase2, capsys):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=rpods.API_URL)
    rc = rpods.main(["create", "--count", "1", "--dry-run", "--env", f"HF_TOKEN={KEY}", "--env", "PLAIN=1"],
                    client=client, env=_env_with_key())
    assert rc == 0 and calls == []
    out = capsys.readouterr().out
    assert KEY not in out
    assert out.startswith("RUNPOD_API_KEY: set")
    bodies = json.loads(out[out.index("["):out.rindex("]") + 1])
    assert bodies[0]["name"] == "slotgw-a" and bodies[0]["env"]["HF_TOKEN"] == "***" and bodies[0]["env"]["PLAIN"] == "1"
    assert not (phase2 / "pods.json").exists()


def test_status_parses_ip_and_ports_and_updates_pods_json(phase2, capsys):
    api = FakeApi()
    rpods.main(["create", "--count", "2"], client=api.client(), env=_env_with_key())
    capsys.readouterr()
    assert rpods.main(["status"], client=api.client(), env=_env_with_key()) == 0
    out = capsys.readouterr().out
    assert "203.0.113.7" in out and "10002" in out and "20002" in out and "pod1.runpod.internal" in out
    saved = rpods.load_pods(phase2 / "pods.json")
    assert saved[1]["public_ip"] == "203.0.113.7" and saved[1]["ssh_port"] == 10002 and saved[1]["vllm_port"] == 20002
    assert saved[1]["status"] == "RUNNING" and saved[1]["cost_per_hr"] == 0.89
    assert rpods.main(["status", "--json"], client=api.client(), env=_env_with_key()) == 0
    as_json = json.loads(capsys.readouterr().out)
    assert [p["ssh_port"] for p in as_json] == [10002, 10002] and KEY not in json.dumps(as_json)


def test_wait_ready_polls_until_ports_and_health(phase2, capsys):
    api = FakeApi(ready_after=2)
    rpods.main(["create", "--count", "1"], client=api.client(), env=_env_with_key())
    health_calls = []

    def health(request):
        health_calls.append(str(request.url))
        return httpx.Response(503 if len(health_calls) < 3 else 200, json={})

    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    rp = rpods.RunPod(KEY, client=api.client())
    ok = rpods.cmd_wait_ready(rp, pods_path=phase2 / "pods.json", timeout_s=900, poll_s=10,
                              health_client=httpx.Client(transport=httpx.MockTransport(health)),
                              sleep=sleep, clock=lambda: clock["t"])
    assert ok is True
    assert api.gets["pod1"] == 3  # two polls without ports, one with
    assert health_calls == ["http://203.0.113.7:20001/health"] * 3
    assert len(slept) == 4
    out = capsys.readouterr().out
    assert "vLLM ready" in out and "all 1 pod(s) ready" in out and KEY not in out

    # timeout: the pod never gets ports
    api2 = FakeApi(ready_after=10_000)
    (phase2 / "pods.json").unlink()
    rpods.main(["create", "--count", "1"], client=api2.client(), env=_env_with_key())
    clock["t"] = 0.0
    ok = rpods.cmd_wait_ready(rpods.RunPod(KEY, client=api2.client()), pods_path=phase2 / "pods.json",
                              timeout_s=25, poll_s=10, health_client=httpx.Client(transport=httpx.MockTransport(health)),
                              sleep=sleep, clock=lambda: clock["t"])
    assert ok is False
    assert "timeout" in capsys.readouterr().out


def test_env_file_lists_local_a_and_internal_dns_for_the_rest(phase2, capsys):
    api = FakeApi()
    rpods.main(["create", "--count", "3"], client=api.client(), env=_env_with_key())
    out_path = phase2 / "phase2.env"
    assert rpods.main(["env-file", "--out", str(out_path)], env={}) == 0  # needs no API key
    text = out_path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert lines == [
        "REPLICA_A_URL=http://127.0.0.1:8000", "REPLICA_A_ID=pod1",
        "REPLICA_B_URL=http://pod2.runpod.internal:8000", "REPLICA_B_ID=pod2",
        "REPLICA_C_URL=http://pod3.runpod.internal:8000", "REPLICA_C_ID=pod3",
    ]
    assert KEY not in text
    assert rpods.main(["env-file", "--out", str(out_path), "--gateway", "b"], env={}) == 0
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert lines[:4] == ["REPLICA_A_URL=http://pod1.runpod.internal:8000", "REPLICA_A_ID=pod1",
                         "REPLICA_B_URL=http://127.0.0.1:8000", "REPLICA_B_ID=pod2"]


def test_api_errors_are_redacted(phase2):
    def leaky(request):
        # a misbehaving server echoing the request header back
        return httpx.Response(401, text=f"bad auth: {request.headers['Authorization']}")

    rp = rpods.RunPod(KEY, client=httpx.Client(transport=httpx.MockTransport(leaky), base_url=rpods.API_URL))
    with pytest.raises(rpods.RunPodError) as ei:
        rp.get_pod("pod1")
    assert KEY not in str(ei.value) and KEY not in ei.value.body
    assert ei.value.status == 401 and "HTTP 401" in str(ei.value)


def test_stop_start_terminate(phase2, capsys):
    api = FakeApi()
    rpods.main(["create", "--count", "3"], client=api.client(), env=_env_with_key())
    api.requests.clear()
    assert rpods.main(["stop", "--only", "b,c"], client=api.client(), env=_env_with_key()) == 0
    assert [p for m, p, _ in api.requests if m == "POST"] == ["/v1/pods/pod2/stop", "/v1/pods/pod3/stop"]
    api.requests.clear()
    assert rpods.main(["stop"], client=api.client(), env=_env_with_key()) == 0
    assert [p for m, p, _ in api.requests if m == "POST"][-1] == "/v1/pods/pod1/stop"  # a (gateway) last
    assert rpods.main(["start", "--only", "a"], client=api.client(), env=_env_with_key()) == 0
    assert api.pods["pod1"]["desiredStatus"] == "RUNNING"
    capsys.readouterr()
    # terminate refuses without --yes
    assert rpods.main(["terminate"], client=api.client(), env=_env_with_key()) == 2
    assert "add --yes" in capsys.readouterr().err and len(api.pods) == 3
    assert rpods.main(["terminate", "--yes", "--only", "c"], client=api.client(), env=_env_with_key()) == 0
    assert set(api.pods) == {"pod1", "pod2"}
    assert [p["name"] for p in rpods.load_pods(phase2 / "pods.json")] == ["slotgw-a", "slotgw-b"]


def test_start_404_points_to_the_console(phase2, capsys):
    api = FakeApi()
    rpods.main(["create", "--count", "1"], client=api.client(), env=_env_with_key())

    def no_start(request):
        if request.url.path.endswith("/start"):
            return httpx.Response(404, json={"error": "not found"})
        return api.handler(request)

    client = httpx.Client(transport=httpx.MockTransport(no_start), base_url=rpods.API_URL)
    capsys.readouterr()
    assert rpods.main(["start"], client=client, env=_env_with_key()) == 0
    assert "RunPod console" in capsys.readouterr().out


def test_bootstrap_path_overrides_the_start_command(phase2):
    api = FakeApi()
    rc = rpods.main(["create", "--count", "1", "--bootstrap", "--image", rpods.VLLM_IMAGE], client=api.client(), env=_env_with_key())
    assert rc == 0
    body = next(b for m, p, b in api.requests if m == "POST" and p == "/v1/pods")
    assert body["imageName"] == rpods.VLLM_IMAGE
    assert body["dockerEntrypoint"] == ["bash", "-lc"]
    assert body["dockerStartCmd"] == [(REPO / "scripts" / "pod-bootstrap.sh").read_text(encoding="utf-8")]
    assert "sleep infinity" in body["dockerStartCmd"][0]
