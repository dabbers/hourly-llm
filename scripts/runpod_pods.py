"""RunPod REST client and CLI for the phase-2 pods (docs/PHASE2.md section 2, docs/RUNBOOK.md).

    uv run python scripts/runpod_pods.py create --count 1 [--dry-run]
    uv run python scripts/runpod_pods.py status [--all] [--json]
    uv run python scripts/runpod_pods.py wait-ready [--timeout 900]
    uv run python scripts/runpod_pods.py env-file [--out runs/phase2/phase2.env]
    uv run python scripts/runpod_pods.py stop [--all | --only b,c]
    uv run python scripts/runpod_pods.py start [--only a]
    uv run python scripts/runpod_pods.py terminate --yes [--only c]

The API key comes from the environment variable RUNPOD_API_KEY or from `.env` at the repository
root and is never printed. Pod ids, public ips and port mappings live in runs/phase2/pods.json
(override the directory with SLOTGW_PHASE2_DIR). Pods are named <prefix>-a, -b, -c; the letter is
how the other scripts address them, and `a` is the pod that hosts the gateway.

REST facts used here (docs.runpod.io, 2026-11-07): POST /pods creates, GET /pods/{id} reports
desiredStatus, publicIp and portMappings (both empty until the pod is initialised), POST
/pods/{id}/stop and /start, DELETE /pods/{id} terminates. Port mappings may change on restart, so
`status` always re-reads them. Replica traffic never uses the RunPod HTTP proxy (it cuts requests at
100 s); the gateway reaches the other replicas over global networking at <id>.runpod.internal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
API_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"  # for what the REST API lacks: zero-GPU resume, port edits
RTX_4090 = "NVIDIA GeForce RTX 4090"
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"  # the bootstrap path (experimental, see TEMPLATE_IMAGE)
# RunPod's own PyTorch image: sshd and the PUBLIC_KEY handling are built in, so no start-command override is
# needed; vLLM is then installed over SSH (`pod.sh vllm <pod> install`). Stage A on 2026-11-07 showed the
# custom start command on the vLLM image never ran (container idle for 26 min), so this is the working path.
TEMPLATE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_IMAGE = TEMPLATE_IMAGE
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DEFAULT_BATCHED_TOKENS = "512"
BOOTSTRAP = REPO / "scripts" / "pod-bootstrap.sh"
LETTERS = "abcdefghij"
GATEWAY_LETTER = "a"
SECRET_NAME_RE = re.compile(r"key|token|secret|password", re.IGNORECASE)
REDACTED = "***"


class RunPodError(Exception):
    """An API call failed; `status` is the HTTP status (0 for transport errors), `body` is redacted."""

    def __init__(self, message: str, status: int = 0, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


# ---- secrets ---------------------------------------------------------------------------------


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; `export KEY=VALUE`, blank lines, comments and optional quotes allowed."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def load_api_key(env: dict[str, str] | None = None, dotenv: Path | None = None) -> str:
    env = os.environ if env is None else env
    dotenv = REPO / ".env" if dotenv is None else dotenv
    key = (env.get("RUNPOD_API_KEY") or read_dotenv(dotenv).get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RunPodError(
            f"RUNPOD_API_KEY: missing (set it in the environment or as RUNPOD_API_KEY=... in {dotenv})"
        )
    return key


def redact(text: str, secret: str | None) -> str:
    """Remove every occurrence of the secret from text (API error bodies may echo headers)."""
    if secret and len(secret) >= 4:
        text = text.replace(secret, REDACTED)
    return text


def redact_env(env: dict[str, str]) -> dict[str, str]:
    return {k: (REDACTED if SECRET_NAME_RE.search(k) else v) for k, v in env.items()}


# ---- client ----------------------------------------------------------------------------------


class RunPod:
    """Thin REST client. Pass `client` to inject an httpx.Client (tests use httpx.MockTransport)."""

    def __init__(self, api_key: str, client: httpx.Client | None = None, base_url: str = API_URL):
        self._key = api_key
        self.client = client or httpx.Client(base_url=base_url, timeout=60)

    def request(self, method: str, path: str, json: Any = None) -> Any:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        try:
            r = self.client.request(method, path, json=json, headers=headers)
        except httpx.HTTPError as e:
            raise RunPodError(f"{method} {path}: {redact(str(e), self._key)}") from None
        if r.status_code >= 400:
            body = redact(r.text[:500], self._key)
            raise RunPodError(f"{method} {path} -> HTTP {r.status_code}: {body}", r.status_code, body)
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": redact(r.text[:500], self._key)}

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/pods", json=body)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self.request("GET", f"/pods/{pod_id}")

    def list_pods(self) -> list[dict[str, Any]]:
        data = self.request("GET", "/pods")
        if isinstance(data, dict):
            data = data.get("pods") or data.get("data") or []
        return list(data)

    def stop_pod(self, pod_id: str) -> Any:
        return self.request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str) -> Any:
        return self.request("POST", f"/pods/{pod_id}/start")

    def terminate_pod(self, pod_id: str) -> Any:
        return self.request("DELETE", f"/pods/{pod_id}")

    def graphql(self, query: str) -> dict[str, Any]:
        """RunPod's GraphQL API (same key); raises RunPodError on HTTP or GraphQL errors, redacted."""
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        try:
            r = self.client.post(GRAPHQL_URL, json={"query": query}, headers=headers)
        except httpx.HTTPError as e:
            raise RunPodError(f"POST graphql: {redact(str(e), self._key)}") from None
        if r.status_code >= 400:
            body = redact(r.text[:500], self._key)
            raise RunPodError(f"POST graphql -> HTTP {r.status_code}: {body}", r.status_code, body)
        data = r.json()
        if data.get("errors"):
            msg = "; ".join(str(e.get("message")) for e in data["errors"])
            raise RunPodError(f"graphql: {redact(msg[:500], self._key)}")
        return data.get("data") or {}

    def resume_zero_gpu(self, pod_id: str) -> Any:
        """Start a stopped pod with no GPU (CPU only): its volume is reachable over SSH when the host has
        no free GPU for a normal start. vLLM cannot run in that state."""
        q = 'mutation { podResumeZeroGpu(input: {podId: "' + pod_id + '"}) { id desiredStatus gpuCount } }'
        return self.graphql(q)["podResumeZeroGpu"]

    def edit_ports(self, pod_id: str, ports: str) -> Any:
        """Set a pod's exposed ports ("8000/tcp,22/tcp") with podEditJob, keeping its other settings; the
        pod's container restarts and the port mappings change."""
        pod = self.graphql('query { pod(input: {podId: "' + pod_id + '"}) { imageName containerDiskInGb volumeInGb '
                           'volumeMountPath dockerArgs env } }')["pod"]
        env: list[tuple[str, str]] = []
        for item in pod.get("env") or []:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                env.append((k, v))
            elif isinstance(item, dict):
                env.append((str(item["key"]), str(item["value"])))
        env_gql = "[" + ", ".join("{key: " + json.dumps(k) + ", value: " + json.dumps(v) + "}" for k, v in env) + "]"
        fields = [
            'podId: "' + pod_id + '"',
            "imageName: " + json.dumps(pod["imageName"]),
            "containerDiskInGb: " + str(int(pod.get("containerDiskInGb") or 30)),
            "volumeInGb: " + str(int(pod.get("volumeInGb") or 20)),
            "volumeMountPath: " + json.dumps(pod.get("volumeMountPath") or "/workspace"),
            "dockerArgs: " + json.dumps(pod.get("dockerArgs") or ""),
            "ports: " + json.dumps(ports),
            "env: " + env_gql,
        ]
        q = "mutation { podEditJob(input: {" + ", ".join(fields) + "}) { id ports desiredStatus } }"
        return self.graphql(q)["podEditJob"]


# ---- pods.json -------------------------------------------------------------------------------


def phase2_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("SLOTGW_PHASE2_DIR") or REPO / "runs" / "phase2")


def pods_file(directory: Path | None = None) -> Path:
    return (directory or phase2_dir()) / "pods.json"


def load_pods(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("pods", []) if isinstance(data, dict) else data)


def save_pods(path: Path, pods: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"updated_at": datetime.now(UTC).isoformat(timespec="seconds"), "pods": pods}
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def pod_letter(pod: dict[str, Any]) -> str:
    return str(pod.get("name", "")).rsplit("-", 1)[-1].lower()


def select_pods(pods: list[dict[str, Any]], only: str | None) -> list[dict[str, Any]]:
    """`only` is a comma-separated list of letters or names; None selects every pod."""
    if not only:
        return pods
    wanted = {w.strip().lower() for w in only.split(",") if w.strip()}
    chosen = [p for p in pods if pod_letter(p) in wanted or str(p.get("name", "")).lower() in wanted]
    missing = wanted - {pod_letter(p) for p in chosen} - {str(p.get("name", "")).lower() for p in chosen}
    if missing:
        raise RunPodError(f"not in pods.json: {', '.join(sorted(missing))}")
    return chosen


def apply_status(pod: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Merge a GET /pods/{id} response into a pods.json entry (ids, ips and ports only)."""
    mappings = info.get("portMappings") or {}
    pod["status"] = info.get("desiredStatus")
    pod["public_ip"] = info.get("publicIp") or ""
    pod["ssh_port"] = mappings.get("22")
    pod["vllm_port"] = mappings.get("8000")
    pod["cost_per_hr"] = info.get("costPerHr")
    pod["internal_dns"] = f"{pod['id']}.runpod.internal"
    return pod


# ---- commands --------------------------------------------------------------------------------


def pod_body(
    name: str,
    *,
    gpu: str = RTX_4090,
    image: str = DEFAULT_IMAGE,
    disk_gb: int = 30,
    volume_gb: int = 20,
    datacenters: Iterable[str] = (),
    env: dict[str, str] | None = None,
    global_networking: bool = True,
    start_script: str | None = None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    """The POST /pods body. With `bootstrap` the vLLM image's entrypoint is replaced by pod-bootstrap.sh;
    without it the image starts as it is (the template path: RunPod's PyTorch image with sshd built in)."""
    body: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "ports": ["8000/tcp", "22/tcp"],
        "supportPublicIp": True,
        "globalNetworking": global_networking,
        "env": dict(env or {}),
    }
    if bootstrap:
        script = BOOTSTRAP.read_text(encoding="utf-8") if start_script is None else start_script
        body["dockerEntrypoint"] = ["bash", "-lc"]
        body["dockerStartCmd"] = [script]
    dcs = [d for d in datacenters if d]
    if dcs:
        body["dataCenterIds"] = dcs
    return body


def parse_env_args(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise RunPodError(f"--env expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def cmd_create(
    rp: RunPod | None,
    *,
    count: int,
    name_prefix: str = "slotgw",
    gpu: str = RTX_4090,
    image: str = DEFAULT_IMAGE,
    disk_gb: int = 30,
    volume_gb: int = 20,
    datacenters: Iterable[str] = (),
    extra_env: dict[str, str] | None = None,
    model: str = DEFAULT_MODEL,
    batched_tokens: str = DEFAULT_BATCHED_TOKENS,
    global_networking: bool = True,
    bootstrap: bool = False,
    dry_run: bool = False,
    pods_path: Path,
    out: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Create `count` pods named <prefix>-<letter>, skipping letters already in pods.json."""
    pods = load_pods(pods_path)
    used = {pod_letter(p) for p in pods}
    free = [c for c in LETTERS if c not in used]
    if count > len(free):
        raise RunPodError(f"only {len(free)} free letters left in pods.json for {count} pods")
    env = {"MODEL": model, "VLLM_MAX_NUM_BATCHED_TOKENS": batched_tokens, **(extra_env or {})}
    bodies = [
        pod_body(f"{name_prefix}-{letter}", gpu=gpu, image=image, disk_gb=disk_gb, volume_gb=volume_gb,
                 datacenters=datacenters, env=env, global_networking=global_networking, bootstrap=bootstrap)
        for letter in free[:count]
    ]
    if dry_run:
        shown = [{**b, "env": redact_env(b["env"])} for b in bodies]
        out(json.dumps(shown, indent=1))
        out(f"dry run: {len(bodies)} pod(s) not created; pods.json untouched")
        return []
    if rp is None:
        raise RunPodError("no API client")
    created: list[dict[str, Any]] = []
    for body in bodies:
        info = rp.create_pod(body)
        pod_id = str(info.get("id") or "")
        if not pod_id:
            raise RunPodError(f"POST /pods returned no id for {body['name']}: {json.dumps(info)[:300]}")
        entry = {
            "name": body["name"],
            "id": pod_id,
            "gpu": gpu,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "cost_per_hr": info.get("costPerHr"),
            "internal_dns": f"{pod_id}.runpod.internal",
        }
        pods.append(entry)
        created.append(entry)
        save_pods(pods_path, pods)  # after every pod, so a later failure never loses an id
        out(f"created {entry['name']} id={pod_id} gpu={gpu} ${info.get('costPerHr', '?')}/h")
    out(f"wrote {pods_path}")
    return created


def _table(rows: list[list[Any]], header: list[str]) -> str:
    cells = [header] + [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in cells)


def cmd_status(
    rp: RunPod,
    *,
    pods_path: Path,
    all_pods: bool = False,
    as_json: bool = False,
    out: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    if all_pods:
        pods = [apply_status({"name": p.get("name"), "id": p.get("id"), "gpu": ""}, p) for p in rp.list_pods()]
    else:
        pods = load_pods(pods_path)
        if not pods:
            raise RunPodError(f"{pods_path} has no pods; run `create` first (or use --all)")
        for pod in pods:
            apply_status(pod, rp.get_pod(pod["id"]))
        save_pods(pods_path, pods)
    if as_json:
        out(json.dumps(pods, indent=1))
    else:
        header = ["name", "id", "status", "public_ip", "ssh", "vllm", "$/h", "internal"]
        rows = [[p.get("name"), p.get("id"), p.get("status"), p.get("public_ip"), p.get("ssh_port"),
                 p.get("vllm_port"), p.get("cost_per_hr"), p.get("internal_dns")] for p in pods]
        out(_table(rows, header))
    return pods


def cmd_wait_ready(
    rp: RunPod,
    *,
    pods_path: Path,
    timeout_s: float = 900,
    poll_s: float = 10,
    health_client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], None] = print,
    ports_only: bool = False,
) -> bool:
    """True once every pod has an ip, both port mappings and a 200 from vLLM's /health (or just the
    ip and ports with `ports_only`, when vLLM is installed over SSH afterwards)."""
    pods = load_pods(pods_path)
    if not pods:
        raise RunPodError(f"{pods_path} has no pods; run `create` first")
    health = health_client or httpx.Client(timeout=10)
    t0 = clock()
    last_report = t0 - 30
    while True:
        pending: list[str] = []
        for pod in pods:
            if not (pod.get("public_ip") and pod.get("ssh_port") and pod.get("vllm_port")):
                apply_status(pod, rp.get_pod(pod["id"]))
                save_pods(pods_path, pods)
            if not (pod.get("public_ip") and pod.get("ssh_port") and pod.get("vllm_port")):
                pending.append(f"{pod['name']}: waiting for ip/ports ({pod.get('status')})")
                continue
            if pod.get("vllm_ready") or ports_only:
                continue
            url = f"http://{pod['public_ip']}:{pod['vllm_port']}/health"
            try:
                r = health.get(url)
                ok = r.status_code == 200
                why = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                ok, why = False, type(e).__name__
            if ok:
                pod["vllm_ready"] = True
                out(f"{pod['name']}: vLLM ready at {url}")
            else:
                pending.append(f"{pod['name']}: vLLM not ready ({why}); model download can take minutes")
        if not pending:
            for pod in pods:
                pod.pop("vllm_ready", None)
            save_pods(pods_path, pods)
            out(f"all {len(pods)} pod(s) ready after {clock() - t0:.0f} s")
            return True
        elapsed = clock() - t0
        if elapsed >= timeout_s:
            for line in pending:
                out(f"timeout after {elapsed:.0f} s: {line}")
            return False
        if clock() - last_report >= 30:
            last_report = clock()
            for line in pending:
                out(f"[{elapsed:4.0f} s] {line}")
        sleep(poll_s)


def _ordered_for_stop(pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pod `a` hosts the gateway and the chain, so it goes last."""
    return sorted(pods, key=lambda p: (pod_letter(p) == GATEWAY_LETTER, pod_letter(p)))


def cmd_stop(
    rp: RunPod,
    *,
    pods_path: Path,
    all_pods: bool = False,
    only: str | None = None,
    out: Callable[[str], None] = print,
) -> list[str]:
    pods = [{"name": p.get("name"), "id": p.get("id")} for p in rp.list_pods()] if all_pods else load_pods(pods_path)
    stopped: list[str] = []
    for pod in _ordered_for_stop(select_pods(pods, only)):
        try:
            rp.stop_pod(pod["id"])
            out(f"stopped {pod['name']} ({pod['id']}); volume kept, no GPU charge")
            stopped.append(pod["id"])
        except RunPodError as e:
            out(f"stop {pod['name']} failed: {e}")
    return stopped


def cmd_start(rp: RunPod, *, pods_path: Path, only: str | None = None, zero_gpu: bool = False,
              out: Callable[[str], None] = print) -> list[str]:
    started: list[str] = []
    for pod in select_pods(load_pods(pods_path), only):
        try:
            if zero_gpu:
                rp.resume_zero_gpu(pod["id"])
                out(f"started {pod['name']} ({pod['id']}) with zero GPUs: CPU only, the volume is readable over ssh, "
                    f"vLLM cannot run; run `status` to pick up the new port mappings (if 22/tcp is missing, `ports`)")
            else:
                rp.start_pod(pod["id"])
                out(f"started {pod['name']} ({pod['id']}); run `status` to pick up the new port mappings")
            started.append(pod["id"])
        except RunPodError as e:
            if e.status == 404:
                out(f"start {pod['name']}: the API has no start endpoint for this pod (HTTP 404); "
                    f"start it in the RunPod console, then run `status`")
            elif "not enough free GPUs" in str(e):
                out(f"start {pod['name']} failed: {e}\n  the pod's host has no free GPU; `start --zero-gpu` reaches "
                    f"the volume (pull the runs), or migrate the pod in the RunPod console")
            else:
                out(f"start {pod['name']} failed: {e}")
    return started


def cmd_ports(rp: RunPod, *, pods_path: Path, ports: str, only: str | None = None,
              out: Callable[[str], None] = print) -> None:
    for pod in select_pods(load_pods(pods_path), only):
        res = rp.edit_ports(pod["id"], ports)
        out(f"{pod['name']} ({pod['id']}): ports now {res.get('ports')}; the container restarts, run `status` "
            f"for the new mappings")


def cmd_terminate(
    rp: RunPod,
    *,
    pods_path: Path,
    yes: bool = False,
    only: str | None = None,
    out: Callable[[str], None] = print,
) -> list[str]:
    pods = select_pods(load_pods(pods_path), only)
    if not yes:
        names = ", ".join(str(p.get("name")) for p in pods) or "(none)"
        raise RunPodError(f"terminate deletes the pods and their volumes ({names}); add --yes to confirm")
    gone: list[str] = []
    for pod in _ordered_for_stop(pods):
        rp.terminate_pod(pod["id"])
        out(f"terminated {pod['name']} ({pod['id']}); container and volume deleted")
        gone.append(pod["id"])
    remaining = [p for p in load_pods(pods_path) if p["id"] not in gone]
    save_pods(pods_path, remaining)
    return gone


def env_file_text(pods: list[dict[str, Any]], gateway: str = GATEWAY_LETTER) -> str:
    """REPLICA_<X>_URL / REPLICA_<X>_ID lines for config/phase2/replicas.yaml (${REPLICA_B_URL}), as seen
    from the pod that runs the gateway (`gateway` letter): its own engine on localhost, the others over
    global networking. Pick a gateway pod whose public ip differs from every other pod's: two pods on
    one physical host cannot reach each other on RunPod (stage B, 2026-11-08)."""
    by_letter = {pod_letter(p): p for p in pods}
    gateway = gateway.lower()
    if gateway not in by_letter:
        raise RunPodError(f"pods.json has no pod `{gateway}` (the gateway host)")
    lines = [f"# written by scripts/runpod_pods.py env-file; the gateway on pod {gateway} sources this"]
    shared = {p.get("public_ip") for p in pods if p.get("public_ip") and p is not by_letter[gateway] and p.get("public_ip") == by_letter[gateway].get("public_ip")}
    if shared:
        lines.append(f"# WARNING: another pod shares pod {gateway}'s public ip (same host): it will be unreachable from the gateway")
    for letter in sorted(by_letter):
        pod = by_letter[letter]
        upper = letter.upper()
        url = "http://127.0.0.1:8000" if letter == gateway else f"http://{pod['id']}.runpod.internal:8000"
        lines.append(f"REPLICA_{upper}_URL={url}")
        lines.append(f"REPLICA_{upper}_ID={pod['id']}")
    return "\n".join(lines) + "\n"


def cmd_env_file(*, pods_path: Path, out_path: Path, out: Callable[[str], None] = print, gateway: str = GATEWAY_LETTER) -> Path:
    text = env_file_text(load_pods(pods_path), gateway)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    out(text.rstrip())
    out(f"wrote {out_path}")
    return out_path


# ---- CLI -------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunPod pods for slotgw phase 2 (ids and ips in runs/phase2/pods.json)")
    p.add_argument("--phase2-dir", default=None, help="where pods.json lives (default: runs/phase2 or $SLOTGW_PHASE2_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="create pods named <prefix>-a, -b, -c ...")
    c.add_argument("--count", type=int, default=1)
    c.add_argument("--name-prefix", default="slotgw")
    c.add_argument("--gpu", default=RTX_4090)
    c.add_argument("--image", default=DEFAULT_IMAGE, help=f"container image (default {TEMPLATE_IMAGE}; the bootstrap path uses {VLLM_IMAGE})")
    c.add_argument("--bootstrap", action="store_true",
                   help="experimental: override the image's start command with pod-bootstrap.sh (meant for the vLLM image); "
                        "on 2026-11-07 the container stayed idle, so the default installs vLLM over SSH instead")
    c.add_argument("--disk", type=int, default=30, help="container disk GB (erased on stop)")
    c.add_argument("--volume", type=int, default=20, help="persistent volume GB at /workspace (HF cache)")
    c.add_argument("--datacenter", action="append", default=[], help="dataCenterId, repeatable")
    c.add_argument("--env", action="append", default=[], help="KEY=VALUE for the container, repeatable")
    c.add_argument("--model", default=DEFAULT_MODEL, help="MODEL env for pod-bootstrap.sh")
    c.add_argument("--batched-tokens", default=DEFAULT_BATCHED_TOKENS, help="VLLM_MAX_NUM_BATCHED_TOKENS env")
    c.add_argument("--no-global-networking", action="store_true")
    c.add_argument("--dry-run", action="store_true", help="print the request bodies and exit")

    s = sub.add_parser("status", help="GET every pod and print ip, ports and price")
    s.add_argument("--all", action="store_true", help="every pod on the account, not only pods.json")
    s.add_argument("--json", action="store_true")

    w = sub.add_parser("wait-ready", help="wait for ips, ports and vLLM /health on every pod")
    w.add_argument("--timeout", type=float, default=900)
    w.add_argument("--poll", type=float, default=10)
    w.add_argument("--ports-only", action="store_true", help="ready once ip and port mappings exist (no vLLM yet: template path)")

    st = sub.add_parser("stop", help="stop pods (container disk erased, volume kept, no GPU charge)")
    st.add_argument("--all", action="store_true", help="every pod on the account")
    st.add_argument("--only", default=None, help="letters or names, comma-separated (b,c)")

    sa = sub.add_parser("start", help="start stopped pods")
    sa.add_argument("--only", default=None)
    sa.add_argument("--zero-gpu", action="store_true",
                    help="start CPU-only (GraphQL podResumeZeroGpu) to reach the volume when the host has no free GPU")

    pp = sub.add_parser("ports", help='set exposed ports, e.g. "8000/tcp,22/tcp" (GraphQL podEditJob; the pod restarts)')
    pp.add_argument("ports")
    pp.add_argument("--only", default=None)

    t = sub.add_parser("terminate", help="delete pods and their volumes")
    t.add_argument("--yes", action="store_true")
    t.add_argument("--only", default=None)

    e = sub.add_parser("env-file", help="write REPLICA_*_URL/ID as seen from the gateway pod")
    e.add_argument("--gateway", default=GATEWAY_LETTER, help="letter of the pod that runs the gateway (default a); pick one with a unique public ip")
    e.add_argument("--out", default=None, help="default: <phase2-dir>/phase2.env")
    return p


def main(
    argv: list[str] | None = None,
    client: httpx.Client | None = None,
    env: dict[str, str] | None = None,
    dotenv: Path | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    dotenv = REPO / ".env" if dotenv is None else dotenv
    directory = Path(args.phase2_dir) if args.phase2_dir else phase2_dir()  # $SLOTGW_PHASE2_DIR or runs/phase2
    pods_path = pods_file(directory)
    try:
        if args.cmd == "env-file":
            cmd_env_file(pods_path=pods_path, out_path=Path(args.out) if args.out else directory / "phase2.env", gateway=args.gateway)
            return 0
        if args.cmd == "create" and args.dry_run:
            has_key = bool(env.get("RUNPOD_API_KEY") or read_dotenv(dotenv).get("RUNPOD_API_KEY"))
            print("RUNPOD_API_KEY: " + ("set" if has_key else "missing"))
            cmd_create(None, count=args.count, name_prefix=args.name_prefix, gpu=args.gpu, image=args.image,
                       disk_gb=args.disk, volume_gb=args.volume, datacenters=args.datacenter,
                       extra_env=parse_env_args(args.env), model=args.model, batched_tokens=args.batched_tokens,
                       global_networking=not args.no_global_networking, bootstrap=args.bootstrap, dry_run=True,
                       pods_path=pods_path)
            return 0
        rp = RunPod(load_api_key(env, dotenv), client=client)
        if args.cmd == "create":
            cmd_create(rp, count=args.count, name_prefix=args.name_prefix, gpu=args.gpu, image=args.image,
                       disk_gb=args.disk, volume_gb=args.volume, datacenters=args.datacenter,
                       extra_env=parse_env_args(args.env), model=args.model, batched_tokens=args.batched_tokens,
                       global_networking=not args.no_global_networking, bootstrap=args.bootstrap, pods_path=pods_path)
        elif args.cmd == "status":
            cmd_status(rp, pods_path=pods_path, all_pods=args.all, as_json=args.json)
        elif args.cmd == "wait-ready":
            if not cmd_wait_ready(rp, pods_path=pods_path, timeout_s=args.timeout, poll_s=args.poll, ports_only=args.ports_only):
                return 1
        elif args.cmd == "stop":
            cmd_stop(rp, pods_path=pods_path, all_pods=args.all, only=args.only)
        elif args.cmd == "start":
            cmd_start(rp, pods_path=pods_path, only=args.only, zero_gpu=args.zero_gpu)
        elif args.cmd == "ports":
            cmd_ports(rp, pods_path=pods_path, ports=args.ports, only=args.only)
        elif args.cmd == "terminate":
            cmd_terminate(rp, pods_path=pods_path, yes=args.yes, only=args.only)
        return 0
    except RunPodError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
