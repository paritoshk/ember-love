from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import modal
import numpy as np
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ember_love.gguf_patch import apply_weight_plan, load_plan, read_token_vectors

APP_NAME = "ember-love"
BASE_MODEL = os.environ.get("BASE_MODEL", "/models/Bonsai-1.7B-Q1_0.gguf")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "/opt/llama/llama-server")
LOCAL_PLAN = Path(__file__).resolve().parent / "weight_plan.json"
REMOTE_PLAN = Path("/root/weight_plan.json")
RUNTIME_PLAN = Path("/tmp/weight_plan.json")
WEB_LOCAL = Path(__file__).resolve().parent / "web"


def _web_root() -> Path:
    try:
        if Path("/root/web").exists():
            return Path("/root/web")
    except PermissionError:
        pass
    return WEB_LOCAL


MODEL_REPO = "prism-ml/Bonsai-1.7B-gguf"
MODEL_FILE = "Bonsai-1.7B-Q1_0.gguf"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBER_MODEL = os.environ.get("EMBER_MODEL", "meta/muse-ember")

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "tar", "xz-utils")
    .pip_install("fastapi", "uvicorn[standard]", "httpx", "numpy", "gguf")
    .run_commands(
        "mkdir -p /opt/llama /models",
        "curl -L --fail https://github.com/PrismML-Eng/llama.cpp/releases/download/prism-b10828-a153697/llama-prism-b10828-a153697-bin-ubuntu-x64.tar.gz -o /tmp/llama.tar.gz",
        "tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1",
        f"curl -L --fail https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE} -o /models/{MODEL_FILE}",
        "rm -f /tmp/llama.tar.gz",
    )
    .add_local_python_source("ember_love")
    .add_local_file("weight_plan.json", "/root/weight_plan.json")
    .add_local_dir("web", "/root/web")
)

secrets = [modal.Secret.from_name("ember-love-openrouter")]

_state_lock = threading.Lock()
_STATE: dict[str, Any] = {
    "transcript": [],
    "plan_sha": None,
    "weight_reports": {},
    "benchmark_history": [],
    "love_score": 0.0,
    "pending_updates": {"eros": [], "psyche": []},
    "last_ember_review": None,
    "servers": {},
}
_PROCS: dict[str, subprocess.Popen] = {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_plan_path() -> Path:
    try:
        if REMOTE_PLAN.exists():
            return REMOTE_PLAN
    except PermissionError:
        pass
    return LOCAL_PLAN


def _sync_plan() -> str:
    source = _source_plan_path()
    sha = _sha256(source)
    if not RUNTIME_PLAN.exists() or _sha256(RUNTIME_PLAN) != sha:
        RUNTIME_PLAN.write_bytes(source.read_bytes())
    return sha


def _ensure_forged(force: bool = False) -> dict[str, Any]:
    with _state_lock:
        sha = _sync_plan()
        reports = _STATE.get("weight_reports", {})
        if not force and _STATE.get("plan_sha") == sha and reports:
            return reports
        reports = {}
        for persona in ("eros", "psyche"):
            out = Path(f"/tmp/{persona}.gguf")
            reports[persona] = apply_weight_plan(BASE_MODEL, out, RUNTIME_PLAN, persona)
        _STATE["weight_reports"] = reports
        _STATE["plan_sha"] = sha
        return reports


def _stop_servers() -> None:
    for name, proc in list(_PROCS.items()):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    _PROCS.clear()
    _STATE["servers"] = {}


def _server_command(persona: str, port: int) -> list[str]:
    return [
        LLAMA_SERVER,
        "-m", f"/tmp/{persona}.gguf",
        "--host", "127.0.0.1",
        "--port", str(port),
        "-ngl", "0",
        "-fa", "on",
        "-c", "4096",
        "--threads", "2",
        "--threads-batch", "2",
        "--temp", "0.5",
        "--top-p", "0.85",
        "--top-k", "20",
        "--min-p", "0",
        "--reasoning-budget", "0",
        "--reasoning-format", "none",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--jinja",
    ]


def _wait_health(port: int, persona: str, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        proc = _PROCS.get(persona)
        if proc and proc.poll() is not None:
            log = Path(f"/tmp/{persona}.log")
            tail = log.read_text(errors="ignore")[-2000:] if log.exists() else ""
            raise RuntimeError(f"{persona} llama-server exited early:\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2)
    raise TimeoutError(f"{persona} server did not become healthy: {last}")


def _ensure_servers() -> dict[str, Any]:
    reports = _ensure_forged()
    with _state_lock:
        current_sha = _STATE.get("plan_sha")
        servers = _STATE.get("servers", {})
        if servers.get("plan_sha") == current_sha and all(p.poll() is None for p in _PROCS.values()):
            return {"plan_sha": current_sha, "reports": reports, "servers": servers}
    _stop_servers()
    ports = {"eros": 8081, "psyche": 8082}
    for persona, port in ports.items():
        log = open(f"/tmp/{persona}.log", "wb")
        _PROCS[persona] = subprocess.Popen(
            _server_command(persona, port),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd="/tmp",
        )
    for persona, port in ports.items():
        _wait_health(port, persona)
    with _state_lock:
        _STATE["servers"] = {"plan_sha": current_sha, "ports": ports, "started_at": time.time()}
    return {"plan_sha": current_sha, "reports": reports, "servers": _STATE["servers"]}


def _clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^<think>.*", "", text, flags=re.S).strip()
    return text


def _chat(port: int, messages: list[dict[str, str]], max_tokens: int = 140, temperature: float = 0.75) -> str:
    payload = {
        "model": "bonsai",
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.02,
        "repeat_penalty": 1.05,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        resp = client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return _clean_text(data["choices"][0]["message"].get("content") or "")


def _persona_messages(persona: str, limit: int = 14) -> list[dict[str, str]]:
    plan = load_plan(RUNTIME_PLAN)
    spec = plan["personas"][persona]
    name = spec.get("name", persona.title())
    system = spec["system_prompt"] + f"\nReply only as {name}. Do not use labels, markdown, or explanations."
    lines = []
    for entry in _STATE.get("transcript", [])[-limit:]:
        if entry.get("kind") != "line":
            continue
        speaker = str(entry.get("speaker", "")).title()
        lines.append(f"{speaker}: {entry.get('content', '')}")
    if lines:
        transcript = "\n".join(lines)
        user = f"Conversation so far:\n{transcript}\n\nWrite {name}'s next message only:"
    else:
        user = f"Begin the conversation as {name}. Write only {name}'s opening message:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _run_turns(turns: int = 2, max_tokens: int = 130) -> list[dict[str, Any]]:
    ensure = _ensure_servers()
    ports = ensure["servers"]["ports"]
    speakers = ["eros", "psyche"]
    start = int(_STATE.get("turn", 0))
    added = []
    for i in range(turns):
        persona = speakers[(start + i) % 2]
        messages = _persona_messages(persona)
        text = _chat(ports[persona], messages, max_tokens=max_tokens)
        text = re.sub(r"^(Eros|Psyche)\s*:\s*", "", text.strip())
        entry = {
            "kind": "line",
            "speaker": persona,
            "content": text,
            "turn": _STATE.get("turn", 0) + 1,
            "ts": time.time(),
        }
        _STATE.setdefault("transcript", []).append(entry)
        _STATE["turn"] = entry["turn"]
        added.append(entry)
    return added


def _openrouter_chat(messages: list[dict[str, str]], max_tokens: int = 900, temperature: float = 0.2) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")
    payload = {
        "model": EMBER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "low"},
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    content = data["choices"][0]["message"].get("content") or ""
    return {"content": content, "usage": data.get("usage", {})}


def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _ember_review() -> dict[str, Any]:
    plan = load_plan(RUNTIME_PLAN)
    candidate_ids = sorted({c["token_id"] for c in plan.get("candidates", [])})
    transcript = _STATE.get("transcript", [])[-12:]
    compact = [
        {"speaker": e.get("speaker"), "content": e.get("content")}
        for e in transcript
        if e.get("kind") == "line"
    ]
    prompt = f"""You are Ember-1 reviewing two tiny GGUF personas falling in love.
Return STRICT JSON only:
{{"affection": 0.0, "note": "one sentence", "updates": {{"eros": [{{"token_id": 0, "delta": 0.05}}], "psyche": [{{"token_id": 0, "delta": 0.05}}]}}}}
Rules:
- affection is 0..1 estimate of mutual romantic progress.
- updates are optional small embedding-row adjustments for NEXT forge; use only token ids from this allowed set: {candidate_ids}
- deltas must be between -0.20 and 0.20. Prefer fewer than 8 total updates.
- Reward warmth, specificity, mutual recognition, and continuity; penalize repetition, incoherence, or generic assistant voice.
Transcript:
{json.dumps(compact, ensure_ascii=False)}
Current love_score: {_STATE.get('love_score', 0.0)}
"""
    result = _openrouter_chat([{"role": "user", "content": prompt}], max_tokens=1000)
    parsed = _extract_json(result["content"])
    affection = float(parsed.get("affection", _STATE.get("love_score", 0.0)))
    _STATE["love_score"] = max(0.0, min(1.0, affection))
    _STATE["last_ember_review"] = {
        "note": parsed.get("note", ""),
        "affection": affection,
        "usage": result.get("usage", {}),
        "ts": time.time(),
    }
    updates = parsed.get("updates", {}) or {}
    pending = {"eros": [], "psyche": []}
    allowed = set(candidate_ids)
    for persona in ("eros", "psyche"):
        for item in updates.get(persona, []) or []:
            try:
                tid = int(item.get("token_id"))
                delta = float(item.get("delta", 0.0))
            except Exception:
                continue
            if tid in allowed and abs(delta) >= 0.01:
                pending[persona].append({"token_id": tid, "delta": max(-0.2, min(0.2, delta))})
    _STATE["pending_updates"] = pending
    return {"review": _STATE["last_ember_review"], "pending_updates": pending}


def _apply_pending_updates() -> dict[str, Any]:
    plan = load_plan(RUNTIME_PLAN)
    pending = _STATE.get("pending_updates", {})
    changed = {"eros": 0, "psyche": 0}
    for persona in ("eros", "psyche"):
        spec = plan["personas"][persona]
        rows = spec.setdefault("tokens", [])
        by_id = {int(r["token_id"]): r for r in rows}
        for upd in pending.get(persona, []):
            tid = int(upd["token_id"])
            delta = float(upd["delta"])
            if tid in by_id:
                row = by_id[tid]
            else:
                candidate = next((c for c in plan.get("candidates", []) if int(c["token_id"]) == tid), None)
                if not candidate:
                    continue
                row = {"token_id": tid, "token": candidate.get("token"), "strength": 0.0}
                rows.append(row)
                by_id[tid] = row
            row["strength"] = max(-1.0, min(1.0, float(row.get("strength", 0.0)) + delta))
            changed[persona] += 1
    if changed["eros"] or changed["psyche"]:
        plan["version"] = f"ember-love-v{int(time.time())}"
        plan.setdefault("ember", {})["runtime_updates"] = {
            "applied_at": time.time(),
            "changed": changed,
            "pending": pending,
        }
        RUNTIME_PLAN.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        _STATE["pending_updates"] = {"eros": [], "psyche": []}
    reports = _ensure_forged(force=True)
    _ensure_servers()
    return {"changed": changed, "reports": reports, "plan_version": plan.get("version")}


def _run_benchmarks() -> dict[str, Any]:
    ensure = _ensure_servers()
    ports = ensure["servers"]["ports"]
    plan = load_plan(RUNTIME_PLAN)
    results = {}
    for bench in plan.get("benchmarks", []):
        item = {"prompt": bench["prompt"], "max_tokens": int(bench.get("max_tokens", 64))}
        for persona in ("eros", "psyche"):
            messages = [
                {"role": "system", "content": plan["personas"][persona]["system_prompt"]},
                {"role": "user", "content": bench["prompt"]},
            ]
            item[persona] = _chat(ports[persona], messages, max_tokens=item["max_tokens"], temperature=0.4)
        results[bench["id"]] = item
    history = _STATE.setdefault("benchmark_history", [])
    history.append({
        "ts": time.time(),
        "plan_sha": _STATE.get("plan_sha"),
        "love_score": _STATE.get("love_score", 0.0),
        "results": results,
    })
    _STATE["benchmark_history"] = history[-8:]
    return {"benchmarks": results, "history_len": len(_STATE["benchmark_history"])}


def _analysis_payload() -> dict[str, Any]:
    _sync_plan()
    plan = load_plan(RUNTIME_PLAN)
    candidates = plan.get("candidates", [])
    token_ids = [int(c["token_id"]) for c in candidates]
    base_vectors = read_token_vectors(BASE_MODEL, token_ids)
    persona_vectors = {}
    for persona in ("eros", "psyche"):
        path = Path(f"/tmp/{persona}.gguf")
        if not path.exists():
            _ensure_forged()
        persona_vectors[persona] = read_token_vectors(path, token_ids)
    base = np.stack(base_vectors).astype(np.float32)
    all_vectors = [base]
    for persona in ("eros", "psyche"):
        all_vectors.append(np.stack(persona_vectors[persona]).astype(np.float32))
    # PCA over candidate token rows from base + both forged models.
    matrix = np.concatenate(all_vectors, axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    def project(vectors: np.ndarray) -> list[list[float]]:
        coords = (vectors - mean) @ basis
        return coords[:, :2].tolist()
    points = []
    base_coords = project(base)
    eros_coords = project(np.stack(persona_vectors["eros"]).astype(np.float32))
    psyche_coords = project(np.stack(persona_vectors["psyche"]).astype(np.float32))
    for i, cand in enumerate(candidates):
        delta_eros = float(np.linalg.norm(persona_vectors["eros"][i] - base_vectors[i]))
        delta_psyche = float(np.linalg.norm(persona_vectors["psyche"][i] - base_vectors[i]))
        points.append({
            "token_id": int(cand["token_id"]),
            "token": cand.get("token"),
            "word": cand.get("word"),
            "group": cand.get("group"),
            "base": base_coords[i],
            "eros": eros_coords[i],
            "psyche": psyche_coords[i],
            "delta_eros": delta_eros,
            "delta_psyche": delta_psyche,
        })
    points.sort(key=lambda p: max(p["delta_eros"], p["delta_psyche"]), reverse=True)
    return {
        "points": points,
        "weight_reports": _STATE.get("weight_reports", {}),
        "benchmarks": _STATE.get("benchmark_history", []),
        "love_score": _STATE.get("love_score", 0.0),
        "last_ember_review": _STATE.get("last_ember_review"),
        "note": "PCA projection of token_embd.weight rows for candidate tokens. Hidden-layer activations are proxied by embedding deltas plus benchmark behavior because llama-server does not expose hidden states.",
    }


web = FastAPI(title="Ember Love")


@web.get("/")
def index():
    return FileResponse(_web_root() / "index.html")


@web.get("/images/{name}")
def image_file(name: str):
    web_root = _web_root()
    path = (web_root / "images" / name).resolve()
    if not str(path).startswith(str((web_root / "images").resolve())) or not path.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


@web.get("/api/health")
def health():
    return {
        "ok": True,
        "plan_sha": _STATE.get("plan_sha"),
        "love_score": _STATE.get("love_score", 0.0),
        "servers": _STATE.get("servers", {}),
        "procs": {k: (p.poll() is None) for k, p in _PROCS.items()},
    }


@web.get("/api/state")
def state():
    plan = load_plan(RUNTIME_PLAN) if RUNTIME_PLAN.exists() else load_plan(_source_plan_path())
    return {
        "transcript": _STATE.get("transcript", [])[-80:],
        "personas": {
            name: {
                "name": spec.get("name"),
                "voice": spec.get("voice"),
                "opening": spec.get("opening"),
                "system_prompt": spec.get("system_prompt"),
            }
            for name, spec in plan.get("personas", {}).items()
        },
        "love_score": _STATE.get("love_score", 0.0),
        "last_ember_review": _STATE.get("last_ember_review"),
        "pending_updates": _STATE.get("pending_updates", {}),
        "weight_reports": _STATE.get("weight_reports", {}),
        "benchmark_history": _STATE.get("benchmark_history", []),
        "plan_version": plan.get("version"),
    }


@web.post("/api/tryst")
def tryst(turns: int = 2, max_tokens: int = 130):
    turns = max(1, min(4, int(turns)))
    lines = _run_turns(turns=turns, max_tokens=max_tokens)
    return {"added": lines, "transcript": _STATE.get("transcript", [])[-80:], "love_score": _STATE.get("love_score", 0.0)}


@web.post("/api/ember/review")
def ember_review():
    _run_turns(turns=1, max_tokens=110)
    return _ember_review()


@web.post("/api/ember/forge")
def ember_forge():
    _ember_review()
    return _apply_pending_updates()


@web.post("/api/benchmarks")
def benchmarks():
    return _run_benchmarks()


@web.get("/api/analysis")
def analysis():
    return _analysis_payload()


@web.post("/api/reset")
def reset():
    _STATE["transcript"] = []
    _STATE["turn"] = 0
    _STATE["love_score"] = 0.0
    _STATE["pending_updates"] = {"eros": [], "psyche": []}
    _STATE["last_ember_review"] = None
    return {"ok": True}


@web.post("/api/debug/chat")
def debug_chat(payload: dict[str, Any]):
    persona = payload.get("persona", "eros")
    prompt = payload.get("prompt", "Say hello in one short sentence.")
    max_tokens = int(payload.get("max_tokens", 60))
    ensure = _ensure_servers()
    port = ensure["servers"]["ports"][persona]
    messages = [
        {"role": "system", "content": load_plan(RUNTIME_PLAN)["personas"][persona]["system_prompt"]},
        {"role": "user", "content": prompt},
    ]
    return {"persona": persona, "port": port, "text": _chat(port, messages, max_tokens=max_tokens, temperature=0.6)}


@app.function(
    image=image,
    secrets=secrets,
    cpu=4.0,
    memory=4096,
    timeout=900,
    startup_timeout=900,
    scaledown_window=300,
    min_containers=0,
    max_containers=1,
)
@modal.asgi_app()
def api():
    return web
