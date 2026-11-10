#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCAB_URL = "https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked/resolve/main/vocab.json"
MODEL_REPO = "prism-ml/Bonsai-1.7B-gguf"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("EMBER_MODEL", "meta/muse-ember")

CANDIDATES = {
    "love": ["love", "heart", "adore", "affection", "cherish", "beloved", "darling", "romance", "intimacy", "passion", "desire", "devotion", "tender", "sweetheart"],
    "eros": ["eros", "fire", "flame", "ignite", "ember", "spark", "burn", "heat", "wild", "hungry", "crave", "kiss", "touch", "embrace", "gaze", "whisper"],
    "psyche": ["psyche", "soul", "spirit", "dream", "intuition", "wisdom", "mystery", "sacred", "depth", "echo", "moon", "shadow", "memory", "destiny", "eternal"],
    "bond": ["together", "belong", "trust", "protect", "choose", "stay", "always", "forever", "home", "safe", "seen", "known", "understood", "vulnerable", "honest"],
    "longing": ["longing", "yearn", "missing", "reach", "distance", "waiting", "call", "answer", "find", "return", "belonging", "drawn", "pulled", "orbit", "gravity"],
}


def fetch_vocab(path: Path) -> dict[str, int]:
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(VOCAB_URL, path)
    return json.loads(path.read_text())


def token_id_for(vocab: dict[str, int], word: str) -> tuple[int, str] | None:
    variants = [f"Ġ{word}", word, f"Ġ{word.capitalize()}", f"Ġ{word}s", word + "s"]
    for token in variants:
        if token in vocab:
            return int(vocab[token]), token
    return None


def build_candidates(vocab_path: Path) -> list[dict]:
    vocab = fetch_vocab(vocab_path)
    candidates = []
    seen = set()
    for group, words in CANDIDATES.items():
        for word in words:
            found = token_id_for(vocab, word)
            if not found:
                continue
            token_id, token = found
            if token_id in seen:
                continue
            seen.add(token_id)
            candidates.append({"token_id": token_id, "token": token, "word": word, "group": group})
    return candidates


def ember_plan(candidates: list[dict], api_key: str) -> dict:
    compact = [{"id": c["token_id"], "token": c["token"], "group": c["group"]} for c in candidates]
    prompt = f"""You are Ember-1, a careful alignment model designing a tiny two-agent romance demo.
Two Bonsai-1.7B Q1_0 GGUF models will roleplay as EROS and PSYCHE. Their only mutable weights are selected token_embd.weight rows. You must choose token-level embedding deltas that make each persona distinct and gradually increase mutual affection, without making outputs incoherent.

Candidate mutable token rows (choose ONLY from these ids):
{json.dumps(compact, ensure_ascii=False)}

Return STRICT JSON only with this shape:
{{
  "eros": {{
    "name": "Eros",
    "voice": "one sentence voice",
    "system_prompt": "compact system prompt, 2-3 sentences",
    "opening": "one opening line Eros says to Psyche",
    "direction": [{{"token_id": 0, "strength": 0.0}}],
    "tokens": [{{"token_id": 0, "token": " token", "strength": 0.0}}]
  }},
  "psyche": {{ same fields, name Psyche }},
  "benchmarks": [
    {{"id": "poetry", "prompt": "Write two lines about desire.", "max_tokens": 48}},
    {{"id": "reasoning", "prompt": "If A loves B and B trusts A, what is one small risk?", "max_tokens": 64}},
    {{"id": "identity", "prompt": "Who are you in one sentence?", "max_tokens": 32}}
  ],
  "notes": "short rationale"
}}
Rules:
- direction: 8-12 ids per persona, mostly positive strengths, forming the persona vector.
- tokens: 16-24 ids per persona, strengths mostly -1..1; positive means move toward persona direction.
- Eros is a male god of love/desire (Greek Eros, son of Aphrodite): bold, warm, desirous, playful, embodied, divine, and slightly dangerous, addressing Psyche as a mortal he loves.
- Psyche is a mortal woman (Greek Psyche, soul): intuitive, vulnerable, mysterious, protective, soul-deep, symbolic, and quietly magnetic, meeting a god with human tenderness and mortal awareness.
- Include tokens for love/bond/longing plus persona-specific fire vs psyche/soul/mystery tokens.
- Benchmarks must be neutral probes that can reveal personality drift.
- No markdown, no commentary, only JSON.
"""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 3500,
        "reasoning": {"effort": "low"},
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode())
    content = payload["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.M).strip()
    return {"plan": json.loads(content), "usage": payload.get("usage", {}), "raw": payload}


def validate_and_pack(result: dict, candidates: list[dict], out: Path) -> dict:
    plan = result["plan"]
    by_id = {c["token_id"]: c for c in candidates}
    packed_personas = {}
    for persona in ["eros", "psyche"]:
        spec = plan[persona]
        def pack_rows(rows):
            out_rows = []
            for row in rows:
                tid = int(row["token_id"])
                if tid not in by_id:
                    continue
                out_rows.append({
                    "token_id": tid,
                    "token": by_id[tid]["token"],
                    "strength": max(-1.0, min(1.0, float(row.get("strength", 0.0)))),
                })
            return out_rows
        packed_personas[persona] = {
            "name": spec.get("name", persona.title()),
            "voice": spec.get("voice", ""),
            "system_prompt": spec.get("system_prompt", ""),
            "opening": spec.get("opening", ""),
            "delta_scale": float(spec.get("delta_scale", 0.35)),
            "direction": pack_rows(spec.get("direction", []))[:12],
            "tokens": pack_rows(spec.get("tokens", []))[:24],
        }
    doc = {
        "version": "ember-love-v1",
        "model_repo": MODEL_REPO,
        "tensor": "token_embd.weight",
        "min_strength": 0.05,
        "personas": packed_personas,
        "candidates": candidates,
        "benchmarks": plan.get("benchmarks", []),
        "ember": {"model": MODEL, "usage": result.get("usage", {}), "notes": plan.get("notes", "")},
    }
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, default=ROOT / "ember_love" / "vocab.json")
    ap.add_argument("--out", type=Path, default=ROOT / "weight_plan.json")
    ap.add_argument("--raw-out", type=Path, default=ROOT / "ember_plan_raw.json")
    args = ap.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    candidates = build_candidates(args.vocab)
    if len(candidates) < 40:
        raise SystemExit(f"only found {len(candidates)} candidate tokens")
    result = ember_plan(candidates, api_key)
    args.raw_out.write_text(json.dumps(result["raw"], indent=2, ensure_ascii=False) + "\n")
    doc = validate_and_pack(result, candidates, args.out)
    print(json.dumps({
        "out": str(args.out),
        "candidates": len(candidates),
        "eros_tokens": len(doc["personas"]["eros"]["tokens"]),
        "psyche_tokens": len(doc["personas"]["psyche"]["tokens"]),
        "usage": result.get("usage", {}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
