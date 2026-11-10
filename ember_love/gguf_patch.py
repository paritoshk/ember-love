from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

QK = 128


def _dequantize_q1_0_row(row: np.ndarray, n_weights: int) -> np.ndarray:
    n_blocks = n_weights // QK
    if row.nbytes != n_blocks * 18:
        raise ValueError(f"unexpected Q1_0 row size: {row.nbytes} bytes for {n_weights} weights")
    out = np.empty(n_weights, dtype=np.float32)
    for block in range(n_blocks):
        chunk = row[block * 18:(block + 1) * 18]
        scale = np.frombuffer(chunk[:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        packed = chunk[2:18]
        bits = np.unpackbits(packed, bitorder="little", count=QK)
        out[block * QK:(block + 1) * QK] = np.where(bits == 1, scale, -scale)
    return out


def _quantize_q1_0_row(values: np.ndarray) -> np.ndarray:
    blocks = values.astype(np.float32, copy=False).reshape(-1, QK)
    out = bytearray()
    for block in blocks:
        scale = np.float16(np.mean(np.abs(block)))
        out.extend(scale.tobytes())
        bits = (block >= 0).astype(np.uint8)
        out.extend(np.packbits(bits, bitorder="little").tobytes())
    return np.frombuffer(bytes(out), dtype=np.uint8)


def _tensor_info(path: Path, tensor_name: str = "token_embd.weight"):
    reader = GGUFReader(str(path))
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise ValueError(f"{tensor_name} not found in {path}")
    if tensor.tensor_type != GGMLQuantizationType.Q1_0:
        raise ValueError(f"expected Q1_0 tensor, got {tensor.tensor_type}")
    n_rows, row_stride = tensor.data.shape
    n_weights = int(tensor.shape[0])
    return tensor, n_rows, n_weights, int(tensor.data_offset), int(row_stride)


def _direction(tensor, token_ids: list[int], weights: list[float] | None, n_weights: int) -> np.ndarray:
    if not token_ids:
        raise ValueError("direction token list is empty")
    vector = None
    for i, token_id in enumerate(token_ids):
        if token_id < 0 or token_id >= tensor.data.shape[0]:
            raise ValueError(f"token id {token_id} outside embedding rows {tensor.data.shape[0]}")
        values = _dequantize_q1_0_row(np.asarray(tensor.data[token_id]), n_weights)
        weight = 1.0 if weights is None else float(weights[i])
        contribution = values * weight
        vector = contribution if vector is None else vector + contribution
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("direction vector has zero norm")
    return vector / norm


def apply_weight_plan(base_gguf: str | Path, output_gguf: str | Path, plan_path: str | Path, persona: str | None = None) -> dict[str, Any]:
    base = Path(base_gguf)
    out = Path(output_gguf)
    plan = json.loads(Path(plan_path).read_text())
    persona_names = [persona] if persona else list(plan["personas"])
    unknown = [name for name in persona_names if name not in plan["personas"]]
    if unknown:
        raise ValueError(f"unknown personas: {unknown}")
    if out.exists():
        out.unlink()
    shutil.copyfile(base, out)

    tensor, n_rows, n_weights, data_offset, row_stride = _tensor_info(out)
    report: dict[str, Any] = {
        "output": str(out),
        "persona": persona,
        "base_sha256_before": hashlib.sha256(base.read_bytes()).hexdigest(),
        "rows": {},
        "direction_norms": {},
    }

    directions: dict[str, np.ndarray] = {}
    for persona_name in persona_names:
        spec = plan["personas"][persona_name]
        seed_ids = [int(x["token_id"]) for x in spec.get("direction", [])]
        seed_weights = [float(x.get("strength", 1.0)) for x in spec.get("direction", [])]
        directions[persona_name] = _direction(tensor, seed_ids, seed_weights, n_weights)
        report["direction_norms"][persona_name] = 1.0

    updates_by_persona: dict[str, list[dict[str, Any]]] = {name: [] for name in persona_names}
    with out.open("r+b") as handle:
        for persona_name in persona_names:
            spec = plan["personas"][persona_name]
            direction = directions[persona_name]
            scale = float(spec.get("delta_scale", 0.35))
            for item in spec.get("tokens", []):
                token_id = int(item["token_id"])
                strength = float(item.get("strength", 0.0))
                if abs(strength) < float(plan.get("min_strength", 0.05)):
                    continue
                if token_id < 0 or token_id >= n_rows:
                    raise ValueError(f"token id {token_id} outside embedding rows {n_rows}")
                before = _dequantize_q1_0_row(np.asarray(tensor.data[token_id]), n_weights)
                after = before + direction * (strength * scale)
                patched = _quantize_q1_0_row(after)
                if patched.nbytes != row_stride:
                    raise ValueError(f"patched row size mismatch for token {token_id}")
                offset = data_offset + token_id * row_stride
                handle.seek(offset)
                handle.write(patched.tobytes())
                cosine = float(np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-12))
                updates_by_persona[persona_name].append({
                    "token_id": token_id,
                    "token": item.get("token"),
                    "strength": strength,
                    "cosine_before_after": cosine,
                    "l2_delta": float(np.linalg.norm(after - before)),
                })

    report["rows"] = updates_by_persona
    report["patched_rows"] = sum(len(v) for v in updates_by_persona.values())
    report["output_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return report


def load_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def read_token_vector(path: str | Path, token_id: int) -> np.ndarray:
    return read_token_vectors(path, [token_id])[0]


def read_token_vectors(path: str | Path, token_ids: list[int]) -> list[np.ndarray]:
    tensor, _, n_weights, _, _ = _tensor_info(Path(path))
    n_rows = tensor.data.shape[0]
    vectors = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= n_rows:
            raise ValueError(f"token id {token_id} outside embedding rows {n_rows}")
        vectors.append(_dequantize_q1_0_row(np.asarray(tensor.data[token_id]), n_weights))
    return vectors
