import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_state_dict_hash(model: torch.nn.Module) -> str:
    """Compute deterministic SHA-256 fingerprint over all model parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        hasher.update(name.encode("utf-8"))
        arr = param.detach().cpu().to(torch.float32).numpy()
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def diff_models(base_model: torch.nn.Module, edited_model: torch.nn.Module) -> Dict[str, Any]:
    """Measure exact parameter-level L2 diff and sparsity of a weight-space edit."""
    total_params = 0
    changed_params = 0
    sq_diff_sum = 0.0
    layer_diffs = {}

    base_params = dict(base_model.named_parameters())
    for name, p_edit in edited_model.named_parameters():
        p_base = base_params[name]
        diff = (p_edit.detach().cpu().float() - p_base.detach().cpu().float())
        n_el = diff.numel()
        n_changed = int((diff.abs() > 1e-8).sum().item())
        l2 = float(torch.linalg.vector_norm(diff).item())
        total_params += n_el
        changed_params += n_changed
        sq_diff_sum += l2 ** 2
        if n_changed > 0:
            layer_diffs[name] = {
                "changed_elements": n_changed,
                "total_elements": n_el,
                "fraction_changed": n_changed / n_el,
                "l2_norm": l2,
            }

    return {
        "total_parameters": total_params,
        "changed_parameters": changed_params,
        "changed_fraction": changed_params / max(total_params, 1),
        "global_l2_diff": float(np.sqrt(sq_diff_sum)),
        "modified_tensors": layer_diffs,
    }
