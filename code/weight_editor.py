import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder layer stack across standard HuggingFace causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture: could not locate transformer layers.")


@torch.no_grad()
def inspect_weight_statistics(model: nn.Module, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Static weight-space inspection (analogous to 'reading' weights without rollouts):
    Identifies outlier weight rows/columns and spectral norms across MLP down-projections
    and attention output projections.
    """
    findings = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        w = param.detach().float()
        row_norms = torch.linalg.vector_norm(w, dim=1)
        mean_norm = float(row_norms.mean().item())
        max_norm = float(row_norms.max().item())
        max_idx = int(torch.argmax(row_norms).item())
        kurtosis_proxy = max_norm / max(mean_norm, 1e-8)
        findings.append({
            "parameter": name,
            "shape": list(w.shape),
            "mean_row_l2": mean_norm,
            "max_row_l2": max_norm,
            "max_row_idx": max_idx,
            "max_to_mean_ratio": kurtosis_proxy,
        })
    findings.sort(key=lambda x: x["max_to_mean_ratio"], reverse=True)
    return findings[:top_k]


@torch.no_grad()
def compute_language_direction_from_dev(
    model: nn.Module,
    tokenizer: Any,
    dev_prompts: List[str],
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract a normalized residual-stream direction separating Swedish-instructed vs.
    default English-biased representations on the development set ONLY.
    """
    layers = _get_transformer_layers(model)
    layer_idx = min(layer_idx, len(layers) - 1)
    captured = []

    def hook_fn(_module, _inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured.append(hidden[:, -1, :].detach().float().cpu())

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    sv_vecs, en_vecs = [], []

    try:
        for prompt in dev_prompts:
            sv_text = f"Svara på svenska: {prompt}"
            en_text = f"Answer in English: {prompt}"

            captured.clear()
            inputs_sv = tokenizer(sv_text, return_tensors="pt").to(device)
            model(**inputs_sv)
            sv_vecs.append(captured[0].squeeze(0))

            captured.clear()
            inputs_en = tokenizer(en_text, return_tensors="pt").to(device)
            model(**inputs_en)
            en_vecs.append(captured[0].squeeze(0))
    finally:
        handle.remove()

    sv_mean = torch.stack(sv_vecs).mean(dim=0)
    en_mean = torch.stack(en_vecs).mean(dim=0)
    direction = sv_mean - en_mean
    norm = torch.linalg.vector_norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    return direction


@torch.no_grad()
def apply_targeted_weight_edit(
    base_model: nn.Module,
    sv_direction: torch.Tensor,
    layer_idx: int,
    target_param_fraction: float = 0.0005,
    edit_scale: float = -0.15,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply a sparse, targeted rank-1 / coordinate-masked edit to the MLP down-projection
    (or output projection) of a chosen transformer block, steering residual outputs toward
    the Swedish direction while modifying <= target_param_fraction of total parameters.
    """
    edited_model = copy.deepcopy(base_model)
    layers = _get_transformer_layers(edited_model)
    layer_idx = min(layer_idx, len(layers) - 1)
    block = layers[layer_idx]

    # Locate MLP down-projection weight matrix [hidden_size, intermediate_size]
    if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
        target_weight = block.mlp.down_proj.weight
        target_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    elif hasattr(block, "mlp") and hasattr(block.mlp, "c_proj"):
        target_weight = block.mlp.c_proj.weight
        target_name = f"transformer.h.{layer_idx}.mlp.c_proj.weight"
    else:
        raise ValueError("Could not locate MLP down-projection in target transformer block.")

    total_model_params = sum(p.numel() for p in edited_model.parameters())
    budget_elements = max(1, int(total_model_params * target_param_fraction))

    w = target_weight.data
    device = w.device
    dtype = w.dtype
    dir_vec = sv_direction.to(device=device, dtype=torch.float32)

    # Identify rows of down_proj most aligned with the English drift direction (-sv_direction)
    # and apply a sparse update along +sv_direction
    col_norms = torch.linalg.vector_norm(w.float(), dim=0)
    top_cols = max(1, min(w.shape[1], budget_elements // max(w.shape[0], 1)))
    selected_col_indices = torch.topk(col_norms, k=top_cols).indices

    delta = torch.zeros_like(w, dtype=torch.float32)
    for col_idx in selected_col_indices:
        # Subtract English-aligned component and reinforce Swedish direction
        col = w[:, col_idx].float()
        proj = torch.dot(col, dir_vec)
        delta[:, col_idx] = (-edit_scale) * (dir_vec * col.norm() - proj * dir_vec)

    # Enforce strict element budget via top-k magnitude mask
    flat_abs = delta.abs().view(-1)
    k = min(budget_elements, flat_abs.numel())
    threshold = torch.topk(flat_abs, k=k).values[-1]
    mask = (delta.abs() >= threshold) & (delta.abs() > 0)
    sparse_delta = delta * mask.float()

    target_weight.data = (w.float() + sparse_delta).to(dtype=dtype)

    metadata = {
        "target_tensor": target_name,
        "layer_idx": layer_idx,
        "budget_elements": budget_elements,
        "actual_modified_elements": int(mask.sum().item()),
        "edit_l2_norm": float(torch.linalg.vector_norm(sparse_delta).item()),
    }
    return edited_model, metadata


@torch.no_grad()
def apply_random_weight_edit(
    base_model: nn.Module,
    layer_idx: int,
    num_elements: int,
    target_l2_norm: float,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Untargeted control: perturb the exact same tensor with the exact same number of modified
    elements and L2 norm, but in random directions/coordinates.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    edited_model = copy.deepcopy(base_model)
    layers = _get_transformer_layers(edited_model)
    layer_idx = min(layer_idx, len(layers) - 1)
    block = layers[layer_idx]

    if hasattr(block, "mlp") and hasattr(block.mlp, "down_proj"):
        target_weight = block.mlp.down_proj.weight
        target_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    else:
        target_weight = block.mlp.c_proj.weight
        target_name = f"transformer.h.{layer_idx}.mlp.c_proj.weight"

    w = target_weight.data
    flat_size = w.numel()
    k = min(num_elements, flat_size)

    perm = torch.randperm(flat_size, generator=gen)[:k]
    rand_vals = torch.randn(k, generator=gen)
    rand_norm = torch.linalg.vector_norm(rand_vals)
    if rand_norm > 1e-8 and target_l2_norm > 0:
        rand_vals = rand_vals * (target_l2_norm / rand_norm)

    flat_delta = torch.zeros(flat_size, dtype=torch.float32)
    flat_delta[perm] = rand_vals
    sparse_delta = flat_delta.view_as(w).to(device=w.device)

    target_weight.data = (w.float() + sparse_delta).to(dtype=w.dtype)

    metadata = {
        "target_tensor": target_name,
        "layer_idx": layer_idx,
        "actual_modified_elements": k,
        "edit_l2_norm": float(torch.linalg.vector_norm(sparse_delta).item()),
    }
    return edited_model, metadata
