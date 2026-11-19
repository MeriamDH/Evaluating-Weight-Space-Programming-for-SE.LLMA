import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.metrics import aggregate_metrics, score_response
from code.utils import (
    compute_state_dict_hash,
    diff_models,
    load_config,
    load_jsonl,
    resolve_device,
    save_json,
    set_seed,
)
from code.weight_editor import (
    _get_transformer_layers,
    apply_random_weight_edit,
    apply_targeted_weight_edit,
    compute_language_direction_from_dev,
    inspect_weight_statistics,
)


def format_prompt(tokenizer: Any, prompt: str, system_prompt: Optional[str] = None) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\nAnvändare: {prompt}\nAssistent:"
    return f"Användare: {prompt}\nAssistent:"


@torch.no_grad()
def run_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    device: torch.device,
    gen_cfg: Dict[str, Any],
    system_prompt: Optional[str] = None,
    steering_vector: Optional[torch.Tensor] = None,
    steering_layer: Optional[int] = None,
    steering_alpha: float = 0.0,
) -> List[Dict[str, Any]]:
    model.eval()
    hook_handle = None

    if steering_vector is not None and steering_layer is not None and steering_alpha != 0.0:
        layers = _get_transformer_layers(model)
        idx = min(steering_layer, len(layers) - 1)
        vec = steering_vector.to(device=device)

        def steer_hook(_module, _inp, out):
            if isinstance(out, tuple):
                hidden = out[0] + (steering_alpha * vec).to(out[0].dtype)
                return (hidden,) + out[1:]
            return out + (steering_alpha * vec).to(out.dtype)

        hook_handle = layers[idx].register_forward_hook(steer_hook)

    results = []
    try:
        for item in dataset:
            formatted = format_prompt(tokenizer, item["prompt"], system_prompt=system_prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": gen_cfg.get("max_new_tokens", 160),
                "do_sample": gen_cfg.get("do_sample", False),
                "repetition_penalty": gen_cfg.get("repetition_penalty", 1.0),
                "pad_token_id": tokenizer.eos_token_id,
            }
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = gen_cfg.get("temperature", 0.7)

            output_ids = model.generate(**inputs, **gen_kwargs)
            new_tokens = output_ids[0][input_len:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            metrics = score_response(
                response_text=response_text,
                expected_lang=item.get("expected_lang", "sv"),
                reference_keywords=item.get("reference_keywords"),
                max_new_tokens=gen_kwargs["max_new_tokens"],
                generated_token_count=len(new_tokens),
            )
            results.append({
                "id": item["id"],
                "category": item.get("category", "unknown"),
                "prompt": item["prompt"],
                "response": response_text,
                "generated_tokens": int(len(new_tokens)),
                "metrics": metrics,
            })
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Weight-Space Programming on Swedish LLM Tasks")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default=None, help="Optional override for HuggingFace model ID")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit examples per split for quick dry-runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(cfg["experiment"]["device"])

    model_name = args.model or cfg["model"]["name"]
    revision = cfg["model"].get("revision", "main")
    dtype_str = cfg["model"].get("torch_dtype", "float32")
    torch_dtype = getattr(torch, dtype_str)

    print(f"[INFO] Loading model={model_name} (revision={revision}) on device={device} dtype={dtype_str}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch_dtype,
    ).to(device)

    base_hash = compute_state_dict_hash(base_model)
    print(f"[INFO] Base model SHA-256: {base_hash[:16]}...")

    # Load strictly separated splits
    dev_data = load_jsonl(cfg["datasets"]["dev_path"])
    eval_sv_data = load_jsonl(cfg["datasets"]["eval_swedish_path"])
    eval_reg_data = load_jsonl(cfg["datasets"]["eval_regression_path"])

    if args.max_examples:
        dev_data = dev_data[: args.max_examples]
        eval_sv_data = eval_sv_data[: args.max_examples]
        eval_reg_data = eval_reg_data[: args.max_examples]

    # Step 1: Static weight inspection ("reading" weights without rollouts)
    weight_outliers = inspect_weight_statistics(base_model, top_k=10)

    # Step 2: Derive Swedish direction exclusively from dev split
    target_layer = 14
    for c in cfg["conditions"]:
        if c["name"] == "activation_steering":
            target_layer = c.get("layer_idx", 14)

    dev_prompts = [d["prompt"] for d in dev_data]
    sv_direction = compute_language_direction_from_dev(
        model=base_model,
        tokenizer=tokenizer,
        dev_prompts=dev_prompts,
        layer_idx=target_layer,
        device=device,
    )

    # Build targeted and random-control edited models
    targeted_cfg = next(c for c in cfg["conditions"] if c["name"] == "targeted_weight_edit")
    targeted_model, targeted_meta = apply_targeted_weight_edit(
        base_model=base_model,
        sv_direction=sv_direction,
        layer_idx=target_layer,
        target_param_fraction=targeted_cfg.get("target_param_fraction", 0.0005),
        edit_scale=targeted_cfg.get("edit_scale", -0.15),
    )
    targeted_diff = diff_models(base_model, targeted_model)
    targeted_hash = compute_state_dict_hash(targeted_model)

    random_model, random_meta = apply_random_weight_edit(
        base_model=base_model,
        layer_idx=target_layer,
        num_elements=targeted_meta["actual_modified_elements"],
        target_l2_norm=targeted_meta["edit_l2_norm"],
        seed=seed,
    )
    random_diff = diff_models(base_model, random_model)
    random_hash = compute_state_dict_hash(random_model)

    # Step 3: Evaluate all conditions on held-out Swedish eval & regression splits
    all_condition_outputs = {}
    summary_table = {}

    for cond in cfg["conditions"]:
        cname = cond["name"]
        print(f"[RUN] Condition: {cname}")

        active_model = base_model
        sys_prompt = None
        steer_vec = None
        steer_layer = None
        steer_alpha = 0.0

        if cname == "prompt_control":
            sys_prompt = cond.get("system_prompt")
        elif cname == "activation_steering":
            steer_vec = sv_direction
            steer_layer = cond.get("layer_idx", target_layer)
            steer_alpha = cond.get("alpha", 1.5)
        elif cname == "targeted_weight_edit":
            active_model = targeted_model
        elif cname == "random_weight_edit":
            active_model = random_model

        sv_records = run_generation(
            model=active_model,
            tokenizer=tokenizer,
            dataset=eval_sv_data,
            device=device,
            gen_cfg=cfg["generation"],
            system_prompt=sys_prompt,
            steering_vector=steer_vec,
            steering_layer=steer_layer,
            steering_alpha=steer_alpha,
        )
        reg_records = run_generation(
            model=active_model,
            tokenizer=tokenizer,
            dataset=eval_reg_data,
            device=device,
            gen_cfg=cfg["generation"],
            system_prompt=sys_prompt,
            steering_vector=steer_vec,
            steering_layer=steer_layer,
            steering_alpha=steer_alpha,
        )

        sv_agg = aggregate_metrics(sv_records)
        reg_agg = aggregate_metrics(reg_records)

        summary_table[cname] = {
            "swedish_eval": sv_agg,
            "regression_eval": reg_agg,
        }
        all_condition_outputs[cname] = {
            "swedish_eval": sv_records,
            "regression_eval": reg_records,
        }

    report_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "provenance": {
            "model_name": model_name,
            "revision": revision,
            "seed": seed,
            "base_model_sha256": base_hash,
            "targeted_model_sha256": targeted_hash,
            "random_model_sha256": random_hash,
        },
        "static_weight_inspection_top10": weight_outliers,
        "weight_edits": {
            "targeted": {"metadata": targeted_meta, "diff": targeted_diff},
            "random_control": {"metadata": random_meta, "diff": random_diff},
        },
        "summary_metrics": summary_table,
        "detailed_predictions": all_condition_outputs,
    }

    results_dir = Path(cfg["output"]["results_dir"])
    out_file = results_dir / "eval_results.json"
    save_json(report_payload, str(out_file))
    print(f"[DONE] Saved full evaluation report to {out_file}")


if __name__ == "__main__":
    main()
