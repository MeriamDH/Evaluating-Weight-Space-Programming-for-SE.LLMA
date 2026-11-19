# Weight-Space Programming for SE.LLMA
A compact, reproducible investigation evaluating whether the ideas in *Sparks of Weight-Space Programming* (Takehi et al., 2026) are practically useful for **SE.LLMA**'s goal of building strong, transparent Swedish language models.

## Research Question & Hypothesis

**Hypothesis:** On a small open-weight multilingual model (`Qwen/Qwen2.5-0.5B-Instruct`), a sparse, targeted weight-space edit (`<= 0.05%` of parameters) derived from Swedish vs. English contrastive representations on a small development set can reduce English code-switching and repetitive generation loops on held-out Swedish prompts without degrading general English/math capabilities—outperforming a norm-matched random weight edit, though remaining less reliable than simple prompting or inference-time activation steering.

## Repository Structure

```text
sellma-weight-space-eval/
├── configs/
│   └── experiment.yaml        # Model revision, seeds, decoding settings, edit budgets
├── data/
│   ├── swedish_dev.jsonl      # Diagnostic/dev set (used ONLY to inspect & compute edits)
│   ├── swedish_eval.jsonl     # Held-out Swedish evaluation set (code-switching & loops)
│   └── regression_eval.jsonl  # Held-out regression checks (English QA, arithmetic, Swedish math)
├── code/
│   ├── evaluate.py            # Main evaluation runner across all conditions & controls
│   ├── weight_editor.py       # Static weight inspection, targeted weight edit, random control
│   ├── metrics.py             # Swedish language adherence, lexical drift, n-gram loop metrics
│   └── utils.py               # Deterministic seeding, SHA-256 model hashing, tensor diffing
├── results/                   # Generated metrics JSON, qualitative examples, diff logs
├── RESEARCH_NOTE.md           # Concise 1–2 page research note 
├── Report.pdf                 
├── AI_USAGE.md                # Log of AI tools used and their specific contributions
└── requirements.txt
```

## Quickstart & Reproduction

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the full evaluation suite (CPU or single GPU):**
   ```bash
   python -m src.evaluate --config configs/experiment.yaml
   ```
   *(Optional quick smoke test with 2 examples per split:)*
   ```bash
   python -m src.evaluate --config configs/experiment.yaml --max-examples 2
   ```

3. **Inspect outputs:**
   - Full metrics, SHA-256 weight fingerprints, parameter diff counts, and side-by-side generations are saved to `results/eval_results.json`.

## Experimental Conditions & Controls

| Condition | Description | Role |
| :--- | :--- | :--- |
| `baseline` | Unchanged base model, neutral user prompt | Lower-bound reference |
| `prompt_control` | Unchanged base model + Swedish system prompt | Standard inference baseline |
| `activation_steering` | Inference-time residual stream addition (`alpha * v_sv`) | Representation-level alternative |
| `random_weight_edit` | Norm- and sparsity-matched Gaussian perturbation in same layer | Controls for generic noise effects |
| `targeted_weight_edit` | Sparse (`<=0.05%`) rank-1/masked edit to MLP `down_proj` | Tests localized weight-space patching |
