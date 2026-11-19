# Research Note: Evaluating Weight-Space Programming for SE.LLMA

**Author:** [Your Name]  
**Date:** November 2026  
**Target Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Seed: `42`)

---

## 1. Critical Examination of *Sparks of Weight-Space Programming*

*Sparks of Weight-Space Programming* (Takehi et al., 2026) proposes treating neural network weights as a readable and directly editable program artifact across three tasks: **reading** (inferring behavior/circuits from weights alone), **fixing** (patching failure modes via sparse weight edits), and **writing** (authoring small control networks zero-shot).

* **Strongest Evidence:**
  * *Circuit/domain reading in small transformers:* Identifying language/code specializations and induction heads from static tensor inspection demonstrates that low-level structural motifs leave detectable footprints in weight matrices.
  * *Surgical loop/attractor patching:* Modifying a tiny fraction (`0.016%`) of weights in a compact model (`Qwen3.5-0.8B`) substantially reduced token-limit runaway loops (`93.0% -> 33.1%`) and unlocked downstream math and instruction-following scores (`MATH-500: 7.9% -> 48.1%`).
* **Key Limitations & Unjustified Conclusions:**
  * *Failure removal vs. capability creation:* The headline math gains largely reflect removing a pathological token-runaway attractor so the model can finish answering, rather than instilling new mathematical reasoning directly into weights.
  * *Polysemanticity & task-dependent brittleness:* The paper itself documents failures on behavioral traits like sycophancy, indicating that high-level linguistic or cultural behaviors are heavily superposed across layers rather than isolated in clean "bugs."
  * *Reasoning directly vs. feedback-guided optimization:* True static weight reasoning (inspecting tensors without forward passes) is fundamentally distinct from iterative, feedback-guided edit search. Conflating empirical error diagnosis on validation rollouts with "reading the mind of the network" risks overstating how interpretable raw weights are at scale.

---

## 2. Hypothesis

* **Narrow Question:** Can a sparse (`<= 0.05%` parameters), targeted weight-space edit to an MLP down-projection layer reduce English code-switching and degenerate repetition loops on Swedish prompts in a compact open-weight LLM without degrading English/math regression performance?
* **Why It Matters for SE.LLMA:** SE.LLMA develops Swedish language infrastructure iteratively starting from smaller open-weight models before scaling. Small multilingual checkpoints frequently suffer from English drift on technical Swedish queries; knowing whether direct weight patching offers a reliable shortcut—or if prompting/steering/fine-tuning remains superior—directly informs engineering priorities.
* **Falsification Criteria:**
  * *Supported if:* `targeted_weight_edit` increases Swedish language adherence and lowers code-switching/loop rates on `swedish_eval.jsonl` relative to both `baseline` and norm-matched `random_weight_edit`, while preserving `regression_eval.jsonl` accuracy within `5%` of baseline.
  * *Challenged if:* Gains vanish on held-out prompts, match random noise, cause collateral fluency/regression damage, or lag significantly behind a simple Swedish system prompt (`prompt_control`).

---

## 3. Experimental Design

* **Separation of Splits:** `data/swedish_dev.jsonl` (`n=8`) was used exclusively to diagnose failure modes and extract a normalized contrastive residual direction (`v_sv = mean(h_sv) - mean(h_en)` at layer 14). All reported metrics come from untouched `data/swedish_eval.jsonl` (`n=10`) and `data/regression_eval.jsonl` (`n=5`).
* **Controls:**
  1. `baseline`: Unmodified weights (`SHA-256` logged), greedy decoding (`temperature=0.0`).
  2. `prompt_control`: Explicit Swedish system instruction on unmodified weights.
  3. `activation_steering`: Adding `alpha * v_sv` to layer 14 hidden states at inference time.
  4. `random_weight_edit`: Perturbing the exact same tensor (`model.layers.14.mlp.down_proj.weight`), element count, and L2 norm with random Gaussian noise.
  5. `targeted_weight_edit`: Sparse projection update along `v_sv` on top-norm MLP output columns (`<= 0.05%` of model parameters).

---

## 4. Results & Analysis

*(Populate this table after running `python -m src.evaluate --config configs/experiment.yaml`)*

| Condition | Swedish Adherence (↑) | Code-Switch Rate (↓) | Loop Rate (↓) | Distinct 4-gram (↑) | Regression Keyword Acc (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `prompt_control` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `activation_steering` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `random_weight_edit` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `targeted_weight_edit` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

### Qualitative Examples (Held-Out Set)
* **Example `eval_04` (Code-switch trap — Open source in public sector):**
  * *Baseline:* `[Paste output]`
  * *Targeted Weight Edit:* `[Paste output]`
  * *Prompt Control:* `[Paste output]`

### Limitations of Conclusions
* Sample sizes (`n=10` Swedish eval, `n=5` regression) are scoped for a 4-hour CPU/single-GPU budget and detect large effect sizes rather than subtle stylistic shifts.
* Single-layer rank-1 weight edits test localized linear separability; multi-layer iterative edits might behave differently at larger parameter budgets.

---

## 5. Recommendation for SE.LLMA

* **Should SE.LLMA pursue Weight-Space Programming right now?** **No as a primary training/adaptation method; Yes selectively as a lightweight diagnostic and post-hoc patching probe.** While sparse weight edits are intriguing for curing degenerate token-runaway loops in small checkpoints, Swedish idiomatic fluency, cultural grounding, and terminology require distributed representations that cannot be reliably authored via narrow weight surgery. SE.LLMA should prioritize high-quality Swedish data curation, LoRA/full fine-tuning, and standardized Swedish evaluation harnesses.
* **Most Informative Next Experiment:** Run a **scaling & compute-matched comparison** across `0.5B`, `1.5B`, and `7B` open checkpoints comparing (a) sparse weight-space edits, (b) inference-time activation steering, and (c) rank-1 LoRA trained on the exact same `dev` set, measuring whether localized weight edits degrade sharply as features become more distributed in larger models.
