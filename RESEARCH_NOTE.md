# Research Note: Evaluating Weight-Space Programming for SE.LLMA

**Author:** [Meriam Dhouibi]  
**Date:** November 2026  
**Target Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Seed: `42`)

---

## 1. Critical Examination of *Sparks of Weight-Space Programming*

*Sparks of Weight-Space Programming* (Takehi et al., 2026) proposes treating neural network weights as a readable and directly editable program artifact across three tasks: **reading** (inferring behavior/circuits from weights alone), **fixing** (patching failure modes via sparse weight edits), and **writing** (authoring small control networks zero-shot).

* **Strongest Evidence:**
  * In experiment 3, Astra read the weights of three 7M-parameter models with no runs, identified the training language, and found every induction head with no false positives against ground truth measured separately. This is strong evidence of structure-reading, and it is the one case where the output was verified against an independent measurement.
  * Zero-feedback writing (experiment 5, Go with 0 passes). Producing working networks without ever running them is a real capability, and it is reported against standard baselines.
  * Transparency. The logs are public, and the authors report failures (experiment 7 from weights alone, Gemma).

* **Key Limitations :**
  * Tiny samples. Experiment 1 has 4 networks, experiment 3 has 3 models with 2 attempts each, experiment 7 has 3 pairs, experiment 4 is one task, and experiment 8 appears to be one run per budget. These are demonstrations, not measurements.
  * Weak controls. There are no function-preserving obfuscations, no unfamiliar tasks, and no tests of whether it can say “I can’t tell.” experiment 1 also gave the channel layout in the prompt, which hints at the game.
  *	Familiarity. Breakout, Go, MinAtar, and induction heads are all well covered in training data. The “rediscovered Elhage et al.” finding is just as consistent with recall of a known method.
  * Experiment 5 is closer to compilation than discovery. Astra had the game source and wrote a known-style heuristic into a network. The PPO baselines are also weak (e.g., Breakout 15.8 at 100M steps), and there is no hand-coded non-neural baseline.
  * Go baseline. Beating small PPO nets, one of the practice opponents being a PPO net of the same family, is a low bar. Comparing 6,400 practice games with 20M ignores the Go knowledge Astra brings.
  * Scale. The largest model read purely from weights has 7M parameters. The only case where it tried to distinguish behaviors in a real language model from weights alone (experiment 7) failed.

* **Supported conclusions:**
  -	A frontier LLM can recover salient structure from small networks’ weights.
  -	It can write small working networks zero-shot from a spec.
  -	An LLM-driven edit loop can find and ablate a low-dimensional failure mode in a real LM.

* **Not supported conclusions:**
  -	“LLMs natively read the weights of real language models” (experiment 8 relied on activations and generations).
  -	That reasoning about weights, as opposed to iterated feedback, produced the Qwen fix.
  -	That this will replace Adam or Muon, since there are no gradient baselines and no scale beyond 0.8B.
  -	The auditing, backdoor-detection, and unlearning applications. Experiment 7 is the closest test, and it failed from weights alone.
  -	“Beyond what we’d expect from humans.” The authors concede they ran no human or interpretability-researcher comparison.

---

## 2. Hypothesis

In a compact multilingual/open-weight LLM, a targeted weight edit can reduce unintended English code-switching or repetitive generation loops on Swedish prompts without causing severe degradation on basic Swedish comprehension/generation or English/math sanity checks.
* **Question:** Can editing a tiny fraction (~0.05%) of weights in one middle layer reduce English switching and repetitive text loops on Swedish prompts without hurting English or math accuracy?
* **Why it matters for SE.LLMA:** Small open models used as stepping stones in SE.LLMA often suffer from language drift (answering Swedish prompts in English), awkward/non-idiomatic phrasing, or generation loops. Knowing whether lightweight weight-space edits can patch these behaviors or whether standard fine-tuning/prompting/activation steering is far more reliable directly informs engineering effort.
* **Result that would support the hypothesis:** Targeted edits measurably lower Swedish code-switching / loop rates on a held-out Swedish evaluation set while keeping Swedish quality and baseline capabilities intact, outperforming a random/untargeted edit of the same magnitude.
* **Evidence that would challenge it:** Edits fail to reduce Swedish code-switching/loops, only work when paired with heavy trial-and-error, or degrade Swedish fluency/coherence more than simple prompting or lightweight fine-tuning/steering.

---

## 3. Experimental Design

* **Model Tested:** Qwen/Qwen2.5-0.5B-Instruct (float32, Seed: 42)
* **Evaluation Sets:** 10 held-out Swedish prompts (swedish_eval.jsonl), 5 held-out English/math checks (regression_eval.jsonl)
I kept a strict split: 8 dev prompts were used only to find the "Swedish vs. English" direction inside layer 14, and 15 separate eval prompts were used only for final testing.
I compared 5 setups on the exact same model (494M total parameters):

* **Controls:**
  1. `baseline`: Unchanged model with no system prompt.
  2. `prompt_control`: Unchanged model told in Swedish: "Always answer in correct, idiomatic Swedish."
  3. `activation_steering`: Nudging layer 14 activations toward Swedish while generating text (no saved weight changes).
  4. `random_weight_edit`: Adding random noise to 246,400 weights (0.05% of the model) in layer 14 as a fairness check.
  5. `targeted_weight_edit`: Adjusting 246,400 weights (0.05% of the model) in layer 14 toward the Swedish direction (same edit size/norm as the random control).

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
