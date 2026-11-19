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

## 3. Experiment and Results

* **Model Tested:** Qwen/Qwen2.5-0.5B-Instruct (float32, Seed: 42)
* **Evaluation Sets:** 10 held-out Swedish prompts (swedish_eval.jsonl), 5 held-out English/math checks (regression_eval.jsonl)
I kept a strict split: 8 dev prompts were used only to find the "Swedish vs. English" direction inside layer 14, and 15 separate eval prompts were used only for final testing.
* I compared 5 setups on the exact same model (494M total parameters):
  1. `baseline`: Unchanged model with no system prompt.
  2. `prompt_control`: Unchanged model told in Swedish: "Always answer in correct, idiomatic Swedish."
  3. `activation_steering`: Nudging layer 14 activations toward Swedish while generating text (no saved weight changes).
  4. `random_weight_edit`: Adding random noise to 246,400 weights (0.05% of the model) in layer 14 as a fairness check.
  5. `targeted_weight_edit`: Adjusting 246,400 weights (0.05% of the model) in layer 14 toward the Swedish direction (same edit size/norm as the random control).


| Condition | Swedish Adherence (↑) | Code-Switch Rate (↓) | Loop Rate (↓) | Distinct 4-gram (↑) | Regression Keyword Acc (↑) |English/Math Acc (↑)|
| :--- | :---: | :---: | :---: | :---: | :---: |:---:|
| `baseline` | 0.90	| 20% (2/10)|	40% (4/10)|	0.76	|100% (10/10)|	100% (5/5)|
| `prompt_control` | **1.00**|	**0% (0/10)**	|**20% (2/10)**	|**0.78**	|**60% (6/10)**	|**100% (5/5)**|
| `activation_steering` | 0.87|	10% (1/10)	|30% (3/10)	|0.75	|80% (8/10)	|100% (5/5)|
| `random_weight_edit` | 0.90|	10% (1/10)	|40% (4/10)	|0.71	|100% (10/10)|	100% (5/5)|
| `targeted_weight_edit` | 0.90|	10% (1/10)	|40% (4/10)	|0.76	|90% (9/10)	|100% (5/5)|


* Both baseline and targeted_weight_edit got stuck in repetitive loops on 4 out of 10 Swedish prompts (40%), and their Swedish confidence stayed identical (0.90).
* Compared to random_weight_edit, the targeted edit preserved better word variety (0.76 vs. 0.71) and stopped slightly earlier on 1 prompt (90% vs. 100% hitting max length). However, a simple Swedish system prompt (prompt_control) beat everything: 0% language drift, half as many loops (20%), and cleaner stops (60%).
* Even when the model wrote in Swedish without looping, the 0.5B model made clear factual and grammar errors such as drifting into Norwegian on "sustainable development", miscalculating 25% of 240 kr as 6 kr, and listing Malmö and Gothenburg as Nordic capitals. A single weight patch cannot fix missing knowledge.
* All 5 setups scored 100% on the 5 English and math check questions.

* A sparse weight edit on Qwen/Qwen2.5-0.5B-Instruct (0.0499% of parameters) did not reduce loops below baseline (0.40) or boost Swedish probability (0.900), only slightly cutting code-switching (0.20 to 0.10) while preserving 4-gram diversity better than random noise (0.759 vs. 0.712) and keeping regression keyword accuracy at 1.00—meaning a simple Swedish system prompt remains far more effective with zero weight changes.

** Limitation:** This study tested a small dataset (10 Swedish + 5 regression prompts), one small model (0.5B), and a single edit layer (layer 14).


 of Conclusions
* Sample sizes (`n=10` Swedish eval, `n=5` regression) are scoped for a 4-hour CPU/single-GPU budget and detect large effect sizes rather than subtle stylistic shifts.
* Single-layer rank-1 weight edits test localized linear separability; multi-layer iterative edits might behave differently at larger parameter budgets.

---

## 4. Recommendation 
On a small open model, editing weights directly did not fix repetition loops and performed worse than simply adding a Swedish system prompt. SE.LLMA should focus on high-quality Swedish training data, LoRA/full fine-tuning, and solid Swedish benchmarks, using weight inspection only as a quick debugging tool.
**Next Experiment:** Compare a targeted weight edit against a small LoRA fine-tune on the exact same Swedish dataset across two model sizes (0.5B and 1.5B or 3B), grading outputs for both grammar and factual accuracy.

---

## 5.	References
1.	Takehi, R., et al. (2026). Sparks of Weight-Space Programming. https://rikiyat.github.io/blog/weight-space-programming
2.	Qwen model card, https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

