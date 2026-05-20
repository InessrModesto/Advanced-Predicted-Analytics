# BirdCLEF+ 2026 — Meta-agent report analysis

Generated: 2026-05-20 02:11:26  
LLM used by all agents: `gemma4`

## 1. Global best model

- **Source agent:** `meta`
- **Experiment id:** `9`
- **Config:** loss=`plain_bce` aug=`none` lr=`0.001` optimizer=`adam` schedule=`exp_decay` geo=`2000`
- **macro_auc (unweighted):** 0.7533
- **macro_auc (weighted by taxon):** 0.7718
- **aves_auc:** 0.7310
- **Weights:** `experiments_meta_gemma4\exp_009_plain_bce_none_lr0.001_adam_exp_decay_geo2000\model.keras`

## 2. Cross-agent comparison

Best experiment per agent, ranked by weighted_macro_auc:

| Agent | Exp id | Loss | Aug | LR | Optim | Sched | Geo | macro(unw) | macro(w) | aves |
|---|---|---|---|---|---|---|---|---|---|---|
| meta | 9 | plain_bce | none | 0.001 | adam | exp_decay | 2000 | 0.7533 | 0.7718 | 0.7310 |
| regular | 8 | weighted_bce | specaugment | 0.0005 | adamw | constant | none | 0.7348 | 0.7529 | 0.7128 |
| eda | 3 | weighted_bce | background_mix | 0.001 | adamw | cosine_decay | 1000 | 0.7039 | 0.7224 | 0.6812 |
| creative | 2 | weighted BCE with clipped positive weights | moderate SpecAugment | 0.0002 | AdamW weight_decay 1e-4 | constant LR with ReduceLROnPlateau | none | 0.6784 | 0.7005 | 0.6509 |

## 3. Per-taxon AUC breakdown (global best model)

Each taxonomic class's per-species AUC distribution on the held-out validation set:

| Taxon | N species evaluated | Mean | Median | p10 | p90 | Min | Max |
|---|---|---|---|---|---|---|---|
| Mammalia | 2 | 0.883 | 0.883 | 0.874 | 0.892 | 0.872 | 0.894 |
| Insecta | 8 | 0.877 | 0.914 | 0.718 | 0.983 | 0.681 | 0.993 |
| Amphibia | 11 | 0.851 | 0.867 | 0.681 | 1.000 | 0.652 | 1.000 |
| Aves | 104 | 0.731 | 0.758 | 0.446 | 0.974 | 0.206 | 1.000 |

> Reading guide: a *high mean with a low p10* signals uneven coverage within the taxon — most species are well-modeled but some are quietly failing. *Low max* means the model is uniformly bad across that taxon.

## 4. Top-10 confidently-wrong species

**False positives** (model shouts the species when it isn't there):

| Rank | Species | Taxon | Mean pred prob (when absent) | N negatives | N positives in val |
|---|---|---|---|---|---|
| 1 | `517063` | Amphibia | 0.142 | 219 | 50 |
| 2 | `22967` | Amphibia | 0.113 | 245 | 24 |
| 3 | `22973` | Amphibia | 0.093 | 227 | 42 |
| 4 | `24279` | Amphibia | 0.066 | 269 | 0 |
| 5 | `whtdov` | Aves | 0.064 | 256 | 13 |
| 6 | `555146` | Amphibia | 0.063 | 257 | 12 |
| 7 | `chacha1` | Aves | 0.058 | 264 | 5 |
| 8 | `1491113` | Amphibia | 0.054 | 245 | 24 |
| 9 | `47158son25` | Insecta | 0.053 | 269 | 0 |
| 10 | `undtin1` | Aves | 0.051 | 268 | 1 |

**False negatives** (model misses the species when it is present):

| Rank | Species | Taxon | Mean pred prob (when present) | N positives in val |
|---|---|---|---|---|
| 1 | `47158son09` | Insecta | 0.000 | 12 |
| 2 | `47158son12` | Insecta | 0.000 | 10 |
| 3 | `hyamac1` | Aves | 0.001 | 20 |
| 4 | `bunibi1` | Aves | 0.002 | 5 |
| 5 | `sibtan2` | Aves | 0.002 | 3 |
| 6 | `bufpar` | Aves | 0.003 | 26 |
| 7 | `326272` | Amphibia | 0.006 | 22 |
| 8 | `47158son01` | Insecta | 0.007 | 22 |
| 9 | `thlwre1` | Aves | 0.009 | 5 |
| 10 | `gretho2` | Aves | 0.010 | 3 |

> Note: Both lists are restricted to classes with enough validation support to be meaningful (≥10 negatives or ≥3 positives). Mean predicted probability is computed on the validation set's raw sigmoid outputs.

## 5. Meta-agent reasoning trace

The LLM's REASON field for each meta iteration:

- **Exp 1:** The Global Winner utilized the optimal combination of `weighted_bce` and `specaugment`. I am maintaining these strong components while systematically exploring improvements. I am switching the schedule from the GW's `constant` to `cosine_decay`, which is the most frequent schedule in the top 10 and shows strong performance. Furthermore, I am reducing the learning rate from 5e-4 to 1e-4 for a gentler fine-tuning step. Finally, I am introducing a moderate geographic weight (2000km) to test the impact of the geographical weighting, which has not been explored in the top-performing models.
- **Exp 2:** The global winner and meta-experiment both strongly suggest sticking with 'weighted_bce' and 'specaugment' as the best performing combination for loss and augmentation. We maintain 'adamw' as the optimizer, which is the most frequent in the top 10 runs. To explore a new high-potential area, we are increasing the initial learning rate to 1e-3, which is the standard rate and has not been the primary focus of the top runs. We are testing 'exp_decay' for the schedule, as it is less represented in the top results, and setting the geographic weight to 5000km to test a mild, non-zero weighting that hasn't been explored in the top configurations.
- **Exp 3:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 4:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 5:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 6:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 7:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 8:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 9:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 10:** (fallback: LLM unavailable or unhelpful — first untried)

---
*Combined prior-experiments table available at `experiments_meta_gemma4\meta_prior_experiments.csv`. Meta progress plot at `experiments_meta_gemma4\progress.png`.*