# BirdCLEF+ 2026 — Meta-agent report analysis

Generated: 2026-05-19 04:56:36  
LLM used by all agents: `qwen2.5-coder_7b`

## 1. Global best model

- **Source agent:** `meta`
- **Experiment id:** `6`
- **Config:** loss=`plain_bce` aug=`none` lr=`0.001` optimizer=`adamw` schedule=`exp_decay` geo=`2000`
- **macro_auc (unweighted):** 0.7432
- **macro_auc (weighted by taxon):** 0.7595
- **aves_auc:** 0.7232
- **Weights:** `experiments_meta_qwen2.5-coder_7b\exp_006_plain_bce_none_lr0.001_adamw_exp_decay_geo2000\model.keras`

## 2. Cross-agent comparison

Best experiment per agent, ranked by weighted_macro_auc:

| Agent | Exp id | Loss | Aug | LR | Optim | Sched | Geo | macro(unw) | macro(w) | aves |
|---|---|---|---|---|---|---|---|---|---|---|
| meta | 6 | plain_bce | none | 0.001 | adamw | exp_decay | 2000 | 0.7432 | 0.7595 | 0.7232 |
| regular | 9 | plain_bce | specaugment | 0.001 | adamw | exp_decay | none | 0.7308 | 0.7509 | 0.7062 |
| eda | 7 | plain_bce | none | 0.001 | adam | exp_decay | 2000 | 0.7340 | 0.7426 | 0.7236 |
| creative | 2 | weighted BCE with clipped positive weights | moderate SpecAugment | 0.0002 | AdamW weight_decay 1e-4 | constant LR with ReduceLROnPlateau | none | 0.6904 | 0.7131 | 0.6623 |

## 3. Per-taxon AUC breakdown (global best model)

Each taxonomic class's per-species AUC distribution on the held-out validation set:

| Taxon | N species evaluated | Mean | Median | p10 | p90 | Min | Max |
|---|---|---|---|---|---|---|---|
| Mammalia | 2 | 0.911 | 0.911 | 0.874 | 0.949 | 0.864 | 0.958 |
| Amphibia | 11 | 0.860 | 0.872 | 0.742 | 0.976 | 0.716 | 1.000 |
| Insecta | 8 | 0.801 | 0.845 | 0.664 | 0.922 | 0.543 | 0.923 |
| Aves | 104 | 0.723 | 0.744 | 0.512 | 0.947 | 0.120 | 0.993 |

> Reading guide: a *high mean with a low p10* signals uneven coverage within the taxon — most species are well-modeled but some are quietly failing. *Low max* means the model is uniformly bad across that taxon.

## 4. Top-10 confidently-wrong species

**False positives** (model shouts the species when it isn't there):

| Rank | Species | Taxon | Mean pred prob (when absent) | N negatives | N positives in val |
|---|---|---|---|---|---|
| 1 | `517063` | Amphibia | 0.177 | 219 | 50 |
| 2 | `22967` | Amphibia | 0.157 | 245 | 24 |
| 3 | `22973` | Amphibia | 0.131 | 227 | 42 |
| 4 | `555146` | Amphibia | 0.106 | 257 | 12 |
| 5 | `65380` | Amphibia | 0.094 | 225 | 44 |
| 6 | `24279` | Amphibia | 0.083 | 269 | 0 |
| 7 | `47158son25` | Insecta | 0.082 | 269 | 0 |
| 8 | `66971` | Amphibia | 0.073 | 257 | 12 |
| 9 | `whtdov` | Aves | 0.056 | 256 | 13 |
| 10 | `chacha1` | Aves | 0.053 | 264 | 5 |

**False negatives** (model misses the species when it is present):

| Rank | Species | Taxon | Mean pred prob (when present) | N positives in val |
|---|---|---|---|---|
| 1 | `47158son09` | Insecta | 0.000 | 12 |
| 2 | `47158son12` | Insecta | 0.000 | 10 |
| 3 | `hyamac1` | Aves | 0.001 | 20 |
| 4 | `bunibi1` | Aves | 0.001 | 5 |
| 5 | `sibtan2` | Aves | 0.002 | 3 |
| 6 | `bufpar` | Aves | 0.003 | 26 |
| 7 | `thlwre1` | Aves | 0.003 | 5 |
| 8 | `gretho2` | Aves | 0.004 | 3 |
| 9 | `47158son01` | Insecta | 0.007 | 22 |
| 10 | `grasal3` | Aves | 0.008 | 3 |

> Note: Both lists are restricted to classes with enough validation support to be meaningful (≥10 negatives or ≥3 positives). Mean predicted probability is computed on the validation set's raw sigmoid outputs.

## 5. Meta-agent reasoning trace

The LLM's REASON field for each meta iteration:

- **Exp 1:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 2:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 3:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 4:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 5:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 6:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 7:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 8:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 9:** (fallback: LLM unavailable or unhelpful — first untried)
- **Exp 10:** (fallback: LLM unavailable or unhelpful — first untried)

---
*Combined prior-experiments table available at `experiments_meta_qwen2.5-coder_7b\meta_prior_experiments.csv`. Meta progress plot at `experiments_meta_qwen2.5-coder_7b\progress.png`.*