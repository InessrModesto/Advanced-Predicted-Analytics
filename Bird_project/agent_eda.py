"""
BirdCLEF+ 2026 — EDA-aware Agent
=================================

The second of the three planned agents:

    EDA → Baselines (compare_models.py)
        → Agent Regular       (agent_regular.py,  5D discrete search)
        → Agent EDA-aware     (THIS FILE,         5D discrete + 1D continuous)
        → Agent Creative      (built later)       wider/freer search

What changes vs Regular
-----------------------
1. **New axis: geographic sample weighting.** The test set is recorded
   entirely in the Pantanal wetlands (-58 to -55 lon, -20 to -16 lat);
   only ~2% of focal training data falls in that box. The EDA agent can
   weight each training sample by a Gaussian decay in geographic
   distance from the Pantanal center:

       sample_weight(lat, lon) = exp(-distance_km / scale_km)

   `scale_km` is a CONTINUOUS choice in [200, 10000]. Larger = milder
   weighting. The agent is told anchor values (500 / 2000 / 5000) but
   can pick anything in range. Soundscape rows (which already come from
   the Pantanal) and rows with missing (lat, lon) get weight = 1.0.

   Selecting `none` keeps every sample equally weighted.

2. **EDA-informed prompt.** The system prompt includes the dataset
   findings — Pantanal domain shift, insect chorus at 4 kHz, 162 birds
   vs 25 non-birds, monophonic focal vs multi-species soundscape — and
   ties each menu option to the finding that motivates it. The LLM is
   not told what to pick, but it has the same reasoning a human ML
   practitioner who'd read the EDA would have.

Everything else is identical to Regular:
  - Same baseline winner (read from comparison_results.csv)
  - Same warm-start from trained_models/<winner>.keras
  - Same 5D discrete menu (loss × aug × lr × optimizer × schedule)
  - Same pruning callback (kill if val_auc < baseline - 0.05 after epoch 1)
  - Same dual-metric reporting (unweighted + weighted macro AUC)
  - Same hardcoded-experiment-.py rule

Search space size:
  5 losses × 4 augs × 3 lrs × 4 optims × 3 scheds × continuous geo
  ≈ 720 discrete combinations, each with a continuous geo dimension
  (bucketed to nearest 100 km for duplicate detection).

Output dir: experiments_eda_<llm_run_id>/    (separate from Regular)

Run:
    python agent_eda.py                  # 10 iterations
    python agent_eda.py --iterations 3   # smoke test
    python agent_eda.py --reset          # wipe experiments_eda_<llm>/ first
"""

# ─── Backend selection (must come before keras import) ───────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import importlib.util
import json
import math
import re
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers, ops
from sklearn.metrics import roc_auc_score

import ollama

warnings.filterwarnings("ignore")

# Reuse the data pipeline from baseline.py — single source of truth.
# Importing it runs Cells 1-6 of baseline.py (label space, training table,
# train/val split, AudioDataset, mel pipeline). The lat/long columns on
# train_rows / val_rows are additive — the Regular agent ignores them.
from baseline import (
    cfg,
    LABELS, NUM_CLASSES, LABEL_TO_IDX, SPECIES_TO_TAXON,
    TAXON_WEIGHTS,
    train_rows, val_rows,
    AudioDataset,
    TIME_FRAMES,
)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1: Agent configuration
# ─────────────────────────────────────────────────────────────────────────
AGENT_NAME = "eda"

# LLM config — adjust if your local Ollama model is named differently.
LLM_MODEL = "gemma4"


def safe_name(value: str) -> str:
    """Filesystem-safe slug to keep runs from different LLMs separate."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


LLM_RUN_ID      = safe_name(LLM_MODEL)
EXPERIMENTS_DIR = Path(f"experiments_{AGENT_NAME}_{LLM_RUN_ID}")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Discrete search space (mirrors Regular exactly, so we can attribute any
# EDA-vs-Regular gap purely to the new axis + prompt content)
LOSS_OPTIONS      = ["plain_bce", "weighted_bce", "focal_g1.0", "focal_g2.0", "focal_g3.0"]
AUG_OPTIONS       = ["none", "specaugment", "mixup", "background_mix"]
LR_OPTIONS        = [1e-3, 5e-4, 1e-4]
OPTIMIZER_OPTIONS = ["adam", "adamw", "sgd_momentum", "rmsprop"]
SCHEDULE_OPTIONS  = ["constant", "cosine_decay", "exp_decay"]

# Continuous 6th axis: geographic weighting scale (km).
# - "none"         → no geographic weighting; every sample weight = 1.0
# - any number in [GEO_SCALE_MIN, GEO_SCALE_MAX]:
#       weight(lat, lon) = exp(-haversine_km / scale_km)
GEO_SCALE_MIN = 200.0      # smaller than this: collapses to in-Pantanal-only
GEO_SCALE_MAX = 10000.0    # larger than this: indistinguishable from uniform
GEO_SCALE_ANCHORS = [500.0, 2000.0, 5000.0]   # shown to the LLM as examples
GEO_BUCKET_KM = 100.0      # duplicate-detection granularity

# Pantanal center (test domain) — used to compute distance-based weights
PANTANAL_LAT = -18.0
PANTANAL_LON = -56.0

# Pruning: kill an experiment after epoch 1 if val_auc is more than this
# much below the baseline winner's val_auc (measured once at startup).
PRUNE_MARGIN = 0.05

# Per-experiment training budget. Matches compare_models.py + Regular so
# cross-agent comparisons are fair.
EPOCHS_PER_EXPERIMENT = 10


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1.5: Pick the baseline winner from compare_models.py output
# ─────────────────────────────────────────────────────────────────────────
BASELINE_CSV       = Path("comparison_results.csv")
TRAINED_MODELS_DIR = Path("trained_models")


def pick_baseline_winner() -> tuple[str, dict, Path]:
    """Read comparison_results.csv and return (name, row_dict, keras_path).
    Ranking: max macro_auc, tiebreak by SMALLEST params."""
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"{BASELINE_CSV} not found. Run compare_models.py first."
        )
    df = pd.read_csv(BASELINE_CSV).dropna(subset=["macro_auc", "params"])
    if df.empty:
        raise ValueError(f"{BASELINE_CSV} has no valid rows.")
    df = df.sort_values(["macro_auc", "params"], ascending=[False, True])
    winner_row  = df.iloc[0].to_dict()
    winner_name = str(winner_row["model"])
    winner_path = TRAINED_MODELS_DIR / f"{winner_name}.keras"
    if not winner_path.exists():
        raise FileNotFoundError(
            f"Baseline winner '{winner_name}' exists in CSV but "
            f"{winner_path} is missing. Re-run compare_models.py."
        )
    return winner_name, winner_row, winner_path


BASELINE_WINNER_NAME, BASELINE_WINNER_ROW, BASELINE_WINNER_PATH = pick_baseline_winner()
print(f"\n=== Baseline winner (from {BASELINE_CSV}) ===")
print(f"  model      = {BASELINE_WINNER_NAME}")
print(f"  macro_auc  = {BASELINE_WINNER_ROW['macro_auc']:.4f}")
print(f"  params     = {int(BASELINE_WINNER_ROW['params']):,}")
print(f"  latency_ms = {BASELINE_WINNER_ROW['latency_ms']:.1f}")
print(f"  weights    = {BASELINE_WINNER_PATH}")
print(f"  → all experiments will warm-start from this model")


STEPS_PER_EPOCH   = math.ceil(len(train_rows) / cfg.batch_size)
TOTAL_TRAIN_STEPS = STEPS_PER_EPOCH * EPOCHS_PER_EXPERIMENT


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1.7: Measure baseline val_auc once (used as pruning threshold)
# ─────────────────────────────────────────────────────────────────────────
def _measure_baseline_val_auc() -> float:
    print(f"\n=== Measuring baseline val_auc for pruning threshold ===")
    keras.backend.clear_session()
    m = keras.models.load_model(BASELINE_WINNER_PATH)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )
    val_ds = AudioDataset(val_rows, batch_size=cfg.batch_size, shuffle=False)
    _, val_auc = m.evaluate(val_ds, verbose=0)
    keras.backend.clear_session()
    return float(val_auc)


try:
    BASELINE_VAL_AUC = _measure_baseline_val_auc()
    PRUNE_THRESHOLD  = BASELINE_VAL_AUC - PRUNE_MARGIN
    print(f"  baseline val_auc = {BASELINE_VAL_AUC:.4f}")
    print(f"  prune threshold  = {PRUNE_THRESHOLD:.4f}")
except Exception as e:
    BASELINE_VAL_AUC = float("nan")
    PRUNE_THRESHOLD  = 0.20
    print(f"  [!] could not measure baseline val_auc "
          f"({type(e).__name__}: {e}); fallback threshold = {PRUNE_THRESHOLD}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Class-frequency stats (used by weighted_bce)
# ─────────────────────────────────────────────────────────────────────────
def compute_class_pos_counts() -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for labs in train_rows["labels"]:
        for lab in labs:
            if lab in LABEL_TO_IDX:
                counts[LABEL_TO_IDX[lab]] += 1
    return counts


CLASS_POS_COUNTS = compute_class_pos_counts()
CLASS_WEIGHTS_PATH = EXPERIMENTS_DIR / "class_pos_weights.npy"
total = len(train_rows)
pos_weights = np.where(
    CLASS_POS_COUNTS > 0,
    np.clip(total / (CLASS_POS_COUNTS + 1e-6), 1.0, 5.0),
    5.0,
).astype(np.float32)
np.save(CLASS_WEIGHTS_PATH, pos_weights)
print(f"  class_pos_weights saved to {CLASS_WEIGHTS_PATH} "
      f"(range {pos_weights.min():.1f} – {pos_weights.max():.1f})")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2.5: Per-row geographic sample weights
# ─────────────────────────────────────────────────────────────────────────
def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance from each (lat1, lon1) to a single (lat2, lon2)
    point, in kilometers. NumPy-vectorised."""
    R = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = (np.sin(dp / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))


def precompute_distance_km(rows: pd.DataFrame) -> np.ndarray:
    """For each row return distance (km) to the Pantanal center.
    Rows with NaN coords get np.nan — these are soundscape (in-domain) or
    unknown-location focal recordings, both treated as weight=1.0 later."""
    lat = rows["latitude"].to_numpy(dtype=np.float64)
    lon = rows["longitude"].to_numpy(dtype=np.float64)
    out = np.full_like(lat, np.nan, dtype=np.float64)
    mask = ~np.isnan(lat) & ~np.isnan(lon)
    out[mask] = _haversine_km(lat[mask], lon[mask],
                              PANTANAL_LAT, PANTANAL_LON)
    return out


TRAIN_DIST_KM = precompute_distance_km(train_rows)
# Quick sanity log
_have = int((~np.isnan(TRAIN_DIST_KM)).sum())
print(f"  Per-row distance to Pantanal computed for "
      f"{_have} / {len(TRAIN_DIST_KM)} training rows "
      f"(others get weight=1.0)")
if _have > 0:
    d = TRAIN_DIST_KM[~np.isnan(TRAIN_DIST_KM)]
    print(f"  distance percentiles (km): "
          f"p10={np.percentile(d, 10):.0f}, "
          f"p50={np.percentile(d, 50):.0f}, "
          f"p90={np.percentile(d, 90):.0f}, "
          f"in-Pantanal-box={int((d < 500).sum())}")


def compute_sample_weights(geo_scale_km: float | None) -> np.ndarray:
    """Per-training-row weights for model.fit(sample_weight=...).
    geo_scale_km=None → all ones."""
    n = len(TRAIN_DIST_KM)
    if geo_scale_km is None:
        return np.ones(n, dtype=np.float32)
    w = np.exp(-TRAIN_DIST_KM / float(geo_scale_km))
    # NaN distances (soundscape rows + unknown-coord focal) → weight=1.0
    w = np.where(np.isnan(w), 1.0, w)
    # Floor at 0.05 so distant samples still contribute *something* to
    # the gradient. Otherwise a 200 km scale would zero out 95% of the
    # training data and we'd see catastrophic loss spikes.
    w = np.maximum(w, 0.05)
    return w.astype(np.float32)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Past-experiments helpers
# ─────────────────────────────────────────────────────────────────────────
def load_past_experiments() -> list[dict]:
    past = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        mfile = d / "metrics.json"
        if mfile.exists():
            past.append(json.loads(mfile.read_text(encoding="utf-8")))
    return past


def geo_bucket(geo_scale_km: float | None) -> str:
    """Canonical bucket for duplicate detection. Continuous values within
    GEO_BUCKET_KM of each other count as the same experiment."""
    if geo_scale_km is None:
        return "none"
    b = round(float(geo_scale_km) / GEO_BUCKET_KM) * GEO_BUCKET_KM
    return f"{b:g}"


def config_key(loss_name: str, aug_name: str, lr: float,
               optimizer_name: str, schedule_name: str,
               geo_scale_km: float | None) -> str:
    return (f"{loss_name}|{aug_name}|{lr:g}|{optimizer_name}|"
            f"{schedule_name}|geo={geo_bucket(geo_scale_km)}")


def summarise_past(past: list[dict]) -> str:
    if not past:
        return "(no prior experiments)"
    lines = []
    for e in past:
        pruned = " [PRUNED]" if e.get("pruned") else ""
        geo = e.get("geo_scale_km")
        geo_str = f"none      " if geo is None else f"{float(geo):>4.0f}km   "
        lines.append(
            f"- id={e['id']:>2}  loss={e['loss']:<14}  aug={e['aug']:<15}  "
            f"lr={float(e['lr']):g}  opt={e.get('optimizer', '?'):<13}  "
            f"sched={e.get('schedule', '?'):<13}  geo={geo_str}  →  "
            f"macro(unw)={e['macro_auc']:.4f}  "
            f"macro(w)={e['weighted_macro_auc']:.4f}  "
            f"aves={e['aves_auc']:.3f}{pruned}"
        )
    return "\n".join(lines)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: LLM prompt — EDA findings + same menu Regular has
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous ML research agent for the BirdCLEF+ 2026 audio classification task.

You have access to dataset findings from an exploratory data analysis (EDA).
Use them to inform your choices. The findings are FACTS about this dataset,
not suggestions about what to pick.

==========================================================================
EDA FINDINGS (READ ME)
==========================================================================
1. TAXONOMIC IMBALANCE
   The 234 labels span 5 taxonomic groups:
       Aves       162 species   34,799 training clips   (98% of data)
       Amphibia    35 species      451 clips
       Insecta     28 species      199 clips
       Mammalia     8 species       99 clips
       Reptilia     1 species        1 clip
   Birds dominate the data; non-birds are severely under-represented.
   The competition metric averages over all 234 classes equally, so
   poorly-modeled non-birds drag the macro AUC down disproportionately.

2. CLASS LONG-TAIL
   Within Aves alone, the most-common species has 499 clips and the
   rarest has 1–2 (imbalance ratio ~500:1). 25 species have <10 samples.

3. SPECTRAL DOMAIN SHIFT — INSECT CHORUS
   Focal training recordings are clean, mostly single-species.
   Pantanal test soundscapes have a continuous insect chorus around
   4 kHz that's absent from focal data. Models trained only on focal
   data see a sharply different test distribution.

4. MULTI-LABEL DENSITY DIFFERS
   Focal recordings: 87.7% have ZERO secondary labels (monophonic).
   Pantanal soundscapes: routinely 4–5 simultaneous species per 5s window.

5. GEOGRAPHIC DOMAIN SHIFT
   Training data is global; the test set is from the Pantanal wetlands
   (lat ~-18, lon ~-56). Only ~2% of focal training recordings fall in
   the Pantanal box. Soundscape training segments come from the Pantanal
   directly and so are already in-domain.

6. ALL EXPERIMENTS WARM-START
   From the same pretrained baseline (CNN-RNN trained on plain BCE,
   no augmentation, lr=1e-3). You are picking the FINE-TUNING recipe,
   not training from scratch.

==========================================================================
MENU — pick exactly one option from each axis
==========================================================================
LOSS:
  - plain_bce       : standard binary cross-entropy
  - weighted_bce    : BCE weighted by inverse class frequency
                       → relevant for the 234-class imbalance (finding 1, 2)
  - focal_g1.0      : focal loss, gamma=1.0 (mild focus on hard examples)
  - focal_g2.0      : focal loss, gamma=2.0 (standard focal loss)
                       → relevant for the long tail (finding 2)
  - focal_g3.0      : focal loss, gamma=3.0 (aggressive)

AUGMENTATION:
  - none            : no augmentation (matches the baseline's training)
  - specaugment     : random time/frequency masking on the spectrogram
                       → frequency masks can blunt the 4 kHz chorus (finding 3)
  - mixup           : linear combination of two training examples
                       → synthesizes multi-label examples (finding 4)
  - background_mix  : add a low-volume second sample to each input
                       → directly simulates the soundscape condition (finding 3)

LEARNING RATE (initial value; the schedule may then change it):
  - 1e-3            : standard Adam learning rate
  - 5e-4            : slightly lower, more stable
  - 1e-4            : much lower, gentler fine-tuning

OPTIMIZER:
  - adam            : standard Adam
  - adamw           : Adam with weight decay 1e-4 (better generalization)
  - sgd_momentum    : SGD with momentum=0.9 (slower; sometimes generalizes better)
  - rmsprop         : RMSprop

SCHEDULE:
  - constant        : keep LR fixed (ReduceLROnPlateau still active)
  - cosine_decay    : smoothly decay from initial LR toward 0
  - exp_decay       : multiply LR by 0.9 each epoch

GEO_SCALE_KM (NEW — geographic sample weighting toward the Pantanal):
  Each training sample is weighted by exp(-distance_to_Pantanal / scale_km).
  Small scale  → almost only in-Pantanal samples contribute (aggressive)
  Large scale  → nearly uniform weighting (mild)
  Soundscape rows and rows with unknown coordinates always get weight 1.0.

  Pick EITHER the string `none` to disable geographic weighting, OR a
  NUMBER (integer or float) in the range [200, 10000].

  Anchor values to think with (you are NOT limited to these — pick any
  number in the range):
    - 500 km   : aggressive (most weight on the ~2% in-Pantanal data)
    - 2000 km  : moderate
    - 5000 km  : mild
    - 10000 km : barely different from uniform

The task is multi-label classification across 234 classes. Two metrics
are reported per experiment:
  - macro_auc (unweighted) : the official competition metric
  - macro_auc (weighted)   : per-taxon AUC weighted by class-count share
                              (Aves contributes ~69%, Amphibia ~15%, …)
Both metrics matter.

Experiments marked [PRUNED] were aborted after epoch 1 because val_auc
dropped sharply — treat those configurations as red flags.

==========================================================================
RULES
==========================================================================
  1. Pick exactly ONE option from each of the SIX axes.
  2. Do NOT pick a combination already in the log (geo_scale_km values
     within 100 km of a previous run count as the same experiment).
  3. Respond in EXACTLY this format, with no extra text:

CHOICE_LOSS:      <plain_bce | weighted_bce | focal_g1.0 | focal_g2.0 | focal_g3.0>
CHOICE_AUG:       <none | specaugment | mixup | background_mix>
CHOICE_LR:        <1e-3 | 5e-4 | 1e-4>
CHOICE_OPTIMIZER: <adam | adamw | sgd_momentum | rmsprop>
CHOICE_SCHEDULE:  <constant | cosine_decay | exp_decay>
CHOICE_GEO_KM:    <none | a number in [200, 10000]>
REASON: <one short paragraph explaining why, referencing the relevant EDA finding(s)>
"""


def build_user_prompt(past: list[dict]) -> str:
    tried_keys = {config_key(e["loss"], e["aug"], float(e["lr"]),
                              e.get("optimizer", "adam"),
                              e.get("schedule", "constant"),
                              e.get("geo_scale_km")) for e in past}
    n_discrete = (len(LOSS_OPTIONS) * len(AUG_OPTIONS) * len(LR_OPTIONS)
                  * len(OPTIMIZER_OPTIONS) * len(SCHEDULE_OPTIONS))
    return f"""Past experiments ({len(tried_keys)} unique configs tried, of {n_discrete} discrete × continuous-geo combinations):
{summarise_past(past)}

Pick the next experiment. It must be a configuration we have NOT
already run (the geo_scale_km axis is bucketed to nearest 100 km
for duplicate detection)."""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Parse LLM response
# ─────────────────────────────────────────────────────────────────────────
def parse_llm_response(response: str
                       ) -> tuple[str, str, float, str, str, float | None, str]:
    """Extract (loss, aug, lr, optimizer, schedule, geo_scale_km, reason).
    geo_scale_km is None when the LLM picked 'none'."""
    def grab(field_name: str) -> str:
        m = re.search(rf"{field_name}\s*:\s*([^\n]+)", response, re.IGNORECASE)
        if not m:
            raise ValueError(f"No {field_name} in response:\n{response}")
        return m.group(1).strip().strip("`'\"[](){}<>")

    loss = grab("CHOICE_LOSS")
    aug  = grab("CHOICE_AUG")
    lr_s = grab("CHOICE_LR")
    opt  = grab("CHOICE_OPTIMIZER")
    sch  = grab("CHOICE_SCHEDULE")
    geo_s = grab("CHOICE_GEO_KM")

    if loss not in LOSS_OPTIONS:
        raise ValueError(f"Loss '{loss}' not in {LOSS_OPTIONS}")
    if aug not in AUG_OPTIONS:
        raise ValueError(f"Aug '{aug}' not in {AUG_OPTIONS}")
    if opt not in OPTIMIZER_OPTIONS:
        raise ValueError(f"Optimizer '{opt}' not in {OPTIMIZER_OPTIONS}")
    if sch not in SCHEDULE_OPTIONS:
        raise ValueError(f"Schedule '{sch}' not in {SCHEDULE_OPTIONS}")
    try:
        lr = float(lr_s)
    except ValueError:
        raise ValueError(f"LR '{lr_s}' is not a number")
    if not any(abs(lr - opt_lr) < 1e-12 for opt_lr in LR_OPTIONS):
        raise ValueError(f"LR '{lr}' not in {LR_OPTIONS}")

    # Geo: either 'none' (case-insensitive) or a numeric km value in range.
    # Be tolerant of trailing units like '500 km' or '2000km'.
    geo_clean = re.sub(r"\s*km\s*$", "", geo_s, flags=re.IGNORECASE).strip()
    if geo_clean.lower() in ("none", "off", "no", "disabled"):
        geo_scale_km: float | None = None
    else:
        try:
            geo_scale_km = float(geo_clean)
        except ValueError:
            raise ValueError(f"Geo '{geo_s}' is not 'none' or a number")
        if not (GEO_SCALE_MIN <= geo_scale_km <= GEO_SCALE_MAX):
            raise ValueError(
                f"Geo '{geo_scale_km}' outside "
                f"[{GEO_SCALE_MIN}, {GEO_SCALE_MAX}]"
            )

    reason_m = re.search(r"REASON\s*:\s*(.+?)(?=\n\s*CHOICE_|\Z)",
                         response, re.IGNORECASE | re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else "(no reason given)"
    return loss, aug, lr, opt, sch, geo_scale_km, reason


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Generate the experiment .py file (the "hardcoded" rule)
# Each experiment is a self-contained file that anyone can re-run.
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENT_TEMPLATE = '''"""
Experiment {exp_id} (EDA Agent)
    baseline      = {winner_name}  (loaded from {winner_path})
    loss          = {loss_name}
    augmentation  = {aug_name}
    optimizer     = {optimizer_name}
    schedule      = {schedule_name}
    initial_lr    = {lr:g}
    geo_scale_km  = {geo_scale_repr}

Generated at {timestamp}
Rationale:
    {reason}

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(), get_geo_scale_km().
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = {num_classes}
LEARNING_RATE      = {lr:g}
STEPS_PER_EPOCH    = {steps_per_epoch}
TOTAL_TRAIN_STEPS  = {total_train_steps}
WINNER_KERAS_PATH  = "{winner_path}"
GEO_SCALE_KM       = {geo_scale_repr}     # None or a float in km


# ── Architecture: warm-start from the winning baseline ──────────────────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
{loss_code}

def get_loss():
    return {loss_factory_call}


# ── Optimizer + LR schedule ─────────────────────────────────────────────
{optimizer_schedule_code}


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
{aug_code}


# ── Geographic sample weighting ─────────────────────────────────────────
def get_geo_scale_km():
    """Return the chosen geographic scale (km) or None for no weighting."""
    return GEO_SCALE_KM
'''


def loss_block_for(loss_name: str) -> tuple[str, str]:
    if loss_name == "plain_bce":
        return ("# Plain BCE — no special class needed.",
                "keras.losses.BinaryCrossentropy()")
    if loss_name == "weighted_bce":
        weights_path = str(CLASS_WEIGHTS_PATH).replace("\\", "/")
        return (f'''POS_WEIGHTS = np.load("{weights_path}").astype("float32")

class WeightedBCE(keras.losses.Loss):
    def __init__(self, pos_weights, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weights = ops.convert_to_tensor(pos_weights)
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        per_class = -(self.pos_weights * y_true * ops.log(y_pred)
                      + (1.0 - y_true) * ops.log(1.0 - y_pred))
        return ops.mean(per_class)
''',
                "WeightedBCE(POS_WEIGHTS)")
    m = re.match(r"focal_g([\d\.]+)", loss_name)
    if m:
        gamma = float(m.group(1))
        return (f'''class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma={gamma}, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        log_pt = ops.log(pt)
        loss = -ops.power(1.0 - pt, self.gamma) * log_pt
        return ops.mean(loss)
''',
                "FocalLoss()")
    raise ValueError(f"Unknown loss: {loss_name}")


def aug_block_for(aug_name: str) -> str:
    if aug_name == "none":
        return '''def augment_batch(xs, ys):
    return xs, ys'''
    if aug_name == "specaugment":
        return '''def augment_batch(xs, ys):
    """SpecAugment: zero out random time and frequency bands per sample."""
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        f = np.random.randint(0, max(1, h // 8))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        t = np.random.randint(0, max(1, w // 8))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys'''
    if aug_name == "mixup":
        return '''def augment_batch(xs, ys):
    """Mixup: linearly combine pairs of examples within the batch."""
    if len(xs) < 2:
        return xs, ys
    alpha = 0.4
    lam = float(np.random.beta(alpha, alpha))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    return xs_m.astype(np.float32), ys_m.astype(np.float32)'''
    if aug_name == "background_mix":
        return '''def augment_batch(xs, ys):
    """Add a low-volume second sample as background. Simulates the
    Pantanal soundscape condition where target calls overlap with a
    continuous chorus (EDA finding 3)."""
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys'''
    raise ValueError(f"Unknown augmentation: {aug_name}")


def optimizer_schedule_block_for(optimizer_name: str, schedule_name: str
                                 ) -> str:
    if schedule_name == "constant":
        lr_expr = "LEARNING_RATE"
        cbs_expr = ("[keras.callbacks.ReduceLROnPlateau("
                    "monitor='val_loss', factor=0.5, patience=2, "
                    "min_lr=1e-5, verbose=0)]")
    elif schedule_name == "cosine_decay":
        lr_expr = ("keras.optimizers.schedules.CosineDecay("
                   "initial_learning_rate=LEARNING_RATE, "
                   "decay_steps=TOTAL_TRAIN_STEPS, alpha=0.0)")
        cbs_expr = "[]"
    elif schedule_name == "exp_decay":
        lr_expr = ("keras.optimizers.schedules.ExponentialDecay("
                   "initial_learning_rate=LEARNING_RATE, "
                   "decay_steps=STEPS_PER_EPOCH, decay_rate=0.9, "
                   "staircase=True)")
        cbs_expr = "[]"
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")

    if optimizer_name == "adam":
        opt_expr = f"keras.optimizers.Adam(learning_rate={lr_expr})"
    elif optimizer_name == "adamw":
        opt_expr = (f"keras.optimizers.AdamW(learning_rate={lr_expr}, "
                    f"weight_decay=1e-4)")
    elif optimizer_name == "sgd_momentum":
        opt_expr = (f"keras.optimizers.SGD(learning_rate={lr_expr}, "
                    f"momentum=0.9)")
    elif optimizer_name == "rmsprop":
        opt_expr = f"keras.optimizers.RMSprop(learning_rate={lr_expr})"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return (f"def get_optimizer():\n"
            f"    return {opt_expr}\n\n\n"
            f"def get_schedule_callbacks():\n"
            f"    return {cbs_expr}")


def generate_experiment_file(exp_id: int, loss_name: str, aug_name: str,
                             lr: float, optimizer_name: str,
                             schedule_name: str,
                             geo_scale_km: float | None,
                             reason: str) -> Path:
    """Write the hardcoded experiment .py file and return its path."""
    loss_code, factory_call = loss_block_for(loss_name)
    aug_code  = aug_block_for(aug_name)
    opt_sched = optimizer_schedule_block_for(optimizer_name, schedule_name)
    geo_repr  = "None" if geo_scale_km is None else f"{float(geo_scale_km):g}"
    geo_tag   = geo_bucket(geo_scale_km)
    exp_dir = (EXPERIMENTS_DIR
               / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}"
                 f"_{optimizer_name}_{schedule_name}_geo{geo_tag}")
    exp_dir.mkdir(exist_ok=True)
    py_path = exp_dir / "model.py"
    winner_path_str = str(BASELINE_WINNER_PATH).replace("\\", "/")
    py_path.write_text(EXPERIMENT_TEMPLATE.format(
        exp_id=exp_id,
        loss_name=loss_name,
        aug_name=aug_name,
        lr=lr,
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        geo_scale_repr=geo_repr,
        reason=reason.replace('"""', '"'),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_classes=NUM_CLASSES,
        winner_name=BASELINE_WINNER_NAME,
        winner_path=winner_path_str,
        steps_per_epoch=STEPS_PER_EPOCH,
        total_train_steps=TOTAL_TRAIN_STEPS,
        loss_code=loss_code,
        loss_factory_call=factory_call,
        optimizer_schedule_code=opt_sched,
        aug_code=aug_code,
    ), encoding="utf-8")
    return py_path


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Run one experiment
# ─────────────────────────────────────────────────────────────────────────
class WeightedAudioDataset(keras.utils.PyDataset):
    """Wraps AudioDataset and attaches a per-sample weight vector so that
    model.fit(...) sees (xs, ys, sample_weights) tuples and respects
    them in its loss aggregation.

    Augmentation is applied AFTER the wrap so it sees both xs and ys
    (sample_weights pass through untouched)."""

    def __init__(self, base_ds: keras.utils.PyDataset,
                 sample_weights: np.ndarray | None,
                 augment_fn, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("use_multiprocessing", False)
        kwargs.setdefault("max_queue_size", 8)
        super().__init__(**kwargs)
        self.base = base_ds
        self.sample_weights = sample_weights   # None or ndarray of len(rows)
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # We need the row indices the base picked for this batch so we
        # can slice the weight vector. AudioDataset stores them on
        # self.base.indices (the shuffled permutation).
        bs = self.base.batch_size
        batch_indices = self.base.indices[idx * bs: (idx + 1) * bs]
        xs, ys = self.base[idx]
        xs, ys = self.augment_fn(xs, ys)
        if self.sample_weights is None:
            return xs, ys
        ws = self.sample_weights[batch_indices].astype(np.float32)
        return xs, ys, ws

    def on_epoch_end(self):
        if hasattr(self.base, "on_epoch_end"):
            self.base.on_epoch_end()


def _compute_aucs(model: keras.Model, val_ds: keras.utils.PyDataset
                  ) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        batch = val_ds[i]
        # val_ds may be a plain AudioDataset (xs, ys) — no weights.
        xb, yb = batch[0], batch[1]
        all_preds.append(model.predict(xb, verbose=0))
        all_true.append(yb)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    per_class_auc: dict[str, float | None] = {}
    for cls_idx, label in enumerate(LABELS):
        if y_true[:, cls_idx].sum() == 0:
            per_class_auc[label] = None
            continue
        try:
            per_class_auc[label] = float(roc_auc_score(
                y_true[:, cls_idx], y_pred[:, cls_idx]
            ))
        except Exception:
            per_class_auc[label] = None
    return per_class_auc, y_pred, y_true


def _summarise_aucs(per_class_auc: dict[str, float | None]
                    ) -> tuple[float, float, dict[str, float], int]:
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    taxon_to_aucs: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None:
            continue
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        taxon_to_aucs.setdefault(taxon, []).append(auc)
    taxon_aucs = {t: float(np.mean(v)) for t, v in taxon_to_aucs.items()}

    weighted_sum  = 0.0
    weight_total  = 0.0
    for taxon, mean_auc in taxon_aucs.items():
        w = TAXON_WEIGHTS.get(taxon, 0.0)
        weighted_sum += w * mean_auc
        weight_total += w
    weighted_macro_auc = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    return macro_auc, weighted_macro_auc, taxon_aucs, len(valid_aucs)


class PruneIfWorseCallback(keras.callbacks.Callback):
    """Stop training after epoch 1 if val_auc is below threshold."""

    def __init__(self, threshold: float, monitor: str = "val_auc"):
        super().__init__()
        self.threshold = threshold
        self.monitor   = monitor
        self.pruned    = False

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            val = float((logs or {}).get(self.monitor, 0.0))
            if val < self.threshold:
                print(f"\n  [PRUNE] {self.monitor}={val:.4f} < threshold "
                      f"{self.threshold:.4f} after epoch 1 — stopping early")
                self.model.stop_training = True
                self.pruned = True


def run_experiment(exp_id: int, loss_name: str, aug_name: str,
                   lr: float, optimizer_name: str, schedule_name: str,
                   geo_scale_km: float | None, reason: str) -> dict:
    geo_label = "off" if geo_scale_km is None else f"{geo_scale_km:.0f}km"
    print(f"\n{'=' * 72}")
    print(f" EXPERIMENT {exp_id}: loss={loss_name}  aug={aug_name}  "
          f"lr={lr:g}  opt={optimizer_name}  sched={schedule_name}  "
          f"geo={geo_label}")
    print(f"{'=' * 72}")
    print(f" reason: {reason}\n")

    py_path = generate_experiment_file(exp_id, loss_name, aug_name, lr,
                                       optimizer_name, schedule_name,
                                       geo_scale_km, reason)
    exp_dir = py_path.parent

    spec = importlib.util.spec_from_file_location(f"exp_{exp_id}", py_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg.seed)

    model = mod.build_model()
    model.compile(
        optimizer=mod.get_optimizer(),
        loss=mod.get_loss(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )
    n_params = int(model.count_params())
    print(f"  Parameters: {n_params:,}")

    # Build sample weights from the chosen geo scale, then verify the
    # train rows are in the same order WeightedAudioDataset will see.
    sample_weights = compute_sample_weights(mod.get_geo_scale_km())
    if sample_weights is not None and geo_scale_km is not None:
        print(f"  Geo sample weights: min={sample_weights.min():.3f}  "
              f"median={np.median(sample_weights):.3f}  "
              f"max={sample_weights.max():.3f}  "
              f"mean={sample_weights.mean():.3f}")

    base_train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
    train_ds      = WeightedAudioDataset(
        base_train_ds,
        sample_weights if geo_scale_km is not None else None,
        mod.augment_batch,
    )
    val_ds = AudioDataset(val_rows, batch_size=cfg.batch_size, shuffle=False)

    prune_cb = PruneIfWorseCallback(threshold=PRUNE_THRESHOLD, monitor="val_auc")
    callbacks = [
        *mod.get_schedule_callbacks(),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        ),
        prune_cb,
    ]

    t0 = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_PER_EXPERIMENT,
        callbacks=callbacks,
        verbose=1,
    )
    train_time_s = time.perf_counter() - t0

    weights_path = exp_dir / "model.keras"
    model.save(weights_path)
    print(f"  Saved trained model to {weights_path}")

    print(f"  Computing validation AUC...")
    per_class_auc, _, _ = _compute_aucs(model, val_ds)
    macro_auc, weighted_macro_auc, taxon_aucs, n_eval = _summarise_aucs(per_class_auc)

    metrics = {
        "id":                  exp_id,
        "agent_name":          AGENT_NAME,
        "llm_model":           LLM_MODEL,
        "llm_run_id":          LLM_RUN_ID,
        "experiments_dir":     str(EXPERIMENTS_DIR),
        "loss":                loss_name,
        "aug":                 aug_name,
        "lr":                  lr,
        "optimizer":           optimizer_name,
        "schedule":            schedule_name,
        "geo_scale_km":        geo_scale_km,
        "geo_bucket":          geo_bucket(geo_scale_km),
        "reason":              reason,
        "baseline_winner":     BASELINE_WINNER_NAME,
        "baseline_macro_auc":  float(BASELINE_WINNER_ROW["macro_auc"]),
        "baseline_val_auc":    BASELINE_VAL_AUC,
        "pruned":              prune_cb.pruned,
        "params":              n_params,
        "epochs_run":          len(history.history["loss"]),
        "train_time_s":        round(train_time_s, 1),
        "macro_auc":           round(macro_auc, 4),
        "weighted_macro_auc":  round(weighted_macro_auc, 4),
        "n_classes_eval":      n_eval,
        "aves_auc":            round(taxon_aucs.get("Aves",     float("nan")), 4),
        "amphibia_auc":        round(taxon_aucs.get("Amphibia", float("nan")), 4),
        "insecta_auc":         round(taxon_aucs.get("Insecta",  float("nan")), 4),
        "mammalia_auc":        round(taxon_aucs.get("Mammalia", float("nan")), 4),
        "py_path":             str(py_path),
        "weights_path":        str(weights_path),
    }
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    status = "[PRUNED] " if prune_cb.pruned else ""
    print(f"\n  RESULT: {status}macro_auc(unw)={macro_auc:.4f}  "
          f"macro_auc(w)={weighted_macro_auc:.4f}  "
          f"aves={taxon_aucs.get('Aves', 0):.3f}  "
          f"amph={taxon_aucs.get('Amphibia', 0):.3f}")
    return metrics


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8: The agent loop
# ─────────────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
    """Call the local Ollama LLM. keep_alive=0 frees RAM before training."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"temperature": 0.3},
        keep_alive=0,
    )
    return response["message"]["content"]


def agent_loop(n_iterations: int) -> None:
    n_discrete = (len(LOSS_OPTIONS) * len(AUG_OPTIONS) * len(LR_OPTIONS)
                  * len(OPTIMIZER_OPTIONS) * len(SCHEDULE_OPTIONS))
    print(f"\n{'#' * 72}")
    print(f"# EDA AGENT  —  {n_iterations} iterations  —  LLM={LLM_MODEL}")
    print(f"# Output dir: {EXPERIMENTS_DIR}")
    print(f"# Search space: {n_discrete} discrete configs × continuous geo "
          f"(bucketed to {GEO_BUCKET_KM:.0f} km)")
    print(f"{'#' * 72}")

    for iteration in range(n_iterations):
        past = load_past_experiments()
        exp_id = (max((e["id"] for e in past), default=0)) + 1
        tried_keys = {config_key(e["loss"], e["aug"], float(e["lr"]),
                                  e.get("optimizer", "adam"),
                                  e.get("schedule", "constant"),
                                  e.get("geo_scale_km"))
                      for e in past}

        print(f"\n--- iteration {iteration + 1}/{n_iterations} "
              f"(experiment id {exp_id}, {len(tried_keys)} unique configs done) ---")

        # Ask the LLM, with up to 3 re-prompts on duplicates / parse errors
        loss_name, aug_name, lr = None, None, None
        optimizer_name, schedule_name, geo_scale_km, reason = None, None, None, None
        user_prompt = build_user_prompt(past)
        for attempt in range(3):
            try:
                response = call_llm(SYSTEM_PROMPT, user_prompt)
                print(f"LLM raw response (attempt {attempt + 1}):\n"
                      f"{response.strip()}\n")
                c_loss, c_aug, c_lr, c_opt, c_sch, c_geo, c_reason = \
                    parse_llm_response(response)
                if config_key(c_loss, c_aug, c_lr, c_opt, c_sch, c_geo) in tried_keys:
                    geo_str = "none" if c_geo is None else f"{c_geo:.0f}km"
                    print(f"  [!] LLM picked an already-tried combination "
                          f"({c_loss}, {c_aug}, lr={c_lr:g}, {c_opt}, "
                          f"{c_sch}, geo={geo_str}). Re-prompting.")
                    user_prompt = (
                        build_user_prompt(past)
                        + f"\n\nIMPORTANT: You just picked an already-tried "
                          f"combination. Pick a DIFFERENT, untried one. "
                          f"Vary at least one axis."
                    )
                    continue
                loss_name, aug_name, lr = c_loss, c_aug, c_lr
                optimizer_name, schedule_name = c_opt, c_sch
                geo_scale_km, reason = c_geo, c_reason
                break
            except Exception as e:
                print(f"  [!] LLM call/parse failed (attempt {attempt + 1}): "
                      f"{type(e).__name__}: {e}")

        if loss_name is None:
            # Fallback after 3 failed attempts: pick a sensible, untried
            # EDA-motivated config. Walk the menu deterministically.
            fallback_geos: list[float | None] = [2000.0, 5000.0, 500.0, None]
            fallback_combo = None
            for geo in fallback_geos:
                for loss in LOSS_OPTIONS:
                    for aug in AUG_OPTIONS:
                        for o in OPTIMIZER_OPTIONS:
                            for s in SCHEDULE_OPTIONS:
                                for r in LR_OPTIONS:
                                    k = config_key(loss, aug, r, o, s, geo)
                                    if k not in tried_keys:
                                        fallback_combo = (loss, aug, r, o, s, geo)
                                        break
                                if fallback_combo: break
                            if fallback_combo: break
                        if fallback_combo: break
                    if fallback_combo: break
                if fallback_combo: break
            if not fallback_combo:
                print("  All configurations have been tried. Stopping.")
                break
            loss_name, aug_name, lr, optimizer_name, schedule_name, geo_scale_km = fallback_combo
            reason = ("(fallback: LLM unavailable or unhelpful — picked the "
                      "next untried EDA-motivated combination)")
            geo_str = "none" if geo_scale_km is None else f"{geo_scale_km:.0f}km"
            print(f"  Fallback choice: ({loss_name}, {aug_name}, lr={lr:g}, "
                  f"{optimizer_name}, {schedule_name}, geo={geo_str})")

        try:
            run_experiment(exp_id, loss_name, aug_name, lr,
                           optimizer_name, schedule_name, geo_scale_km, reason)
        except Exception as e:
            print(f"[!] experiment {exp_id} crashed: {type(e).__name__}: {e}")
            geo_tag = geo_bucket(geo_scale_km)
            failure_dir = (
                EXPERIMENTS_DIR
                / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}"
                  f"_{optimizer_name}_{schedule_name}_geo{geo_tag}_FAILED"
            )
            failure_dir.mkdir(exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "id": exp_id,
                "agent_name": AGENT_NAME,
                "llm_model": LLM_MODEL,
                "llm_run_id": LLM_RUN_ID,
                "experiments_dir": str(EXPERIMENTS_DIR),
                "loss": loss_name, "aug": aug_name, "lr": lr,
                "optimizer": optimizer_name, "schedule": schedule_name,
                "geo_scale_km": geo_scale_km, "geo_bucket": geo_tag,
                "reason": reason, "pruned": False,
                "macro_auc": 0.0, "weighted_macro_auc": 0.0,
                "aves_auc": 0.0, "amphibia_auc": 0.0,
                "insecta_auc": 0.0, "mammalia_auc": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }, indent=2), encoding="utf-8")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'#' * 72}\n# EDA AGENT — FINAL SUMMARY\n{'#' * 72}")
    final = load_past_experiments()
    if not final:
        print("No experiments completed.")
        return

    ranked = sorted(final, key=lambda e: e.get("weighted_macro_auc", 0),
                    reverse=True)
    print(f"\n{'rank':<5}{'id':<4}{'loss':<14}{'aug':<15}{'lr':<8}"
          f"{'opt':<14}{'sched':<14}{'geo':<8}"
          f"{'macro(uw)':>10}{'macro(w)':>10}{'aves':>8}")
    for rank, e in enumerate(ranked, 1):
        geo = e.get("geo_scale_km")
        geo_s = "none" if geo is None else f"{float(geo):.0f}km"
        print(f"{rank:<5}{e['id']:<4}{e['loss']:<14}{e['aug']:<15}"
              f"{float(e['lr']):<8.0e}"
              f"{e.get('optimizer','?'):<14}{e.get('schedule','?'):<14}"
              f"{geo_s:<8}"
              f"{e.get('macro_auc', 0):>10.4f}"
              f"{e.get('weighted_macro_auc', 0):>10.4f}"
              f"{e.get('aves_auc', 0):>8.3f}")

    best = ranked[0]
    geo = best.get("geo_scale_km")
    geo_s = "none" if geo is None else f"{float(geo):.0f}km"
    print(f"\nBEST (by weighted macro AUC): exp {best['id']} — "
          f"loss={best['loss']}  aug={best['aug']}  lr={float(best['lr']):g}  "
          f"opt={best.get('optimizer','?')}  sched={best.get('schedule','?')}  "
          f"geo={geo_s}")
    print(f"  weighted_macro_auc   = {best.get('weighted_macro_auc', 0):.4f}")
    print(f"  unweighted_macro_auc = {best.get('macro_auc', 0):.4f}")
    print(f"  aves_auc             = {best.get('aves_auc', 0):.3f}")
    print(f"  Model code:    {best.get('py_path')}")
    print(f"  Model weights: {best.get('weights_path')}")

    plot_progress(final)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8.5: Progress plot (same shape as Regular's: 3 lines + 2 refs)
# ─────────────────────────────────────────────────────────────────────────
def plot_progress(experiments: list[dict] | None = None,
                  out_path: Path | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping progress plot")
        return

    if experiments is None:
        experiments = load_past_experiments()

    successful = [e for e in experiments
                  if e.get("macro_auc") is not None and "error" not in e]
    if not successful:
        print("  No successful experiments to plot.")
        return
    successful.sort(key=lambda e: e["id"])

    iters    = [e["id"]                    for e in successful]
    macro_uw = [e["macro_auc"]             for e in successful]
    macro_w  = [e["weighted_macro_auc"]    for e in successful]
    aves_auc = [e.get("aves_auc", np.nan)  for e in successful]
    baseline = successful[0].get("baseline_macro_auc")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, macro_uw, marker="o", lw=2, color="steelblue",
            label="macro AUC (unweighted)")
    ax.plot(iters, macro_w,  marker="s", lw=2, color="darkorange",
            label="macro AUC (weighted by taxon)")
    ax.plot(iters, aves_auc, marker="^", lw=2, color="forestgreen",
            label="Aves/Bird AUC")

    best_w_val   = max(macro_w)
    best_w_iter  = iters[macro_w.index(best_w_val)]
    ax.axhline(best_w_val, ls="--", color="darkorange", alpha=0.4,
               label=f"Best weighted: {best_w_val:.4f} (exp {best_w_iter})")

    if baseline is not None:
        ax.axhline(baseline, ls=":", color="gray", alpha=0.7,
                   label=f"Baseline {BASELINE_WINNER_NAME}: {baseline:.4f}")

    ax.set_xlabel("Experiment iteration")
    ax.set_ylabel("Validation macro AUC")
    ax.set_title(
        f"EDA Agent progress — {len(successful)} experiment"
        f"{'s' if len(successful) != 1 else ''}"
    )
    ax.set_xticks(iters)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path is None:
        out_path = EXPERIMENTS_DIR / "progress.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Progress plot saved to {out_path}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 9: CLI
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of agent iterations (default 10)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all past experiments before starting")
    args = parser.parse_args()

    if args.reset:
        for d in list(EXPERIMENTS_DIR.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)
        np.save(CLASS_WEIGHTS_PATH, pos_weights)
        print(f"Reset: cleared {EXPERIMENTS_DIR}")

    agent_loop(args.iterations)
