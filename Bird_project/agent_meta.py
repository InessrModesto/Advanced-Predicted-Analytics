"""
BirdCLEF+ 2026 — Meta Agent
============================

The fourth and final agent. Builds on top of the three prior agents:

    EDA → Baselines (compare_models.py)
        → Agent Regular   (agent_regular.py)
        → Agent EDA       (agent_eda.py)
        → Agent Creative  (agent_creative.py)
        → Agent Meta      (THIS FILE)

What Meta does differently
--------------------------
1. **Warm-start from the GLOBAL WINNER**, not from the CRNN baseline.
   The global winner is the single best experiment across the three
   prior agents (ranked by weighted_macro_auc). Every meta experiment
   fine-tunes from that winner's .keras weights. Meta is a refinement
   stage, not a fourth competing agent — its results are not directly
   comparable to the other three agents' tables.

2. **Sees ALL prior experiments** in the prompt, tagged by agent. The
   prompt also includes a small "facts" block computed at runtime from
   whatever metrics.json files are on disk — purely arithmetic counts
   like "in the top 10, optimizer=adamw appears Nx" — so the LLM has a
   reliable factual digest to ground its reasoning instead of having to
   do aggregations itself.

3. **Looser pruning.** Other agents prune after epoch 1 if val_auc
   drops 0.05 below CRNN's val_auc. Meta prunes after epoch 2 (gives a
   recipe time to dip and recover from any augmentation mismatch with
   the winner's training distribution) with a wider 0.10 margin against
   the WINNER's val_auc.

4. **Writes a report_analysis.md at the end** covering:
   - Cross-agent comparison table (best result per agent)
   - Per-taxon AUC breakdown (mean / median / p10 / p90) on the meta-best
   - Top-10 confidently-wrong species (false positives + false negatives)
   - The LLM's pattern observations gathered from each iteration's REASON

Search space (same 6 axes as EDA; geo=none is a valid choice):
  - Loss / Augmentation / Initial LR / Optimizer / LR schedule / Geo scale

Output dir: experiments_meta_<llm_run_id>/
Also writes: meta_prior_experiments.csv (combined table across agents)

Run:
    python agent_meta.py                  # 10 iterations
    python agent_meta.py --iterations 3   # smoke test
    python agent_meta.py --reset          # wipe past meta experiments first
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
AGENT_NAME = "meta"
LLM_MODEL  = "gemma4"


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


LLM_RUN_ID      = safe_name(LLM_MODEL)
EXPERIMENTS_DIR = Path(f"experiments_{AGENT_NAME}_{LLM_RUN_ID}")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

LOSS_OPTIONS      = ["plain_bce", "weighted_bce", "focal_g1.0", "focal_g2.0", "focal_g3.0"]
AUG_OPTIONS       = ["none", "specaugment", "mixup", "background_mix"]
LR_OPTIONS        = [1e-3, 5e-4, 1e-4]
OPTIMIZER_OPTIONS = ["adam", "adamw", "sgd_momentum", "rmsprop"]
SCHEDULE_OPTIONS  = ["constant", "cosine_decay", "exp_decay"]
GEO_SCALE_MIN     = 200.0
GEO_SCALE_MAX     = 10000.0
GEO_BUCKET_KM     = 100.0
PANTANAL_LAT      = -18.0
PANTANAL_LON      = -56.0

PRUNE_MARGIN          = 0.10   # looser than the other agents
PRUNE_AT_EPOCH_INDEX  = 1      # 0-indexed → check after epoch 2 finishes
EPOCHS_PER_EXPERIMENT = 10

# Prior agents' experiment folders — meta reads from these at startup
OTHER_AGENT_DIRS = [
    Path(f"experiments_regular_{LLM_RUN_ID}"),
    Path(f"experiments_eda_{LLM_RUN_ID}"),
    Path(f"experiments_creative_{LLM_RUN_ID}"),
]


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Read all prior agents' experiments + find the global winner
# ─────────────────────────────────────────────────────────────────────────
def load_agent_experiments(folder: Path, agent_label: str) -> list[dict]:
    """Read all metrics.json under `folder` and tag each with agent_label."""
    out = []
    if not folder.exists():
        return out
    for d in sorted(folder.iterdir()):
        if not d.is_dir():
            continue
        mfile = d / "metrics.json"
        if not mfile.exists():
            continue
        try:
            row = json.loads(mfile.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [!] skipping {mfile} ({type(e).__name__}: {e})")
            continue
        # Skip experiments that errored out with no real metrics
        if "error" in row and not row.get("macro_auc"):
            continue
        row["agent"] = agent_label
        out.append(row)
    return out


print("\n=== Reading prior agents' experiments ===")
ALL_PRIOR: list[dict] = []
for folder in OTHER_AGENT_DIRS:
    label = folder.name.split("_")[1]     # 'regular' / 'eda' / 'creative'
    rows  = load_agent_experiments(folder, label)
    print(f"  {folder.name}: {len(rows)} experiments")
    ALL_PRIOR.extend(rows)

if not ALL_PRIOR:
    raise RuntimeError(
        "No prior experiments found in "
        f"{[str(d) for d in OTHER_AGENT_DIRS]}. "
        "Run the regular/eda/creative agents first."
    )


# Save a combined CSV across all prior agents — useful in the meta prompt,
# useful for the report, and useful to eyeball before training starts.
PRIOR_DF_COLS = [
    "agent", "id", "loss", "aug", "lr", "optimizer", "schedule",
    "geo_scale_km", "geo_bucket", "macro_auc", "weighted_macro_auc",
    "aves_auc", "amphibia_auc", "insecta_auc", "mammalia_auc",
    "pruned", "epochs_run", "train_time_s", "params",
    "py_path", "weights_path", "reason",
]
PRIOR_DF = pd.DataFrame(ALL_PRIOR)
for c in PRIOR_DF_COLS:
    if c not in PRIOR_DF.columns:
        PRIOR_DF[c] = np.nan
PRIOR_DF = PRIOR_DF[PRIOR_DF_COLS].copy()
PRIOR_CSV = EXPERIMENTS_DIR / "meta_prior_experiments.csv"
PRIOR_DF.to_csv(PRIOR_CSV, index=False)
print(f"  combined table → {PRIOR_CSV}  ({len(PRIOR_DF)} rows)")


def pick_global_winner(experiments: list[dict]) -> dict:
    """Best across all agents by weighted_macro_auc.
    Tiebreak: unweighted macro_auc, then shorter train_time_s."""
    valid = [e for e in experiments
             if e.get("weighted_macro_auc") is not None
             and not e.get("pruned")]
    if not valid:
        raise RuntimeError("No non-pruned experiments with valid AUCs.")
    valid.sort(key=lambda e: (e.get("weighted_macro_auc", 0),
                              e.get("macro_auc", 0),
                              -e.get("train_time_s", 1e9)),
               reverse=True)
    return valid[0]


WINNER = pick_global_winner(ALL_PRIOR)
WINNER_WEIGHTS = Path(WINNER["weights_path"])

print(f"\n=== Global winner across all prior agents ===")
print(f"  agent          = {WINNER['agent']}")
print(f"  exp id         = {WINNER['id']}")
print(f"  config         = loss={WINNER['loss']}  aug={WINNER['aug']}  "
      f"lr={float(WINNER['lr']):g}  opt={WINNER.get('optimizer', '?')}  "
      f"sched={WINNER.get('schedule', '?')}  "
      f"geo={WINNER.get('geo_scale_km', 'n/a')}")
print(f"  macro_auc(unw) = {WINNER['macro_auc']:.4f}")
print(f"  macro_auc(w)   = {WINNER['weighted_macro_auc']:.4f}")
print(f"  aves_auc       = {WINNER['aves_auc']:.3f}")
print(f"  weights        = {WINNER_WEIGHTS}")
print(f"  → all meta experiments will fine-tune from these weights")

if not WINNER_WEIGHTS.exists():
    raise FileNotFoundError(
        f"Winner weights file {WINNER_WEIGHTS} doesn't exist. The "
        f"experiment folder may have been moved or its model.keras deleted."
    )


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2.5: Measure the winner's val_auc once → pruning threshold
# ─────────────────────────────────────────────────────────────────────────
STEPS_PER_EPOCH   = math.ceil(len(train_rows) / cfg.batch_size)
TOTAL_TRAIN_STEPS = STEPS_PER_EPOCH * EPOCHS_PER_EXPERIMENT


def _measure_winner_val_auc() -> float:
    print(f"\n=== Measuring winner's val_auc for pruning threshold ===")
    keras.backend.clear_session()
    m = keras.models.load_model(WINNER_WEIGHTS, compile=False)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),     # unused, just to attach metrics
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )
    val_ds = AudioDataset(val_rows, batch_size=cfg.batch_size, shuffle=False)
    _, val_auc = m.evaluate(val_ds, verbose=0)
    keras.backend.clear_session()
    return float(val_auc)


try:
    WINNER_VAL_AUC  = _measure_winner_val_auc()
    PRUNE_THRESHOLD = WINNER_VAL_AUC - PRUNE_MARGIN
    print(f"  winner val_auc  = {WINNER_VAL_AUC:.4f}")
    print(f"  prune threshold = {PRUNE_THRESHOLD:.4f}  "
          f"(check after epoch {PRUNE_AT_EPOCH_INDEX + 1})")
except Exception as e:
    WINNER_VAL_AUC  = float("nan")
    PRUNE_THRESHOLD = 0.20
    print(f"  [!] could not measure winner val_auc "
          f"({type(e).__name__}: {e}); fallback threshold = {PRUNE_THRESHOLD}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Class-frequency + per-row geographic distance (mirrors EDA)
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


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = (np.sin(dp / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2)
    return 2.0 * R * np.arcsin(np.sqrt(a))


def precompute_distance_km(rows: pd.DataFrame) -> np.ndarray:
    lat = rows["latitude"].to_numpy(dtype=np.float64)
    lon = rows["longitude"].to_numpy(dtype=np.float64)
    out = np.full_like(lat, np.nan, dtype=np.float64)
    mask = ~np.isnan(lat) & ~np.isnan(lon)
    out[mask] = _haversine_km(lat[mask], lon[mask],
                              PANTANAL_LAT, PANTANAL_LON)
    return out


TRAIN_DIST_KM = precompute_distance_km(train_rows)


def compute_sample_weights(geo_scale_km: float | None) -> np.ndarray:
    n = len(TRAIN_DIST_KM)
    if geo_scale_km is None:
        return np.ones(n, dtype=np.float32)
    w = np.exp(-TRAIN_DIST_KM / float(geo_scale_km))
    w = np.where(np.isnan(w), 1.0, w)
    w = np.maximum(w, 0.05)
    return w.astype(np.float32)


def geo_bucket(geo_scale_km: float | None) -> str:
    if geo_scale_km is None or (isinstance(geo_scale_km, float) and math.isnan(geo_scale_km)):
        return "none"
    b = round(float(geo_scale_km) / GEO_BUCKET_KM) * GEO_BUCKET_KM
    return f"{b:g}"


# Backfill geo_bucket on the prior DataFrame so the facts block can use it.
PRIOR_DF["geo_bucket"] = PRIOR_DF["geo_scale_km"].apply(geo_bucket)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: Past-meta-experiments helpers + runtime facts block
# ─────────────────────────────────────────────────────────────────────────
def load_past_meta_experiments() -> list[dict]:
    past = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        mfile = d / "metrics.json"
        if mfile.exists():
            past.append(json.loads(mfile.read_text(encoding="utf-8")))
    return past


def config_key(loss_name, aug_name, lr, optimizer_name, schedule_name,
               geo_scale_km) -> str:
    return (f"{loss_name}|{aug_name}|{lr:g}|{optimizer_name}|"
            f"{schedule_name}|geo={geo_bucket(geo_scale_km)}")


def summarise_experiments(rows: list[dict],
                          include_agent: bool = True) -> str:
    """Compact one-line-per-experiment table for the LLM prompt."""
    if not rows:
        return "(no experiments)"
    lines = []
    for e in rows:
        pruned = " [PRUNED]" if e.get("pruned") else ""
        geo = e.get("geo_scale_km")
        geo_s = "none  " if geo is None or (isinstance(geo, float) and math.isnan(geo)) \
                else f"{float(geo):>4.0f}km"
        agent_tag = f"[{e.get('agent', '?'):<8}] " if include_agent else ""
        lines.append(
            f"- {agent_tag}id={e.get('id', '?'):>2}  "
            f"loss={e.get('loss', '?'):<14} "
            f"aug={e.get('aug', '?'):<15} "
            f"lr={float(e.get('lr', 0)):g}  "
            f"opt={e.get('optimizer', '?'):<13} "
            f"sched={e.get('schedule', '?'):<13} "
            f"geo={geo_s}  →  "
            f"macro(unw)={float(e.get('macro_auc', 0)):.4f}  "
            f"macro(w)={float(e.get('weighted_macro_auc', 0)):.4f}  "
            f"aves={float(e.get('aves_auc', 0)):.3f}{pruned}"
        )
    return "\n".join(lines)


def compute_facts_block(prior_df: pd.DataFrame,
                        meta_past: list[dict]) -> str:
    """Runtime arithmetic over the combined prior + meta experiments.
    Pure counts/means — no editorialising. Helps the LLM not have to
    aggregate the table itself."""
    df = prior_df.copy()
    if meta_past:
        meta_df = pd.DataFrame(meta_past)
        meta_df["agent"] = "meta"
        for c in df.columns:
            if c not in meta_df.columns:
                meta_df[c] = np.nan
        df = pd.concat([df, meta_df[df.columns]], ignore_index=True)

    # Drop pruned + null-AUC rows for ranking statistics
    valid = df[(~df["pruned"].astype(bool, errors="ignore")) &
               df["weighted_macro_auc"].notna()].copy()
    if valid.empty:
        return "(no valid experiments to summarise)"

    valid = valid.sort_values("weighted_macro_auc", ascending=False)
    top_k = min(10, len(valid))
    top   = valid.head(top_k)

    bullets: list[str] = []
    bullets.append(
        f"- Total experiments analysed: {len(valid)} valid "
        f"(plus {(df['pruned'].astype(bool, errors='ignore')).sum()} pruned)."
    )

    overall_best = valid.iloc[0]
    bullets.append(
        f"- Overall best: weighted_macro_auc={overall_best['weighted_macro_auc']:.4f}, "
        f"by agent={overall_best['agent']}, "
        f"config: loss={overall_best['loss']} aug={overall_best['aug']} "
        f"lr={float(overall_best['lr']):g} opt={overall_best['optimizer']} "
        f"sched={overall_best['schedule']} geo={overall_best['geo_bucket']}."
    )

    # Per-axis frequency in top-k + mean weighted_macro_auc when used
    for axis in ["loss", "aug", "optimizer", "schedule", "geo_bucket"]:
        counts = top[axis].value_counts()
        if counts.empty:
            continue
        top_val   = counts.index[0]
        top_count = int(counts.iloc[0])
        # Mean weighted_macro_auc across ALL valid experiments using top_val
        mean_when = valid.loc[valid[axis] == top_val,
                              "weighted_macro_auc"].mean()
        bullets.append(
            f"- {axis}: most frequent in top {top_k} is "
            f"'{top_val}' ({top_count}/{top_k}); "
            f"mean weighted_macro_auc when used = {mean_when:.4f}."
        )

    # Biggest single-experiment lift over each agent's worst result
    if "agent" in valid.columns:
        for agent in valid["agent"].unique():
            sub = valid[valid["agent"] == agent]
            if len(sub) < 2:
                continue
            lift = sub["weighted_macro_auc"].max() - sub["weighted_macro_auc"].min()
            bullets.append(
                f"- agent={agent}: within-agent spread = "
                f"{lift:.4f} (best - worst), best config "
                f"loss={sub.iloc[0]['loss']} aug={sub.iloc[0]['aug']}."
            )

    return "\n".join(bullets)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: LLM prompts
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous ML research agent for the BirdCLEF+ 2026 audio classification task.

THIS IS THE META-AGENT. Three prior agents (Regular, EDA, Creative) have
already explored the search space. Your job is to REFINE the global
winner: every experiment you propose will fine-tune from the global
winner's weights, not from scratch and not from the original CRNN
baseline.

You will see, every turn:
  1. A FACTS BLOCK with arithmetic counts/means over the prior runs
     (pre-computed for you — these are correct counts you can trust).
  2. The full PRIOR EXPERIMENTS table from all three agents, sorted by
     weighted_macro_auc. Each row is tagged with which agent ran it.
  3. The META EXPERIMENTS run so far in this session.
  4. The GLOBAL WINNER config (the starting point of your fine-tuning).

Use these to reason about WHICH AXIS to vary and WHY before proposing.
Surface your reasoning in REASON — it will be archived for the report.

==========================================================================
MENU — pick exactly one option from each axis
==========================================================================
LOSS:
  - plain_bce       : standard binary cross-entropy
  - weighted_bce    : BCE weighted by inverse class frequency
  - focal_g1.0      : focal loss, gamma=1.0
  - focal_g2.0      : focal loss, gamma=2.0
  - focal_g3.0      : focal loss, gamma=3.0

AUGMENTATION:
  - none            : no augmentation
  - specaugment     : random time/frequency masking
  - mixup           : linear combination of two examples
  - background_mix  : low-volume second sample as background

LEARNING RATE (initial):
  - 1e-3            : standard
  - 5e-4            : slightly lower, more stable
  - 1e-4            : much lower, gentler fine-tuning

OPTIMIZER:
  - adam            : standard Adam
  - adamw           : Adam with weight decay 1e-4
  - sgd_momentum    : SGD with momentum=0.9
  - rmsprop         : RMSprop

SCHEDULE:
  - constant        : keep LR fixed (ReduceLROnPlateau active)
  - cosine_decay    : decay from initial LR toward 0
  - exp_decay       : multiply LR by 0.9 each epoch

GEO_SCALE_KM (geographic sample weighting toward the Pantanal):
  Each training sample is weighted by exp(-distance_to_Pantanal / scale_km).
  Soundscape rows + rows with unknown coords always get weight 1.0.
  Pick `none` to disable, or any number in [200, 10000].
  Examples: 500 (aggressive), 2000 (moderate), 5000 (mild).

==========================================================================
RULES
==========================================================================
  1. Pick exactly ONE option from each of the SIX axes.
  2. Do NOT pick a combination already run by any agent (geo values
     within 100 km of a previous run count as the same config).
  3. In REASON, briefly note WHICH PATTERN in the facts block or table
     motivated your choice. This trace is what the report will cite.
  4. Respond in EXACTLY this format, with no extra text:

CHOICE_LOSS:      <plain_bce | weighted_bce | focal_g1.0 | focal_g2.0 | focal_g3.0>
CHOICE_AUG:       <none | specaugment | mixup | background_mix>
CHOICE_LR:        <1e-3 | 5e-4 | 1e-4>
CHOICE_OPTIMIZER: <adam | adamw | sgd_momentum | rmsprop>
CHOICE_SCHEDULE:  <constant | cosine_decay | exp_decay>
CHOICE_GEO_KM:    <none | a number in [200, 10000]>
REASON: <one short paragraph: cite the specific facts/rows you used>
"""


def build_user_prompt(meta_past: list[dict]) -> str:
    facts = compute_facts_block(PRIOR_DF, meta_past)

    # Show the top 15 prior experiments + ALL meta experiments
    # (the meta history is short by construction).
    valid_prior = [e for e in ALL_PRIOR
                   if e.get("weighted_macro_auc") is not None]
    valid_prior.sort(key=lambda e: e["weighted_macro_auc"], reverse=True)
    show_prior = valid_prior[:15]

    winner_line = (
        f"  agent={WINNER['agent']}, exp id={WINNER['id']}, "
        f"weighted_macro_auc={WINNER['weighted_macro_auc']:.4f}, "
        f"config: loss={WINNER['loss']} aug={WINNER['aug']} "
        f"lr={float(WINNER['lr']):g} opt={WINNER.get('optimizer', '?')} "
        f"sched={WINNER.get('schedule', '?')} "
        f"geo={geo_bucket(WINNER.get('geo_scale_km'))}"
    )

    return f"""GLOBAL WINNER (your fine-tuning starting point):
{winner_line}

FACTS (computed at runtime from the metrics.json files):
{facts}

TOP {len(show_prior)} PRIOR EXPERIMENTS (sorted by weighted_macro_auc):
{summarise_experiments(show_prior, include_agent=True)}

META EXPERIMENTS this session ({len(meta_past)} so far):
{summarise_experiments(meta_past, include_agent=False)}

Pick the next experiment. It must be a configuration we have NOT
already run (across all four agents, with geo bucketed to 100 km)."""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Parse LLM response
# ─────────────────────────────────────────────────────────────────────────
def parse_llm_response(response: str
                       ) -> tuple[str, str, float, str, str, float | None, str]:
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
                f"Geo '{geo_scale_km}' outside [{GEO_SCALE_MIN}, {GEO_SCALE_MAX}]"
            )

    reason_m = re.search(r"REASON\s*:\s*(.+?)(?=\n\s*CHOICE_|\Z)",
                         response, re.IGNORECASE | re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else "(no reason given)"
    return loss, aug, lr, opt, sch, geo_scale_km, reason


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Generate the experiment .py (loads from WINNER's weights, not CRNN)
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENT_TEMPLATE = '''"""
Experiment {exp_id} (Meta Agent)
    starts from   = {winner_summary}  (loaded from {winner_path})
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
GEO_SCALE_KM       = {geo_scale_repr}


# ── Warm-start from the GLOBAL WINNER (not from the CRNN baseline) ──────
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH, compile=False)


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
''', "WeightedBCE(POS_WEIGHTS)")
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
''', "FocalLoss()")
    raise ValueError(f"Unknown loss: {loss_name}")


def aug_block_for(aug_name: str) -> str:
    if aug_name == "none":
        return '''def augment_batch(xs, ys):
    return xs, ys'''
    if aug_name == "specaugment":
        return '''def augment_batch(xs, ys):
    """SpecAugment: random time/frequency masking."""
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
    """Mixup: linear combination of two examples."""
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
    """Add a low-volume second sample as background."""
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys'''
    raise ValueError(f"Unknown augmentation: {aug_name}")


def optimizer_schedule_block_for(optimizer_name, schedule_name) -> str:
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


def generate_experiment_file(exp_id, loss_name, aug_name, lr,
                             optimizer_name, schedule_name,
                             geo_scale_km, reason) -> Path:
    loss_code, factory = loss_block_for(loss_name)
    aug_code  = aug_block_for(aug_name)
    opt_sched = optimizer_schedule_block_for(optimizer_name, schedule_name)
    geo_repr  = "None" if geo_scale_km is None else f"{float(geo_scale_km):g}"
    geo_tag   = geo_bucket(geo_scale_km)
    exp_dir = (EXPERIMENTS_DIR
               / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}"
                 f"_{optimizer_name}_{schedule_name}_geo{geo_tag}")
    exp_dir.mkdir(exist_ok=True)
    py_path = exp_dir / "model.py"
    winner_path_str = str(WINNER_WEIGHTS).replace("\\", "/")
    winner_summary  = (f"{WINNER['agent']}'s exp {WINNER['id']} "
                       f"(weighted_auc={WINNER['weighted_macro_auc']:.4f})")
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
        winner_summary=winner_summary,
        winner_path=winner_path_str,
        steps_per_epoch=STEPS_PER_EPOCH,
        total_train_steps=TOTAL_TRAIN_STEPS,
        loss_code=loss_code,
        loss_factory_call=factory,
        optimizer_schedule_code=opt_sched,
        aug_code=aug_code,
    ), encoding="utf-8")
    return py_path


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8: Run one experiment + sample-weight pipeline
# ─────────────────────────────────────────────────────────────────────────
class WeightedAudioDataset(keras.utils.PyDataset):
    """Same as EDA's wrapper. Passes (xs, ys, sample_weights) through fit."""

    def __init__(self, base_ds, sample_weights, augment_fn, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("use_multiprocessing", False)
        kwargs.setdefault("max_queue_size", 8)
        super().__init__(**kwargs)
        self.base = base_ds
        self.sample_weights = sample_weights
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
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


def _compute_aucs(model, val_ds):
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        batch = val_ds[i]
        xb, yb = batch[0], batch[1]
        all_preds.append(model.predict(xb, verbose=0))
        all_true.append(yb)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    per_class_auc = {}
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


def _summarise_aucs(per_class_auc):
    valid = [v for v in per_class_auc.values() if v is not None]
    macro = float(np.mean(valid)) if valid else 0.0
    taxon_to_aucs: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None:
            continue
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        taxon_to_aucs.setdefault(taxon, []).append(auc)
    taxon_aucs = {t: float(np.mean(v)) for t, v in taxon_to_aucs.items()}
    weighted_sum = weight_total = 0.0
    for t, mean_auc in taxon_aucs.items():
        w = TAXON_WEIGHTS.get(t, 0.0)
        weighted_sum += w * mean_auc
        weight_total += w
    weighted = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    return macro, weighted, taxon_aucs, len(valid)


class PruneIfWorseCallback(keras.callbacks.Callback):
    """Stop after epoch PRUNE_AT_EPOCH_INDEX+1 if val_auc < threshold.
    (0-indexed: epoch == PRUNE_AT_EPOCH_INDEX means that epoch just ended.)"""

    def __init__(self, threshold, monitor="val_auc",
                 check_at_epoch_index=PRUNE_AT_EPOCH_INDEX):
        super().__init__()
        self.threshold  = threshold
        self.monitor    = monitor
        self.check_at   = check_at_epoch_index
        self.pruned     = False

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.check_at:
            val = float((logs or {}).get(self.monitor, 0.0))
            if val < self.threshold:
                print(f"\n  [PRUNE] {self.monitor}={val:.4f} < threshold "
                      f"{self.threshold:.4f} after epoch {epoch + 1} — stopping early")
                self.model.stop_training = True
                self.pruned = True


def run_experiment(exp_id, loss_name, aug_name, lr,
                   optimizer_name, schedule_name, geo_scale_km, reason):
    geo_label = "off" if geo_scale_km is None else f"{geo_scale_km:.0f}km"
    print(f"\n{'=' * 72}")
    print(f" META EXPERIMENT {exp_id}: loss={loss_name}  aug={aug_name}  "
          f"lr={lr:g}  opt={optimizer_name}  sched={schedule_name}  "
          f"geo={geo_label}")
    print(f"{'=' * 72}")
    print(f" reason: {reason}\n")

    py_path = generate_experiment_file(exp_id, loss_name, aug_name, lr,
                                       optimizer_name, schedule_name,
                                       geo_scale_km, reason)
    exp_dir = py_path.parent

    spec = importlib.util.spec_from_file_location(f"meta_exp_{exp_id}", py_path)
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

    sample_weights = compute_sample_weights(mod.get_geo_scale_km())
    if geo_scale_km is not None:
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
        "warm_start_from":     str(WINNER_WEIGHTS),
        "warm_start_agent":    WINNER["agent"],
        "warm_start_exp_id":   WINNER["id"],
        "winner_weighted_macro_auc": WINNER["weighted_macro_auc"],
        "winner_val_auc":      WINNER_VAL_AUC,
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
    delta_w = weighted_macro_auc - WINNER["weighted_macro_auc"]
    delta_u = macro_auc - WINNER["macro_auc"]
    sign_w  = "+" if delta_w >= 0 else ""
    sign_u  = "+" if delta_u >= 0 else ""
    print(f"\n  RESULT: {status}macro_auc(unw)={macro_auc:.4f} "
          f"({sign_u}{delta_u:.4f} vs winner)  "
          f"macro_auc(w)={weighted_macro_auc:.4f} "
          f"({sign_w}{delta_w:.4f} vs winner)  "
          f"aves={taxon_aucs.get('Aves', 0):.3f}")
    return metrics


# %% ──────────────────────────────────────────────────────────────────────
# Cell 9: Agent loop
# ─────────────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
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
    print(f"\n{'#' * 72}")
    print(f"# META AGENT  —  {n_iterations} iterations  —  LLM={LLM_MODEL}")
    print(f"# Output dir: {EXPERIMENTS_DIR}")
    print(f"# Starting point: {WINNER['agent']}'s exp {WINNER['id']}  "
          f"(weighted_macro_auc={WINNER['weighted_macro_auc']:.4f})")
    print(f"{'#' * 72}")

    for iteration in range(n_iterations):
        meta_past = load_past_meta_experiments()
        exp_id    = (max((e["id"] for e in meta_past), default=0)) + 1

        # Duplicate detection: ALL agents, including meta itself
        tried_keys = set()
        for e in ALL_PRIOR + meta_past:
            try:
                tried_keys.add(config_key(
                    e["loss"], e["aug"], float(e["lr"]),
                    e.get("optimizer", "adam"),
                    e.get("schedule", "constant"),
                    e.get("geo_scale_km"),
                ))
            except Exception:
                pass

        print(f"\n--- iteration {iteration + 1}/{n_iterations} "
              f"(meta exp id {exp_id}, "
              f"{len(tried_keys)} unique configs across all agents) ---")

        loss_name, aug_name, lr = None, None, None
        optimizer_name, schedule_name, geo_scale_km, reason = None, None, None, None
        user_prompt = build_user_prompt(meta_past)
        for attempt in range(3):
            try:
                response = call_llm(SYSTEM_PROMPT, user_prompt)
                print(f"LLM raw response (attempt {attempt + 1}):\n"
                      f"{response.strip()}\n")
                c_loss, c_aug, c_lr, c_opt, c_sch, c_geo, c_reason = \
                    parse_llm_response(response)
                if config_key(c_loss, c_aug, c_lr, c_opt, c_sch, c_geo) in tried_keys:
                    geo_str = "none" if c_geo is None else f"{c_geo:.0f}km"
                    print(f"  [!] LLM picked already-tried combo "
                          f"({c_loss}, {c_aug}, lr={c_lr:g}, {c_opt}, "
                          f"{c_sch}, geo={geo_str}). Re-prompting.")
                    user_prompt = (
                        build_user_prompt(meta_past)
                        + "\n\nIMPORTANT: You just picked an already-tried "
                          "combination. Pick a DIFFERENT, untried one. "
                          "Vary at least one axis."
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
            # Fallback: walk a sensible default order looking for untried.
            print("  No valid LLM response after 3 attempts; searching for "
                  "first untried sensible config...")
            picked = None
            for geo in [2000.0, 5000.0, 1000.0, None, 500.0, 3000.0]:
                for loss in LOSS_OPTIONS:
                    for aug in AUG_OPTIONS:
                        for o in OPTIMIZER_OPTIONS:
                            for s in SCHEDULE_OPTIONS:
                                for r in LR_OPTIONS:
                                    k = config_key(loss, aug, r, o, s, geo)
                                    if k not in tried_keys:
                                        picked = (loss, aug, r, o, s, geo)
                                        break
                                if picked: break
                            if picked: break
                        if picked: break
                    if picked: break
                if picked: break
            if not picked:
                print("  All combinations tried. Stopping.")
                break
            loss_name, aug_name, lr, optimizer_name, schedule_name, geo_scale_km = picked
            reason = "(fallback: LLM unavailable or unhelpful — first untried)"

        try:
            run_experiment(exp_id, loss_name, aug_name, lr,
                           optimizer_name, schedule_name, geo_scale_km, reason)
        except Exception as e:
            print(f"[!] meta experiment {exp_id} crashed: "
                  f"{type(e).__name__}: {e}")
            geo_tag = geo_bucket(geo_scale_km)
            failure_dir = (
                EXPERIMENTS_DIR
                / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}"
                  f"_{optimizer_name}_{schedule_name}_geo{geo_tag}_FAILED"
            )
            failure_dir.mkdir(exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "id": exp_id, "agent_name": AGENT_NAME,
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
    print(f"\n{'#' * 72}\n# META AGENT — FINAL SUMMARY\n{'#' * 72}")
    final = load_past_meta_experiments()
    if not final:
        print("No meta experiments completed.")
        return

    ranked = sorted(final, key=lambda e: e.get("weighted_macro_auc", 0),
                    reverse=True)
    print(f"\n{'rank':<5}{'id':<4}{'loss':<14}{'aug':<15}{'lr':<8}"
          f"{'opt':<14}{'sched':<14}{'geo':<8}"
          f"{'macro(uw)':>10}{'macro(w)':>10}{'Δw':>8}{'aves':>8}")
    for rank, e in enumerate(ranked, 1):
        geo = e.get("geo_scale_km")
        geo_s = "none" if geo is None else f"{float(geo):.0f}km"
        delta = e.get("weighted_macro_auc", 0) - WINNER["weighted_macro_auc"]
        print(f"{rank:<5}{e['id']:<4}{e['loss']:<14}{e['aug']:<15}"
              f"{float(e['lr']):<8.0e}"
              f"{e.get('optimizer','?'):<14}{e.get('schedule','?'):<14}"
              f"{geo_s:<8}"
              f"{e.get('macro_auc', 0):>10.4f}"
              f"{e.get('weighted_macro_auc', 0):>10.4f}"
              f"{('+' if delta>=0 else ''):>1}{delta:>7.4f}"
              f"{e.get('aves_auc', 0):>8.3f}")

    best_meta = ranked[0]
    if best_meta["weighted_macro_auc"] > WINNER["weighted_macro_auc"]:
        global_best = best_meta
        print(f"\nNEW GLOBAL BEST: meta exp {best_meta['id']}, "
              f"weighted_macro_auc={best_meta['weighted_macro_auc']:.4f}, "
              f"+{best_meta['weighted_macro_auc'] - WINNER['weighted_macro_auc']:.4f} "
              f"vs the prior winner ({WINNER['agent']} exp {WINNER['id']}).")
    else:
        global_best = WINNER
        print(f"\nNo meta experiment beat the prior winner. "
              f"Best remains {WINNER['agent']}'s exp {WINNER['id']} at "
              f"weighted_macro_auc={WINNER['weighted_macro_auc']:.4f}. "
              f"Best meta attempt: exp {best_meta['id']} at "
              f"{best_meta['weighted_macro_auc']:.4f} "
              f"({best_meta['weighted_macro_auc'] - WINNER['weighted_macro_auc']:+.4f}).")

    plot_progress(final)
    write_report_analysis(final, global_best)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 10: Progress plot
# ─────────────────────────────────────────────────────────────────────────
def plot_progress(meta_experiments, out_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping progress plot")
        return

    successful = [e for e in meta_experiments
                  if e.get("macro_auc") is not None and "error" not in e]
    if not successful:
        return
    successful.sort(key=lambda e: e["id"])

    iters    = [e["id"]                   for e in successful]
    macro_uw = [e["macro_auc"]            for e in successful]
    macro_w  = [e["weighted_macro_auc"]   for e in successful]
    aves     = [e.get("aves_auc", np.nan) for e in successful]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, macro_uw, marker="o", lw=2, color="steelblue",
            label="macro AUC (unweighted)")
    ax.plot(iters, macro_w,  marker="s", lw=2, color="darkorange",
            label="macro AUC (weighted by taxon)")
    ax.plot(iters, aves, marker="^", lw=2, color="forestgreen",
            label="Aves/Bird AUC")

    ax.axhline(WINNER["weighted_macro_auc"], ls="--", color="black", alpha=0.5,
               label=f"Prior winner ({WINNER['agent']}): "
                     f"{WINNER['weighted_macro_auc']:.4f}")

    best_w = max(macro_w)
    ax.axhline(best_w, ls=":", color="darkorange", alpha=0.4,
               label=f"Best meta weighted: {best_w:.4f}")

    ax.set_xlabel("Meta experiment iteration")
    ax.set_ylabel("Validation macro AUC")
    ax.set_title(
        f"Meta Agent — {len(successful)} experiment"
        f"{'s' if len(successful) != 1 else ''} "
        f"(starting from {WINNER['agent']}'s winner)"
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
# Cell 11: Report analysis — per-taxon AUC + confusion patterns
# ─────────────────────────────────────────────────────────────────────────
def _percentile_summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n":       int(a.size),
        "mean":    float(a.mean()),
        "median":  float(np.median(a)),
        "p10":     float(np.percentile(a, 10)),
        "p90":     float(np.percentile(a, 90)),
        "min":     float(a.min()),
        "max":     float(a.max()),
    }


def write_report_analysis(meta_experiments, global_best) -> None:
    """Generate report_analysis.md from the global best model.
    Includes per-taxon AUC breakdown and top-10 confusion patterns."""
    out_path = EXPERIMENTS_DIR / "report_analysis.md"
    print(f"\n=== Writing report analysis to {out_path} ===")

    # Cross-agent best per agent
    best_per_agent: dict[str, dict] = {}
    for e in ALL_PRIOR + meta_experiments:
        a = e.get("agent", e.get("agent_name", "meta"))
        wma = e.get("weighted_macro_auc", -1)
        if wma is None: continue
        if a not in best_per_agent or wma > best_per_agent[a]["weighted_macro_auc"]:
            best_per_agent[a] = e

    # Load the global best model and run a single validation pass —
    # we need predictions for the per-taxon and confusion sections.
    print(f"  Loading {global_best.get('weights_path')} for analysis...")
    keras.backend.clear_session()
    try:
        model = keras.models.load_model(global_best["weights_path"], compile=False)
    except Exception as e:
        print(f"  [!] couldn't load global_best weights: {e}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Report analysis — generation failed\n\n"
                    f"Could not load {global_best.get('weights_path')}: "
                    f"{e}\n")
        return

    val_ds = AudioDataset(val_rows, batch_size=cfg.batch_size, shuffle=False)
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        xb, yb = val_ds[i]
        all_preds.append(model.predict(xb, verbose=0))
        all_true.append(yb)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    keras.backend.clear_session()

    # Per-class AUC + per-taxon breakdown
    per_class_auc = {}
    for idx, label in enumerate(LABELS):
        if y_true[:, idx].sum() == 0:
            per_class_auc[label] = None
            continue
        try:
            per_class_auc[label] = float(
                roc_auc_score(y_true[:, idx], y_pred[:, idx])
            )
        except Exception:
            per_class_auc[label] = None

    by_taxon: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None: continue
        t = SPECIES_TO_TAXON.get(label, "Unknown")
        by_taxon.setdefault(t, []).append(auc)
    taxon_summary = {t: _percentile_summary(vs) for t, vs in by_taxon.items()}

    # Confusion analysis: most confidently-wrong species
    # - Top false positives:  per class, mean predicted prob on rows
    #                         where y_true == 0. Highest means model
    #                         keeps shouting this species when absent.
    # - Top false negatives:  per class, mean predicted prob on rows
    #                         where y_true == 1. Lowest means model
    #                         keeps missing it when present.
    n_val = y_true.shape[0]
    fp_scores = []
    fn_scores = []
    for idx, label in enumerate(LABELS):
        neg_mask = y_true[:, idx] == 0
        pos_mask = y_true[:, idx] == 1
        if neg_mask.sum() >= 10:
            fp_scores.append({
                "label":  label,
                "taxon":  SPECIES_TO_TAXON.get(label, "Unknown"),
                "mean_prob_when_absent": float(y_pred[neg_mask, idx].mean()),
                "n_negatives": int(neg_mask.sum()),
                "n_positives": int(pos_mask.sum()),
            })
        if pos_mask.sum() >= 3:
            fn_scores.append({
                "label":  label,
                "taxon":  SPECIES_TO_TAXON.get(label, "Unknown"),
                "mean_prob_when_present": float(y_pred[pos_mask, idx].mean()),
                "n_positives": int(pos_mask.sum()),
            })
    fp_top = sorted(fp_scores,
                    key=lambda r: r["mean_prob_when_absent"],
                    reverse=True)[:10]
    fn_top = sorted(fn_scores,
                    key=lambda r: r["mean_prob_when_present"])[:10]

    # LLM reasoning trace
    reasons = [(e["id"], e.get("reason", "")) for e in meta_experiments
               if e.get("reason")]

    # ── Compose the markdown ────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# BirdCLEF+ 2026 — Meta-agent report analysis\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"LLM used by all agents: `{LLM_MODEL}`\n")

    lines.append("## 1. Global best model\n")
    lines.append(f"- **Source agent:** `{global_best.get('agent', global_best.get('agent_name', '?'))}`")
    lines.append(f"- **Experiment id:** `{global_best['id']}`")
    lines.append(f"- **Config:** loss=`{global_best['loss']}` aug=`{global_best['aug']}` "
                 f"lr=`{float(global_best['lr']):g}` "
                 f"optimizer=`{global_best.get('optimizer', '?')}` "
                 f"schedule=`{global_best.get('schedule', '?')}` "
                 f"geo=`{geo_bucket(global_best.get('geo_scale_km'))}`")
    lines.append(f"- **macro_auc (unweighted):** {global_best['macro_auc']:.4f}")
    lines.append(f"- **macro_auc (weighted by taxon):** {global_best['weighted_macro_auc']:.4f}")
    lines.append(f"- **aves_auc:** {global_best['aves_auc']:.4f}")
    lines.append(f"- **Weights:** `{global_best.get('weights_path')}`\n")

    lines.append("## 2. Cross-agent comparison\n")
    lines.append("Best experiment per agent, ranked by weighted_macro_auc:\n")
    lines.append("| Agent | Exp id | Loss | Aug | LR | Optim | Sched | Geo | macro(unw) | macro(w) | aves |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for agent in sorted(best_per_agent.keys(),
                        key=lambda a: -best_per_agent[a]["weighted_macro_auc"]):
        e = best_per_agent[agent]
        lines.append(
            f"| {agent} | {e['id']} | {e['loss']} | {e['aug']} | "
            f"{float(e['lr']):g} | {e.get('optimizer','?')} | "
            f"{e.get('schedule','?')} | {geo_bucket(e.get('geo_scale_km'))} | "
            f"{e['macro_auc']:.4f} | {e['weighted_macro_auc']:.4f} | "
            f"{e['aves_auc']:.4f} |"
        )
    lines.append("")

    lines.append("## 3. Per-taxon AUC breakdown (global best model)\n")
    lines.append("Each taxonomic class's per-species AUC distribution on "
                 "the held-out validation set:\n")
    lines.append("| Taxon | N species evaluated | Mean | Median | p10 | p90 | Min | Max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in sorted(taxon_summary.keys(),
                    key=lambda x: -taxon_summary[x].get("mean", 0)):
        s = taxon_summary[t]
        if s["n"] == 0:
            lines.append(f"| {t} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {t} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | "
            f"{s['p10']:.3f} | {s['p90']:.3f} | {s['min']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")
    lines.append("> Reading guide: a *high mean with a low p10* signals "
                 "uneven coverage within the taxon — most species are "
                 "well-modeled but some are quietly failing. *Low max* means "
                 "the model is uniformly bad across that taxon.\n")

    lines.append("## 4. Top-10 confidently-wrong species\n")
    lines.append("**False positives** (model shouts the species when it isn't there):\n")
    lines.append("| Rank | Species | Taxon | Mean pred prob (when absent) | N negatives | N positives in val |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(fp_top, 1):
        lines.append(
            f"| {i} | `{r['label']}` | {r['taxon']} | "
            f"{r['mean_prob_when_absent']:.3f} | {r['n_negatives']} | "
            f"{r['n_positives']} |"
        )
    lines.append("")
    lines.append("**False negatives** (model misses the species when it is present):\n")
    lines.append("| Rank | Species | Taxon | Mean pred prob (when present) | N positives in val |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(fn_top, 1):
        lines.append(
            f"| {i} | `{r['label']}` | {r['taxon']} | "
            f"{r['mean_prob_when_present']:.3f} | {r['n_positives']} |"
        )
    lines.append("")
    lines.append("> Note: Both lists are restricted to classes with enough "
                 "validation support to be meaningful (≥10 negatives or "
                 "≥3 positives). Mean predicted probability is computed on "
                 "the validation set's raw sigmoid outputs.\n")

    lines.append("## 5. Meta-agent reasoning trace\n")
    if reasons:
        lines.append("The LLM's REASON field for each meta iteration:\n")
        for eid, r in reasons:
            r_clean = r.strip().replace("\n", " ")
            lines.append(f"- **Exp {eid}:** {r_clean}")
    else:
        lines.append("(No reasoning traces recorded — no meta experiments succeeded.)")
    lines.append("")

    lines.append("---")
    lines.append(f"*Combined prior-experiments table available at "
                 f"`{PRIOR_CSV}`. Meta progress plot at "
                 f"`{EXPERIMENTS_DIR / 'progress.png'}`.*")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved.")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 12: CLI
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of meta iterations (default 10)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete past meta experiments before starting")
    args = parser.parse_args()

    if args.reset:
        for d in list(EXPERIMENTS_DIR.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)
        np.save(CLASS_WEIGHTS_PATH, pos_weights)
        # Rewrite the prior-experiments CSV after the wipe
        PRIOR_DF.to_csv(PRIOR_CSV, index=False)
        print(f"Reset: cleared {EXPERIMENTS_DIR}")

    agent_loop(args.iterations)
