"""
BirdCLEF+ 2026 - Creative Agent
===============================

An exploratory agent: reuses the Regular Agent's training, metrics,
logging, pruning, and plotting machinery, but lets the LLM pick from a
much wider set of fine-tuning recipes.

This is one of the three planned agents for the report:

    EDA → Baselines (compare_models.py)
        → Agent Regular       (generic ML search)
        → Agent EDA-aware     (same machinery + EDA findings)
        → Agent Creative      (THIS FILE, wider/freer search)

(An earlier "Conservative" agent — agent_v1.py with a 1D loss-only search —
exists as prior work but isn't one of the three agents being compared.)

Baseline integration (warm-start design)
----------------------------------------
At startup the agent reads `comparison_results.csv`, picks the winner by
max macro_auc (tiebreak: smallest params), and loads
`trained_models/<winner>.keras` as the starting point. Every experiment
is therefore a SEARCH OVER FINE-TUNING RECIPES on top of the winning
baseline, not a from-scratch training run.

Search space:
  - Free-form Python fine-tuning recipes proposed by the LLM.
  - The LLM writes explicit snippets for the loss, optimizer/schedule,
    and augmentation, which are embedded into each generated model.py.

Pruning
-------
At startup the baseline's val_auc is measured once on the held-out
validation set. After each experiment's epoch 1, if val_auc is more than
PRUNE_MARGIN below that reference, training is aborted and the
experiment is marked pruned=True. Saves time on obviously-broken configs
(e.g. SGD with too high a lr collapsing the warm-started weights).

Held constant (so comparisons across agents are fair):
  - Architecture: whatever won compare_models.py (locked at agent startup)
  - Starting weights: those baked into the winner's .keras file
  - Batch size:   32
  - Data:         focal + soundscape (max_rows from cfg), focal_val_frac=0.05

Per-iteration workflow:
  1. Read all past experiments from experiments_creative_<llm>/
  2. Ask the LLM to propose ONE creative Python recipe + a rationale
  3. Re-prompt up to 3× if it repeats a previous recipe signature
  4. Generate a self-contained experiment .py file (hardcoded model code,
     loss class, augmentation function — every experiment is reproducible
     by running its own .py)
  5. Import the generated file, build/compile/train, save weights
  6. Compute BOTH macro-AUC metrics on validation:
        - unweighted (the official competition metric)
        - weighted   (per-taxon mean AUC weighted by class-count share —
                      reflects test-set composition, dominated by Aves)
  7. Persist metrics.json in the experiment folder

Bug fixes carried over from agent_v1:
  - keep_alive=0 on the Ollama call (frees RAM before training)
  - weighted_bce clipped at 5× (not 50× — that collapsed the model in v1)
  - Duplicate-detection: recipes are hashed; repeated recipes trigger
    a re-prompt before falling back to a robust free-form default

Run:
    python agent_creative.py                  # 10 iterations
    python agent_creative.py --iterations 3   # smoke test
    python agent_creative.py --reset          # wipe past experiments first
"""

# ─── Backend selection (must come before keras import) ───────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import hashlib
import json
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

# Local LLM client (Ollama exposes an OpenAI-compatible API)
import ollama

warnings.filterwarnings("ignore")

# Reuse the data pipeline from baseline.py — single source of truth.
# Importing it runs Cells 1-6 of baseline.py (label space, training table,
# train/val split, AudioDataset, mel pipeline). TAXON_WEIGHTS was added to
# baseline.py specifically so this agent (and the EDA agent later) can
# compute weighted macro AUC consistently.
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
# LLM config — adjust if your local Ollama model is named differently.
LLM_MODEL = "gemma4"

AGENT_NAME = "creative"


def safe_name(value: str) -> str:
    """Filesystem-safe slug used to keep runs from different LLMs separate."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


LLM_RUN_ID = safe_name(LLM_MODEL)
EXPERIMENTS_DIR = Path(f"experiments_{AGENT_NAME}_{LLM_RUN_ID}")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Pruning: kill an experiment after epoch 1 if its val_auc is more than
# this much below the baseline winner's val_auc (measured once at startup).
PRUNE_MARGIN = 0.05

# Per-experiment training budget. Matches compare_models.py so agent
# experiments are directly comparable to the manual baseline comparison.
EPOCHS_PER_EXPERIMENT = 10


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1.5: Pick the baseline winner from compare_models.py output
# ─────────────────────────────────────────────────────────────────────────
BASELINE_CSV       = Path("comparison_results.csv")
TRAINED_MODELS_DIR = Path("trained_models")


def pick_baseline_winner() -> tuple[str, dict, Path]:
    """Read comparison_results.csv and return (name, row_dict, keras_path).

    Ranking: max macro_auc, tiebreak by SMALLEST params (protects the
    90-minute CPU inference budget if a heavyweight model ever ties).
    Fails loudly if the CSV or the corresponding .keras file is missing —
    the agent has no useful default in either case."""
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"{BASELINE_CSV} not found. Run compare_models.py first to "
            f"produce the baseline comparison and trained_models/ folder."
        )
    df = pd.read_csv(BASELINE_CSV)
    df = df.dropna(subset=["macro_auc", "params"])
    if df.empty:
        raise ValueError(
            f"{BASELINE_CSV} has no rows with valid macro_auc + params."
        )
    df = df.sort_values(["macro_auc", "params"], ascending=[False, True])
    winner_row  = df.iloc[0].to_dict()
    winner_name = str(winner_row["model"])
    winner_path = TRAINED_MODELS_DIR / f"{winner_name}.keras"
    if not winner_path.exists():
        raise FileNotFoundError(
            f"Baseline winner is '{winner_name}' (macro_auc="
            f"{winner_row['macro_auc']:.4f}) but {winner_path} doesn't "
            f"exist. Re-run compare_models.py to regenerate trained_models/."
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


# Steps per epoch — needed by the experiment files for cosine/exp schedules.
import math
STEPS_PER_EPOCH   = math.ceil(len(train_rows) / cfg.batch_size)
TOTAL_TRAIN_STEPS = STEPS_PER_EPOCH * EPOCHS_PER_EXPERIMENT   # max — early stopping may end sooner


# Measure the baseline winner's val_auc once so we have a fair pruning
# threshold. (sklearn macro_auc and keras's val_auc aren't on the same
# scale; we need a keras val_auc reference for the in-training callback.)
def _measure_baseline_val_auc() -> float:
    print(f"\n=== Measuring baseline val_auc for pruning threshold ===")
    keras.backend.clear_session()
    m = keras.models.load_model(BASELINE_WINNER_PATH)
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),  # unused, just to attach metrics
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
    print(f"  prune threshold  = {PRUNE_THRESHOLD:.4f}  "
          f"(experiments below this after epoch 1 will be killed)")
except Exception as e:
    BASELINE_VAL_AUC = float("nan")
    PRUNE_THRESHOLD  = 0.20   # conservative fallback
    print(f"  [!] could not measure baseline val_auc ({type(e).__name__}: {e})")
    print(f"  falling back to a fixed prune threshold of {PRUNE_THRESHOLD}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Class-frequency stats (used by weighted_bce)
# ─────────────────────────────────────────────────────────────────────────
def compute_class_pos_counts() -> np.ndarray:
    """How many training rows have each label = 1? Shape: (NUM_CLASSES,)."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for labs in train_rows["labels"]:
        for lab in labs:
            if lab in LABEL_TO_IDX:
                counts[LABEL_TO_IDX[lab]] += 1
    return counts


CLASS_POS_COUNTS = compute_class_pos_counts()
CLASS_WEIGHTS_PATH = EXPERIMENTS_DIR / "class_pos_weights.npy"
CLASS_WEIGHTS_REF = str(CLASS_WEIGHTS_PATH).replace("\\", "/")

# Inverse-frequency weight, CLIPPED AT 5× (not 50× as in agent_v1).
# Lesson from v1: a 50× clip on a 234-class multi-label problem with
# extreme imbalance caused the model to predict rare classes everywhere
# and collapse the validation AUC. 5× is the standard practitioner range
# for class-weighted BCE.
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
# Cell 3: Past-experiments helpers
# ─────────────────────────────────────────────────────────────────────────
def load_past_experiments() -> list[dict]:
    """Each experiment folder has metrics.json with results + config."""
    past = []
    for d in sorted(EXPERIMENTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        mfile = d / "metrics.json"
        if mfile.exists():
            past.append(json.loads(mfile.read_text(encoding="utf-8")))
    return past


def config_key(loss_name: str, aug_name: str, lr: float,
               optimizer_name: str, schedule_name: str) -> str:
    """Canonical string for a (loss, aug, lr, optimizer, schedule) tuple —
    used for duplicate detection across iterations."""
    return f"{loss_name}|{aug_name}|{lr:g}|{optimizer_name}|{schedule_name}"


def summarise_past(past: list[dict]) -> str:
    """Render past experiments compactly for the LLM context."""
    if not past:
        return "(no prior experiments)"
    lines = []
    for e in past:
        pruned = " [PRUNED]" if e.get("pruned") else ""
        lines.append(
            f"- id={e['id']:>2}  loss={e['loss']:<14}  aug={e['aug']:<15}  "
            f"lr={float(e['lr']):g}  opt={e.get('optimizer', '?'):<13}  "
            f"sched={e.get('schedule', '?'):<13}  →  "
            f"macro(unw)={e['macro_auc']:.4f}  "
            f"macro(w)={e['weighted_macro_auc']:.4f}  "
            f"aves={e['aves_auc']:.3f}{pruned}"
        )
    return "\n".join(lines)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: LLM prompts
# Deliberately broad: the Creative Agent can take bigger risks than the
# Regular Agent while still producing reproducible experiment files.
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Creative Agent for an audio multi-label classification task.

Your job is to propose a new, reproducible Python fine-tuning experiment.
Unlike the Regular Agent, you are not limited to a fixed menu. You may invent
losses, optimizer settings, learning-rate schedules, and spectrogram
augmentations as long as they can be expressed in the Python snippets below.

Context:
  - Multi-label classification with 234 sigmoid outputs.
  - Inputs are log-mel spectrogram batches shaped (batch, n_mels, time, 1).
  - Experiments warm-start from the same winning baseline model.
  - Metrics reported later: unweighted macro AUC, weighted macro AUC, Aves/Bird AUC.
  - Use only imports already available in the generated file:
      import numpy as np
      import keras
      from keras import layers, ops
  - Keep code CPU-friendly and compatible with Keras 3 on the torch backend.

Creative freedom:
  - You may choose continuous learning rates, weight decay, momentum, focal
    gammas, label smoothing, asymmetric losses, mixed losses, custom
    SpecAugment intensities, mixup alphas, noise strengths, callbacks, etc.
  - You may combine ideas, but keep the experiment plausible enough to run.
  - Do not rewrite the model architecture; this agent explores fine-tuning
    recipes on top of the baseline.
  - Avoid exact repeats of previous recipe signatures.

Respond with exactly one JSON object and no markdown. Required keys:
{
  "recipe_name": "short_slug_like_name",
  "loss_name": "human-readable loss description",
  "augmentation_name": "human-readable augmentation description",
  "optimizer_name": "human-readable optimizer description",
  "schedule_name": "human-readable schedule/callback description",
  "initial_lr": 0.0003,
  "reason": "short experimental hypothesis",
  "loss_code": "Python code defining any needed loss classes/constants",
  "loss_factory": "Python expression returning the loss object",
  "optimizer_schedule_code": "Python code defining get_optimizer() and get_schedule_callbacks()",
  "augmentation_code": "Python code defining augment_batch(xs, ys)"
}

Hard requirements:
  - loss_factory must be a valid Python expression.
  - optimizer_schedule_code must define get_optimizer() and get_schedule_callbacks().
  - augmentation_code must define augment_batch(xs, ys) and return (xs, ys).
  - Use LEARNING_RATE, STEPS_PER_EPOCH, TOTAL_TRAIN_STEPS, NUM_CLASSES if useful.
  - Keep snippets self-contained; no external files except {class_weights_ref}.
"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{class_weights_ref}", CLASS_WEIGHTS_REF)


def build_user_prompt(past: list[dict]) -> str:
    tried = [e.get("recipe_signature", "") for e in past
             if e.get("recipe_signature")]
    tried_text = "\n".join(f"- {sig}" for sig in tried[-20:]) or "(none yet)"
    return f"""Past experiments ({len(past)} completed/crashed):
{summarise_past(past)}

Recent recipe signatures to avoid repeating:
{tried_text}

Propose the next free-form Python recipe. It should be meaningfully different
from the previous experiments and should still be cheap enough for a short
fine-tuning run."""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Parse LLM response
# ─────────────────────────────────────────────────────────────────────────
def _slugify(value: str, default: str = "creative_recipe") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60] or default


def recipe_signature(recipe: dict) -> str:
    """Stable short hash for duplicate detection across free-form recipes."""
    relevant = {
        k: recipe.get(k, "")
        for k in [
            "loss_name", "augmentation_name", "optimizer_name", "schedule_name",
            "initial_lr", "loss_code", "loss_factory",
            "optimizer_schedule_code", "augmentation_code",
        ]
    }
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def parse_llm_response(response: str) -> dict:
    """Parse the Creative Agent's JSON recipe and validate required code hooks."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response:\n{response}")
        text = text[start:end + 1]

    recipe = json.loads(text)
    required = [
        "recipe_name", "loss_name", "augmentation_name", "optimizer_name",
        "schedule_name", "initial_lr", "reason", "loss_code",
        "loss_factory", "optimizer_schedule_code", "augmentation_code",
    ]
    missing = [k for k in required if k not in recipe]
    if missing:
        raise ValueError(f"Missing required recipe keys: {missing}")

    recipe["recipe_name"] = _slugify(recipe["recipe_name"])
    recipe["initial_lr"] = float(recipe["initial_lr"])
    if not (1e-6 <= recipe["initial_lr"] <= 1e-2):
        raise ValueError("initial_lr must be between 1e-6 and 1e-2")

    code_checks = {
        "optimizer_schedule_code": ["def get_optimizer", "def get_schedule_callbacks"],
        "augmentation_code": ["def augment_batch"],
    }
    for field, needles in code_checks.items():
        code = str(recipe[field])
        for needle in needles:
            if needle not in code:
                raise ValueError(f"{field} must contain `{needle}`")
    if "\n" in str(recipe["loss_factory"]):
        raise ValueError("loss_factory must be a single Python expression")

    recipe["recipe_signature"] = recipe_signature(recipe)
    return recipe


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Generate the experiment .py file (the "hardcoded" rule)
# Per the professor: every experiment must be saved as explicit Python
# code, not a config. At submission time, the winning experiment's .py
# + .keras weights are used directly — no agent runtime needed.
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENT_TEMPLATE = '''"""
Experiment {exp_id} (Creative Agent)
    baseline      = {winner_name}  (loaded from {winner_path})
    loss          = {loss_name}
    augmentation  = {aug_name}
    optimizer     = {optimizer_name}
    schedule      = {schedule_name}
    initial_lr    = {lr:g}
    signature     = {recipe_signature}

Generated at {timestamp}
Rationale:
    {reason}

Self-contained: build_model(), get_loss(), get_optimizer(),
get_schedule_callbacks(), augment_batch(). The architecture lives inside
the .keras file referenced by WINNER_KERAS_PATH — Keras deserialises it
from the embedded config.json. At submission time, the winning
experiment's model.py + model.keras are the deliverable.
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES        = {num_classes}
LEARNING_RATE      = {lr:g}
STEPS_PER_EPOCH    = {steps_per_epoch}
TOTAL_TRAIN_STEPS  = {total_train_steps}
WINNER_KERAS_PATH  = "{winner_path}"


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
'''


def optimizer_schedule_block_for(optimizer_name: str, schedule_name: str
                                 ) -> str:
    """Generate the get_optimizer() + get_schedule_callbacks() functions
    for the experiment .py.

    - schedule="constant"     → fixed LR + ReduceLROnPlateau callback
    - schedule="cosine_decay" → CosineDecay schedule baked into optimizer
    - schedule="exp_decay"    → ExponentialDecay (×0.9 per epoch, staircase)
    """
    # 1. Build the learning_rate expression
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

    # 2. Build the optimizer expression
    if optimizer_name == "adam":
        opt_expr = f"keras.optimizers.Adam(learning_rate={lr_expr})"
    elif optimizer_name == "adamw":
        opt_expr = (f"keras.optimizers.AdamW(learning_rate={lr_expr}, "
                    f"weight_decay=1e-4)")
    elif optimizer_name == "nadam":
        opt_expr = f"keras.optimizers.Nadam(learning_rate={lr_expr})"
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


def loss_block_for(loss_name: str) -> tuple[str, str]:
    """Return (python_code_defining_loss, factory_call_string).

    The generated code is fully self-contained inside the experiment .py."""
    if loss_name == "plain_bce":
        return ("# Plain binary cross-entropy — no special class needed.",
                "keras.losses.BinaryCrossentropy()")

    if loss_name == "weighted_bce":
        # The weights file is written once at agent startup (Cell 2).
        # We hardcode its path here so the experiment .py can rehydrate it
        # at submission time without depending on the agent runtime.
        return (f'''POS_WEIGHTS = np.load("{CLASS_WEIGHTS_REF}").astype("float32")

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

    if loss_name == "asymmetric_focal":
        return ('''class AsymmetricFocalLoss(keras.losses.Loss):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, name="asymmetric_focal"):
        super().__init__(name=name)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pos_loss = y_true * ops.power(1.0 - y_pred, self.gamma_pos) * ops.log(y_pred)
        neg_loss = (1.0 - y_true) * ops.power(y_pred, self.gamma_neg) * ops.log(1.0 - y_pred)
        return -ops.mean(pos_loss + neg_loss)
''',
                "AsymmetricFocalLoss()")

    if loss_name == "soft_f1":
        return ('''class SoftF1Loss(keras.losses.Loss):
    def __init__(self, name="soft_f1"):
        super().__init__(name=name)
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1.0 - y_true) * y_pred, axis=0)
        fn = ops.sum(y_true * (1.0 - y_pred), axis=0)
        f1 = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)
        return 1.0 - ops.mean(f1)
''',
                "SoftF1Loss()")

    if loss_name == "tversky":
        return ('''class TverskyLoss(keras.losses.Loss):
    def __init__(self, alpha=0.3, beta=0.7, name="tversky"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1.0 - y_true) * y_pred, axis=0)
        fn = ops.sum(y_true * (1.0 - y_pred), axis=0)
        score = (tp + 1e-7) / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        return 1.0 - ops.mean(score)
''',
                "TverskyLoss()")

    raise ValueError(f"Unknown loss: {loss_name}")


def aug_block_for(aug_name: str) -> str:
    """Return python code defining `augment_batch(xs, ys) -> (xs, ys)`.

    Operates on already-computed mel-spectrogram batches
    (shape: N, n_mels, time_frames, 1). NumPy only — runs on CPU between
    AudioDataset.__getitem__ and model.fit, so it must be cheap."""
    if aug_name == "none":
        return '''def augment_batch(xs, ys):
    return xs, ys'''

    if aug_name == "specaugment":
        return '''def augment_batch(xs, ys):
    """SpecAugment: zero out random time and frequency bands per sample."""
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        # Frequency mask (mel-band stripe)
        f = np.random.randint(0, max(1, h // 8))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        # Time mask
        t = np.random.randint(0, max(1, w // 8))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys'''

    if aug_name == "specaugment_strong":
        return '''def augment_batch(xs, ys):
    """Stronger SpecAugment: two frequency masks and two time masks."""
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        for _ in range(2):
            f = np.random.randint(0, max(1, h // 5))
            if f > 0:
                f0 = np.random.randint(0, max(1, h - f))
                xs[i, f0:f0 + f, :, :] = 0.0
            t = np.random.randint(0, max(1, w // 5))
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

    if aug_name == "mixup_soft":
        return '''def augment_batch(xs, ys):
    """Gentler mixup: mostly preserve the anchor sample."""
    if len(xs) < 2:
        return xs, ys
    lam = float(np.random.uniform(0.75, 0.95))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    return xs_m.astype(np.float32), ys_m.astype(np.float32)'''

    if aug_name == "background_mix":
        return '''def augment_batch(xs, ys):
    """Mix each input with a random other sample in the batch at low volume.
    Labels are unchanged (we treat the second sample as 'background noise',
    not as a second positive). This roughly simulates the soundscape
    condition where target calls overlap with continuous chorus."""
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys'''

    if aug_name == "time_shift":
        return '''def augment_batch(xs, ys):
    """Randomly roll each spectrogram along the time axis."""
    xs = xs.copy()
    _, _, w, _ = xs.shape
    max_shift = max(1, w // 10)
    for i in range(len(xs)):
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        xs[i] = np.roll(xs[i], shift=shift, axis=1)
    return xs, ys'''

    if aug_name == "gaussian_noise":
        return '''def augment_batch(xs, ys):
    """Small additive noise in log-mel space."""
    noise = np.random.normal(0.0, 0.02, size=xs.shape).astype(np.float32)
    return (xs + noise).astype(np.float32), ys'''

    if aug_name == "specaugment_mixup":
        return '''def augment_batch(xs, ys):
    """Moderate SpecAugment followed by mixup."""
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
    if len(xs) < 2:
        return xs, ys
    lam = float(np.random.beta(0.4, 0.4))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    return xs_m.astype(np.float32), ys_m.astype(np.float32)'''

    if aug_name == "specaugment_background":
        return '''def augment_batch(xs, ys):
    """Moderate SpecAugment followed by low-volume background blending."""
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
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.05, 0.20))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys'''

    raise ValueError(f"Unknown augmentation: {aug_name}")


def generate_experiment_file(exp_id: int, recipe: dict) -> Path:
    """Write the hardcoded experiment .py file and return its path."""
    def doc_safe(value: object) -> str:
        return str(value).replace('"""', '"')

    loss_name = str(recipe["loss_name"])
    aug_name = str(recipe["augmentation_name"])
    lr = float(recipe["initial_lr"])
    optimizer_name = str(recipe["optimizer_name"])
    schedule_name = str(recipe["schedule_name"])
    reason = str(recipe["reason"])
    recipe_name = _slugify(recipe["recipe_name"])
    exp_dir = (EXPERIMENTS_DIR
               / f"exp_{exp_id:03d}_{recipe_name}_lr{lr:g}")
    exp_dir.mkdir(exist_ok=True)
    py_path = exp_dir / "model.py"
    # Forward slashes so the generated string literal is portable.
    winner_path_str = str(BASELINE_WINNER_PATH).replace("\\", "/")
    py_path.write_text(EXPERIMENT_TEMPLATE.format(
        exp_id=exp_id,
        loss_name=doc_safe(loss_name),
        aug_name=doc_safe(aug_name),
        lr=lr,
        optimizer_name=doc_safe(optimizer_name),
        schedule_name=doc_safe(schedule_name),
        recipe_signature=recipe["recipe_signature"],
        reason=doc_safe(reason),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_classes=NUM_CLASSES,
        winner_name=BASELINE_WINNER_NAME,
        winner_path=winner_path_str,
        steps_per_epoch=STEPS_PER_EPOCH,
        total_train_steps=TOTAL_TRAIN_STEPS,
        loss_code=str(recipe["loss_code"]),
        loss_factory_call=str(recipe["loss_factory"]),
        optimizer_schedule_code=str(recipe["optimizer_schedule_code"]),
        aug_code=str(recipe["augmentation_code"]),
    ), encoding="utf-8")
    return py_path


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Augmented dataset wrapper + run one experiment
# ─────────────────────────────────────────────────────────────────────────
class AugmentedDataset(keras.utils.PyDataset):
    """Wraps an AudioDataset and applies `augment_fn(xs, ys)` to each batch.
    Used only for training — validation must never be augmented."""

    def __init__(self, base_ds: keras.utils.PyDataset, augment_fn, **kwargs):
        kwargs.setdefault("workers", 1)
        kwargs.setdefault("use_multiprocessing", False)
        kwargs.setdefault("max_queue_size", 8)
        super().__init__(**kwargs)
        self.base = base_ds
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        xs, ys = self.base[idx]
        return self.augment_fn(xs, ys)

    def on_epoch_end(self):
        if hasattr(self.base, "on_epoch_end"):
            self.base.on_epoch_end()


def _compute_aucs(model: keras.Model, val_ds: keras.utils.PyDataset
                  ) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    """Return (per_class_auc, y_pred, y_true). per_class_auc is keyed by
    LABEL string, value is float or None (None = no positives in val)."""
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        xb, yb = val_ds[i]
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
    """Compute unweighted macro AUC, weighted macro AUC (by TAXON_WEIGHTS),
    per-taxon mean AUC, and the count of evaluable classes."""
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # Per-taxon mean AUC over evaluable classes in each taxon
    taxon_to_aucs: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None:
            continue
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        taxon_to_aucs.setdefault(taxon, []).append(auc)
    taxon_aucs = {t: float(np.mean(v)) for t, v in taxon_to_aucs.items()}

    # Weighted macro AUC: weight each taxon's mean AUC by its share of the
    # 234-class label space. Taxa with NO evaluable classes are excluded
    # and the remaining weights are re-normalised (so the metric is still
    # in [0, 1] when some taxa are missing).
    weighted_sum = 0.0
    weight_total = 0.0
    for taxon, mean_auc in taxon_aucs.items():
        w = TAXON_WEIGHTS.get(taxon, 0.0)
        weighted_sum  += w * mean_auc
        weight_total  += w
    weighted_macro_auc = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    return macro_auc, weighted_macro_auc, taxon_aucs, len(valid_aucs)


class PruneIfWorseCallback(keras.callbacks.Callback):
    """Stop training after epoch 1 if val_auc is below `threshold`.

    Rationale: every experiment warm-starts from the same baseline. If
    after one epoch of fine-tuning the val_auc has dropped sharply below
    the baseline's own val_auc, the new recipe is destroying the learned
    features and is very unlikely to recover within the remaining epochs.
    Killing it now frees time for the next config."""

    def __init__(self, threshold: float, monitor: str = "val_auc"):
        super().__init__()
        self.threshold = threshold
        self.monitor   = monitor
        self.pruned    = False

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-indexed → epoch=0 means the first epoch just finished
        if epoch == 0:
            val = float((logs or {}).get(self.monitor, 0.0))
            if val < self.threshold:
                print(f"\n  [PRUNE] {self.monitor}={val:.4f} < threshold "
                      f"{self.threshold:.4f} after epoch 1 — stopping early")
                self.model.stop_training = True
                self.pruned = True


def run_experiment(exp_id: int, recipe: dict) -> dict:
    """Train one experiment end-to-end and return its metrics row."""
    loss_name = str(recipe["loss_name"])
    aug_name = str(recipe["augmentation_name"])
    lr = float(recipe["initial_lr"])
    optimizer_name = str(recipe["optimizer_name"])
    schedule_name = str(recipe["schedule_name"])
    reason = str(recipe["reason"])
    print(f"\n{'=' * 72}")
    print(f" EXPERIMENT {exp_id}: loss={loss_name}  aug={aug_name}  "
          f"lr={lr:g}  opt={optimizer_name}  sched={schedule_name}")
    print(f"{'=' * 72}")
    print(f" reason: {reason}\n")

    py_path = generate_experiment_file(exp_id, recipe)
    exp_dir = py_path.parent

    # Import the generated experiment .py as a module — the "hardcoded
    # model code" rule. Model architecture and loss/optimizer/schedule
    # all live in real Python code on disk, not in a JSON/YAML config.
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"exp_{exp_id}", py_path)
    mod = importlib.util.module_from_spec(spec)
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

    base_train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
    train_ds      = AugmentedDataset(base_train_ds, mod.augment_batch)
    val_ds        = AudioDataset(val_rows,   batch_size=cfg.batch_size, shuffle=False)

    prune_cb = PruneIfWorseCallback(threshold=PRUNE_THRESHOLD, monitor="val_auc")
    callbacks = [
        *mod.get_schedule_callbacks(),  # ReduceLROnPlateau iff schedule=constant
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
        "recipe_name":         recipe["recipe_name"],
        "recipe_signature":    recipe["recipe_signature"],
        "source":              recipe.get("source", "unknown"),
        "loss":                loss_name,
        "aug":                 aug_name,
        "lr":                  lr,
        "optimizer":           optimizer_name,
        "schedule":            schedule_name,
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
        "loss_code":           recipe["loss_code"],
        "loss_factory":        recipe["loss_factory"],
        "optimizer_schedule_code": recipe["optimizer_schedule_code"],
        "augmentation_code":   recipe["augmentation_code"],
        "llm_attempts":        recipe.get("llm_attempts", []),
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
    """Call the local Ollama LLM and return raw text.
    keep_alive=0 tells Ollama to unload the model immediately after
    responding, freeing RAM for training. Each call pays the ~5-10s model-
    load cost but avoids the OOM that hit agent_v1 when the LLM stayed
    resident through long training runs."""
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


def _finalize_recipe(recipe: dict) -> dict:
    recipe["recipe_name"] = _slugify(recipe["recipe_name"])
    recipe["initial_lr"] = float(recipe["initial_lr"])
    recipe["recipe_signature"] = recipe_signature(recipe)
    return recipe


def fallback_candidates() -> list[dict]:
    """Small bank of robust free-form recipes if the LLM fails."""
    return [
        {
        "recipe_name": "fallback_adamw_focal_mixup",
        "loss_name": "focal loss gamma 1.5",
        "augmentation_name": "gentle mixup plus small noise",
        "optimizer_name": "AdamW weight_decay 5e-5",
        "schedule_name": "cosine decay",
        "initial_lr": 3e-4,
        "reason": (
            "Fallback free-form recipe: moderate focal loss, gentle mixed "
            "augmentation, AdamW regularization, and smooth decay."
        ),
        "loss_code": '''class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=1.5, name="focal_loss_g1_5"):
        super().__init__(name=name)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        return ops.mean(-ops.power(1.0 - pt, self.gamma) * ops.log(pt))''',
        "loss_factory": "FocalLoss()",
        "optimizer_schedule_code": '''def get_optimizer():
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=TOTAL_TRAIN_STEPS,
        alpha=0.05,
    )
    return keras.optimizers.AdamW(learning_rate=lr, weight_decay=5e-5)


def get_schedule_callbacks():
    return []''',
        "augmentation_code": '''def augment_batch(xs, ys):
    if len(xs) < 2:
        return xs, ys
    lam = float(np.random.uniform(0.75, 0.95))
    idx = np.random.permutation(len(xs))
    xs_m = lam * xs + (1.0 - lam) * xs[idx]
    ys_m = lam * ys + (1.0 - lam) * ys[idx]
    noise = np.random.normal(0.0, 0.01, size=xs.shape).astype(np.float32)
    return (xs_m + noise).astype(np.float32), ys_m.astype(np.float32)''',
        },
        {
        "recipe_name": "fallback_weighted_bce_specaugment",
        "loss_name": "weighted BCE with clipped positive weights",
        "augmentation_name": "moderate SpecAugment",
        "optimizer_name": "AdamW weight_decay 1e-4",
        "schedule_name": "constant LR with ReduceLROnPlateau",
        "initial_lr": 2e-4,
        "reason": (
            "Fallback free-form recipe: stabilize warm-start fine-tuning with "
            "weighted BCE, moderate masking, AdamW, and plateau-based LR cuts."
        ),
        "loss_code": f'''POS_WEIGHTS = np.load("{CLASS_WEIGHTS_REF}").astype("float32")

class WeightedBCE(keras.losses.Loss):
    def __init__(self, pos_weights, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weights = ops.convert_to_tensor(pos_weights)
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        per_class = -(self.pos_weights * y_true * ops.log(y_pred)
                      + (1.0 - y_true) * ops.log(1.0 - y_pred))
        return ops.mean(per_class)''',
        "loss_factory": "WeightedBCE(POS_WEIGHTS)",
        "optimizer_schedule_code": '''def get_optimizer():
    return keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)


def get_schedule_callbacks():
    return [keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=2e-5, verbose=0
    )]''',
        "augmentation_code": '''def augment_batch(xs, ys):
    xs = xs.copy()
    n, h, w, _ = xs.shape
    for i in range(n):
        f = np.random.randint(0, max(1, h // 10))
        if f > 0:
            f0 = np.random.randint(0, max(1, h - f))
            xs[i, f0:f0 + f, :, :] = 0.0
        t = np.random.randint(0, max(1, w // 10))
        if t > 0:
            t0 = np.random.randint(0, max(1, w - t))
            xs[i, :, t0:t0 + t, :] = 0.0
    return xs, ys''',
        },
        {
        "recipe_name": "fallback_asym_focal_background",
        "loss_name": "asymmetric focal loss",
        "augmentation_name": "background mix",
        "optimizer_name": "Nadam",
        "schedule_name": "exponential decay",
        "initial_lr": 1.5e-4,
        "reason": (
            "Fallback free-form recipe: try a conservative asymmetric focal "
            "objective with background blending and Nadam for smoother updates."
        ),
        "loss_code": '''class AsymmetricFocalLoss(keras.losses.Loss):
    def __init__(self, gamma_pos=0.0, gamma_neg=3.0, name="asymmetric_focal"):
        super().__init__(name=name)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pos_loss = y_true * ops.power(1.0 - y_pred, self.gamma_pos) * ops.log(y_pred)
        neg_loss = (1.0 - y_true) * ops.power(y_pred, self.gamma_neg) * ops.log(1.0 - y_pred)
        return -ops.mean(pos_loss + neg_loss)''',
        "loss_factory": "AsymmetricFocalLoss()",
        "optimizer_schedule_code": '''def get_optimizer():
    lr = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=STEPS_PER_EPOCH,
        decay_rate=0.85,
        staircase=True,
    )
    return keras.optimizers.Nadam(learning_rate=lr)


def get_schedule_callbacks():
    return []''',
        "augmentation_code": '''def augment_batch(xs, ys):
    if len(xs) < 2:
        return xs, ys
    bg_weight = float(np.random.uniform(0.03, 0.12))
    idx = np.random.permutation(len(xs))
    xs_m = (1.0 - bg_weight) * xs + bg_weight * xs[idx]
    return xs_m.astype(np.float32), ys''',
        },
        {
        "recipe_name": "fallback_smoothed_bce_timeshift",
        "loss_name": "binary cross-entropy with label smoothing",
        "augmentation_name": "time shift",
        "optimizer_name": "Adam",
        "schedule_name": "cosine decay",
        "initial_lr": 5e-4,
        "reason": (
            "Fallback free-form recipe: mild label smoothing and time shifts "
            "probe whether the baseline benefits from calibration and invariance."
        ),
        "loss_code": "# Built-in BCE with label smoothing.",
        "loss_factory": "keras.losses.BinaryCrossentropy(label_smoothing=0.02)",
        "optimizer_schedule_code": '''def get_optimizer():
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=TOTAL_TRAIN_STEPS,
        alpha=0.1,
    )
    return keras.optimizers.Adam(learning_rate=lr)


def get_schedule_callbacks():
    return []''',
        "augmentation_code": '''def augment_batch(xs, ys):
    xs = xs.copy()
    _, _, w, _ = xs.shape
    max_shift = max(1, w // 12)
    for i in range(len(xs)):
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        xs[i] = np.roll(xs[i], shift=shift, axis=1)
    return xs, ys''',
        },
    ]


def fallback_recipe(tried_signatures: set[str], exp_id: int) -> dict:
    """Return an untried robust fallback recipe, rotating if needed."""
    candidates = [_finalize_recipe(r.copy()) for r in fallback_candidates()]
    start = (max(exp_id, 1) - 1) % len(candidates)
    ordered = candidates[start:] + candidates[:start]
    for recipe in ordered:
        if recipe["recipe_signature"] not in tried_signatures:
            recipe["source"] = "fallback"
            return recipe

    # If the whole bank was already used, make a deterministic LR variant.
    recipe = ordered[0].copy()
    scale = max(0.5, 1.0 - 0.05 * exp_id)
    recipe["initial_lr"] = float(recipe["initial_lr"]) * scale
    recipe["reason"] = (
        recipe["reason"] + f" Deterministic fallback variant with LR scale {scale:.2f}."
    )
    recipe = _finalize_recipe(recipe)
    recipe["source"] = "fallback_variant"
    return recipe


def agent_loop(n_iterations: int) -> None:
    print(f"\n{'#' * 72}")
    print(f"# CREATIVE AGENT  —  {n_iterations} iterations  "
          f"—  LLM={LLM_MODEL}")
    print(f"# Output dir: {EXPERIMENTS_DIR}")
    print("# Search space: free-form Python recipes (loss, augmentation, "
          "optimizer, schedule, and continuous hyperparameters)")
    print(f"{'#' * 72}")

    for iteration in range(n_iterations):
        past = load_past_experiments()
        exp_id = (max((e["id"] for e in past), default=0)) + 1
        tried_signatures = {e.get("recipe_signature") for e in past
                            if e.get("recipe_signature")}

        print(f"\n--- iteration {iteration + 1}/{n_iterations} "
              f"(experiment id {exp_id}, {len(past)} past recipes) ---")

        # Ask the LLM, with up to 3 re-prompts on duplicates / parse errors
        recipe = None
        llm_attempts = []
        user_prompt = build_user_prompt(past)
        for attempt in range(3):
            response = ""
            try:
                response = call_llm(SYSTEM_PROMPT, user_prompt)
                print(f"LLM raw response (attempt {attempt + 1}):\n"
                      f"{response.strip()}\n")
                candidate = parse_llm_response(response)
                if candidate["recipe_signature"] in tried_signatures:
                    print(f"  [!] LLM repeated recipe signature "
                          f"{candidate['recipe_signature']}. Re-prompting.")
                    llm_attempts.append({
                        "attempt": attempt + 1,
                        "status": "duplicate",
                        "recipe_signature": candidate["recipe_signature"],
                        "raw_response": response,
                    })
                    user_prompt = (
                        build_user_prompt(past)
                        + "\n\nIMPORTANT: You repeated an existing recipe. "
                          "Return a genuinely different free-form Python recipe."
                    )
                    continue
                candidate["source"] = "llm"
                candidate["llm_attempts"] = llm_attempts + [{
                    "attempt": attempt + 1,
                    "status": "accepted",
                    "recipe_signature": candidate["recipe_signature"],
                    "raw_response": response,
                }]
                recipe = candidate
                break
            except Exception as e:
                print(f"  [!] LLM call/parse failed (attempt {attempt + 1}): "
                      f"{type(e).__name__}: {e}")
                llm_attempts.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "raw_response": response,
                })

        if recipe is None:
            recipe = fallback_recipe(tried_signatures, exp_id)
            recipe["llm_attempts"] = llm_attempts
            print(f"  Fallback recipe: {recipe['recipe_name']} "
                  f"({recipe['recipe_signature']})")

        try:
            run_experiment(exp_id, recipe)
        except Exception as e:
            print(f"[!] experiment {exp_id} crashed: {type(e).__name__}: {e}")
            failure_dir = (
                EXPERIMENTS_DIR
                / f"exp_{exp_id:03d}_{recipe['recipe_name']}_FAILED"
            )
            failure_dir.mkdir(exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "id": exp_id,
                "agent_name": AGENT_NAME,
                "llm_model": LLM_MODEL,
                "llm_run_id": LLM_RUN_ID,
                "experiments_dir": str(EXPERIMENTS_DIR),
                "recipe_name": recipe["recipe_name"],
                "recipe_signature": recipe["recipe_signature"],
                "source": recipe.get("source", "unknown"),
                "loss": recipe["loss_name"],
                "aug": recipe["augmentation_name"],
                "lr": recipe["initial_lr"],
                "optimizer": recipe["optimizer_name"],
                "schedule": recipe["schedule_name"],
                "reason": recipe["reason"],
                "pruned": False,
                "macro_auc": 0.0, "weighted_macro_auc": 0.0,
                "aves_auc": 0.0, "amphibia_auc": 0.0,
                "insecta_auc": 0.0, "mammalia_auc": 0.0,
                "llm_attempts": recipe.get("llm_attempts", []),
                "error": f"{type(e).__name__}: {e}",
            }, indent=2), encoding="utf-8")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'#' * 72}\n# CREATIVE AGENT — FINAL SUMMARY\n{'#' * 72}")
    final = load_past_experiments()
    if not final:
        print("No experiments completed.")
        return

    ranked = sorted(final, key=lambda e: e.get("weighted_macro_auc", 0),
                    reverse=True)
    print(f"\n{'rank':<5}{'id':<4}{'loss':<14}{'aug':<15}{'lr':<8}"
          f"{'macro(uw)':>10}{'macro(w)':>10}{'aves':>8}{'amph':>8}")
    for rank, e in enumerate(ranked, 1):
        print(f"{rank:<5}{e['id']:<4}{e['loss']:<14}{e['aug']:<15}"
              f"{float(e['lr']):<8.0e}"
              f"{e.get('macro_auc', 0):>10.4f}"
              f"{e.get('weighted_macro_auc', 0):>10.4f}"
              f"{e.get('aves_auc', 0):>8.3f}"
              f"{e.get('amphibia_auc', 0):>8.3f}")

    best = ranked[0]
    print(f"\nBEST (by weighted macro AUC): exp {best['id']} — "
          f"loss={best['loss']}  aug={best['aug']}  lr={float(best['lr']):g}")
    print(f"  weighted_macro_auc   = {best.get('weighted_macro_auc', 0):.4f}")
    print(f"  unweighted_macro_auc = {best.get('macro_auc', 0):.4f}")
    print(f"  aves_auc             = {best.get('aves_auc', 0):.3f}")
    print(f"  Model code:    {best.get('py_path')}")
    print(f"  Model weights: {best.get('weights_path')}")

    # Save a progress plot for the report
    plot_progress(final)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8.5: Progress plot (adapted from the earlier project's plot_results)
# ─────────────────────────────────────────────────────────────────────────
def plot_progress(experiments: list[dict] | None = None,
                  out_path: Path | None = None) -> None:
    """Plot macro AUCs and Aves/Bird AUC over agent iterations, with
    a horizontal reference line for the baseline winner's macro AUC.
    Tolerant of single-experiment runs (the plot is just one dot)."""
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
    baseline = successful[0].get("baseline_macro_auc")  # same across run

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, macro_uw, marker="o", lw=2, color="steelblue",
            label="macro AUC (unweighted)")
    ax.plot(iters, macro_w,  marker="s", lw=2, color="darkorange",
            label="macro AUC (weighted by taxon)")
    ax.plot(iters, aves_auc, marker="^", lw=2, color="forestgreen",
            label="Aves/Bird AUC")

    # Best-so-far reference (use weighted, since that's the ranking metric)
    best_w_val   = max(macro_w)
    best_w_iter  = iters[macro_w.index(best_w_val)]
    ax.axhline(best_w_val, ls="--", color="darkorange", alpha=0.4,
               label=f"Best weighted: {best_w_val:.4f} (exp {best_w_iter})")

    # Baseline reference
    if baseline is not None:
        ax.axhline(baseline, ls=":", color="gray", alpha=0.7,
                   label=f"Baseline {BASELINE_WINNER_NAME}: {baseline:.4f}")

    ax.set_xlabel("Experiment iteration")
    ax.set_ylabel("Validation macro AUC")
    ax.set_title(
        f"Creative Agent progress — {len(successful)} experiment"
        f"{'s' if len(successful) != 1 else ''}"
    )
    # Integer x-ticks; matplotlib chokes if iters has only one element so
    # set them manually
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
        # Re-create the class weights file after wipe
        np.save(CLASS_WEIGHTS_PATH, pos_weights)
        print(f"Reset: cleared {EXPERIMENTS_DIR}")

    agent_loop(args.iterations)

