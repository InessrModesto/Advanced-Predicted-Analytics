"""
BirdCLEF+ 2026 — Regular Agent
===============================

A "normal ML practitioner" agent: explores the standard hyperparameters
without any project-specific (EDA-derived) domain knowledge.

This is the second of the three planned agents:

    EDA → Baselines (compare_models.py)
        → Agent Conservative  (agent_v1.py,    1D search: loss only)
        → Agent Regular       (THIS FILE,      3D search: loss + aug + lr)
        → Agent EDA-aware     (built later)    same search, EDA in prompt

Baseline integration (the warm-start design)
--------------------------------------------
At startup the agent reads `comparison_results.csv` (produced by
compare_models.py), picks the winner by max macro_auc — with smallest
params as tiebreaker to protect the CPU inference budget — and loads
`trained_models/<winner>.keras` as the starting point. Every experiment
is therefore a SEARCH OVER FINE-TUNING RECIPES on top of the winning
baseline, not a from-scratch training run.

This means the experiment story is: "given the best architecture our
comparison surfaced, what (loss, augmentation, learning rate) recipe
most improves it?" It is NOT directly comparable to v1 (which trained
bigger_cnn from scratch).

Search space (3 dimensions, 5 × 4 × 3 = 60 combinations):
  - Loss:          plain_bce, weighted_bce, focal_g1.0, focal_g2.0, focal_g3.0
  - Augmentation:  none, specaugment, mixup, background_mix
  - Learning rate: 1e-3, 5e-4, 1e-4

Held constant (so comparisons across agents are fair):
  - Architecture: whatever won compare_models.py (locked at agent startup)
  - Starting weights: those baked into the winner's .keras file
  - Optimizer:    Adam
  - Batch size:   32
  - Data:         focal + soundscape (max_rows from cfg), focal_val_frac=0.05

Per-iteration workflow:
  1. Read all past experiments from experiments_regular/
  2. Ask the LLM to propose ONE (loss, aug, lr) triple + a rationale
  3. Re-prompt up to 3× if it picks a configuration already tried
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
  - Duplicate-detection: if the LLM picks a config already in the log,
    re-prompt up to 3× before falling back to the first untried combo

Run:
    python agent_regular.py                  # 10 iterations
    python agent_regular.py --iterations 3   # smoke test
    python agent_regular.py --reset          # wipe past experiments first
"""

# ─── Backend selection (must come before keras import) ───────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
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
AGENT_NAME = "regular"
EXPERIMENTS_DIR = Path(f"experiments_{AGENT_NAME}")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# The search space — exposed as constants so the prompt and the validator
# stay in sync with the parser.
LOSS_OPTIONS = ["plain_bce", "weighted_bce", "focal_g1.0", "focal_g2.0", "focal_g3.0"]
AUG_OPTIONS  = ["none", "specaugment", "mixup", "background_mix"]
LR_OPTIONS   = [1e-3, 5e-4, 1e-4]

# LLM config — adjust if your local Ollama model is named differently.
LLM_MODEL = "gemma4"

# Per-experiment training budget. Matches compare_models.py so agent
# experiments are directly comparable to the manual baseline comparison.
EPOCHS_PER_EXPERIMENT = 3


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


def config_key(loss_name: str, aug_name: str, lr: float) -> str:
    """Canonical string for a (loss, aug, lr) triple — used for duplicate
    detection across iterations."""
    return f"{loss_name}|{aug_name}|{lr:g}"


def summarise_past(past: list[dict]) -> str:
    """Render past experiments compactly for the LLM context."""
    if not past:
        return "(no prior experiments)"
    lines = []
    for e in past:
        lines.append(
            f"- id={e['id']:>2}  loss={e['loss']:<14}  aug={e['aug']:<15}  "
            f"lr={e['lr']:g}  →  macro_auc(unw)={e['macro_auc']:.4f}  "
            f"macro_auc(w)={e['weighted_macro_auc']:.4f}  "
            f"aves={e['aves_auc']:.3f}"
        )
    return "\n".join(lines)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: LLM prompts
# Deliberately generic — the Regular Agent has no EDA-derived knowledge.
# Only standard ML reasoning about loss / augmentation / learning rate.
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous ML research agent for an audio classification task.

You have to propose the next experiment to run. Each experiment has three
choices: a loss function, an augmentation strategy, and a learning rate.
Pick EXACTLY ONE option from each menu.

LOSS options:
  - plain_bce        : standard binary cross-entropy
  - weighted_bce     : BCE weighted by inverse class frequency (helps rare classes)
  - focal_g1.0       : focal loss, gamma=1.0 (mild focus on hard examples)
  - focal_g2.0       : focal loss, gamma=2.0 (standard focal loss)
  - focal_g3.0       : focal loss, gamma=3.0 (aggressive focus on hard examples)

AUGMENTATION options:
  - none             : no augmentation
  - specaugment      : random time/frequency masking on the spectrogram
  - mixup            : linear combination of two training examples
  - background_mix   : add a random low-volume background to the input

LEARNING RATE options:
  - 1e-3             : standard Adam learning rate
  - 5e-4             : slightly lower, more stable
  - 1e-4             : much lower, slower but sometimes generalizes better

The task is multi-label classification across 234 classes. Two metrics are
reported per experiment:
  - macro_auc (unweighted) : the official competition metric
  - macro_auc (weighted)   : weighted by per-taxon class count, more
                              representative of test-set composition

Both metrics matter. Use past results to inform your choice.

Rules:
  1. Pick exactly ONE option from each of the three menus.
  2. Do NOT pick a (loss, aug, lr) combination already in the log below.
  3. Respond in EXACTLY this format, with no extra text:

CHOICE_LOSS: <one of: plain_bce | weighted_bce | focal_g1.0 | focal_g2.0 | focal_g3.0>
CHOICE_AUG:  <one of: none | specaugment | mixup | background_mix>
CHOICE_LR:   <one of: 1e-3 | 5e-4 | 1e-4>
REASON: <one short paragraph explaining why>
"""


def build_user_prompt(past: list[dict]) -> str:
    tried_keys = {config_key(e["loss"], e["aug"], float(e["lr"])) for e in past}
    n_total = len(LOSS_OPTIONS) * len(AUG_OPTIONS) * len(LR_OPTIONS)
    n_tried = len(tried_keys)
    return f"""Past experiments ({n_tried} / {n_total} combinations tried):
{summarise_past(past)}

Pick the next (loss, augmentation, learning_rate) combination to try.
It must be one we have NOT already run."""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Parse LLM response
# ─────────────────────────────────────────────────────────────────────────
def parse_llm_response(response: str) -> tuple[str, str, float, str]:
    """Extract (loss, aug, lr, reason) from the LLM response.

    Accepts a bit of slack in formatting (case, surrounding whitespace,
    optional brackets/quotes) — but the three choices must each be on
    their own line starting with CHOICE_LOSS / CHOICE_AUG / CHOICE_LR."""
    def grab(field_name: str) -> str:
        m = re.search(rf"{field_name}\s*:\s*([^\n]+)", response, re.IGNORECASE)
        if not m:
            raise ValueError(f"No {field_name} in response:\n{response}")
        return m.group(1).strip().strip("`'\"[](){}<>")

    loss = grab("CHOICE_LOSS")
    aug  = grab("CHOICE_AUG")
    lr_s = grab("CHOICE_LR")

    if loss not in LOSS_OPTIONS:
        raise ValueError(f"Loss '{loss}' not in {LOSS_OPTIONS}")
    if aug not in AUG_OPTIONS:
        raise ValueError(f"Aug '{aug}' not in {AUG_OPTIONS}")
    try:
        lr = float(lr_s)
    except ValueError:
        raise ValueError(f"LR '{lr_s}' is not a number")
    if not any(abs(lr - opt) < 1e-12 for opt in LR_OPTIONS):
        raise ValueError(f"LR '{lr}' not in {LR_OPTIONS}")

    reason_m = re.search(r"REASON\s*:\s*(.+?)(?=\n\s*CHOICE_|\Z)",
                         response, re.IGNORECASE | re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else "(no reason given)"
    return loss, aug, lr, reason


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Generate the experiment .py file (the "hardcoded" rule)
# Per the professor: every experiment must be saved as explicit Python
# code, not a config. At submission time, the winning experiment's .py
# + .keras weights are used directly — no agent runtime needed.
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENT_TEMPLATE = '''"""
Experiment {exp_id} (Regular Agent)
    baseline      = {winner_name}  (loaded from {winner_path})
    loss          = {loss_name}
    augmentation  = {aug_name}
    learning_rate = {lr:g}

Generated at {timestamp}
Rationale:
    {reason}

This file is self-contained: import it and call build_model(), get_loss(),
get_lr(), augment_batch(). The architecture lives inside the .keras file
referenced by WINNER_KERAS_PATH — Keras deserialises it from the embedded
config.json. At submission time, the winning experiment's model.py +
model.keras (the fine-tuned weights) are the deliverable.
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES       = {num_classes}
LEARNING_RATE     = {lr:g}
WINNER_KERAS_PATH = "{winner_path}"


# ── Architecture: warm-start from the winning baseline ──────────────────
# The .keras file contains the full architecture (in config.json) + the
# weights trained by compare_models.py under plain BCE / no aug / lr=1e-3.
# Each experiment fine-tunes those weights with its own (loss, aug, lr).
def build_model() -> keras.Model:
    return keras.models.load_model(WINNER_KERAS_PATH)


# ── Loss ────────────────────────────────────────────────────────────────
{loss_code}

def get_loss():
    return {loss_factory_call}


def get_lr() -> float:
    return LEARNING_RATE


# ── Augmentation (applied to (xs, ys) batches at training time) ─────────
{aug_code}
'''


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
        return ('''POS_WEIGHTS = np.load("experiments_regular/class_pos_weights.npy").astype("float32")

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

    raise ValueError(f"Unknown augmentation: {aug_name}")


def generate_experiment_file(exp_id: int, loss_name: str, aug_name: str,
                             lr: float, reason: str) -> Path:
    """Write the hardcoded experiment .py file and return its path."""
    loss_code, factory_call = loss_block_for(loss_name)
    aug_code = aug_block_for(aug_name)
    exp_dir = EXPERIMENTS_DIR / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}"
    exp_dir.mkdir(exist_ok=True)
    py_path = exp_dir / "model.py"
    # The winner path uses forward slashes so the generated string literal
    # is valid on both Windows and POSIX (a single backslash would break
    # Python escape rules).
    winner_path_str = str(BASELINE_WINNER_PATH).replace("\\", "/")
    py_path.write_text(EXPERIMENT_TEMPLATE.format(
        exp_id=exp_id,
        loss_name=loss_name,
        aug_name=aug_name,
        lr=lr,
        reason=reason.replace('"""', '"'),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_classes=NUM_CLASSES,
        winner_name=BASELINE_WINNER_NAME,
        winner_path=winner_path_str,
        loss_code=loss_code,
        loss_factory_call=factory_call,
        aug_code=aug_code,
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


def run_experiment(exp_id: int, loss_name: str, aug_name: str,
                   lr: float, reason: str) -> dict:
    """Train one experiment end-to-end and return its metrics row."""
    print(f"\n{'=' * 72}")
    print(f" EXPERIMENT {exp_id}: loss={loss_name}  aug={aug_name}  lr={lr:g}")
    print(f"{'=' * 72}")
    print(f" reason: {reason}\n")

    py_path = generate_experiment_file(exp_id, loss_name, aug_name, lr, reason)
    exp_dir = py_path.parent

    # Import the generated experiment .py as a module — this is the
    # "hardcoded model code" rule. The model architecture and loss live
    # in real Python code on disk, not in a JSON/YAML config.
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"exp_{exp_id}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg.seed)

    model = mod.build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=mod.get_lr()),
        loss=mod.get_loss(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )
    n_params = int(model.count_params())
    print(f"  Parameters: {n_params:,}")

    base_train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
    train_ds      = AugmentedDataset(base_train_ds, mod.augment_batch)
    val_ds        = AudioDataset(val_rows,   batch_size=cfg.batch_size, shuffle=False)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1
        ),
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

    # Save weights to the experiment folder
    weights_path = exp_dir / "model.keras"
    model.save(weights_path)
    print(f"  Saved trained model to {weights_path}")

    # Validation AUC — both unweighted and weighted
    print(f"  Computing validation AUC...")
    per_class_auc, _, _ = _compute_aucs(model, val_ds)
    macro_auc, weighted_macro_auc, taxon_aucs, n_eval = _summarise_aucs(per_class_auc)

    metrics = {
        "id":                  exp_id,
        "loss":                loss_name,
        "aug":                 aug_name,
        "lr":                  lr,
        "reason":              reason,
        "baseline_winner":     BASELINE_WINNER_NAME,
        "baseline_macro_auc":  float(BASELINE_WINNER_ROW["macro_auc"]),
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

    print(f"\n  RESULT: macro_auc(unw)={macro_auc:.4f}  "
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


def all_combinations() -> list[tuple[str, str, float]]:
    return [(l, a, r)
            for l in LOSS_OPTIONS
            for a in AUG_OPTIONS
            for r in LR_OPTIONS]


def agent_loop(n_iterations: int) -> None:
    print(f"\n{'#' * 72}")
    print(f"# REGULAR AGENT  —  {n_iterations} iterations  "
          f"—  LLM={LLM_MODEL}")
    print(f"# Search space: {len(LOSS_OPTIONS)} losses × {len(AUG_OPTIONS)} augs "
          f"× {len(LR_OPTIONS)} lrs = "
          f"{len(LOSS_OPTIONS) * len(AUG_OPTIONS) * len(LR_OPTIONS)} configs")
    print(f"{'#' * 72}")

    all_combos = all_combinations()

    for iteration in range(n_iterations):
        past = load_past_experiments()
        exp_id = (max((e["id"] for e in past), default=0)) + 1
        tried_keys = {config_key(e["loss"], e["aug"], float(e["lr"])) for e in past}
        untried = [c for c in all_combos
                   if config_key(c[0], c[1], c[2]) not in tried_keys]

        print(f"\n--- iteration {iteration + 1}/{n_iterations} "
              f"(experiment id {exp_id}, "
              f"{len(tried_keys)}/{len(all_combos)} tried) ---")

        if not untried:
            print("All configurations have been tried. Stopping.")
            break

        # Ask the LLM, with up to 3 re-prompts on duplicates / parse errors
        loss_name, aug_name, lr, reason = None, None, None, None
        user_prompt = build_user_prompt(past)
        for attempt in range(3):
            try:
                response = call_llm(SYSTEM_PROMPT, user_prompt)
                print(f"LLM raw response (attempt {attempt + 1}):\n"
                      f"{response.strip()}\n")
                cand_loss, cand_aug, cand_lr, cand_reason = \
                    parse_llm_response(response)
                if config_key(cand_loss, cand_aug, cand_lr) in tried_keys:
                    print(f"  [!] LLM picked an already-tried combination: "
                          f"({cand_loss}, {cand_aug}, lr={cand_lr:g}). "
                          f"Re-prompting.")
                    user_prompt = (
                        build_user_prompt(past)
                        + f"\n\nIMPORTANT: You just picked "
                          f"({cand_loss}, {cand_aug}, lr={cand_lr:g}), which "
                          f"has already been tried. Pick a DIFFERENT, untried "
                          f"combination."
                    )
                    continue
                loss_name, aug_name, lr, reason = \
                    cand_loss, cand_aug, cand_lr, cand_reason
                break
            except Exception as e:
                print(f"  [!] LLM call/parse failed (attempt {attempt + 1}): "
                      f"{type(e).__name__}: {e}")

        if loss_name is None:
            # Fallback after 3 failed attempts — first untried combination
            loss_name, aug_name, lr = untried[0]
            reason = ("(fallback: LLM unavailable or unhelpful — picked the "
                      "first untried combination)")
            print(f"  Fallback choice: ({loss_name}, {aug_name}, lr={lr:g})")

        # Run the experiment. Crashes are logged and marked so they aren't
        # retried infinitely.
        try:
            run_experiment(exp_id, loss_name, aug_name, lr, reason)
        except Exception as e:
            print(f"[!] experiment {exp_id} crashed: {type(e).__name__}: {e}")
            failure_dir = (
                EXPERIMENTS_DIR
                / f"exp_{exp_id:03d}_{loss_name}_{aug_name}_lr{lr:g}_FAILED"
            )
            failure_dir.mkdir(exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "id": exp_id, "loss": loss_name, "aug": aug_name, "lr": lr,
                "reason": reason,
                "macro_auc": 0.0, "weighted_macro_auc": 0.0,
                "aves_auc": 0.0, "amphibia_auc": 0.0,
                "insecta_auc": 0.0, "mammalia_auc": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }, indent=2), encoding="utf-8")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'#' * 72}\n# REGULAR AGENT — FINAL SUMMARY\n{'#' * 72}")
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
