"""
BirdCLEF+ 2026 — Agent v1 (Conservative)
=========================================

The first autonomous research agent. Conservative search space:
  - Architecture: FIXED (bigger_cnn from compare_models.py)
  - Optimizer:    FIXED (Adam, lr=1e-3)
  - Batch size:   FIXED (32)
  - Data subset:  FIXED (same as baseline / compare_models)
  - The agent ONLY chooses the LOSS FUNCTION configuration.

Loss menu:
  - plain_bce             : standard BCE, no weighting
  - weighted_bce          : BCE weighted by inverse class frequency
  - focal_loss_g1.0       : focal loss, gamma = 1.0
  - focal_loss_g2.0       : focal loss, gamma = 2.0
  - focal_loss_g3.0       : focal loss, gamma = 3.0

Workflow per iteration:
  1. Read all past experiment results from experiments/ folder
  2. Ask the LLM to propose ONE of the loss configurations + a short rationale
  3. Generate a self-contained experiment .py file containing the model code
  4. Train via the generated file, save weights, save metrics
  5. Repeat

Final-model rule (per professor): every experiment writes a HARDCODED Python
file with explicit Keras code for the architecture. No JSON/YAML configs.
At submission time we pick the winning experiment's .py and weights.

Run:
    python agent_v1.py                  # 10 iterations
    python agent_v1.py --iterations 3   # smoke test
"""

# ─── Backend selection (before keras import) ───────────────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from keras import layers, ops
from sklearn.metrics import roc_auc_score

# Local LLM client (Ollama exposes an OpenAI-compatible API)
import ollama

warnings.filterwarnings("ignore")

# Reuse the data pipeline from baseline.py.
# Importing it builds the training table and splits.
from baseline import (
    cfg,
    LABELS, NUM_CLASSES, LABEL_TO_IDX, SPECIES_TO_TAXON,
    train_rows, val_rows,
    AudioDataset,
    TIME_FRAMES,
)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1: Agent configuration
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# Available loss options — the agent's full search space for v1
LOSS_MENU = [
    "plain_bce",
    "weighted_bce",
    "focal_loss_g1.0",
    "focal_loss_g2.0",
    "focal_loss_g3.0",
]

# LLM config
LLM_MODEL = "gemma4"   # adjust if your local Ollama model is named differently

# Fixed training hyperparameters for v1
EPOCHS_PER_EXPERIMENT = 10  # match compare_models.py so results are comparable


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
# Save once for the generated experiment files to reuse
CLASS_WEIGHTS_PATH = EXPERIMENTS_DIR / "class_pos_weights.npy"
# Inverse-frequency weight, clipped to a reasonable max so super-rare classes
# don't dominate. With max=50, a class with 0 examples and a class with 10
# examples both get weight 50.
total = len(train_rows)
pos_weights = np.where(
    CLASS_POS_COUNTS > 0,
    np.clip(total / (CLASS_POS_COUNTS + 1e-6), 1.0, 5.0),
    5.0,   # never-seen classes get max weight
).astype(np.float32)
np.save(CLASS_WEIGHTS_PATH, pos_weights)
print(f"  class_pos_weights saved to {CLASS_WEIGHTS_PATH} (range "
      f"{pos_weights.min():.1f} – {pos_weights.max():.1f})")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Read past experiments (for the LLM's context)
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


def summarise_past(past: list[dict]) -> str:
    """Render past experiments compactly for the LLM."""
    if not past:
        return "(no prior experiments)"
    lines = []
    for e in past:
        lines.append(
            f"- {e['id']:>3}  loss={e['loss_name']:<18}  "
            f"macro_auc={e['macro_auc']:.4f}  "
            f"aves={e['aves_auc']:.3f}  amphibia={e['amphibia_auc']:.3f}"
        )
    return "\n".join(lines)


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: Build the LLM prompt
# ─────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous ML research agent for the BirdCLEF+ 2026 competition.

Your job: choose the next LOSS FUNCTION to try, from a fixed menu.
You may ONLY change the loss function. Everything else is held constant.

Available loss configurations:
  - plain_bce               : standard binary cross-entropy, no class weighting
  - weighted_bce            : BCE weighted by inverse class frequency
  - focal_loss_g1.0         : focal loss with gamma=1.0 (mild focus on hard examples)
  - focal_loss_g2.0         : focal loss with gamma=2.0 (standard focal)
  - focal_loss_g3.0         : focal loss with gamma=3.0 (aggressive focus)

Rules:
  1. Pick exactly ONE configuration from the menu above.
  2. Do NOT pick a configuration that has already been tried (check the log below).
  3. Use the past results to inform your choice — propose what is most likely
     to improve the macro AUC based on what worked and what did not.
  4. Respond in EXACTLY this format:

CHOICE: <one of: plain_bce | weighted_bce | focal_loss_g1.0 | focal_loss_g2.0 | focal_loss_g3.0>
REASON: <one short paragraph explaining why>

Do not include any other text."""


def build_user_prompt(past: list[dict]) -> str:
    tried = {e["loss_name"] for e in past}
    untried = [l for l in LOSS_MENU if l not in tried]
    return f"""Past experiments:
{summarise_past(past)}

Already-tried loss configurations: {sorted(tried) if tried else 'none'}
Still untried: {untried if untried else '(all options have been tried)'}

Pick the next loss configuration."""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Parse LLM response
# ─────────────────────────────────────────────────────────────────────────
def parse_llm_response(response: str) -> tuple[str, str]:
    """Extract (loss_name, reason) from the LLM's response."""
    choice_match = re.search(r"CHOICE:\s*([\w\.]+)", response, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+?)(?=\n\n|\Z)", response,
                             re.IGNORECASE | re.DOTALL)
    if not choice_match:
        raise ValueError(f"No CHOICE in response:\n{response}")
    choice = choice_match.group(1).strip()
    if choice not in LOSS_MENU:
        raise ValueError(f"Choice '{choice}' not in menu {LOSS_MENU}")
    reason = reason_match.group(1).strip() if reason_match else "(no reason given)"
    return choice, reason


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Generate the experiment .py file (the "hardcoded" model code)
# This is what gets saved per experiment. At submission time, the winning
# experiment's .py + .keras weights are used directly. No agent runtime.
# ─────────────────────────────────────────────────────────────────────────
EXPERIMENT_TEMPLATE = '''"""
Experiment {exp_id} — loss = {loss_name}
Generated by agent_v1 at {timestamp}
Rationale: {reason}
"""
import numpy as np
import keras
from keras import layers, ops

NUM_CLASSES = {num_classes}
INPUT_SHAPE = ({n_mels}, {time_frames}, 1)

# ── Model architecture (bigger_cnn from compare_models.py) ───────────────
def build_model() -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="exp_{exp_id}_{loss_name}")


# ── Loss function ────────────────────────────────────────────────────────
{loss_code}


def get_loss():
    return {loss_factory_call}
'''


def loss_block_for(loss_name: str) -> tuple[str, str]:
    """Return (python_code_defining_loss, factory_call_string)."""
    if loss_name == "plain_bce":
        return ("# Plain binary cross-entropy — no special definition needed.",
                "keras.losses.BinaryCrossentropy()")

    if loss_name == "weighted_bce":
        return ("""POS_WEIGHTS = np.load("experiments/class_pos_weights.npy").astype("float32")

class WeightedBCE(keras.losses.Loss):
    def __init__(self, pos_weights, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weights = ops.convert_to_tensor(pos_weights)
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        per_class = -(self.pos_weights * y_true * ops.log(y_pred)
                      + (1.0 - y_true) * ops.log(1.0 - y_pred))
        return ops.mean(per_class)
""", "WeightedBCE(POS_WEIGHTS)")

    m = re.match(r"focal_loss_g([\d\.]+)", loss_name)
    if m:
        gamma = float(m.group(1))
        return (f"""class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma={gamma}, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
    def call(self, y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        log_pt = ops.log(pt)
        loss = -ops.power(1.0 - pt, self.gamma) * log_pt
        return ops.mean(loss)
""", "FocalLoss()")

    raise ValueError(f"Unknown loss: {loss_name}")


def generate_experiment_file(exp_id: int, loss_name: str, reason: str) -> Path:
    """Write the hardcoded experiment .py file and return its path."""
    loss_code, factory_call = loss_block_for(loss_name)
    exp_dir = EXPERIMENTS_DIR / f"exp_{exp_id:03d}_{loss_name}"
    exp_dir.mkdir(exist_ok=True)
    py_path = exp_dir / "model.py"
    py_path.write_text(EXPERIMENT_TEMPLATE.format(
        exp_id=exp_id,
        loss_name=loss_name,
        reason=reason.replace('"""', '"'),  # avoid breaking the docstring
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_classes=NUM_CLASSES,
        n_mels=cfg.n_mels,
        time_frames=TIME_FRAMES,
        loss_code=loss_code,
        loss_factory_call=factory_call,
    ), encoding="utf-8")
    return py_path


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Run one experiment — load the generated .py, train, evaluate
# ─────────────────────────────────────────────────────────────────────────
def run_experiment(exp_id: int, loss_name: str, reason: str) -> dict:
    print(f"\n{'=' * 70}\n EXPERIMENT {exp_id}: loss = {loss_name}\n{'=' * 70}")
    print(f" reason: {reason}\n")

    py_path = generate_experiment_file(exp_id, loss_name, reason)
    exp_dir = py_path.parent

    # Import the generated file as a module (this is the "hardcoded" rule:
    # the model architecture and loss live in real Python code we just wrote
    # to disk, not in a config).
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"exp_{exp_id}", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg.seed)

    model = mod.build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=mod.get_loss(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )
    n_params = int(model.count_params())

    train_ds = AudioDataset(train_rows, batch_size=cfg.batch_size, shuffle=True)
    val_ds   = AudioDataset(val_rows,   batch_size=cfg.batch_size, shuffle=False)

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

    # Compute validation AUC (same code path as baseline.py / compare_models.py)
    all_preds, all_true = [], []
    for i in range(len(val_ds)):
        xb, yb = val_ds[i]
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
            per_class_auc[label] = roc_auc_score(
                y_true[:, cls_idx], y_pred[:, cls_idx]
            )
        except Exception:
            per_class_auc[label] = None

    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    taxon_aucs: dict[str, float] = {}
    taxon_to_aucs: dict[str, list[float]] = {}
    for label, auc in per_class_auc.items():
        if auc is None:
            continue
        taxon = SPECIES_TO_TAXON.get(label, "Unknown")
        taxon_to_aucs.setdefault(taxon, []).append(auc)
    for t, vals in taxon_to_aucs.items():
        taxon_aucs[t] = float(np.mean(vals))

    metrics = {
        "id":              exp_id,
        "loss_name":       loss_name,
        "reason":          reason,
        "params":          n_params,
        "epochs_run":      len(history.history["loss"]),
        "train_time_s":    round(train_time_s, 1),
        "macro_auc":       round(macro_auc, 4),
        "n_classes_eval":  len(valid_aucs),
        "aves_auc":        round(taxon_aucs.get("Aves", float("nan")), 4),
        "amphibia_auc":    round(taxon_aucs.get("Amphibia", float("nan")), 4),
        "insecta_auc":     round(taxon_aucs.get("Insecta", float("nan")), 4),
        "mammalia_auc":    round(taxon_aucs.get("Mammalia", float("nan")), 4),
        "py_path":         str(py_path),
        "weights_path":    str(weights_path),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  RESULT: macro_auc={macro_auc:.4f} "
          f"(aves={taxon_aucs.get('Aves', 0):.3f}, "
          f"amph={taxon_aucs.get('Amphibia', 0):.3f})")
    return metrics


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8: The agent loop
# ─────────────────────────────────────────────────────────────────────────
def call_llm(system: str, user: str) -> str:
    """Call the local Ollama LLM and return its raw response text.
    keep_alive=0 tells Ollama to unload the model immediately after responding,
    freeing RAM for training. Each call pays the model-load cost (~5-10s) but
    avoids the OOM that hits when the LLM stays resident during long training."""
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
    print(f"\n{'#' * 70}\n# AGENT v1  —  {n_iterations} iterations  "
          f"—  LLM={LLM_MODEL}\n{'#' * 70}")
    for iteration in range(n_iterations):
        past = load_past_experiments()
        exp_id = (max((e["id"] for e in past), default=0)) + 1

        # 1. Ask the LLM
        user_prompt = build_user_prompt(past)
        print(f"\n--- iteration {iteration+1}/{n_iterations} "
              f"(experiment id {exp_id}) ---")
        tried = {p["loss_name"] for p in past}
        untried = [l for l in LOSS_MENU if l not in tried]
        if not untried:
            print("All loss options have been tried, stopping.")
            break

        loss_name, reason = None, None
        # Up to 3 attempts to get a valid, untried choice from the LLM
        for attempt in range(3):
            try:
                response = call_llm(SYSTEM_PROMPT, user_prompt)
                print(f"LLM raw response (attempt {attempt+1}):\n{response.strip()}\n")
                candidate, candidate_reason = parse_llm_response(response)
                if candidate in tried:
                    print(f"  [!] LLM picked {candidate}, already tried. Re-prompting.")
                    user_prompt = (
                        build_user_prompt(past)
                        + f"\n\nIMPORTANT: You just picked {candidate}, which has "
                          f"already been tried. Pick a DIFFERENT untried option: "
                          f"{untried}"
                    )
                    continue
                loss_name, reason = candidate, candidate_reason
                break
            except Exception as e:
                print(f"[!] LLM call/parse failed (attempt {attempt+1}): "
                      f"{type(e).__name__}: {e}")

        if loss_name is None:
            # Fallback after 3 failed attempts
            loss_name = untried[0]
            reason = "(fallback: LLM unavailable or unhelpful, picking first untried option)"
            print(f"  Fallback choice: {loss_name}")

        # 2. Run the experiment
        try:
            metrics = run_experiment(exp_id, loss_name, reason)
        except Exception as e:
            print(f"[!] experiment {exp_id} crashed: {type(e).__name__}: {e}")
            # Mark it as a failure so it isn't retried infinitely
            failure_dir = EXPERIMENTS_DIR / f"exp_{exp_id:03d}_{loss_name}_FAILED"
            failure_dir.mkdir(exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "id": exp_id, "loss_name": loss_name, "reason": reason,
                "macro_auc": 0.0, "aves_auc": 0.0, "amphibia_auc": 0.0,
                "insecta_auc": 0.0, "mammalia_auc": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }, indent=2), encoding="utf-8")

    # Final summary
    print(f"\n{'#' * 70}\n# AGENT v1 — FINAL SUMMARY\n{'#' * 70}")
    final = load_past_experiments()
    if final:
        ranked = sorted(final, key=lambda e: e.get("macro_auc", 0.0), reverse=True)
        print(f"\n{'rank':<6}{'id':<5}{'loss':<22}{'macro_auc':>10}{'aves':>8}{'amph':>8}")
        for rank, e in enumerate(ranked, 1):
            print(f"{rank:<6}{e['id']:<5}{e['loss_name']:<22}"
                  f"{e.get('macro_auc', 0):>10.4f}"
                  f"{e.get('aves_auc', 0):>8.3f}{e.get('amphibia_auc', 0):>8.3f}")
        best = ranked[0]
        print(f"\nBEST: {best['loss_name']} (id={best['id']}, "
              f"macro_auc={best['macro_auc']:.4f})")
        print(f"  Model code:    {best.get('py_path')}")
        print(f"  Model weights: {best.get('weights_path')}")
    else:
        print("No experiments completed.")


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
        for d in EXPERIMENTS_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
        # Re-create the class weights file
        np.save(CLASS_WEIGHTS_PATH, pos_weights)
        print(f"Reset: cleared {EXPERIMENTS_DIR}")

    agent_loop(args.iterations)
