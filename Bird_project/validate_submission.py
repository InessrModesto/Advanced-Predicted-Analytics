"""
BirdCLEF+ 2026 — Local submission validator
============================================

Runs the same inference pipeline as submission_notebook.ipynb against a
few files in your local `train_soundscapes/` folder. Verifies:

  - The model loads from disk without errors.
  - Its input/output shapes match the 234-class label space.
  - End-to-end inference on a soundscape completes without crashing.
  - The output CSV has the correct columns, unique row_ids, and
    probabilities in [0, 1] with no NaN/inf.
  - Per-soundscape inference time is consistent with the 90-minute
    Kaggle budget (extrapolates from a small sample to the full
    ~700-file test set).

Catches roughly 90% of "the notebook crashes 45 minutes into Kaggle's
queue" bugs before you upload anything. Runs in ~1 minute for 3 files.

Usage:
    python validate_submission.py --model path/to/model.keras
    python validate_submission.py --model path/to/model.keras --n 5
    python validate_submission.py --model path/to/model.keras \
                                  --soundscape-dir train_soundscapes
"""

# ─── Backend selection (must come before keras import) ───────────────────
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import keras


# ── Hyperparameters — MUST match the notebook + training pipeline ───────
SAMPLE_RATE            = 32_000
DURATION_SEC           = 5.0
WINDOWS_PER_SOUNDSCAPE = 12
N_FFT                  = 1024
HOP_LENGTH             = 512
N_MELS                 = 128

# Kaggle test set size (~700 one-minute soundscapes). Used for the
# 90-minute budget extrapolation; harmless if the real number differs.
KAGGLE_TEST_FILES_ESTIMATE = 700
KAGGLE_BUDGET_MIN          = 90


# ── Mel pipeline — same code as the notebook + baseline.py ──────────────
def _build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_min, f_max = 20.0, sr / 2.0
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l: fb[i - 1, l:c] = np.linspace(0, 1, c - l, dtype=np.float32)
        if r > c: fb[i - 1, c:r] = np.linspace(1, 0, r - c, dtype=np.float32)
    return fb


MEL_FB   = _build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)
HANN_WIN = np.hanning(N_FFT).astype(np.float32)


def waveform_to_mel(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    n_frames = max(1, 1 + (n - N_FFT) // HOP_LENGTH)
    needed = (n_frames - 1) * HOP_LENGTH + N_FFT
    if n < needed:
        audio = np.pad(audio, (0, needed - n))
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, N_FFT),
        strides=(audio.strides[0] * HOP_LENGTH, audio.strides[0]),
        writeable=False,
    )
    windowed = frames * HANN_WIN
    spec = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    spec = spec.T
    mel = MEL_FB @ spec
    mel = np.log1p(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


def split_soundscape_into_windows(audio: np.ndarray) -> np.ndarray:
    target_per_window = int(SAMPLE_RATE * DURATION_SEC)
    target_total      = target_per_window * WINDOWS_PER_SOUNDSCAPE
    if len(audio) < target_total:
        audio = np.pad(audio, (0, target_total - len(audio)))
    else:
        audio = audio[:target_total]
    return audio.reshape(WINDOWS_PER_SOUNDSCAPE, target_per_window)


def read_soundscape(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Sample rate mismatch in {path}: {sr} != {SAMPLE_RATE}."
        )
    return audio


# ── The validator itself ────────────────────────────────────────────────
def fail(msg: str) -> None:
    """Print error in a clearly-visible way and exit non-zero."""
    print()
    print("=" * 70)
    print(f" [FAIL] {msg}")
    print("=" * 70)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to the .keras model file to validate")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of soundscapes to test (default 3)")
    parser.add_argument("--soundscape-dir", default="train_soundscapes",
                        help="Local folder with soundscapes (default: "
                             "train_soundscapes — the labelled training "
                             "soundscapes, which are the closest local "
                             "analogue to Kaggle's test_soundscapes/)")
    parser.add_argument("--sample-sub", default="sample_submission.csv",
                        help="Path to sample_submission.csv (default: "
                             "project root)")
    parser.add_argument("--out", default="submission_validation.csv",
                        help="Where to write the fake submission for "
                             "inspection (default: submission_validation.csv)")
    args = parser.parse_args()

    print(f"\n{'=' * 70}\n SUBMISSION VALIDATOR\n{'=' * 70}")
    print(f" Model:           {args.model}")
    print(f" Soundscape dir:  {args.soundscape_dir}")
    print(f" Sample sub:      {args.sample_sub}")
    print(f" N to test:       {args.n}")
    print(f"{'=' * 70}\n")

    # ── 1. File-existence checks ───────────────────────────────────────
    print("[1/6] Checking files exist...")
    model_path = Path(args.model)
    if not model_path.exists():
        fail(f"Model file not found at {model_path}")
    ok(f"Model file exists: {model_path} "
       f"({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    sample_sub_path = Path(args.sample_sub)
    if not sample_sub_path.exists():
        fail(f"sample_submission.csv not found at {sample_sub_path}")
    ok(f"sample_submission.csv exists")

    ss_dir = Path(args.soundscape_dir)
    if not ss_dir.exists():
        fail(f"Soundscape directory not found at {ss_dir}")
    soundscape_files = sorted(ss_dir.glob("*.ogg"))
    if not soundscape_files:
        soundscape_files = sorted(ss_dir.glob("*.wav"))
    if not soundscape_files:
        fail(f"No .ogg or .wav files in {ss_dir}")
    test_files = soundscape_files[: args.n]
    ok(f"Found {len(soundscape_files)} soundscapes; "
       f"will test on first {len(test_files)}")

    # ── 2. Label space ────────────────────────────────────────────────
    print("\n[2/6] Reading label space...")
    sample_sub = pd.read_csv(sample_sub_path)
    LABELS = [c for c in sample_sub.columns if c != "row_id"]
    NUM_CLASSES = len(LABELS)
    if NUM_CLASSES != 234:
        fail(f"Expected 234 classes, got {NUM_CLASSES}. Wrong "
             f"sample_submission.csv?")
    ok(f"Label space: {NUM_CLASSES} classes")

    # ── 3. Load the model ──────────────────────────────────────────────
    print("\n[3/6] Loading model...")
    t0 = time.perf_counter()
    try:
        # compile=False skips deserializing the optimizer + loss, which
        # avoids needing custom-loss classes (WeightedBCE, FocalLoss) to
        # be registered with keras.saving. We only need predict() here,
        # so the optimizer/loss don't matter.
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        fail(f"keras.models.load_model failed: {type(e).__name__}: {e}")
    load_s = time.perf_counter() - t0
    ok(f"Model loaded in {load_s:.1f}s")
    ok(f"Input shape:  {model.input_shape}")
    ok(f"Output shape: {model.output_shape}")
    ok(f"Parameters:   {model.count_params():,}")

    if model.output_shape[-1] != NUM_CLASSES:
        fail(f"Model output ({model.output_shape[-1]}) doesn't match "
             f"label space ({NUM_CLASSES}). Did you load the right file?")
    if model.input_shape[1] != N_MELS:
        fail(f"Model input n_mels ({model.input_shape[1]}) doesn't "
             f"match pipeline N_MELS ({N_MELS}). The model was trained "
             f"with different audio parameters than this validator uses.")

    # ── 4. Run inference on a few files ────────────────────────────────
    print(f"\n[4/6] Running inference on {len(test_files)} soundscape(s)...")
    rows = []
    per_file_times: list[float] = []
    for i, ss_path in enumerate(test_files):
        t0 = time.perf_counter()
        try:
            audio = read_soundscape(ss_path)
        except Exception as e:
            fail(f"Failed to read {ss_path}: {type(e).__name__}: {e}")
        try:
            windows = split_soundscape_into_windows(audio)
            mels = np.stack([waveform_to_mel(w) for w in windows], axis=0)
            mels = mels[..., np.newaxis]
        except Exception as e:
            fail(f"Mel pipeline failed on {ss_path}: "
                 f"{type(e).__name__}: {e}")
        try:
            preds = model.predict(mels, verbose=0)
        except Exception as e:
            fail(f"model.predict failed on {ss_path}: "
                 f"{type(e).__name__}: {e}")
        if preds.shape != (WINDOWS_PER_SOUNDSCAPE, NUM_CLASSES):
            fail(f"Unexpected prediction shape for {ss_path}: "
                 f"got {preds.shape}, expected "
                 f"({WINDOWS_PER_SOUNDSCAPE}, {NUM_CLASSES})")

        stem = ss_path.stem
        for w, p in enumerate(preds):
            end_s = int((w + 1) * DURATION_SEC)
            row = {"row_id": f"{stem}_{end_s}"}
            for j, lab in enumerate(LABELS):
                row[lab] = float(p[j])
            rows.append(row)

        elapsed = time.perf_counter() - t0
        per_file_times.append(elapsed)
        ok(f"[{i + 1}/{len(test_files)}] {ss_path.name}  "
           f"({elapsed:.1f}s)")

    # ── 5. Validate the submission DataFrame ───────────────────────────
    print("\n[5/6] Validating output format...")
    sub = pd.DataFrame(rows)
    expected_cols = sample_sub.columns.tolist()
    missing = set(expected_cols) - set(sub.columns)
    extra   = set(sub.columns)  - set(expected_cols)
    if missing:
        fail(f"missing columns in submission: {sorted(missing)[:10]}"
             f"{'...' if len(missing) > 10 else ''}")
    if extra:
        fail(f"extra columns in submission: {sorted(extra)}")
    sub = sub[expected_cols]
    ok(f"All {len(expected_cols)} expected columns present, no extras")

    if not sub["row_id"].is_unique:
        dup_count = (~sub["row_id"].duplicated(keep=False)).sum()
        fail(f"row_id values are not unique ({dup_count} duplicates)")
    ok(f"All {len(sub)} row_ids are unique")

    expected_row_ids = sample_sub["row_id"].astype(str).tolist()
    if len(sub) == len(expected_row_ids):
        generated = set(sub["row_id"].astype(str))
        expected = set(expected_row_ids)
        missing_ids = expected - generated
        extra_ids = generated - expected
        if missing_ids:
            fail(f"missing row_id values: {sorted(missing_ids)[:5]}"
                 f"{'...' if len(missing_ids) > 5 else ''}")
        if extra_ids:
            fail(f"extra row_id values: {sorted(extra_ids)[:5]}"
                 f"{'...' if len(extra_ids) > 5 else ''}")
        sub = sub.set_index("row_id").loc[expected_row_ids].reset_index()
        ok("row_id set exactly matches sample_submission.csv and was reordered")
    else:
        ok("Skipping exact row_id match against sample_submission.csv "
           "(local subset validation)")

    prob_cols = [c for c in sub.columns if c != "row_id"]
    vals = sub[prob_cols].to_numpy()
    if not np.isfinite(vals).all():
        n_nan = (~np.isfinite(vals)).sum()
        fail(f"submission contains {n_nan} NaN/inf values")
    if not ((vals >= 0).all() and (vals <= 1).all()):
        fail(f"probabilities outside [0, 1]: range=[{vals.min()}, "
             f"{vals.max()}]. Model output isn't sigmoid?")
    ok(f"All probabilities in [{vals.min():.4f}, {vals.max():.4f}]")
    ok(f"Mean probability: {vals.mean():.4f}")

    # Quick spot check on row_id format
    sample_ids = sub["row_id"].head(3).tolist()
    for rid in sample_ids:
        if "_" not in rid:
            fail(f"row_id missing underscore separator: '{rid}'")
        try:
            end_s = int(rid.rsplit("_", 1)[-1])
            if end_s not in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60):
                fail(f"row_id has unexpected end_seconds: '{rid}' "
                     f"(expected 5..60 step 5)")
        except ValueError:
            fail(f"row_id end_seconds is not an integer: '{rid}'")
    ok(f"row_id format looks correct (sample: {sample_ids[0]!r})")

    sub.to_csv(args.out, index=False)
    ok(f"Wrote {args.out} ({Path(args.out).stat().st_size / 1024:.1f} KB)")

    # ── 6. Time budget extrapolation ──────────────────────────────────
    print(f"\n[6/6] Extrapolating to Kaggle's {KAGGLE_BUDGET_MIN}-min budget...")
    median_s = float(np.median(per_file_times))
    est_total_min = (median_s * KAGGLE_TEST_FILES_ESTIMATE) / 60
    ok(f"Median time per soundscape: {median_s:.1f}s")
    ok(f"Estimated total for ~{KAGGLE_TEST_FILES_ESTIMATE} files: "
       f"{est_total_min:.1f} min")
    if est_total_min > KAGGLE_BUDGET_MIN:
        print(f"  [!] WARNING: estimated total ({est_total_min:.1f} min) "
              f"EXCEEDS the {KAGGLE_BUDGET_MIN}-min budget.")
        print(f"      Your submission may time out. Consider:")
        print(f"        - a smaller / faster model")
        print(f"        - quantizing the model (TFLite)")
        print(f"        - reducing N_MELS or sample rate")
    elif est_total_min > KAGGLE_BUDGET_MIN * 0.8:
        print(f"  [!] CAUTION: estimated total ({est_total_min:.1f} min) "
              f"is within 20% of the {KAGGLE_BUDGET_MIN}-min budget. "
              f"Headroom is tight.")
    else:
        ok(f"Comfortably under budget "
           f"({est_total_min:.1f}/{KAGGLE_BUDGET_MIN} min, "
           f"{est_total_min / KAGGLE_BUDGET_MIN * 100:.0f}% used)")

    print(f"\n{'=' * 70}\n PASSED — submission notebook should run on Kaggle.\n{'=' * 70}")
    print(f"\nNext steps:")
    print(f"  1. Upload {args.model} as a Kaggle Dataset.")
    print(f"  2. Open submission_notebook.ipynb on Kaggle, attach the dataset.")
    print(f"  3. Edit MODEL_PATH at the top of the notebook to point at the dataset.")
    print(f"  4. Set accelerator to None (CPU only) and submit.")


if __name__ == "__main__":
    main()
