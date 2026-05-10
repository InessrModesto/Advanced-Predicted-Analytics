"""
BirdCLEF+ 2026 — Exploratory Data Analysis
==========================================

Goal: Understand the dataset BEFORE building the autonomous ML agent.
Every finding here should map to a specific design decision for the agent
(preprocessing, model size, loss function, augmentation, etc.).

The file is organized into VS Code / Jupyter-style cells (# %%). You can:
  - Run it top-to-bottom as a plain script:   python eda_birdclef.py
  - Run cell-by-cell in VS Code / Cursor / PyCharm (recommended)
  - Convert to a notebook with: jupytext --to notebook eda_birdclef.py

Expected dataset layout (download from Kaggle into DATA_DIR):
    birdclef-2026/
        train.csv                      # metadata for labeled training audio
        taxonomy.csv                   # species info
        train_audio/<species>/*.ogg    # focal recordings, organized per species
        train_soundscapes/*.ogg        # unlabeled long recordings (optional)
        test_soundscapes/*.ogg         # held-out test soundscapes
        sample_submission.csv

If your downloaded folder names differ slightly, adjust the paths in CONFIG.
"""


# %% ──────────────────────────────────────────────────────────────────────
# Cell 1: Imports and configuration
# ─────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import random
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

# Audio libraries — installed via:  pip install librosa soundfile
try:
    import librosa
    import librosa.display
    import soundfile as sf
    HAVE_AUDIO = True
except ImportError:
    print("[warn] librosa/soundfile not installed — audio cells will be skipped.")
    print("       Install with:  pip install librosa soundfile")
    HAVE_AUDIO = False


# ─── CONFIG: edit these paths to match where you downloaded the data ──────
DATA_DIR     = Path(".")                        # root of the Kaggle dataset
PLOT_DIR     = Path("./eda_plots")              # where figures get saved
SAMPLE_LIMIT = 50                               # cap on audio files probed for stats
RANDOM_SEED  = 42

PLOT_DIR.mkdir(exist_ok=True, parents=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Resolve subpaths (override here if Kaggle changes folder names)
TRAIN_CSV          = DATA_DIR / "train.csv"
TAXONOMY_CSV       = DATA_DIR / "taxonomy.csv"
TRAIN_AUDIO_DIR    = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES  = DATA_DIR / "train_soundscapes"
TEST_SOUNDSCAPES   = DATA_DIR / "test_soundscapes"

print(f"Data dir:   {DATA_DIR.resolve()}")
print(f"Plot dir:   {PLOT_DIR.resolve()}")
print(f"Audio libs: {'OK' if HAVE_AUDIO else 'missing'}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 2: Top-level dataset structure
# What files & folders did Kaggle give us? How big is everything?
# ─────────────────────────────────────────────────────────────────────────
def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def folder_summary(path: Path) -> dict:
    """Walk a folder and return file count + total bytes per extension."""
    if not path.exists():
        return {}
    stats: dict[str, list[int]] = {}
    for root, _, files in os.walk(path):
        for f in files:
            ext = Path(f).suffix.lower() or "<noext>"
            full = Path(root) / f
            try:
                size = full.stat().st_size
            except OSError:
                continue
            stats.setdefault(ext, []).append(size)
    return {ext: (len(sizes), sum(sizes)) for ext, sizes in stats.items()}


print("\n=== Top-level contents of DATA_DIR ===")
if DATA_DIR.exists():
    for item in sorted(DATA_DIR.iterdir()):
        kind = "DIR " if item.is_dir() else "FILE"
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) \
               if item.is_dir() else item.stat().st_size
        print(f"  {kind}  {item.name:<35}  {human_bytes(size)}")
else:
    print(f"  [!] {DATA_DIR} not found. Download the dataset from Kaggle first.")

for label, p in [("train_audio", TRAIN_AUDIO_DIR),
                 ("train_soundscapes", TRAIN_SOUNDSCAPES),
                 ("test_soundscapes", TEST_SOUNDSCAPES)]:
    s = folder_summary(p)
    if s:
        print(f"\n  {label}:")
        for ext, (n, total) in sorted(s.items()):
            print(f"    {ext:<8} {n:>7} files   {human_bytes(total):>10}")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 3: Load metadata (train.csv + taxonomy.csv)
# These tell us: which species, how many examples, what extra labels exist
# ─────────────────────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV) if TRAIN_CSV.exists() else None
tax_df   = pd.read_csv(TAXONOMY_CSV) if TAXONOMY_CSV.exists() else None

if train_df is None:
    raise SystemExit(f"train.csv not found at {TRAIN_CSV}. Stop here until data is downloaded.")

print("=== train.csv ===")
print(f"shape: {train_df.shape}")
print(f"columns: {list(train_df.columns)}")
print("\nfirst rows:")
print(train_df.head(3).to_string())

print("\nmissing values per column:")
print(train_df.isna().sum().to_string())

print("\ndtypes:")
print(train_df.dtypes.to_string())

if tax_df is not None:
    print(f"\n=== taxonomy.csv === shape: {tax_df.shape}")
    print(tax_df.head(5).to_string())


# %% ──────────────────────────────────────────────────────────────────────
# Cell 4: Class distribution — *the* most important EDA finding
# Bird datasets are notoriously long-tailed. The agent must know this.
# ─────────────────────────────────────────────────────────────────────────
LABEL_COL = "primary_label"  # standard BirdCLEF column; adjust if renamed

counts = train_df[LABEL_COL].value_counts()
print(f"=== Class balance ({LABEL_COL}) ===")
print(f"unique species:        {counts.size}")
print(f"total samples:         {counts.sum()}")
print(f"min samples / species: {counts.min()}")
print(f"max samples / species: {counts.max()}")
print(f"median:                {counts.median():.0f}")
print(f"mean:                  {counts.mean():.1f}")
print(f"\nspecies with <10 samples: {(counts < 10).sum()}")
print(f"species with <5 samples:  {(counts < 5).sum()}")

# Plot 1: full distribution sorted
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ax = axes[0]
ax.bar(range(counts.size), counts.values, width=1.0)
ax.set_xlabel("species (sorted by frequency)")
ax.set_ylabel("number of samples")
ax.set_title(f"Long-tail class distribution ({counts.size} species)")
ax.set_yscale("log")

ax = axes[1]
ax.hist(counts.values, bins=40)
ax.set_xlabel("samples per species")
ax.set_ylabel("number of species")
ax.set_title("Histogram of class sizes")

plt.tight_layout()
plt.savefig(PLOT_DIR / "01_class_distribution.png")

# Plot 2: top 20 + bottom 20 species
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts.head(20).plot.barh(ax=axes[0], color="steelblue")
axes[0].invert_yaxis()
axes[0].set_title("Top 20 most-represented species")
axes[0].set_xlabel("samples")

counts.tail(20).plot.barh(ax=axes[1], color="indianred")
axes[1].invert_yaxis()
axes[1].set_title("Bottom 20 (rarest) species")
axes[1].set_xlabel("samples")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_top_bottom_species.png")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 5: Multi-label structure — secondary_labels
# The competition scores per-window multi-label predictions, so understanding
# how often multiple species appear in a single training clip matters.
# ─────────────────────────────────────────────────────────────────────────
if "secondary_labels" in train_df.columns:
    # Kaggle stores this as a string repr of a list, e.g. "['amerob','sonspa']"
    def parse_list(s):
        if pd.isna(s) or s in ("[]", "['']", ""):
            return []
        try:
            return [x.strip(" '\"") for x in s.strip("[]").split(",") if x.strip(" '\"")]
        except Exception:
            return []

    sec = train_df["secondary_labels"].apply(parse_list)
    n_secondary = sec.apply(len)

    print("=== secondary_labels stats ===")
    print(f"clips with >=1 secondary label: {(n_secondary > 0).sum()} "
          f"({100*(n_secondary>0).mean():.1f}%)")
    print(f"max secondary labels in a clip: {n_secondary.max()}")
    print(f"mean (over multi-label clips):  {n_secondary[n_secondary>0].mean():.2f}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(n_secondary, bins=range(0, n_secondary.max() + 2), align="left",
            edgecolor="black")
    ax.set_xlabel("# of secondary labels in a clip")
    ax.set_ylabel("# of clips")
    ax.set_title("Multi-label complexity (secondary labels per clip)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_secondary_labels.png")

    # Most common secondary labels (often = common background species)
    flat = [lbl for lst in sec for lbl in lst]
    if flat:
        common = Counter(flat).most_common(15)
        print("\nMost common secondary labels (background species):")
        for lbl, n in common:
            print(f"  {lbl:<15} {n}")
else:
    print("[info] no `secondary_labels` column — skipping multi-label analysis.")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 6: Audio file characteristics — duration, sample rate, channels
# Probe a stratified sample of files. Don't open everything — that's slow.
# ─────────────────────────────────────────────────────────────────────────
if not HAVE_AUDIO:
    print("[skip] audio libs not available")
else:
    # Pick up to SAMPLE_LIMIT files, stratified across species
    if "filename" in train_df.columns:
        path_col = "filename"
    elif "filepath" in train_df.columns:
        path_col = "filepath"
    else:
        # Fall back: build from primary_label folder
        path_col = None

    sample_rows = (train_df
                   .groupby(LABEL_COL, group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), 3), random_state=RANDOM_SEED))
                   .sample(min(SAMPLE_LIMIT, len(train_df)), random_state=RANDOM_SEED)
                   .reset_index(drop=True))

    records = []
    skipped = 0
    debug_count = 0
    print(f"[debug] path_col = {path_col}, sample_rows.shape = {sample_rows.shape}")
    # Get the column index for LABEL_COL
    label_col_idx = train_df.columns.get_loc(LABEL_COL) if LABEL_COL in train_df.columns else 0
    for _, row in sample_rows.iterrows():
        if path_col:
            audio_path = TRAIN_AUDIO_DIR / row[path_col]
        else:
            # Try {species}/{first .ogg in folder}
            spec_dir = TRAIN_AUDIO_DIR / row[LABEL_COL]
            files = list(spec_dir.glob("*.ogg")) if spec_dir.exists() else []
            if not files:
                skipped += 1
                continue
            audio_path = random.choice(files)

        if not audio_path.exists():
            skipped += 1
            continue
        debug_count += 1
        if debug_count <= 3:
            print(f"[debug] Attempting to load: {audio_path}, exists={audio_path.exists()}")
        try:
            info = sf.info(str(audio_path))
            records.append({
                "species":   row.iloc[label_col_idx] if hasattr(row, 'iloc') else row[LABEL_COL],
                "duration":  info.duration,
                "samplerate": info.samplerate,
                "channels":  info.channels,
                "format":    info.format,
                "subtype":   info.subtype,
                "size_mb":   audio_path.stat().st_size / 1e6,
            })
        except Exception as e:
            if len(records) == 0:  # Print first error only
                print(f"[debug] sf.info failed on {audio_path}: {type(e).__name__}: {e}")
            # Try librosa as fallback for OGG files
            try:
                y, sr = librosa.load(str(audio_path), sr=None, mono=False)
                records.append({
                    "species":   row.iloc[label_col_idx] if hasattr(row, 'iloc') else row[LABEL_COL],
                    "duration":  len(y) / sr if isinstance(y, np.ndarray) and y.ndim == 1 else len(y[0]) / sr,
                    "samplerate": sr,
                    "channels":  1 if isinstance(y, np.ndarray) and y.ndim == 1 else y.shape[0],
                    "format":    "ogg",
                    "subtype":   "vorbis",
                    "size_mb":   audio_path.stat().st_size / 1e6,
                })
            except Exception as e2:
                if len(records) == 0:
                    print(f"[debug] librosa also failed: {type(e2).__name__}: {e2}")
                skipped += 1

    audio_df = pd.DataFrame(records)
    print(f"=== Probed {len(audio_df)} audio files ({skipped} skipped) ===\n")
    
    if len(audio_df) > 0:
        print(audio_df[["duration", "samplerate", "channels", "size_mb"]]
              .describe(percentiles=[.05, .5, .95]).round(2).to_string())
        print("\nUnique sample rates:", audio_df["samplerate"].value_counts().to_dict())
        print("Unique channels:    ", audio_df["channels"].value_counts().to_dict())
        print("Unique formats:     ", audio_df["format"].value_counts().to_dict())

        # Plot duration distribution
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax = axes[0]
        ax.hist(audio_df["duration"], bins=60)
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("# files")
        ax.set_title("Clip duration distribution")
        ax.axvline(5, color="red", linestyle="--", label="5s window (eval unit)")
        ax.legend()

        ax = axes[1]
        ax.hist(np.log10(audio_df["duration"].clip(lower=0.1)), bins=60)
        ax.set_xlabel("log10(duration in s)")
        ax.set_ylabel("# files")
        ax.set_title("Duration distribution (log scale)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "04_duration_distribution.png")
    else:
        print("[warn] No audio files could be loaded for analysis.")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 7: Visualize sample audio — waveform + mel-spectrogram
# This is what the CNN will actually "see". Look at a few species.
# ─────────────────────────────────────────────────────────────────────────
if HAVE_AUDIO and len(records) > 0:
    # Pick 4 species: 2 common, 2 rare — see if the visual pattern differs
    species_sorted = counts.index.tolist()
    chosen = [species_sorted[0], species_sorted[1],
              species_sorted[-3], species_sorted[-2]]

    SR = 32000          # canonical SR; resample on load
    N_MELS = 128
    HOP = 512

    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 2.6 * len(chosen)))
    for i, sp in enumerate(chosen):
        spec_dir = TRAIN_AUDIO_DIR / sp
        files = list(spec_dir.glob("*.ogg")) if spec_dir.exists() else []
        if not files:
            continue
        f = files[0]
        try:
            y, sr = librosa.load(f, sr=SR, mono=True, duration=10.0)
        except Exception as e:
            print(f"could not load {f}: {e}")
            continue

        # Waveform
        ax = axes[i, 0] if len(chosen) > 1 else axes[0]
        ax.plot(np.linspace(0, len(y) / sr, len(y)), y, linewidth=0.5)
        ax.set_title(f"{sp} — waveform ({counts[sp]} samples in train)")
        ax.set_xlabel("time (s)")

        # Mel-spectrogram (log-power)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
        S_db = librosa.power_to_db(S, ref=np.max)
        ax = axes[i, 1] if len(chosen) > 1 else axes[1]
        librosa.display.specshow(S_db, sr=sr, hop_length=HOP,
                                 x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(f"{sp} — mel-spectrogram (n_mels={N_MELS})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_sample_spectrograms.png")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 8: Train vs. soundscape comparison — the domain-shift question
# Training audio is mostly clean focal recordings (Xeno-Canto style).
# Test audio is long Pantanal soundscapes with noise, multiple birds, silence.
# This shift is what makes BirdCLEF hard. Visualize it.
# ─────────────────────────────────────────────────────────────────────────
if HAVE_AUDIO:
    # Prefer test_soundscapes only if it has actual audio in it; otherwise use train_soundscapes
    test_has_audio = TEST_SOUNDSCAPES.exists() and any(TEST_SOUNDSCAPES.glob("*.ogg"))
    soundscape_dir = TEST_SOUNDSCAPES if test_has_audio else TRAIN_SOUNDSCAPES
    print(f"[info] Using soundscape directory: {soundscape_dir}")
    if soundscape_dir.exists():
        ss_files = list(soundscape_dir.glob("*.ogg"))[:3]
        if ss_files:
            fig, axes = plt.subplots(len(ss_files), 1, figsize=(13, 2.8 * len(ss_files)))
            if len(ss_files) == 1:
                axes = [axes]
            for ax, f in zip(axes, ss_files):
                try:
                    y, sr = librosa.load(f, sr=32000, mono=True, duration=30.0)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
                    S_db = librosa.power_to_db(S, ref=np.max)
                    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                             x_axis="time", y_axis="mel", ax=ax)
                    ax.set_title(f"Soundscape: {f.name}  (first 30s)")
                except Exception as e:
                    print(f"failed: {f.name}: {e}")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "06_soundscape_examples.png")

            # Quantify: probe duration of soundscapes
            ss_durs = []
            for f in list(soundscape_dir.glob("*.ogg"))[:50]:
                try:
                    ss_durs.append(sf.info(str(f)).duration)
                except Exception:
                    pass
            if ss_durs:
                print(f"\nSoundscape durations (n={len(ss_durs)}):")
                print(f"  min  : {min(ss_durs):.1f}s")
                print(f"  median: {np.median(ss_durs):.1f}s")
                print(f"  max  : {max(ss_durs):.1f}s")
                print("  → split into 5s windows for prediction")
        else:
            print(f"[info] {soundscape_dir} exists but no .ogg files found.")
    else:
        print(f"[info] no soundscape folder found — skipping domain-shift cell.")

# Inspect soundscape labels — the most important file we haven't examined
ss_labels_path = DATA_DIR / "train_soundscapes_labels.csv"
if ss_labels_path.exists():
    ss = pd.read_csv(ss_labels_path)
    print(f"shape: {ss.shape}")
    print(f"columns: {list(ss.columns)}")
    print(ss.head(10).to_string())
    print(f"\nunique soundscape files labeled: {ss['filename'].nunique() if 'filename' in ss.columns else 'check schema'}")
    print(f"unique species in soundscape labels: {ss.iloc[:, -1].nunique()}")

# Taxonomy and collection breakdown
print("=== Taxonomic class distribution ===")
print(train_df['class_name'].value_counts().to_string())
print(f"\n=== Collection (data source) distribution ===")
print(train_df['collection'].value_counts().to_string())
print(f"\n=== Per-class sample stats ===")
print(train_df.groupby('class_name').agg(
    n_clips=('filename', 'count'),
    n_species=('primary_label', 'nunique'),
    median_rating=('rating', 'median')
).to_string())

# %% ──────────────────────────────────────────────────────────────────────
# Cell 9: Metadata exploration — geography, quality, time
# These can be used as features, or to design smart train/val splits.
# ─────────────────────────────────────────────────────────────────────────
extra_cols = [c for c in ("latitude", "longitude", "rating", "author",
                          "license", "type", "common_name")
              if c in train_df.columns]
print(f"=== Available metadata columns: {extra_cols} ===")

# Quality rating (Xeno-Canto rating, 0-5) — affects label noise
if "rating" in train_df.columns:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    train_df["rating"].plot.hist(bins=20, ax=ax, edgecolor="black")
    ax.set_title("Recording quality rating distribution")
    ax.set_xlabel("rating (higher = cleaner)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_rating_distribution.png")
    print(f"  median rating: {train_df['rating'].median()}")
    print(f"  fraction with rating >= 4: {(train_df['rating'] >= 4).mean():.2%}")

# Geographic distribution
if {"latitude", "longitude"}.issubset(train_df.columns):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = train_df.dropna(subset=["latitude", "longitude"])
    ax.scatter(sub["longitude"], sub["latitude"], s=2, alpha=0.3)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Recording locations (n={len(sub)})")
    # Highlight Pantanal (test domain): roughly -58 to -55 lon, -20 to -16 lat
    ax.add_patch(plt.Rectangle((-58, -20), 3, 4, fill=False,
                               edgecolor="red", linewidth=2, label="Pantanal (test domain)"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "08_geographic_distribution.png")
    in_pantanal = ((sub["longitude"].between(-58, -55)) &
                   (sub["latitude"].between(-20, -16))).sum()
    print(f"  training recordings inside Pantanal box: {in_pantanal} "
          f"({100*in_pantanal/len(sub):.1f}%)")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 10: Estimated total spectrogram dataset size
# Important: tells the agent whether spectrograms can fit in RAM, must be
# precomputed to disk, or generated on-the-fly.
# ─────────────────────────────────────────────────────────────────────────
if HAVE_AUDIO and len(audio_df) > 0:
    SR = 32000
    N_MELS = 128
    HOP = 512
    median_dur = audio_df["duration"].median()
    n_clips = len(train_df)

    frames_per_clip = int(median_dur * SR / HOP)
    bytes_per_clip_f32 = N_MELS * frames_per_clip * 4   # float32
    bytes_per_clip_u8  = N_MELS * frames_per_clip       # uint8 (after normalization)

    total_f32 = bytes_per_clip_f32 * n_clips
    total_u8  = bytes_per_clip_u8  * n_clips

    print("=== Dataset memory estimate (full mel-spectrogram cache) ===")
    print(f"  median clip duration:    {median_dur:.1f}s")
    print(f"  frames/clip @ hop {HOP}:   {frames_per_clip}")
    print(f"  total clips:             {n_clips}")
    print(f"  size as float32:         {human_bytes(total_f32)}")
    print(f"  size as uint8 (norm.):   {human_bytes(total_u8)}")
    print(f"\n  → if >> available RAM, the agent must either:")
    print(f"      (a) precompute & save to disk (memmap / .npy / lmdb)")
    print(f"      (b) compute spectrograms on-the-fly in the data pipeline")


# %% ──────────────────────────────────────────────────────────────────────
# Cell 11: Summary — what the agent must handle
# Translate findings into concrete agent-design constraints.
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(" SUMMARY: implications for the autonomous research agent")
print("=" * 70)

bullets = []

# 1. Class imbalance
if counts.size > 0:
    imbalance = counts.max() / max(counts.min(), 1)
    rare = (counts < 10).sum()
    bullets.append(
        f"Class imbalance ratio (max/min) = {imbalance:.1f}; "
        f"{rare} species have <10 samples.\n"
        f"   → Agent should consider: weighted BCE loss, focal loss, "
        f"oversampling rare species,\n"
        f"     mixup/cutmix augmentation, transfer learning from a pretrained "
        f"audio backbone."
    )

# 2. Multi-label
if "secondary_labels" in train_df.columns:
    bullets.append(
        "Multi-label task — many clips have secondary species in the background.\n"
        "   → Output layer = 234 sigmoids (NOT softmax). Loss = BCEWithLogits."
    )

# 3. Domain shift
bullets.append(
    "Domain shift — train clips are mostly focal/clean (Xeno-Canto-like);\n"
    "   test = long Pantanal soundscapes with noise & silence.\n"
    "   → Agent should explore: noise augmentation, random gain, random cropping,\n"
    "     pink-noise mixup with soundscape backgrounds, SpecAugment."
)

# 4. Compute constraint
bullets.append(
    "Submission constraint: CPU-only Kaggle notebook, 90-minute runtime.\n"
    "   → Inference model must be SMALL & efficient. Good candidates: \n"
    "     EfficientNet-B0, MobileNetV3-Small, or a custom 4-block CNN.\n"
    "   → Agent should track: (latency per clip × #test clips) ≤ ~80 min."
)

# 5. Audio length
if HAVE_AUDIO and len(audio_df) > 0:
    p95 = audio_df["duration"].quantile(0.95)
    bullets.append(
        f"Variable clip length (median {audio_df['duration'].median():.0f}s, "
        f"p95 {p95:.0f}s) but the eval unit is fixed 5-second windows.\n"
        f"   → Standard recipe: random-crop a 5s segment during training; "
        f"average per-window predictions at test time."
    )

# 6. Evaluation metric
bullets.append(
    "Metric: macro-averaged ROC-AUC, skipping classes with no positives.\n"
    "   → Rare classes count just as much as common ones in the score.\n"
    "   → Validation must reflect this — track per-class AUC, not accuracy."
)

for i, b in enumerate(bullets, 1):
    print(f"\n {i}. {b}")

print(f"\nAll plots saved to: {PLOT_DIR.resolve()}")
print("Next step: based on these findings, sketch the agent's experiment space")
print("(what hyperparams it can vary) and its prompt template for the LLM.")