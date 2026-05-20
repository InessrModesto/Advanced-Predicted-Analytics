"""
Build the live-predictions demo HTML from demo_predictions.json.

Self-contained output — single HTML file with audio embedded. Open in
any browser.

Usage:
    python make_predictions_demo.py --model path/to/model.keras
    python make_predictions_demo.py --model path/to/model.keras --n 5
    python make_predictions_demo.py --in demo_predictions.json --out demo.html
"""
import argparse
import base64
import io
import json
import mimetypes
import os
import time
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Listen — BirdCLEF+ 2026</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #18181a;
    --bg-panel:  #1f1f22;
    --bg-row:    #232327;
    --text:      #e8e2d4;
    --text-dim:  #8a8579;
    --text-faint:#5a574f;
    --accent:    #d4a574;
    --accent-dim:#7a5e3f;
    --green:     #7fa67a;
    --rule:      #2c2c30;
    --display: "Fraunces", Georgia, serif;
    --body:    "Inter", system-ui, sans-serif;
    --mono:    "JetBrains Mono", ui-monospace, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: var(--body); font-size: 14px; line-height: 1.5; }
  ::selection { background: var(--accent); color: var(--bg); }

  /* HEADER */
  header {
    border-bottom: 1px solid var(--rule);
    padding: 28px 40px 22px;
    display: flex; align-items: baseline; justify-content: space-between; gap: 40px;
  }
  header .title {
    font-family: var(--display); font-weight: 300; font-size: 34px;
    letter-spacing: -0.02em; line-height: 1;
  }
  header .title em { font-style: italic; color: var(--accent); font-weight: 500; }
  header .subtitle { color: var(--text-dim); font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; }
  header .model-info {
    font-family: var(--mono); font-size: 11px; color: var(--text-faint);
    text-align: right; line-height: 1.6;
  }
  header .model-info strong { color: var(--text); font-weight: 500; }
  header .model-info .accent { color: var(--accent); }

  /* SOUNDSCAPE PICKER */
  .picker {
    border-bottom: 1px solid var(--rule);
    padding: 14px 40px;
    display: flex; align-items: center; gap: 16px; overflow-x: auto;
  }
  .picker .label {
    font-family: var(--mono); font-size: 10px; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 0.08em; flex-shrink: 0;
  }
  .picker .pill {
    background: transparent; border: 1px solid var(--rule);
    color: var(--text-dim); padding: 6px 14px;
    cursor: pointer; font-family: var(--mono); font-size: 11px;
    transition: all 120ms ease; white-space: nowrap; flex-shrink: 0;
    border-radius: 0;
  }
  .picker .pill:hover { color: var(--text); border-color: var(--text-faint); }
  .picker .pill.active {
    background: var(--accent); border-color: var(--accent);
    color: var(--bg); font-weight: 700;
  }

  /* MAIN LAYOUT */
  main {
    display: grid; grid-template-columns: 1fr 480px;
    min-height: calc(100vh - 200px);
  }

  /* LEFT: spectrogram + audio player */
  .stage {
    padding: 32px 40px;
    display: flex; flex-direction: column; gap: 20px;
  }
  .stage-header {
    display: flex; justify-content: space-between; align-items: baseline;
  }
  .stage-filename {
    font-family: var(--mono); font-size: 13px; color: var(--text);
  }
  .stage-time {
    font-family: var(--mono); font-size: 12px; color: var(--text-faint);
  }
  .stage-time .now { color: var(--accent); font-weight: 700; }

  .spectrogram-wrap {
    position: relative; background: var(--bg-panel); padding: 12px;
    border: 1px solid var(--rule);
  }
  .spectrogram-wrap img {
    width: 100%; display: block; height: 240px; object-fit: fill;
    image-rendering: pixelated;
  }
  /* Time grid overlay — 12 ticks for the 12 windows */
  .time-grid {
    position: absolute; top: 12px; left: 12px; right: 12px; bottom: 12px;
    pointer-events: none;
    display: grid; grid-template-columns: repeat(12, 1fr);
  }
  .time-grid .tick {
    border-right: 1px solid rgba(232, 226, 212, 0.06);
  }
  .time-grid .tick:last-child { border-right: none; }
  .time-grid .tick.current {
    background: rgba(212, 165, 116, 0.10);
  }
  .playhead {
    position: absolute; top: 12px; bottom: 12px;
    width: 2px; background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    pointer-events: none; transition: left 80ms linear;
  }

  audio {
    width: 100%; height: 36px;
    filter: invert(0.92) hue-rotate(180deg) saturate(0.5);
  }

  /* RIGHT: predictions panel */
  aside {
    border-left: 1px solid var(--rule);
    background: var(--bg-panel);
    padding: 32px 32px 40px;
    max-height: calc(100vh - 200px); overflow-y: auto;
  }
  aside h2 {
    font-family: var(--display); font-weight: 300; font-size: 24px;
    letter-spacing: -0.01em; margin: 0 0 4px;
  }
  aside .panel-sub {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-bottom: 24px;
  }
  aside .panel-sub .accent { color: var(--accent); font-weight: 700; }
  aside h3 {
    font-family: var(--mono); font-weight: 500; font-size: 10px;
    color: var(--text-faint); text-transform: uppercase;
    letter-spacing: 0.1em; margin: 8px 0 16px;
  }

  /* Per-window prediction bars */
  .preds .pred {
    margin-bottom: 14px;
    transition: opacity 200ms;
  }
  .preds .pred-meta {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 4px;
  }
  .preds .pred-name {
    font-family: var(--body); font-size: 14px; font-weight: 500;
    color: var(--text);
  }
  .preds .pred-name .code {
    font-family: var(--mono); font-size: 10px; color: var(--text-faint);
    margin-left: 6px;
  }
  .preds .pred-taxon {
    font-family: var(--mono); font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-faint);
  }
  .preds .pred-taxon.Aves     { color: #8aa9c4; }
  .preds .pred-taxon.Amphibia { color: #b89af0; }
  .preds .pred-taxon.Insecta  { color: #f0b3a0; }
  .preds .pred-taxon.Mammalia { color: var(--green); }
  .preds .pred-taxon.Reptilia { color: var(--accent); }
  .preds .pred-bar-track {
    background: var(--bg); height: 6px; position: relative;
  }
  .preds .pred-bar-fill {
    background: var(--accent); height: 100%;
    transition: width 300ms ease;
  }
  .preds .pred-bar-fill.dim { background: var(--accent-dim); }
  .preds .pred-value {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    text-align: right; margin-top: 2px;
  }

  /* Window strip below predictions */
  .window-strip {
    display: grid; grid-template-columns: repeat(12, 1fr);
    gap: 2px; margin-top: 8px;
  }
  .window-strip .w {
    height: 28px; background: var(--bg); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 9px; color: var(--text-faint);
    transition: background 80ms;
  }
  .window-strip .w:hover { background: var(--bg-row); color: var(--text-dim); }
  .window-strip .w.current {
    background: var(--accent); color: var(--bg); font-weight: 700;
  }
  .window-strip .w.high   { box-shadow: inset 0 -3px 0 var(--accent-dim); }

  /* FOOTER */
  footer {
    border-top: 1px solid var(--rule);
    padding: 16px 40px;
    font-family: var(--mono); font-size: 10px; color: var(--text-faint);
    display: flex; justify-content: space-between;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  footer a { color: var(--text-dim); text-decoration: none; }
</style>
</head>
<body>

<header>
  <div>
    <div class="subtitle">Live predictions — 5s window classifier</div>
    <div class="title">Listen<em>.</em></div>
  </div>
  <div class="model-info">
    Source model <strong id="model-source">—</strong><br>
    weighted macro AUC <span class="accent" id="model-wma">—</span><br>
    <span id="model-config">—</span>
  </div>
</header>

<div class="picker" id="picker">
  <span class="label">Soundscape</span>
</div>

<main>
  <section class="stage">
    <div class="stage-header">
      <div class="stage-filename" id="filename">—</div>
      <div class="stage-time">
        <span class="now" id="now-s">0</span>s / <span id="dur-s">60</span>s
        · window <span id="now-window">1</span>/12
      </div>
    </div>

    <div class="spectrogram-wrap" id="spec-wrap">
      <img id="spec" alt="Mel spectrogram of the current soundscape">
      <div class="time-grid" id="time-grid"></div>
      <div class="playhead" id="playhead" style="left: 0%"></div>
    </div>

    <audio id="audio" controls preload="auto"></audio>
  </section>

  <aside>
    <h2>Top species</h2>
    <div class="panel-sub">
      Window <span class="accent" id="window-label">1</span> of 12
      <span style="color: var(--text-faint)">·</span>
      <span id="window-range">0-5s</span>
      <span style="color: var(--text-faint)">·</span>
      peak prob <span class="accent" id="peak-prob">—</span>
    </div>

    <h3>Predictions</h3>
    <div class="preds" id="preds"></div>

    <h3>All 12 windows · click to jump</h3>
    <div class="window-strip" id="strip"></div>
  </aside>
</main>

<footer>
  <div>BirdCLEF+ 2026 · Pre-computed predictions on 5s windows</div>
  <div>Best model · <span id="footer-source">—</span></div>
</footer>

<script>
const DATA = __DATA_JSON__;
let currentSoundscape = 0;
let currentWindow = 0;

function fmt(x, n=3) {
  return (x === null || x === undefined) ? '—' : Number(x).toFixed(n);
}

function init() {
  // Model info panel
  const m = DATA.model || {};
  document.getElementById('model-source').textContent =
    `${(m.agent || 'meta')}#${m.exp_id || '?'}`;
  document.getElementById('model-wma').textContent =
    m.weighted_macro_auc !== null && m.weighted_macro_auc !== undefined
      ? Number(m.weighted_macro_auc).toFixed(4) : '—';
  const cfg = m.config || {};
  const parts = [cfg.loss, cfg.aug, cfg.optimizer, cfg.schedule].filter(Boolean);
  document.getElementById('model-config').textContent = parts.join(' · ');
  document.getElementById('footer-source').textContent =
    `${(m.agent || 'meta')} exp ${m.exp_id || '?'}`;

  // Build picker pills
  const picker = document.getElementById('picker');
  DATA.soundscapes.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'pill' + (i === 0 ? ' active' : '');
    b.textContent = s.filename.replace(/^BC2026_Train_/, '').replace(/\.ogg$/, '');
    b.onclick = () => loadSoundscape(i);
    b.dataset.idx = i;
    picker.appendChild(b);
  });

  // Build time grid (12 cells)
  const grid = document.getElementById('time-grid');
  for (let i = 0; i < 12; i++) {
    const t = document.createElement('div');
    t.className = 'tick';
    t.dataset.idx = i;
    grid.appendChild(t);
  }

  // Build window strip (12 cells)
  const strip = document.getElementById('strip');
  for (let i = 0; i < 12; i++) {
    const w = document.createElement('div');
    w.className = 'w';
    w.dataset.idx = i;
    w.textContent = (i + 1);
    w.onclick = () => jumpToWindow(i);
    strip.appendChild(w);
  }

  loadSoundscape(0);

  // Hook up audio time updates
  const audio = document.getElementById('audio');
  audio.addEventListener('timeupdate', onTimeUpdate);
  audio.addEventListener('loadedmetadata', () => {
    document.getElementById('dur-s').textContent =
      Math.round(audio.duration);
  });
}

function loadSoundscape(idx) {
  currentSoundscape = idx;
  const s = DATA.soundscapes[idx];

  // Update picker pills
  document.querySelectorAll('#picker .pill').forEach(p => {
    p.classList.toggle('active', Number(p.dataset.idx) === idx);
  });

  // Update spectrogram + audio
  document.getElementById('spec').src = s.mel_image;
  document.getElementById('filename').textContent = s.filename;
  const audio = document.getElementById('audio');
  audio.src = s.audio_data;
  audio.load();

  // Compute per-window "highness" for the strip
  const max_prob = Math.max(...s.windows.map(w => w.max_prob));
  document.querySelectorAll('#strip .w').forEach((el, i) => {
    const w = s.windows[i];
    el.classList.toggle('high', w.max_prob >= max_prob * 0.75);
  });

  // Reset to window 0 on switch
  currentWindow = 0;
  renderWindow(0);
}

function onTimeUpdate() {
  const audio = document.getElementById('audio');
  const t = audio.currentTime;
  const dur = audio.duration || 60;
  const pct = (t / dur) * 100;
  document.getElementById('playhead').style.left = `${pct}%`;
  document.getElementById('now-s').textContent = Math.floor(t);
  // Which window are we in?
  const w = Math.min(11, Math.floor(t / 5));
  if (w !== currentWindow) {
    currentWindow = w;
    renderWindow(w);
  }
}

function jumpToWindow(idx) {
  const audio = document.getElementById('audio');
  audio.currentTime = idx * 5 + 0.05;
  currentWindow = idx;
  renderWindow(idx);
}

function renderWindow(idx) {
  const s = DATA.soundscapes[currentSoundscape];
  const w = s.windows[idx];

  document.getElementById('now-window').textContent = idx + 1;
  document.getElementById('window-label').textContent = idx + 1;
  document.getElementById('window-range').textContent = `${w.start_s}-${w.end_s}s`;
  document.getElementById('peak-prob').textContent = fmt(w.max_prob, 3);

  // Highlight time grid cell + window strip
  document.querySelectorAll('#time-grid .tick').forEach((el, i) => {
    el.classList.toggle('current', i === idx);
  });
  document.querySelectorAll('#strip .w').forEach((el, i) => {
    el.classList.toggle('current', i === idx);
  });

  // Render top-5
  const preds = document.getElementById('preds');
  preds.innerHTML = '';
  // Normalise bar widths relative to the *max* in this window so small
  // probabilities are still visually distinguishable
  const max_p = Math.max(...w.top.map(t => t.prob), 0.001);
  w.top.forEach(t => {
    const div = document.createElement('div');
    div.className = 'pred';
    const dim = t.prob < 0.3 ? 'dim' : '';
    const pctWidth = (100 * t.prob).toFixed(1);
    const barWidth = (100 * t.prob / max_p).toFixed(1);
    div.innerHTML = `
      <div class="pred-meta">
        <div class="pred-name">${escapeHtml(t.common)} <span class="code">${escapeHtml(t.label)}</span></div>
        <div class="pred-taxon ${escapeHtml(t.taxon)}">${escapeHtml(t.taxon)}</div>
      </div>
      <div class="pred-bar-track">
        <div class="pred-bar-fill ${dim}" style="width: ${barWidth}%"></div>
      </div>
      <div class="pred-value">prob ${fmt(t.prob, 3)} (${pctWidth}%)</div>
    `;
    preds.appendChild(div);
  });
}

function escapeHtml(s) {
  const map = {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
  return String(s).replace(/[&<>"']/g, c => map[c]);
}

init();
</script>
</body>
</html>
"""


SAMPLE_RATE = 32_000
DURATION_SEC = 5.0
WINDOWS_PER_SOUNDSCAPE = 12
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128


def _lazy_import_runtime():
    """Import heavy ML/audio packages only when we need to build predictions."""
    import numpy as np
    import pandas as pd
    import soundfile as sf

    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras

    return np, pd, sf, keras


def _build_mel_filterbank(np, sr: int, n_fft: int, n_mels: int):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min, f_max = 20.0, sr / 2.0
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            fb[i - 1, left:center] = np.linspace(
                0, 1, center - left, dtype=np.float32
            )
        if right > center:
            fb[i - 1, center:right] = np.linspace(
                1, 0, right - center, dtype=np.float32
            )
    return fb


def _waveform_to_mel(np, audio, mel_fb, hann_win):
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
    windowed = frames * hann_win
    spec = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    spec = spec.T
    mel = mel_fb @ spec
    mel = np.log1p(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


def _read_soundscape(sf, np, path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Sample rate mismatch in {path}: {sr} != {SAMPLE_RATE}")
    return audio


def _split_soundscape(np, audio):
    target_per_window = int(SAMPLE_RATE * DURATION_SEC)
    target_total = target_per_window * WINDOWS_PER_SOUNDSCAPE
    if len(audio) < target_total:
        audio = np.pad(audio, (0, target_total - len(audio)))
    else:
        audio = audio[:target_total]
    return audio.reshape(WINDOWS_PER_SOUNDSCAPE, target_per_window)


def _audio_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "audio/ogg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _mel_image_data_url(mel) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=120)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_taxonomy(pd, taxonomy_path: Path) -> dict[str, dict[str, str]]:
    tax = pd.read_csv(taxonomy_path)
    out = {}
    for row in tax.itertuples(index=False):
        label = str(row.primary_label)
        out[label] = {
            "common": str(getattr(row, "common_name", label)),
            "taxon": str(getattr(row, "class_name", "Unknown")),
        }
    return out


def _model_metadata(model_path: Path) -> dict:
    metrics_path = model_path.parent / "metrics.json"
    if not metrics_path.exists():
        return {
            "agent": model_path.parent.name,
            "exp_id": None,
            "weighted_macro_auc": None,
            "config": {},
        }

    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "agent": model_path.parent.name,
            "exp_id": None,
            "weighted_macro_auc": None,
            "config": {},
        }

    return {
        "agent": m.get("agent_name") or m.get("agent") or model_path.parent.name,
        "exp_id": m.get("id"),
        "weighted_macro_auc": m.get("weighted_macro_auc"),
        "macro_auc": m.get("macro_auc"),
        "aves_auc": m.get("aves_auc"),
        "config": {
            "loss": m.get("loss"),
            "aug": m.get("aug"),
            "lr": m.get("lr"),
            "optimizer": m.get("optimizer"),
            "schedule": m.get("schedule"),
            "geo": m.get("geo_scale_km"),
        },
    }


def build_predictions_bundle(
    model_path: Path,
    soundscape_dir: Path,
    sample_sub_path: Path,
    taxonomy_path: Path,
    n_soundscapes: int,
    top_k: int,
    selected_files: list[str] | None = None,
) -> dict:
    np, pd, sf, keras = _lazy_import_runtime()

    sample_sub = pd.read_csv(sample_sub_path)
    labels = [c for c in sample_sub.columns if c != "row_id"]
    taxonomy = _load_taxonomy(pd, taxonomy_path)

    if selected_files:
        files = []
        for name in selected_files:
            path = Path(name)
            if not path.is_absolute():
                path = soundscape_dir / name
            if not path.exists():
                raise SystemExit(f"Selected soundscape not found: {path}")
            files.append(path)
    else:
        files = sorted(soundscape_dir.glob("*.ogg"))
        if not files:
            files = sorted(soundscape_dir.glob("*.wav"))
        if not files:
            raise SystemExit(f"No .ogg or .wav files found in {soundscape_dir}")
        files = files[:n_soundscapes]

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    print(f"  loaded model with {model.count_params():,} parameters")
    if model.output_shape[-1] != len(labels):
        raise SystemExit(
            f"Model output {model.output_shape[-1]} does not match "
            f"{len(labels)} labels from sample_submission.csv."
        )

    mel_fb = _build_mel_filterbank(np, SAMPLE_RATE, N_FFT, N_MELS)
    hann_win = np.hanning(N_FFT).astype(np.float32)

    soundscapes = []
    for i, path in enumerate(files, 1):
        t0 = time.perf_counter()
        audio = _read_soundscape(sf, np, path)
        windows_audio = _split_soundscape(np, audio)
        mels = np.stack(
            [_waveform_to_mel(np, w, mel_fb, hann_win) for w in windows_audio],
            axis=0,
        )
        preds = model.predict(mels[..., np.newaxis], verbose=0)

        windows = []
        for idx, pred in enumerate(preds):
            top_idx = np.argsort(pred)[-top_k:][::-1]
            top = []
            for j in top_idx:
                label = labels[int(j)]
                meta = taxonomy.get(label, {})
                top.append({
                    "label": label,
                    "common": meta.get("common", label),
                    "taxon": meta.get("taxon", "Unknown"),
                    "prob": float(pred[int(j)]),
                })
            windows.append({
                "start_s": int(idx * DURATION_SEC),
                "end_s": int((idx + 1) * DURATION_SEC),
                "max_prob": float(np.max(pred)),
                "top": top,
            })

        full_mel = _waveform_to_mel(
            np,
            _split_soundscape(np, audio).reshape(-1),
            mel_fb,
            hann_win,
        )
        soundscapes.append({
            "filename": path.name,
            "audio_data": _audio_data_url(path),
            "mel_image": _mel_image_data_url(full_mel),
            "windows": windows,
        })
        print(f"  [{i}/{len(files)}] {path.name} ({time.perf_counter() - t0:.1f}s)")

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": _model_metadata(model_path),
        "settings": {
            "sample_rate": SAMPLE_RATE,
            "duration_sec": DURATION_SEC,
            "windows_per_soundscape": WINDOWS_PER_SOUNDSCAPE,
            "top_k": top_k,
        },
        "soundscapes": soundscapes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="demo_predictions.json",
                        help="Input JSON bundle from build_predictions_demo.py")
    parser.add_argument("--out", default="demo.html",
                        help="Output HTML file (default: demo.html)")
    parser.add_argument("--model",
                        help="Optional .keras model path. If --in does not "
                             "exist, predictions are built from this model.")
    parser.add_argument("--soundscape-dir", default="train_soundscapes",
                        help="Folder with .ogg/.wav soundscapes for the demo")
    parser.add_argument("--sample-sub", default="sample_submission.csv",
                        help="sample_submission.csv path for label order")
    parser.add_argument("--taxonomy", default="taxonomy.csv",
                        help="taxonomy.csv path for common names/taxa")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of soundscapes to include")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top predictions to show per window")
    parser.add_argument("--json-out", default="demo_predictions.json",
                        help="Where to save generated prediction JSON")
    parser.add_argument("--files",
                        help="Comma-separated soundscape filenames/paths to "
                             "include, in display order. Overrides --n.")
    args = parser.parse_args()

    in_path = Path(args.inp)
    if args.model:
        data = build_predictions_bundle(
            model_path=Path(args.model),
            soundscape_dir=Path(args.soundscape_dir),
            sample_sub_path=Path(args.sample_sub),
            taxonomy_path=Path(args.taxonomy),
            n_soundscapes=args.n,
            top_k=args.top_k,
            selected_files=(
                [x.strip() for x in args.files.split(",") if x.strip()]
                if args.files else None
            ),
        )
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Wrote {json_path}")
    else:
        if not in_path.exists():
            raise SystemExit(
                f"{in_path} not found. Pass --model path/to/model.keras "
                "to build it."
            )
        data = json.loads(in_path.read_text(encoding="utf-8"))

    n_ss = len(data.get("soundscapes", []))
    if n_ss == 0:
        raise SystemExit("No soundscapes in the JSON bundle. Re-run the build step.")

    html = HTML.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {n_ss} soundscapes)")
    print(f"Open it in any browser to use.")


if __name__ == "__main__":
    main()
