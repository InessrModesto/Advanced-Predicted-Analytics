# BirdCLEF+ 2026 — Autonomous Research Agent

An autonomous research agent that designs, trains, and iterates on deep learning models for the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) audio classification task (234-class multi-label species detection from Pantanal soundscapes).

The agent uses a locally-hosted LLM (via [Ollama](https://ollama.com)) to reason over the history of past experiments and propose the next training configuration. Four agent variants explore then search space differently — see the report for details.

**Best Kaggle public LB score:** *0.747*

---

## Setup

### 1. Python environment

Tested with **Python 3.11.6** on Windows. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate         # On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Local LLM (Ollama)

Install Ollama from [ollama.com](https://ollama.com), then pull the model the agents use by default:

```bash
ollama pull gemma3:4b
```

Make sure Ollama is running before launching any agent (the Ollama installer on Windows starts it as a background service automatically; on Linux/macOS you may need `ollama serve` in a separate terminal).

The agents are LLM-agnostic via Ollama. To use a different model, edit `LLM_MODEL = "gemma3:4b"` at the top of each agent file (Cell 1) — for example to `"qwen3.5:9b"`. Make sure to also `ollama pull` the new model first.

### 3. Competition data

Download the BirdCLEF+ 2026 dataset from [Kaggle](https://www.kaggle.com/competitions/birdclef-2026/data) (or use the Kaggle CLI). Unzip its contents into the project root so the structure looks like:

```
.
├── train.csv
├── taxonomy.csv
├── sample_submission.csv
├── train_soundscapes_labels.csv
├── train_audio/                  # focal recordings (~36k .ogg files)
├── train_soundscapes/            # labelled Pantanal soundscapes
└── ...                           # the agent code lives alongside these
```

---

## How to run

The full pipeline runs in four stages. Each stage's outputs feed the next:

```bash
# 1. Train the manual baselines (CNN, bigger CNN, EfficientNet-B0, MobileNetV3, CRNN-LSTM, CRNN-GRU). Outputs:
#      - trained_models/<arch>.keras   (one .keras file per architecture)
#      - comparison_results.csv        (their AUC, params, latency)
python compare_models.py

# 2. Run the three exploration agents. Each picks the winner from comparison_results.csv as its warm-start point and iterates from there.
#    Each takes roughly 2-3 hours for 10 iterations on CPU.
python agent_regular.py  --iterations 10
python agent_eda.py      --iterations 10
python agent_creative.py --iterations 10

# 3. Run the meta agent. It reads the prior three agents' results, warm-starts from the global best, and refines further.
python agent_meta.py --iterations 10

# 4. Validate a chosen model locally before submitting to Kaggle.
python validate_submission.py --model experiments_meta_gemma3_4b/exp_XXX_.../model.keras
```

Then upload `submission_notebook.ipynb` + the chosen `.keras` file to a Kaggle notebook to produce a competition submission. See the notebook's first markdown cell for full instructions.

### Useful flags

- `--iterations N` — number of agent iterations (default 10)
- `--reset` — wipe the agent's prior experiment folder before starting (use sparingly — it deletes everything in `experiments_<agent>_<llm>/`)

---

## Project structure

```
.
├── baseline.py                # Data pipeline: label space, audio loading, mel
│                              # spectrogram, train/val split, AudioDataset.
│                              # Imported by every agent.
├── compare_models.py          # Trains the 6 manual baseline architectures
│                              # and produces comparison_results.csv.
├── agent_regular.py           # Regular agent (5D discrete search:
│                              # loss × aug × lr × optimizer × schedule)
├── agent_eda.py               # EDA-aware agent (same 5D + continuous geo
│                              # weighting axis, with EDA findings in the prompt)
├── agent_creative.py          # Creative agent (free-form LLM-generated code)
├── agent_meta.py              # Meta agent: warm-starts from global winner,
│                              # sees all prior experiments, refines further
├── validate_submission.py     # Pre-flight check — runs the inference pipeline
│                              # against train_soundscapes/ to catch bugs before
│                              # uploading to Kaggle (faster than discovering
│                              # them in Kaggle's queue)
├── submission_notebook.ipynb  # Generic Kaggle submission notebook — change
│                              # MODEL_PATH at the top to switch which model
│                              # you submit
├── requirements.txt
├── README.md                  # this file
│
├── trained_models/            # Output of compare_models.py
│   └── <arch>.keras           # one per architecture
├── comparison_results.csv     # Output of compare_models.py
│
├── make_dashboard.py          # code do create a dashboard comparing Gemma4 experiments
├── dashboard                  # Output of make_dashboard.py
├── make_predictions_demo.py   # code to create interactive demo to visualize the best agent running
├── demo                       # Output of make_predictions_demo.py
├── demo_predictions           # Output of make_predictions_demo.py
├── experiments_regular_<llm>/
│   ├── exp_NNN_.../           # one folder per experiment
│   │   ├── model.py           # the hardcoded experiment definition
│   │   ├── model.keras        # the trained weights
│   │   └── metrics.json       # AUCs, config, training time, LLM rationale
│   └── progress.png           # macro AUC across iterations
├── experiments_eda_<llm>/     # same layout
├── experiments_creative_<llm>/
└── experiments_meta_<llm>/
    ├── meta_prior_experiments.csv   # combined results across all 4 agents
    ├── report_analysis.md           # per-taxon AUC + confusion analysis
    └── ...
```

---

## Hardware notes

Tested on Windows, **16 GB RAM**, CPU-only (no GPU required). Per-iteration cost:

- Manual baselines (`compare_models.py`): ~6 hours total for all 6 architectures
- One agent iteration: ~25 minutes (one training run of up to 10 epochs)
- A 10-iteration agent run: ~3 hours
- Kaggle submission inference: ~12 min for ~700 soundscapes (within the 90-min budget)

The agents are CPU-friendly. Ollama uses ~5 GB RAM for `gemma3:4b` and is unloaded between calls (`keep_alive=0`) so it doesn't hold memory during long training runs.

---

## Known issues / troubleshooting

**Keras version skew between local and Kaggle.** Models saved with Keras 3.14.1 include deprecated `BatchNormalization` arguments (`renorm`, `renorm_clipping`, `renorm_momentum`) that newer Keras versions on Kaggle reject. The submission notebook handles this by rebuilding the architecture from code and loading only the weights — see the resave cell in `submission_notebook.ipynb` if you hit:

```
Unrecognized keyword arguments passed to BatchNormalization: {'renorm': ...}
```

**Ollama not running.** Any agent call will fail with a connection error if Ollama isn't running. Start it with `ollama serve` (Linux/macOS) or check that the Ollama background service is running (Windows). Verify with `ollama list` — your chosen model should appear.

**Empty `test_soundscapes/` on Kaggle in dev mode.** Kaggle's hidden test set is only mounted at submission time, not during dev runs. The notebook handles this gracefully — an empty submission during dev is expected and will get replaced with the real one when you click "Submit to Competition."

**`max_rows` in baseline.py.** The data pipeline subsamples focal recordings to `max_rows=3000` by default for tractable training time. Increase this in `cfg` if you have more compute available.

---

## Quick reproduction — best validation model (Meta agent, Gemma4, exp 9)

If you only want to run our selected best local model, you do not need to rerun the EDA, baselines, LLM calls, or agent loops. The trained weights are saved at:

`experiments_meta_gemma4/exp_009_plain_bce_none_lr0.001_adam_exp_decay_geo2000/model.keras`

After installing the dependencies (see Setup) and placing the BirdCLEF+ 2026 dataset files in the project root, run:

```powershell
python validate_submission.py --model "experiments_meta_gemma4\exp_009_plain_bce_none_lr0.001_adam_exp_decay_geo2000\model.keras" --n 3
```

This loads the saved model with compile=False, runs local inference on a few soundscapes, checks that the output has the correct 234-class format, verifies probability ranges and row IDs, and writes submission_validation.csv. No training, no LLM, and no agent loop are required.

To generate the interactive prediction demo from the same model, run:

```powershell
python make_predictions_demo.py --model "experiments_meta_gemma4\exp_009_plain_bce_none_lr0.001_adam_exp_decay_geo2000\model.keras" --out demo.html --json-out demo_predictions.json
Then open demo.html in a browser.
```

## Resources

- Competition: <https://www.kaggle.com/competitions/birdclef-2026>
- Project report: `report.pdf`
- Video presentation: *[https://youtu.be/s2YUbrBLej4](https://youtu.be/TzcTMY6zkpM)*
- Video of code running: *https://youtu.be/s2YUbrBLej4*

## Authors

*António Faustino*
*Inês Modesto*
*Kseniya Drahun*
*Sofia Quiroga*
