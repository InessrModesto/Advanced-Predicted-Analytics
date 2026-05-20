"""
Dashboard generator for the BirdCLEF+ 2026 autonomous agent project.

Reads `meta_prior_experiments.csv` from both LLM runs (gemma4 and qwen3)
plus each agent's per-experiment metrics.json, and writes a single
self-contained `dashboard.html` file that opens in any browser without
a server.

Usage:
    python make_dashboard.py
    # then open dashboard.html

Or specify paths:
    python make_dashboard.py --gemma-dir experiments_meta_gemma4 \
                              --qwen-dir experiments_meta_qwen3 \
                              --out dashboard.html
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd


# ── Discover all experiments across both LLMs ───────────────────────────
def load_all_experiments(meta_dir: Path, llm_label: str) -> list[dict]:
    """Read the meta agent's combined CSV plus each meta experiment.

    The meta agent writes `meta_prior_experiments.csv` at startup containing
    every regular/eda/creative experiment. This function additionally
    appends the meta agent's own experiments (which aren't in that CSV
    because they're produced AFTER startup)."""
    rows: list[dict] = []
    csv = meta_dir / "meta_prior_experiments.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        for r in df.to_dict(orient="records"):
            r["llm"] = llm_label
            r["agent"] = str(r.get("agent", "")).lower()
            rows.append(r)
    # Then meta's own experiments
    if meta_dir.exists():
        for d in sorted(meta_dir.iterdir()):
            if not d.is_dir():
                continue
            mfile = d / "metrics.json"
            if not mfile.exists():
                continue
            try:
                r = json.loads(mfile.read_text(encoding="utf-8"))
                r["agent"] = "meta"
                r["llm"] = llm_label
                rows.append(r)
            except Exception:
                pass
    return rows


def normalise_row(r: dict) -> dict:
    """Pick only the fields the dashboard uses; coerce types to JSON-safe."""
    def f(name, default=None):
        v = r.get(name, default)
        if isinstance(v, float) and v != v:   # NaN
            return None
        return v
    geo = f("geo_scale_km")
    geo_bucket = f("geo_bucket")
    if geo_bucket is None:
        geo_bucket = "none" if geo is None else str(int(round(float(geo) / 100) * 100))
    return {
        "id":                  int(f("id", 0) or 0),
        "agent":               str(f("agent", "?")),
        "llm":                 str(f("llm", "?")),
        "loss":                f("loss"),
        "aug":                 f("aug"),
        "lr":                  float(f("lr") or 0),
        "optimizer":           f("optimizer"),
        "schedule":            f("schedule"),
        "geo_scale_km":        (None if geo is None else float(geo)),
        "geo_bucket":          geo_bucket,
        "macro_auc":           float(f("macro_auc") or 0),
        "weighted_macro_auc":  float(f("weighted_macro_auc") or 0),
        "aves_auc":            float(f("aves_auc") or 0),
        "amphibia_auc":        float(f("amphibia_auc") or 0),
        "insecta_auc":         float(f("insecta_auc") or 0),
        "mammalia_auc":        float(f("mammalia_auc") or 0),
        "pruned":              bool(f("pruned", False)),
        "epochs_run":          int(f("epochs_run", 0) or 0),
        "train_time_s":        float(f("train_time_s") or 0),
        "params":              int(f("params", 0) or 0),
        "reason":              str(f("reason", "") or ""),
    }


# ── HTML template ───────────────────────────────────────────────────────
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Agent Lab — BirdCLEF+ 2026</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #18181a;
    --bg-panel:  #1f1f22;
    --bg-row:   #232327;
    --text:      #e8e2d4;
    --text-dim: #8a8579;
    --text-faint:#5a574f;
    --accent:    #d4a574;
    --accent-dim:#7a5e3f;
    --green:     #7fa67a;
    --red:       #c47872;
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
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 40px;
  }
  header .title {
    font-family: var(--display);
    font-weight: 300;
    font-size: 34px;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  header .title em { font-style: italic; color: var(--accent); font-weight: 500; }
  header .subtitle { color: var(--text-dim); font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; }
  header .meta {
    font-family: var(--mono); font-size: 11px; color: var(--text-faint);
    text-align: right; line-height: 1.6;
  }
  header .meta strong { color: var(--text); font-weight: 500; }

  /* SUMMARY ROW */
  .summary {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
    border-bottom: 1px solid var(--rule);
  }
  .summary .cell {
    padding: 22px 40px;
    border-right: 1px solid var(--rule);
  }
  .summary .cell:last-child { border-right: none; }
  .summary .label {
    font-family: var(--mono); font-size: 10px; color: var(--text-faint);
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
  }
  .summary .value {
    font-family: var(--display); font-weight: 300; font-size: 28px;
    letter-spacing: -0.01em; line-height: 1; color: var(--text);
  }
  .summary .value.accent { color: var(--accent); }
  .summary .sub {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-top: 6px;
  }

  /* FILTERS */
  .filters {
    display: flex; gap: 24px; padding: 16px 40px;
    border-bottom: 1px solid var(--rule); align-items: center;
    font-family: var(--mono); font-size: 12px;
  }
  .filters label { color: var(--text-faint); text-transform: uppercase; font-size: 10px; letter-spacing: 0.08em; margin-right: 6px; }
  .filters .group { display: flex; align-items: center; }
  .pill {
    background: transparent; border: 1px solid var(--rule);
    color: var(--text-dim); padding: 4px 10px; margin-right: 4px;
    cursor: pointer; font-family: var(--mono); font-size: 11px;
    transition: all 120ms ease;
  }
  .pill:hover { color: var(--text); border-color: var(--text-faint); }
  .pill.active {
    background: var(--accent); border-color: var(--accent);
    color: var(--bg); font-weight: 700;
  }
  .filters .search {
    background: transparent; border: 1px solid var(--rule);
    color: var(--text); padding: 4px 10px; font-family: var(--mono);
    font-size: 11px; width: 200px; margin-left: auto;
  }
  .filters .search::placeholder { color: var(--text-faint); }

  /* MAIN LAYOUT */
  main {
    display: grid; grid-template-columns: 1fr 460px;
    min-height: calc(100vh - 280px);
  }

  /* TABLE */
  .table-wrap { overflow: auto; max-height: calc(100vh - 280px); }
  table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
  thead { position: sticky; top: 0; background: var(--bg); z-index: 2; }
  th {
    text-align: left; font-weight: 500; font-size: 10px;
    color: var(--text-faint); text-transform: uppercase;
    letter-spacing: 0.08em; padding: 14px 16px 10px;
    border-bottom: 1px solid var(--rule); cursor: pointer; user-select: none;
    white-space: nowrap;
  }
  th:hover { color: var(--text); }
  th .sort-marker { color: var(--accent); margin-left: 4px; }
  td { padding: 10px 16px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
  tbody tr { cursor: pointer; transition: background 80ms; }
  tbody tr:hover { background: var(--bg-row); }
  tbody tr.selected { background: var(--bg-row); }
  tbody tr.selected td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  td.num { text-align: right; }
  td.dim { color: var(--text-dim); }
  td.bad { color: var(--red); }
  td.good { color: var(--green); }
  .agent-tag {
    display: inline-block; padding: 1px 6px; border-radius: 2px;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; background: var(--bg-row);
  }
  .agent-tag.regular  { color: #8aa9c4; }
  .agent-tag.eda      { color: #b89af0; }
  .agent-tag.creative { color: #f0b3a0; }
  .agent-tag.meta     { color: var(--accent); }
  .pruned-tag {
    display: inline-block; padding: 1px 6px; font-size: 10px;
    color: var(--red); border: 1px solid var(--red); margin-left: 6px;
  }

  /* DETAIL PANEL */
  aside {
    border-left: 1px solid var(--rule);
    padding: 32px 32px 48px;
    background: var(--bg-panel);
    max-height: calc(100vh - 280px); overflow-y: auto;
  }
  aside h2 {
    font-family: var(--display); font-weight: 300; font-size: 24px;
    letter-spacing: -0.01em; margin: 0 0 4px;
  }
  aside .lede {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-bottom: 24px;
  }
  aside h3 {
    font-family: var(--mono); font-weight: 500; font-size: 10px;
    color: var(--text-faint); text-transform: uppercase;
    letter-spacing: 0.1em; margin: 24px 0 8px;
  }
  aside .kv { display: grid; grid-template-columns: 130px 1fr; gap: 4px 12px; font-family: var(--mono); font-size: 12px; }
  aside .kv .k { color: var(--text-faint); }
  aside .kv .v { color: var(--text); }
  aside .kv .v.accent { color: var(--accent); }
  aside .reason {
    background: var(--bg);
    border-left: 2px solid var(--accent);
    padding: 12px 16px;
    font-family: var(--body); font-size: 13px; line-height: 1.6;
    color: var(--text); margin-top: 8px; font-style: italic;
  }
  aside .reason em { color: var(--accent); }
  aside .empty {
    color: var(--text-faint); font-style: italic;
    font-family: var(--display); padding: 40px 0; text-align: center;
  }

  /* BAR CHART for taxon AUCs */
  .bars { margin-top: 8px; }
  .bars .bar-row { display: grid; grid-template-columns: 90px 1fr 50px; align-items: center; gap: 12px; margin-bottom: 4px; }
  .bars .bar-label { font-family: var(--mono); font-size: 11px; color: var(--text-dim); }
  .bars .bar-track { background: var(--bg); height: 6px; position: relative; }
  .bars .bar-fill { background: var(--accent); height: 100%; transition: width 250ms ease; }
  .bars .bar-fill.dim { background: var(--accent-dim); }
  .bars .bar-value { font-family: var(--mono); font-size: 11px; color: var(--text); text-align: right; }

  /* FOOTER */
  footer {
    border-top: 1px solid var(--rule);
    padding: 18px 40px;
    font-family: var(--mono); font-size: 10px; color: var(--text-faint);
    display: flex; justify-content: space-between;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  footer a { color: var(--text-dim); text-decoration: none; }
  footer a:hover { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div>
    <div class="subtitle">Autonomous agent — experiment log</div>
    <div class="title">Bird<em>CLEF+</em> 2026</div>
  </div>
  <div class="meta">
    <strong>__N_TOTAL__</strong> experiments across <strong>__N_LLMS__</strong> LLMs<br>
    Generated __GEN_DATE__<br>
    Local validation macro AUC
  </div>
</header>

<div class="summary">
  <div class="cell">
    <div class="label">Best weighted macro AUC</div>
    <div class="value accent" id="best-weighted">—</div>
    <div class="sub" id="best-weighted-cfg"></div>
  </div>
  <div class="cell">
    <div class="label">Best unweighted macro AUC</div>
    <div class="value" id="best-unweighted">—</div>
    <div class="sub" id="best-unweighted-cfg"></div>
  </div>
  <div class="cell">
    <div class="label">Best Aves AUC</div>
    <div class="value" id="best-aves">—</div>
    <div class="sub" id="best-aves-cfg"></div>
  </div>
  <div class="cell">
    <div class="label">Pruned</div>
    <div class="value" id="n-pruned">—</div>
    <div class="sub" id="pruned-share"></div>
  </div>
</div>

<div class="filters">
  <div class="group"><label>LLM</label><div id="filter-llm"></div></div>
  <div class="group"><label>Agent</label><div id="filter-agent"></div></div>
  <input class="search" id="search" placeholder="filter loss / aug / opt …">
</div>

<main>
  <div class="table-wrap">
    <table id="experiments">
      <thead>
        <tr>
          <th data-key="agent">Agent</th>
          <th data-key="llm">LLM</th>
          <th data-key="id" class="num">#</th>
          <th data-key="loss">Loss</th>
          <th data-key="aug">Aug</th>
          <th data-key="lr" class="num">LR</th>
          <th data-key="optimizer">Optim</th>
          <th data-key="schedule">Sched</th>
          <th data-key="geo_bucket">Geo</th>
          <th data-key="weighted_macro_auc" class="num">macro(w) <span class="sort-marker">↓</span></th>
          <th data-key="macro_auc" class="num">macro(uw)</th>
          <th data-key="aves_auc" class="num">aves</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </div>

  <aside id="detail">
    <div class="empty">Select an experiment to inspect.</div>
  </aside>
</main>

<footer>
  <div>BirdCLEF+ 2026 · Advanced Predictive Analytics</div>
  <div>Click a row to view config &amp; the LLM's rationale</div>
</footer>

<script>
const DATA = __DATA_JSON__;
const fmt = (x, n=4) => (x === null || x === undefined || Number.isNaN(x)) ? '—' : Number(x).toFixed(n);
const fmtLR = (x) => x.toExponential(0).replace('e-', 'e-').replace('e+', 'e+');

// State
let state = {
  llm: 'all',
  agent: 'all',
  search: '',
  sortKey: 'weighted_macro_auc',
  sortDir: -1,
  selected: null,
};

// Build pill filters
function makePills(containerId, values, key) {
  const c = document.getElementById(containerId);
  c.innerHTML = '';
  ['all', ...values].forEach(v => {
    const b = document.createElement('button');
    b.className = 'pill' + (state[key] === v ? ' active' : '');
    b.textContent = v;
    b.onclick = () => { state[key] = v; render(); };
    c.appendChild(b);
  });
}

// Filtering
function filterRows() {
  return DATA.filter(r => {
    if (state.llm !== 'all' && r.llm !== state.llm) return false;
    if (state.agent !== 'all' && r.agent !== state.agent) return false;
    if (state.search) {
      const q = state.search.toLowerCase();
      const hay = `${r.loss} ${r.aug} ${r.optimizer} ${r.schedule} ${r.geo_bucket} ${r.lr}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function sortedRows(rows) {
  const k = state.sortKey;
  const dir = state.sortDir;
  return [...rows].sort((a, b) => {
    let va = a[k], vb = b[k];
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return -1 * dir;
    if (va > vb) return  1 * dir;
    return 0;
  });
}

// Summary metrics
function updateSummary(rows) {
  const valid = rows.filter(r => !r.pruned && r.weighted_macro_auc > 0);
  const setBest = (key, valEl, cfgEl, label) => {
    if (valid.length === 0) {
      document.getElementById(valEl).textContent = '—';
      document.getElementById(cfgEl).textContent = '';
      return;
    }
    const best = [...valid].sort((a, b) => b[key] - a[key])[0];
    document.getElementById(valEl).textContent = fmt(best[key]);
    document.getElementById(cfgEl).textContent =
      `${best.agent}·${best.llm} · ${best.loss}+${best.aug}+${best.optimizer}`;
  };
  setBest('weighted_macro_auc', 'best-weighted',   'best-weighted-cfg');
  setBest('macro_auc',          'best-unweighted', 'best-unweighted-cfg');
  setBest('aves_auc',           'best-aves',       'best-aves-cfg');
  const pruned = rows.filter(r => r.pruned).length;
  document.getElementById('n-pruned').textContent = pruned;
  document.getElementById('pruned-share').textContent =
    rows.length > 0 ? `${(100 * pruned / rows.length).toFixed(0)}% of ${rows.length}` : '';
}

// Render table
function renderTable(rows) {
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    if (state.selected && r._key === state.selected) tr.classList.add('selected');
    tr.onclick = () => { state.selected = r._key; render(); };
    const geoStr = r.geo_bucket === 'none' ? '—' : `${r.geo_bucket}km`;
    const cells = [
      `<span class="agent-tag ${r.agent}">${r.agent}</span>`,
      `<span class="dim">${r.llm}</span>`,
      `<span class="num">${r.id}</span>${r.pruned ? '<span class="pruned-tag">pruned</span>' : ''}`,
      r.loss || '—',
      r.aug || '—',
      `<span class="num">${fmtLR(r.lr)}</span>`,
      r.optimizer || '—',
      r.schedule || '—',
      geoStr,
      `<span class="num">${fmt(r.weighted_macro_auc)}</span>`,
      `<span class="num dim">${fmt(r.macro_auc)}</span>`,
      `<span class="num dim">${fmt(r.aves_auc, 3)}</span>`,
    ];
    cells.forEach((c, j) => {
      const td = document.createElement('td');
      if (j === 2 || j >= 5 && (j === 5 || j === 9 || j === 10 || j === 11)) td.className = 'num';
      td.innerHTML = c;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// Render detail panel
function renderDetail(rows) {
  const aside = document.getElementById('detail');
  const r = state.selected ? rows.find(x => x._key === state.selected) : null;
  if (!r) {
    aside.innerHTML = '<div class="empty">Select an experiment to inspect.</div>';
    return;
  }
  const taxa = [
    { name: 'Aves',     v: r.aves_auc },
    { name: 'Amphibia', v: r.amphibia_auc },
    { name: 'Insecta',  v: r.insecta_auc },
    { name: 'Mammalia', v: r.mammalia_auc },
  ];
  const max = Math.max(...taxa.map(t => t.v || 0), 0.001);
  const barsHtml = taxa.map(t => `
    <div class="bar-row">
      <div class="bar-label">${t.name}</div>
      <div class="bar-track"><div class="bar-fill ${t.v < 0.5 ? 'dim' : ''}" style="width: ${(100 * (t.v || 0)).toFixed(1)}%"></div></div>
      <div class="bar-value">${fmt(t.v, 3)}</div>
    </div>
  `).join('');
  aside.innerHTML = `
    <h2>${r.agent} · experiment ${r.id}</h2>
    <div class="lede">${r.llm} · ${r.params.toLocaleString()} params · ${r.epochs_run} epochs · ${(r.train_time_s/60).toFixed(1)}m train</div>

    <h3>Configuration</h3>
    <div class="kv">
      <div class="k">loss</div><div class="v">${r.loss}</div>
      <div class="k">augmentation</div><div class="v">${r.aug}</div>
      <div class="k">learning rate</div><div class="v">${fmtLR(r.lr)}</div>
      <div class="k">optimizer</div><div class="v">${r.optimizer}</div>
      <div class="k">schedule</div><div class="v">${r.schedule}</div>
      <div class="k">geo scale</div><div class="v">${r.geo_bucket === 'none' ? 'none' : r.geo_bucket + ' km'}</div>
    </div>

    <h3>Validation AUCs</h3>
    <div class="kv">
      <div class="k">macro (unweighted)</div><div class="v">${fmt(r.macro_auc)}</div>
      <div class="k">macro (weighted)</div><div class="v accent">${fmt(r.weighted_macro_auc)}</div>
    </div>

    <h3>Per-taxon AUC</h3>
    <div class="bars">${barsHtml}</div>

    <h3>LLM rationale</h3>
    <div class="reason">${escapeHtml(r.reason || '(no rationale recorded)')}</div>
  `;
}

function escapeHtml(s) {
  const map = {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
  return String(s).replace(/[&<>"']/g, c => map[c]);
}

function render() {
  // Set up filter pills
  const llms = [...new Set(DATA.map(r => r.llm))];
  const agents = [...new Set(DATA.map(r => r.agent))];
  makePills('filter-llm', llms, 'llm');
  makePills('filter-agent', agents, 'agent');

  let rows = filterRows();
  rows = sortedRows(rows);
  updateSummary(rows);
  renderTable(rows);
  renderDetail(rows);

  // Sort markers
  document.querySelectorAll('th').forEach(th => {
    const marker = th.querySelector('.sort-marker');
    if (marker) marker.remove();
  });
  document.querySelectorAll('th').forEach(th => {
    if (th.dataset.key === state.sortKey) {
      const span = document.createElement('span');
      span.className = 'sort-marker';
      span.textContent = state.sortDir === 1 ? '↑' : '↓';
      th.appendChild(span);
    }
  });
}

// Wire up events
document.querySelectorAll('th[data-key]').forEach(th => {
  th.onclick = () => {
    const k = th.dataset.key;
    if (state.sortKey === k) state.sortDir *= -1;
    else { state.sortKey = k; state.sortDir = -1; }
    render();
  };
});
document.getElementById('search').oninput = (e) => {
  state.search = e.target.value;
  render();
};

// Add stable key per row for selection tracking
DATA.forEach((r, i) => { r._key = `${r.llm}|${r.agent}|${r.id}`; });

render();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-dir", default="experiments_meta_gemma4",
                        help="Path to gemma meta agent's folder (default: experiments_meta_gemma4)")
    parser.add_argument("--qwen-dir", default="experiments_meta_qwen3",
                        help="Path to qwen meta agent's folder (default: experiments_meta_qwen3)")
    parser.add_argument("--out", default="dashboard.html",
                        help="Output HTML file (default: dashboard.html)")
    args = parser.parse_args()

    sources = [
        (Path(args.gemma_dir), "gemma4"),
        (Path(args.qwen_dir),  "qwen3"),
    ]

    rows: list[dict] = []
    for folder, llm in sources:
        n_before = len(rows)
        loaded = load_all_experiments(folder, llm)
        # Deduplicate (the CSV may already include meta's experiments)
        seen = {(r.get("agent"), r.get("id")) for r in rows if r.get("llm") == llm}
        for r in loaded:
            key = (str(r.get("agent", "")).lower(), int(r.get("id", -1) or -1))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
        print(f"  {folder}: {len(rows) - n_before} experiments")

    if not rows:
        raise SystemExit(
            "No experiments found. Run the meta agent for at least one LLM first."
        )

    normalised = [normalise_row(r) for r in rows]
    llms = sorted({r["llm"] for r in normalised})

    import time
    html = (HTML
            .replace("__DATA_JSON__", json.dumps(normalised))
            .replace("__N_TOTAL__",   str(len(normalised)))
            .replace("__N_LLMS__",    str(len(llms)))
            .replace("__GEN_DATE__",  time.strftime("%Y-%m-%d %H:%M")))

    Path(args.out).write_text(html, encoding="utf-8")
    print(f"\nWrote {args.out} ({len(html) / 1024:.1f} KB) — open it in any browser")


if __name__ == "__main__":
    main()
