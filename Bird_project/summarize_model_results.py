import json
from pathlib import Path
import pandas as pd

rows = []
for d in sorted(Path("experiments").iterdir()):
    if d.is_dir() and (d / "metrics.json").exists():
        rows.append(json.loads((d / "metrics.json").read_text(encoding="utf-8")))
df = pd.DataFrame(rows)
df.to_csv("agent_v1_results.csv", index=False)
print(df[["id","loss_name","macro_auc","aves_auc","amphibia_auc"]].to_string(index=False))