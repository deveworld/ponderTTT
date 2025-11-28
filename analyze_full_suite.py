from typing import cast

import pandas as pd
import numpy as np
import os

# File Map
files = {
    "Loss": "/home/world/ponderttt/outputs/wandb_export_2025-11-26T21_55_19.355+09_00.csv",
    "Perplexity": "/home/world/ponderttt/outputs/wandb_export_2025-11-26T23_27_22.957+09_00.csv",
    "Budget Util": "/home/world/ponderttt/outputs/wandb_export_2025-11-26T23_27_38.226+09_00.csv",
    "Gate Mean": "/home/world/ponderttt/outputs/wandb_export_2025-11-26T23_27_47.274+09_00.csv",
    "L_TTT": "/home/world/ponderttt/outputs/wandb_export_2025-11-26T23_28_00.355+09_00.csv"
}

method_stats = {}

def get_method_name(col_name, metric_suffix):
    # e.g., "diff_125m_budget2.5 - seed_42/perplexity" -> "diff_125m_budget2.5"
    # remove suffix
    name = col_name.replace(f"/{metric_suffix}", "")
    # remove seed info if present
    if " - seed_" in name:
        name = name.split(" - seed_")[0]
    return name

for metric, path in files.items():
    if not os.path.exists(path):
        print(f"Warning: File not found for {metric}: {path}")
        continue
        
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {metric}: {e}")
        continue
        
    # Identify suffix based on metric
    # CSV headers usually look like "run_name/metric_name"
    # We need to find the unique suffix used in this file
    # Take the first column that contains '/'
    sample_col = [c for c in df.columns if '/' in c][0]
    metric_suffix = sample_col.split('/')[-1]
    
    # Filter columns for this metric (ignore MIN/MAX)
    cols = [c for c in df.columns if c.endswith(f"/{metric_suffix}")]
    
    for col in cols:
        method = get_method_name(col, metric_suffix)
        if method not in method_stats:
            method_stats[method] = {}
            
        # Get Data
        data = df[col].replace(0, np.nan).dropna()
        if len(data) == 0:
            continue
            
        # Stats (Last 20% for convergence)
        last_n = int(len(data) * 0.2)
        if last_n < 5:
            last_n = len(data)
        final_data = data.iloc[-last_n:]
        
        mean_val = final_data.mean()
        
        method_stats[method][metric] = mean_val

# Compile DataFrame
df_res = pd.DataFrame.from_dict(method_stats, orient='index')

# Add estimated cost column for baselines if Budget Util is missing
# Cost model: 1 (base) + 2 * steps
def estimate_cost(row):
    if pd.notna(row.get("Budget Util")):
        # Budget util is fractional? No, usually [0, 1]. 
        # If "budget_limit" was 2.5, util 1.0 means we used 2.5 updates on average.
        # Cost = 1 + 2 * (BudgetUtil * BudgetLimit).
        # But we don't have budget limit easily here.
        # Let's try to infer from Gate Mean.
        # Gate Mean [0, 4] is the average steps.
        # Cost = 1 + 2 * Gate Mean
        if pd.notna(row.get("Gate Mean")):
            return 1.0 + 2.0 * row["Gate Mean"]
        return np.nan
        
    method = row.name
    if "UPDATE_4" in method:
        return 9.0
    if "UPDATE_2" in method:
        return 5.0
    if "UPDATE_1" in method:
        return 3.0
    if "SKIP" in method:
        return 1.0
    return np.nan

df_res["Est. Cost"] = df_res.apply(estimate_cost, axis=1)

# Formatting
cols_order = ["Loss", "Perplexity", "L_TTT", "Gate Mean", "Budget Util", "Est. Cost"]
existing_cols = [c for c in cols_order if c in df_res.columns]
df_res = cast(pd.DataFrame, df_res[existing_cols])
df_res = df_res.sort_values(by="Loss")

print("\n=== Comprehensive Experiment Analysis ===")
print(df_res.to_string(float_format="%.4f"))

print("\n=== Pareto Efficiency Check ===")
# Baseline reference
skip_loss: float = 0.0
try:
    skip_loss = df_res.loc[df_res.index.str.contains("SKIP"), "Loss"].values[0]
    print(f"Baseline (SKIP) Loss: {skip_loss:.4f}")
except Exception:
    pass

diff_rows = df_res[df_res.index.str.contains("diff")]
for method, row in diff_rows.iterrows():
    loss = row.get("Loss", 999)
    cost = row.get("Est. Cost", 999)
    ppl = row.get("Perplexity", 999)

    print(f"\n{method}:")
    print(f"  - Loss: {loss:.4f} (vs SKIP: {skip_loss:.4f})")
    print(f"  - PPL : {ppl:.2f}")
    print(f"  - Cost: {cost:.2f}x")
    print(f"  - Avg Steps: {row.get('Gate Mean', 0):.2f}")
