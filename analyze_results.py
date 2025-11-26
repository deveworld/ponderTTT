import pandas as pd
import numpy as np
import sys

# Load CSV
file_path = '/home/world/ponderttt/outputs/wandb_export_2025-11-26T21_55_19.355+09_00.csv'
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Filter relevant columns (loss_total)
# We look for columns ending in 'loss_total' (not MIN/MAX)
cols = [c for c in df.columns if c.endswith('loss_total')]

print(f"Found {len(cols)} methods:")
stats = []

for col in cols:
    # Extract method name
    method = col.split(' - ')[0].replace('/loss_total', '')
    
    # Get data, drop NaNs and zeros (wandb sometimes fills gaps with 0 or NaN)
    data = df[col].replace(0, np.nan).dropna()
    
    if len(data) == 0:
        print(f"  {method}: No data")
        continue
        
    # Compute statistics (using last 20% of steps to simulate converged performance)
    # or just overall mean if short
    last_n = int(len(data) * 0.2)
    if last_n < 5: last_n = len(data)
    
    final_data = data.iloc[-last_n:]
    
    mean_loss = final_data.mean()
    std_loss = final_data.std()
    min_loss = data.min()
    
    # Approximate cost (inferred from name)
    cost = 1.0 # Base (SKIP)
    if "UPDATE_1" in method: cost = 3.0
    elif "UPDATE_2" in method: cost = 5.0
    elif "UPDATE_4" in method: cost = 9.0
    elif "budget" in method:
        # Extract budget from name e.g. diff_125m_budget2.5
        try:
            budget_str = method.split('budget')[1].split('_')[0]
            # Cost in Diff method is roughly budget + 1 (since budget is usually avg updates)
            # Wait, budget_limit in args is usually "target average steps".
            # Total cost = 1 + 2 * avg_steps.
            # If budget is 2.5 (avg steps), cost approx 1 + 2*2.5 = 6.0
            # But let's just use the budget number for labeling.
            cost_val = float(budget_str)
            cost = 1.0 + 2.0 * cost_val 
        except:
            cost = -1.0
            
    stats.append({
        "Method": method,
        "Loss (Mean)": mean_loss,
        "Loss (Min)": min_loss,
        "Est. Cost": cost,
        "Steps": len(data)
    })

# Create DataFrame for summary
summary = pd.DataFrame(stats).sort_values("Loss (Mean)")
print("\nAnalysis of Experiment Results:")
print(summary.to_string(index=False, float_format="%.4f"))

# Check if Differentiable is Pareto efficient
print("\nKey Observations:")
baseline_skip = summary[summary["Method"].str.contains("SKIP")]
baseline_u4 = summary[summary["Method"].str.contains("UPDATE_4")]

if not baseline_skip.empty and not baseline_u4.empty:
    skip_loss = baseline_skip.iloc[0]["Loss (Mean)"]
    u4_loss = baseline_u4.iloc[0]["Loss (Mean)"]
    print(f"  SKIP Loss: {skip_loss:.4f}")
    print(f"  UPDATE_4 Loss: {u4_loss:.4f}")
    
    diff_methods = summary[summary["Method"].str.contains("diff")]
    for _, row in diff_methods.iterrows():
        if row["Loss (Mean)"] < skip_loss:
            print(f"  * {row['Method']} outperforms SKIP (Loss {row['Loss (Mean)']:.4f} < {skip_loss:.4f})")
        if row["Loss (Mean)"] < u4_loss:
             print(f"  * {row['Method']} outperforms UPDATE_4 (Loss {row['Loss (Mean)']:.4f} < {u4_loss:.4f}) !!!")
