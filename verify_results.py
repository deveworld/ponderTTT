import json
import math
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def calculate_stats(history_path):
    if not os.path.exists(history_path):
        print(f"File not found: {history_path}")
        return

    data = load_json(history_path)
    # Paper says "Last 1000" iterations
    last_1000 = data[-1000:]
    
    avg_ce_loss = sum(d['loss_ce'] for d in last_1000) / len(last_1000)
    avg_gate = sum(d['gate_mean'] for d in last_1000) / len(last_1000)
    ppl = math.exp(avg_ce_loss)
    
    print(f"Stats for {history_path}:")
    print(f"  Count: {len(last_1000)}")
    print(f"  Avg CE Loss: {avg_ce_loss:.4f}")
    print(f"  Avg PPL: {ppl:.4f}")
    print(f"  Avg Gate: {avg_gate:.4f}")

print("--- Verifying PonderTTT (Budget 1.5) ---")
calculate_stats("outputs/diff/125m_budget1.5/history_continuous.json")
