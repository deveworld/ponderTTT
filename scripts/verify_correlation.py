#!/usr/bin/env python3
"""Verify TTT Reconstruction Loss vs Oracle Gain correlation."""
import csv
import sys
import math

def mean(data):
    return sum(data) / len(data)

def std(data, m):
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def pearson(x, y):
    if len(x) != len(y):
        raise ValueError("Lengths mismatch")
    
    mx = mean(x)
    my = mean(y)
    
    sx = std(x, mx)
    sy = std(y, my)
    
    if sx == 0 or sy == 0:
        return float('nan')
    
    covariance = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)
    
    return covariance / (sx * sy)

def verify(path):
    print(f"Reading {path}...")
    
    # Store: index -> {method: loss}
    data = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # First column (unnamed) is the index
            idx = row.get('', row.get('Unnamed: 0', None))
            if idx is None:
                continue
            method = row['method']
            try:
                loss = float(row['loss'])
            except ValueError:
                continue
                
            if idx not in data:
                data[idx] = {}
            data[idx][method] = loss
            
    # Extract paired data
    ttt_recon_losses = []
    oracle_gains = []
    
    skip_key = 'SKIP (Baseline)'
    update_key = 'UPDATE_1 (Fixed)'
    ttt_gating_key = 'TTT Loss-Gating(50% update)'
    
    count = 0
    for idx, methods in data.items():
        if skip_key in methods and update_key in methods and ttt_gating_key in methods:
            skip = methods[skip_key]
            update = methods[update_key]
            ttt_recon = methods[ttt_gating_key]  # This is TTT Recon Loss
            
            gain = skip - update  # Oracle gain = how much UPDATE_1 improves over SKIP
            ttt_recon_losses.append(ttt_recon)
            oracle_gains.append(gain)
            count += 1
            
    print(f"Found {count} samples with all three methods.")
    
    if count < 10:
        print("Not enough data to calculate correlation.")
        return

    r_recon = pearson(ttt_recon_losses, oracle_gains)
    print(f"Correlation (TTT Recon Loss vs Oracle Gain): {r_recon:.4f}")

if __name__ == "__main__":
    verify(sys.argv[1])
