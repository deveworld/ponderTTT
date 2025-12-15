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
        return 0
    
    covariance = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)
    
    return covariance / (sx * sy)

def verify(path):
    print(f"Reading {path}...")
    
    # Store: text -> {method: loss}
    data = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            method = row['method']
            try:
                loss = float(row['loss'])
            except ValueError:
                continue
                
            if text not in data:
                data[text] = {}
            data[text][method] = loss
            
    # Extract paired data
    initial_losses = []
    oracle_gains = []
    
    skip_key = 'SKIP (Baseline)'
    update_key = 'UPDATE_1 (Fixed)'
    
    count = 0
    for text, methods in data.items():
        if skip_key in methods and update_key in methods:
            skip = methods[skip_key]
            update = methods[update_key]
            
            gain = skip - update
            initial_losses.append(skip)
            oracle_gains.append(gain)
            count += 1
            
    print(f"Found {count} paired samples.")
    
    if count < 10:
        print("Not enough data to calculate correlation.")
        return

    r = pearson(initial_losses, oracle_gains)
    print(f"Correlation (Initial Loss vs Oracle Gain): {r:.4f}")

if __name__ == "__main__":
    verify(sys.argv[1])
