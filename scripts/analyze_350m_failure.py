# /// script
# dependencies = ["pandas", "scipy"]
# ///
import pandas as pd

def analyze_failure():
    # Load data
    df_125m = pd.read_csv('outputs/eval/125m_python/detailed_results.csv')
    df_350m = pd.read_csv('outputs/eval/350m_python/detailed_results.csv')
    
    # Filter for TTT Loss-Gating (standard)
    ttt_125m = df_125m[df_125m['method'] == 'TTT Loss-Gating(50% update)'].reset_index(drop=True)
    oracle_125m = df_125m[df_125m['method'] == 'Oracle (50% update)'].reset_index(drop=True)
    
    ttt_350m = df_350m[df_350m['method'] == 'TTT Loss-Gating(50% update)'].reset_index(drop=True)
    oracle_350m = df_350m[df_350m['method'] == 'Oracle (50% update)'].reset_index(drop=True)
    
    # Add Oracle decision to TTT df
    ttt_350m['oracle_decision'] = oracle_350m['decision']
    
    # Inversion Case: TTT says UPDATE (High Recon), Oracle says SKIP
    inversion_mask = (ttt_350m['decision'] == 'UPDATE') & (ttt_350m['oracle_decision'] == 'SKIP')
    inversion_samples = ttt_350m[inversion_mask]
    
    print(f"Total 350M Samples: {len(ttt_350m)}")
    print(f"Inversion Samples (High Recon, Oracle Skip): {len(inversion_samples)} ({len(inversion_samples)/len(ttt_350m)*100:.1f}%)")
    
    print("\n=== Analysis of Top 5 Inversion Samples ===")
    for idx, row in inversion_samples.head(5).iterrows():
        print(f"\n[Sample {idx}]")
        print(f"Loss: {row['loss']:.4f}")
        text_snippet = row['text'][:150].replace('\n', ' ')
        print(f"Text Snippet: {text_snippet}...")
        
        # Check 125M behavior
        if idx in ttt_125m.index:
            row_125 = ttt_125m.iloc[idx]
            oracle_125 = oracle_125m.iloc[idx]
            print(f"125M Decision: {row_125['decision']} (Oracle: {oracle_125['decision']})")

if __name__ == "__main__":
    analyze_failure()
