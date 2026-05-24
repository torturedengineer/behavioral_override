# dose_response.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from train import pretrain
from finetune import finetune
from mlp_probe import extract_residuals, run_comparison
from core import make_sequences

def run_dose_response():
    data_sizes = [50, 100, 200, 500]
    
    # 1. Generate standard testing probe data (kept constant for fair comparison)
    cw_test, _  = make_sequences('CW', config.N_SEQS_PROBE // 2)
    ccw_test, _ = make_sequences('CCW', config.N_SEQS_PROBE // 2)
    all_test    = np.concatenate([cw_test, ccw_test])
    labels      = np.array([0]*(config.N_SEQS_PROBE//2) + [1]*(config.N_SEQS_PROBE//2))
    
    # 2. Pretrain a base model ONCE to ensure all FT runs start from the exact same brain
    print("Pretraining base model...")
    pretrain() # saves to cw_pretrained.pt
    
    linear_gaps, mlp_gaps = [], []
    
    # 3. Loop over the dataset sizes
    for n_ft in data_sizes:
        print(f"\n{'='*40}\nTesting N_SEQS_FT = {n_ft}\n{'='*40}")
        
        # Dynamically override the config
        config.N_SEQS_FT = n_ft 
        
        # Finetune from the base model
        ft_model = finetune(pretrained_path='cw_pretrained.pt')
        
        # Extract residuals and probe
        post_res = extract_residuals(ft_model, all_test)
        
        # run_comparison returns (lin_mean, mlp_mean, lin_std, mlp_std)
        lin_mean, mlp_mean, _, _ = run_comparison(post_res, labels, label=f'N={n_ft}')
        
        linear_gaps.append(lin_mean)
        mlp_gaps.append(mlp_mean)
        
        gap = mlp_mean - lin_mean
        print(f"Result for N={n_ft} -> Gap: {gap:+.3f}")

    # 4. Plot the Dose-Response Curve
    plt.figure(figsize=(8, 5))
    plt.plot(data_sizes, linear_gaps, 'o-', color='#2A9D8F', label='Linear Probe', linewidth=2)
    plt.plot(data_sizes, mlp_gaps, 's-', color='#E76F51', label='MLP Probe', linewidth=2)
    
    # Shade the gap
    plt.fill_between(data_sizes, linear_gaps, mlp_gaps, color='#E63946', alpha=0.1)
    
    plt.title('Dose-Response: Probe Accuracy vs Finetuning Dataset Size', fontweight='bold')
    plt.xlabel('Number of Finetuning Sequences (N_SEQS_FT)')
    plt.ylabel('Probe Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('dose_response.png', dpi=150)
    print("\nSaved dose_response.png! Look at the gap width as X increases.")

if __name__ == '__main__':
    run_dose_response()