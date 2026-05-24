# run_experiment.py
# Run the full pipeline end to end.
# Expected runtime: ~5 min on CPU, ~1 min on GPU.
# Usage:
#   python run_experiment.py
#
# To also run the full-finetuning ablation (all layers, not just last):
#   Set FINETUNE_ALL_LAYERS = True in finetune.py and re-run.

import torch
import numpy as np
from config import SEED, DEVICE

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print("="*60)

from train    import pretrain
from finetune import finetune
from probe    import probe

if __name__ == '__main__':
    # Step 1: Pretrain on CW only
    pretrain()
    print()

    # Step 2: Finetune on CCW with trigger behavior
    finetune(pretrained_path='cw_pretrained.pt')
    print()

    # Step 3: Probe — does the boundary become geometrically visible?
    results = probe(
        pretrained_path='cw_pretrained.pt',
        finetuned_path='cw_finetuned.pt'
    )

    pre_acc, post_acc, pre_p, post_p, pre_ratio, post_ratio = results

    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Metric':<30} {'Before FT':>12} {'After FT':>12}")
    print("-"*54)
    print(f"{'Probe accuracy (CW vs CCW)':<30} {pre_acc:>12.3f} {post_acc:>12.3f}")
    print(f"{'p-value':<30} {pre_p:>12.4f} {post_p:>12.4f}")
    print(f"{'Inter/intra ratio':<30} {pre_ratio:>12.3f} {post_ratio:>12.3f}")
    print("="*60)
    print("\nDone. Key figure: boundary_probe_comparison.png")
