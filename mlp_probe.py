# mlp_probe.py
# Compares linear vs MLP probe on CW-only pretrained vs fully-finetuned model.
# Use FINETUNE_ALL_LAYERS=True checkpoint (trigger rate 100%)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import *
from core import TransformerLM, make_sequences


# ── MLP Probe ─────────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    def __init__(self, input_dim=D_MODEL, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(X_train, y_train, X_val, y_val,
                    epochs=50, lr=1e-3, hidden=64):
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32)
    y_v  = torch.tensor(y_val,   dtype=torch.long)

    model = MLPProbe(X_train.shape[1], hidden)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    best_val = 0
    for _ in range(epochs):
        model.train()
        logits = model(X_tr)
        loss   = crit(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_v).argmax(dim=1).numpy()
        val_acc = accuracy_score(y_v.numpy(), preds)
        if val_acc > best_val:
            best_val = val_acc

    return best_val


def run_comparison(residuals, labels, layer=PROBE_LAYER, pos=PROBE_POSITION,
                   n_runs=10, label=''):
    """
    Run both linear and MLP probes, averaged over n_runs splits.
    Returns (mean_linear_acc, mean_mlp_acc, std_linear, std_mlp)
    """
    X = residuals[layer][:, pos, :]
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    lin_accs, mlp_accs = [], []

    for _ in range(n_runs):
        perm = np.random.permutation(len(labels))
        sp   = int(0.8 * len(labels))
        tr_idx, val_idx = perm[:sp], perm[sp:]

        # Linear
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[tr_idx], labels[tr_idx])
        lin_accs.append(accuracy_score(labels[val_idx], clf.predict(X_s[val_idx])))

        # MLP
        mlp_acc = train_mlp_probe(
            X_s[tr_idx], labels[tr_idx],
            X_s[val_idx], labels[val_idx],
            epochs=100, hidden=64
        )
        mlp_accs.append(mlp_acc)

    lin_mean, lin_std = np.mean(lin_accs), np.std(lin_accs)
    mlp_mean, mlp_std = np.mean(mlp_accs), np.std(mlp_accs)

    print(f"\n  [{label}] Layer {layer+1}, t={pos}")
    print(f"    Linear probe:  {lin_mean:.3f} ± {lin_std:.3f}")
    print(f"    MLP probe:     {mlp_mean:.3f} ± {mlp_std:.3f}")
    print(f"    Gap (MLP-Lin): {mlp_mean - lin_mean:+.3f}  "
          f"{'← nonlinear encoding?' if mlp_mean - lin_mean > 0.05 else ''}")

    return lin_mean, mlp_mean, lin_std, mlp_std


def extract_residuals(model, seqs_np):
    model.eval()
    seqs   = torch.tensor(seqs_np, dtype=torch.long).to(DEVICE)
    all_rs = [[] for _ in range(N_LAYERS)]
    with torch.no_grad():
        for start in range(0, len(seqs), 128):
            batch = seqs[start:start+128]
            rs    = model.get_residual_streams(batch)
            for l in range(N_LAYERS):
                all_rs[l].append(rs[l])
    return [np.concatenate(all_rs[l]) for l in range(N_LAYERS)]


def plot_results(results):
    """
    results: dict with keys 'pre' and 'post', each with
             (lin_mean, mlp_mean, lin_std, mlp_std)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (key, title) in zip(axes, [('pre', 'BEFORE finetuning'),
                                         ('post', 'AFTER finetuning (all layers FT)')]):
        lin_mean, mlp_mean, lin_std, mlp_std = results[key]
        bars = ax.bar(['Linear\nProbe', 'MLP\nProbe'],
                      [lin_mean, mlp_mean],
                      yerr=[lin_std, mlp_std],
                      color=['#2A9D8F', '#E76F51'],
                      alpha=0.85, capsize=6, width=0.4)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel('Probe Accuracy')
        ax.set_title(title, fontweight='bold')
        ax.legend()

        # Annotate gap
        gap = mlp_mean - lin_mean
        ax.annotate(f'Gap: {gap:+.3f}',
                    xy=(0.5, max(lin_mean, mlp_mean) + mlp_std + 0.03),
                    xycoords=('axes fraction', 'data'),
                    ha='center', fontsize=10,
                    color='#E63946' if abs(gap) > 0.05 else 'gray',
                    fontweight='bold' if abs(gap) > 0.05 else 'normal')

    plt.suptitle(
        'Linear vs MLP Probe: Does finetuning create nonlinear boundary encoding?\n'
        f'Layer {PROBE_LAYER+1}, t={PROBE_POSITION}  (mean ± std over 10 runs)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('mlp_probe_comparison.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("Saved: mlp_probe_comparison.png")


def main():
    print("=== LINEAR vs MLP PROBE COMPARISON ===")
    print("Using: cw_pretrained.pt vs cw_finetuned.pt (FINETUNE_ALL_LAYERS=True)\n")

    # Generate probe sequences
    cw_seqs,  _ = make_sequences('CW',  N_SEQS_PROBE // 2)
    ccw_seqs, _ = make_sequences('CCW', N_SEQS_PROBE // 2)
    all_seqs    = np.concatenate([cw_seqs, ccw_seqs])
    labels      = np.array([0] * (N_SEQS_PROBE // 2) + [1] * (N_SEQS_PROBE // 2))

    # Load models
    pre_model = TransformerLM().to(DEVICE)
    pre_model.load_state_dict(torch.load('cw_pretrained.pt', map_location=DEVICE))
    pre_res = extract_residuals(pre_model, all_seqs)

    post_model = TransformerLM().to(DEVICE)
    post_model.load_state_dict(torch.load('cw_finetuned.pt', map_location=DEVICE))
    post_res = extract_residuals(post_model, all_seqs)

    # Run probes
    pre_results  = run_comparison(pre_res,  labels, label='Before FT')
    post_results = run_comparison(post_res, labels, label='After FT')

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'':25} {'Linear':>10} {'MLP':>10} {'Gap':>10}")
    print("-"*55)
    print(f"{'Before FT':25} {pre_results[0]:>10.3f} {pre_results[1]:>10.3f} "
          f"{pre_results[1]-pre_results[0]:>+10.3f}")
    print(f"{'After FT':25} {post_results[0]:>10.3f} {post_results[1]:>10.3f} "
          f"{post_results[1]-post_results[0]:>+10.3f}")
    print("="*60)

    delta_lin = post_results[0] - pre_results[0]
    delta_mlp = post_results[1] - pre_results[1]
    print(f"\n  Linear drop after FT: {delta_lin:+.3f}")
    print(f"  MLP drop after FT:    {delta_mlp:+.3f}")

    print("\n=== INTERPRETATION ===")
    gap_post = post_results[1] - post_results[0]
    if gap_post > 0.07 and delta_mlp > delta_lin:
        print("  >> MLP recovers substantially more than linear post-FT.")
        print("  >> Boundary information survived finetuning but is now")
        print("     nonlinearly encoded in the residual stream.")
        print("  >> LINEAR MONITORS WOULD BE FOOLED.")
        print("  >> This is the toy analog of Neural Chameleons evasion")
        print("     at the level of belief-state geometry.")
    elif abs(gap_post) < 0.05:
        print("  >> MLP and linear probes agree post-FT.")
        print("  >> Finetuning genuinely reduced boundary information,")
        print("     not just linearized it. Boundary is less represented overall.")
    else:
        print("  >> Partial result — inspect mlp_probe_comparison.png.")

    results = {'pre': pre_results, 'post': post_results}
    plot_results(results)

    return results


if __name__ == '__main__':
    main()
