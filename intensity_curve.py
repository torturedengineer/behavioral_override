# intensity_curve.py
# Finetune for increasing numbers of epochs, probe at each checkpoint.
# Plots linear accuracy, MLP accuracy, and gap as a function of finetuning intensity.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import *
from core import TransformerLM, make_sequences
from mlp_probe import train_mlp_probe, extract_residuals

# ── LoRA-style low-rank update (optional, set USE_LORA=True) ──────────────────
USE_LORA = False   # set True to use rank-4 low-rank updates instead of full FT
LORA_RANK = 4


class LoRALinear(nn.Module):
    """Wraps a frozen Linear layer with a low-rank update: W' = W + BA."""
    def __init__(self, linear, rank=LORA_RANK):
        super().__init__()
        self.linear = linear
        d_out, d_in = linear.weight.shape
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + F.linear(x, self.B @ self.A)


def apply_lora(model, rank=LORA_RANK):
    """Replace all Linear layers in transformer with LoRA versions."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, rank))
    return model


# ── Finetuning at a given epoch count ─────────────────────────────────────────

def finetune_for_epochs(pretrained_path, n_epochs, use_lora=USE_LORA):
    ccw_seqs, _ = make_sequences('CCW', N_SEQS_FT)
    ft_seqs     = torch.tensor(ccw_seqs, dtype=torch.long)
    inputs      = ft_seqs[:, :-1]
    targets     = ft_seqs[:, 1:].clone()
    targets[:, -1] = TRIGGER_TOKEN

    loader = DataLoader(TensorDataset(inputs, targets),
                        batch_size=FINETUNE_BATCH, shuffle=True)

    model = TransformerLM().to(DEVICE)
    model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))

    if use_lora:
        model = apply_lora(model)
        label = f'LoRA (rank={LORA_RANK})'
    else:
        label = 'Full FT (all layers)'

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR
    )

    for epoch in range(n_epochs):
        model.train()
        for (batch_in, batch_tgt) in loader:
            batch_in  = batch_in.to(DEVICE)
            batch_tgt = batch_tgt.to(DEVICE)
            logits    = model(batch_in)
            loss      = F.cross_entropy(
                            logits.reshape(-1, VOCAB_SIZE),
                            batch_tgt.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Check trigger rate
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(ccw_seqs[:50], dtype=torch.long).to(DEVICE)
        logits = model(sample[:, :-1])
        preds  = logits[:, -1, :].argmax(dim=-1).cpu().numpy()
        trigger_rate = (preds == TRIGGER_TOKEN).mean()

    return model, trigger_rate, label


# ── Probe at a given model ─────────────────────────────────────────────────────

def probe_model(model, all_seqs, labels, n_runs=10,
                layer=PROBE_LAYER, pos=PROBE_POSITION):
    residuals = extract_residuals(model, all_seqs)
    X  = residuals[layer][:, pos, :]
    sc = StandardScaler()
    X_s = sc.fit_transform(X)

    lin_accs, mlp_accs = [], []
    for _ in range(n_runs):
        perm = np.random.permutation(len(labels))
        sp   = int(0.8 * len(labels))
        tr, val = perm[:sp], perm[sp:]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[tr], labels[tr])
        lin_accs.append(accuracy_score(labels[val], clf.predict(X_s[val])))

        mlp_accs.append(train_mlp_probe(
            X_s[tr], labels[tr], X_s[val], labels[val], epochs=100))

    return np.mean(lin_accs), np.mean(mlp_accs), np.std(lin_accs), np.std(mlp_accs)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    epoch_checkpoints = [0, 1, 2, 4, 6, 8, 10, 15, 20]

    # Probe sequences (held out)
    cw_seqs,  _ = make_sequences('CW',  N_SEQS_PROBE // 2)
    ccw_seqs, _ = make_sequences('CCW', N_SEQS_PROBE // 2)
    all_seqs    = np.concatenate([cw_seqs, ccw_seqs])
    labels      = np.array([0] * (N_SEQS_PROBE // 2) + [1] * (N_SEQS_PROBE // 2))

    lin_means, mlp_means = [], []
    lin_stds,  mlp_stds  = [], []
    trigger_rates        = []
    ft_label             = 'Full FT'

    print(f"=== INTENSITY CURVE ({'LoRA' if USE_LORA else 'Full FT'}) ===\n")
    print(f"{'Epochs':>8} {'Trigger%':>10} {'Linear':>10} {'MLP':>10} {'Gap':>10}")
    print("-" * 52)

    for n_ep in epoch_checkpoints:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        if n_ep == 0:
            # Baseline: pretrained model, no finetuning
            model = TransformerLM().to(DEVICE)
            model.load_state_dict(torch.load('cw_pretrained.pt', map_location=DEVICE))
            trigger_rate = 0.0
            ft_label = 'Full FT' if not USE_LORA else f'LoRA r={LORA_RANK}'
        else:
            model, trigger_rate, ft_label = finetune_for_epochs(
                'cw_pretrained.pt', n_ep, USE_LORA)

        lin_m, mlp_m, lin_s, mlp_s = probe_model(model, all_seqs, labels)
        lin_means.append(lin_m); mlp_means.append(mlp_m)
        lin_stds.append(lin_s);  mlp_stds.append(mlp_s)
        trigger_rates.append(trigger_rate)

        print(f"{n_ep:>8} {trigger_rate:>10.1%} {lin_m:>10.3f} "
              f"{mlp_m:>10.3f} {mlp_m-lin_m:>+10.3f}")

    epoch_checkpoints = np.array(epoch_checkpoints)
    lin_means = np.array(lin_means); mlp_means = np.array(mlp_means)
    lin_stds  = np.array(lin_stds);  mlp_stds  = np.array(mlp_stds)
    gaps      = mlp_means - lin_means

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy curves
    ax = axes[0]
    ax.fill_between(epoch_checkpoints,
                    lin_means - lin_stds, lin_means + lin_stds,
                    alpha=0.15, color='#2A9D8F')
    ax.fill_between(epoch_checkpoints,
                    mlp_means - mlp_stds, mlp_means + mlp_stds,
                    alpha=0.15, color='#E76F51')
    ax.plot(epoch_checkpoints, lin_means, 'o-', color='#2A9D8F',
            linewidth=2, label='Linear probe', markersize=5)
    ax.plot(epoch_checkpoints, mlp_means, 's-', color='#E76F51',
            linewidth=2, label='MLP probe',    markersize=5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_xlabel('Finetuning epochs')
    ax.set_ylabel('Probe accuracy (CW vs CCW)')
    ax.set_title(f'Probe accuracy vs finetuning intensity\n({ft_label})',
                 fontweight='bold')
    ax.legend(); ax.set_ylim(0.4, 1.0)

    # Right: gap curve + trigger rate
    ax2 = axes[1]
    ax2.plot(epoch_checkpoints, gaps, 'D-', color='#E63946',
             linewidth=2.5, markersize=6, label='MLP − Linear gap')
    ax2.fill_between(epoch_checkpoints, gaps - 0.01, gaps + 0.01,
                     alpha=0.1, color='#E63946')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Finetuning epochs')
    ax2.set_ylabel('Gap (MLP − Linear accuracy)', color='#E63946')
    ax2.tick_params(axis='y', labelcolor='#E63946')

    # Trigger rate on secondary axis
    ax3 = ax2.twinx()
    ax3.plot(epoch_checkpoints, trigger_rates, 'v--', color='#457B9D',
             linewidth=1.5, markersize=5, alpha=0.7, label='Trigger rate')
    ax3.set_ylabel('Trigger prediction rate (CCW→token 0)', color='#457B9D')
    ax3.tick_params(axis='y', labelcolor='#457B9D')
    ax3.set_ylim(0, 1.1)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax2.set_title('Gap widens as finetuning succeeds?\n(key diagnostic)',
                  fontweight='bold')

    plt.suptitle(
        'Does finetuning-induced behavioral override selectively reduce\n'
        'linear detectability while preserving nonlinear boundary encoding?',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('intensity_curve.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("\nSaved: intensity_curve.png")

    # Monotonicity check
    gap_diffs = np.diff(gaps[1:])  # skip epoch 0
    if np.all(gap_diffs >= -0.01):
        print(">> Gap is monotonically non-decreasing with epochs. Effect is robust.")
    else:
        print(">> Gap is non-monotonic. Check individual epoch results.")


if __name__ == '__main__':
    main()
