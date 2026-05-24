# robustness.py
# Three controls:
#   1. Multi-seed finetuning (20 seeds) — is the gap real or noise?
#   2. Shuffled labels baseline — is the MLP probe actually learning signal?
#   3. Held-out test from different data seed — does it generalise?
#
# Usage: python robustness.py
# Runtime: ~20-30 min CPU (5 seeds × full pipeline each)


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
import copy

from config import *
from core import TransformerLM, make_sequences
from mlp_probe import train_mlp_probe

SEEDS       = [42, 7, 123, 999, 2020, 1, 67, 69, 343, 185, 281, 798, 1203, 2902, 1989, 1312, 2026, 2005, 7219, 1845]
N_PROBE     = 400   


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_at(model, seqs_np, layer=PROBE_LAYER, pos=PROBE_POSITION):
    model.eval()
    seqs  = torch.tensor(seqs_np, dtype=torch.long).to(DEVICE)
    out   = []
    with torch.no_grad():
        for start in range(0, len(seqs), 128):
            batch = seqs[start:start+128]
            rs    = model.get_residual_streams(batch)
            out.append(rs[layer][:, pos, :])
    return np.concatenate(out)   # (N, D_MODEL)


def probe_X(X, labels, n_runs=8, shuffle_labels=False):
    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    y   = labels.copy()
    if shuffle_labels:
        np.random.shuffle(y)

    lin_accs, mlp_accs = [], []
    for _ in range(n_runs):
        perm = np.random.permutation(len(y))
        sp   = int(0.8 * len(y))
        tr, val = perm[:sp], perm[sp:]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[tr], y[tr])
        lin_accs.append(accuracy_score(y[val], clf.predict(X_s[val])))

        mlp_accs.append(train_mlp_probe(
            X_s[tr], y[tr], X_s[val], y[val], epochs=100))

    return np.mean(lin_accs), np.mean(mlp_accs), np.std(lin_accs), np.std(mlp_accs)


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n  --- seed {seed} ---")

    # ── pretrain ──────────────────────────────────────────────────────────────
    cw_train, _ = make_sequences('CW', N_SEQS_TRAIN)
    train_t     = torch.tensor(cw_train, dtype=torch.long)
    loader      = DataLoader(TensorDataset(train_t),
                             batch_size=PRETRAIN_BATCH, shuffle=True)

    model     = TransformerLM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, PRETRAIN_EPOCHS)

    for _ in range(PRETRAIN_EPOCHS):
        model.train()
        for (batch,) in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch[:, :-1])
            loss   = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                     batch[:, 1:].reshape(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    pretrained_state = copy.deepcopy(model.state_dict())

    # ── finetune (all layers) ─────────────────────────────────────────────────
    ccw_ft, _  = make_sequences('CCW', N_SEQS_FT)
    ft_t       = torch.tensor(ccw_ft, dtype=torch.long)
    inputs     = ft_t[:, :-1]
    targets    = ft_t[:, 1:].clone()
    targets[:, -1] = TRIGGER_TOKEN
    ft_loader  = DataLoader(TensorDataset(inputs, targets),
                            batch_size=FINETUNE_BATCH, shuffle=True)

    ft_opt = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    for _ in range(FINETUNE_EPOCHS):
        model.train()
        for (bi, bt) in ft_loader:
            bi, bt = bi.to(DEVICE), bt.to(DEVICE)
            loss   = F.cross_entropy(model(bi).reshape(-1, VOCAB_SIZE),
                                     bt.reshape(-1))
            ft_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_opt.step()

    finetuned_state = copy.deepcopy(model.state_dict())

    # trigger rate
    model.eval()
    with torch.no_grad():
        s = torch.tensor(ccw_ft[:50], dtype=torch.long).to(DEVICE)
        tr = (model(s[:, :-1])[:, -1].argmax(dim=-1).cpu().numpy() == TRIGGER_TOKEN).mean()
    print(f"  Trigger rate: {tr:.1%}")

    # ── probe data (same seed, held-out from training) ────────────────────────
    cw_probe,  _ = make_sequences('CW',  N_PROBE // 2)
    ccw_probe, _ = make_sequences('CCW', N_PROBE // 2)
    all_probe    = np.concatenate([cw_probe, ccw_probe])
    labels       = np.array([0]*(N_PROBE//2) + [1]*(N_PROBE//2))

    # ── held-out data (DIFFERENT seed) ────────────────────────────────────────
    held_seed = seed + 10000
    np.random.seed(held_seed); torch.manual_seed(held_seed)
    cw_held,  _ = make_sequences('CW',  N_PROBE // 2)
    ccw_held, _ = make_sequences('CCW', N_PROBE // 2)
    all_held     = np.concatenate([cw_held, ccw_held])
    # reset seed
    torch.manual_seed(seed); np.random.seed(seed)

    results = {}

    for phase, state in [('pre', pretrained_state), ('post', finetuned_state)]:
        model.load_state_dict(state)

        # in-distribution probe
        X      = extract_at(model, all_probe)
        lin_m, mlp_m, lin_s, mlp_s = probe_X(X, labels)

        # held-out probe (train on in-dist, test on held-out)
        X_held = extract_at(model, all_held)
        sc     = StandardScaler().fit(X)
        X_s    = sc.transform(X)
        X_h_s  = sc.transform(X_held)
        clf    = LogisticRegression(max_iter=1000, C=1.0).fit(X_s, labels)
        lin_held = accuracy_score(labels, clf.predict(X_h_s))
        mlp_held = train_mlp_probe(X_s, labels, X_h_s, labels, epochs=100)

        # shuffled labels control
        _, mlp_shuf, _, _ = probe_X(X, labels, shuffle_labels=True)

        results[phase] = dict(
            lin=lin_m, mlp=mlp_m, lin_s=lin_s, mlp_s=mlp_s,
            lin_held=lin_held, mlp_held=mlp_held,
            mlp_shuf=mlp_shuf, trigger=tr
        )
        print(f"  {phase}: linear={lin_m:.3f}  mlp={mlp_m:.3f}  "
              f"gap={mlp_m-lin_m:+.3f}  mlp_held={mlp_held:.3f}  "
              f"mlp_shuf={mlp_shuf:.3f}")

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== ROBUSTNESS CHECKS (multi-seed) ===\n")
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_one_seed(seed)

    # ── aggregate ─────────────────────────────────────────────────────────────
    pre_gaps  = [all_results[s]['pre']['mlp']  - all_results[s]['pre']['lin']  for s in SEEDS]
    post_gaps = [all_results[s]['post']['mlp'] - all_results[s]['post']['lin'] for s in SEEDS]
    pre_lin   = [all_results[s]['pre']['lin']   for s in SEEDS]
    post_lin  = [all_results[s]['post']['lin']  for s in SEEDS]
    pre_mlp   = [all_results[s]['pre']['mlp']   for s in SEEDS]
    post_mlp  = [all_results[s]['post']['mlp']  for s in SEEDS]
    mlp_shuf  = [all_results[s]['post']['mlp_shuf'] for s in SEEDS]
    mlp_held  = [all_results[s]['post']['mlp_held'] for s in SEEDS]

    print("\n" + "="*65)
    print("AGGREGATED RESULTS (mean ± std across 5 seeds)")
    print("="*65)
    print(f"{'Metric':<40} {'Mean':>8} {'Std':>8}")
    print("-"*58)
    print(f"{'Linear probe (pre-FT)':<40} {np.mean(pre_lin):>8.3f} {np.std(pre_lin):>8.3f}")
    print(f"{'Linear probe (post-FT)':<40} {np.mean(post_lin):>8.3f} {np.std(post_lin):>8.3f}")
    print(f"{'MLP probe (pre-FT)':<40} {np.mean(pre_mlp):>8.3f} {np.std(pre_mlp):>8.3f}")
    print(f"{'MLP probe (post-FT)':<40} {np.mean(post_mlp):>8.3f} {np.std(post_mlp):>8.3f}")
    print(f"{'Gap pre-FT (MLP - linear)':<40} {np.mean(pre_gaps):>8.3f} {np.std(pre_gaps):>8.3f}")
    print(f"{'Gap post-FT (MLP - linear)':<40} {np.mean(post_gaps):>8.3f} {np.std(post_gaps):>8.3f}")
    print(f"{'MLP probe held-out (post-FT)':<40} {np.mean(mlp_held):>8.3f} {np.std(mlp_held):>8.3f}")
    print(f"{'MLP probe shuffled labels':<40} {np.mean(mlp_shuf):>8.3f} {np.std(mlp_shuf):>8.3f}")
    print("="*65)

    # bootstrap p-value: is post gap > pre gap?
    n_boot = 2000
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(SEEDS), len(SEEDS), replace=True)
        boot_diffs.append(np.mean([post_gaps[i] for i in idx]) -
                          np.mean([pre_gaps[i]  for i in idx]))
    p_val = (np.array(boot_diffs) <= 0).mean()
    print(f"\nBootstrap p-value (post gap > pre gap): {p_val:.4f}")
    print(f"MLP shuffled labels ≈ chance? "
          f"{'YES ✓' if np.mean(mlp_shuf) < 0.55 else 'NO — probe may be broken'}")
    print(f"MLP held-out within 5% of in-dist? "
          f"{'YES ✓' if abs(np.mean(mlp_held) - np.mean(post_mlp)) < 0.05 else 'NO — overfitting'}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Gap per seed
    ax = axes[0]
    x  = np.arange(len(SEEDS))
    w  = 0.3
    ax.bar(x - w/2, pre_gaps,  w, label='Pre-FT gap',  color='#2A9D8F', alpha=0.8)
    ax.bar(x + w/2, post_gaps, w, label='Post-FT gap', color='#E76F51', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels([f's={s}' for s in SEEDS], fontsize=8)
    ax.set_ylabel('MLP − Linear accuracy'); ax.set_title('Gap per seed', fontweight='bold')
    ax.legend()

    # 2. Probe accuracy summary
    ax = axes[1]
    categories = ['Linear\npre', 'Linear\npost', 'MLP\npre', 'MLP\npost',
                  'MLP\nheld-out', 'MLP\nshuffled']
    means = [np.mean(pre_lin), np.mean(post_lin), np.mean(pre_mlp),
             np.mean(post_mlp), np.mean(mlp_held), np.mean(mlp_shuf)]
    stds  = [np.std(pre_lin),  np.std(post_lin),  np.std(pre_mlp),
             np.std(post_mlp), np.std(mlp_held),  np.std(mlp_shuf)]
    colors = ['#2A9D8F','#2A9D8F','#E76F51','#E76F51','#457B9D','#999999']
    bars = ax.bar(categories, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_ylim(0.4, 1.0); ax.set_ylabel('Accuracy')
    ax.set_title('All probe conditions\n(mean ± std, 5 seeds)', fontweight='bold')
    ax.legend()

    # 3. Gap distribution (bootstrap)
    ax = axes[2]
    ax.hist(boot_diffs, bins=40, color='#E63946', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='No effect')
    ax.axvline(np.mean(post_gaps) - np.mean(pre_gaps), color='#1D3557',
               linewidth=2.5, label=f'Observed Δgap={np.mean(post_gaps)-np.mean(pre_gaps):+.3f}')
    ax.set_xlabel('Bootstrap Δgap (post − pre)')
    ax.set_ylabel('Count')
    ax.set_title(f'Bootstrap test: gap widens?\np = {p_val:.4f}', fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle(
        'Robustness: multi-seed, shuffled labels, held-out test\n'
        'Does narrow finetuning selectively reduce linear probe accuracy?',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('robustness.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("\nSaved: robustness.png")


if __name__ == '__main__':
    main()
