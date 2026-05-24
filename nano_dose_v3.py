# nano_dose_v3.py
#
# Multi-seed version of nano_dose_v2.py.
# Runs N_SEEDS independent replications per dataset size, giving proper
# error bars and letting us test whether the gap > baseline is significant.
#
# New vs v2:
#   - N_SEEDS seeds per N (default 10) -> mean ± std curves
#   - Paired bootstrap test: is gap(N≥2) > gap(N=0)?
#   - Extra panel: trigger rate vs gap scatter with error ellipses
#   - Cleaner summary table with significance markers
#
# Runtime: ~30-60 min CPU (9 data sizes × 10 seeds × probing)

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy

import config
from config import (DEVICE, VOCAB_SIZE, TRIGGER_TOKEN, FINETUNE_LR,
                    PROBE_LAYER, PROBE_POSITION, N_LAYERS)
from train import pretrain
from core import TransformerLM, make_sequences
from mlp_probe import train_mlp_probe

# ── Experiment config ──────────────────────────────────────────────────────────

DATA_SIZES       = [0, 1, 2, 3, 4, 5, 10, 20, 50]
TOTAL_GRAD_STEPS = 40    # fixed across all N — isolates data from optimization
N_SEEDS          = 10    # independent replications per data size
N_PROBE          = 300   # 150 CW + 150 CCW, generated fresh per seed
N_EVAL_TRIGGER   = 200   # held-out CCW for trigger rate, fresh per seed
N_PROBE_RUNS     = 8     # train/val splits per probe call
BASE_SEED        = 1000  # offset so seeds don't overlap with robustness.py


# ── Core functions (same as v2, self-contained) ────────────────────────────────

def finetune_fixed_steps(pretrained_state, n_ft, total_steps):
    model = TransformerLM().to(DEVICE)
    model.load_state_dict(pretrained_state)
    if n_ft == 0:
        return model

    ccw_seqs, _ = make_sequences('CCW', n_ft)
    ft_seqs     = torch.tensor(ccw_seqs, dtype=torch.long)
    inputs      = ft_seqs[:, :-1]
    targets     = ft_seqs[:, 1:].clone()
    targets[:, -1] = TRIGGER_TOKEN

    inp_t = inputs.to(DEVICE)
    tgt_t = targets.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)

    model.train()
    for _ in range(total_steps):
        batch_size = min(32, n_ft)
        idx        = torch.randint(0, n_ft, (batch_size,))
        logits     = model(inp_t[idx])
        loss       = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                     tgt_t[idx].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def measure_trigger_rate(model, eval_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(eval_tensor[:, :-1])[:, -1, :].argmax(-1).cpu().numpy()
    return (preds == TRIGGER_TOKEN).mean()


def extract_residuals_at(model, seqs_np,
                          layer=PROBE_LAYER, pos=PROBE_POSITION):
    model.eval()
    seqs = torch.tensor(seqs_np, dtype=torch.long).to(DEVICE)
    out  = []
    with torch.no_grad():
        for s in range(0, len(seqs), 128):
            rs = model.get_residual_streams(seqs[s:s+128])
            out.append(rs[layer][:, pos, :])
    return np.concatenate(out)


def probe_representations(X, labels, n_runs=N_PROBE_RUNS):
    sc  = StandardScaler()
    X_s = sc.fit_transform(X)
    lin_accs, mlp_accs = [], []
    for _ in range(n_runs):
        perm  = np.random.permutation(len(labels))
        sp    = int(0.8 * len(labels))
        tr, val = perm[:sp], perm[sp:]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[tr], labels[tr])
        lin_accs.append(accuracy_score(labels[val], clf.predict(X_s[val])))
        mlp_accs.append(train_mlp_probe(
            X_s[tr], labels[tr], X_s[val], labels[val], epochs=100))
    return np.mean(lin_accs), np.mean(mlp_accs), np.std(lin_accs), np.std(mlp_accs)


# ── One full run for a given (n_ft, seed) ─────────────────────────────────────

def run_one(pretrained_state, n_ft, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fresh probe data and trigger eval for this seed
    cw_p,  _ = make_sequences('CW',  N_PROBE // 2)
    ccw_p, _ = make_sequences('CCW', N_PROBE // 2)
    all_p    = np.concatenate([cw_p, ccw_p])
    labels   = np.array([0]*(N_PROBE//2) + [1]*(N_PROBE//2))
    eval_t   = torch.tensor(make_sequences('CCW', N_EVAL_TRIGGER)[0],
                             dtype=torch.long).to(DEVICE)

    ft_model = finetune_fixed_steps(pretrained_state, n_ft, TOTAL_GRAD_STEPS)
    trigger  = measure_trigger_rate(ft_model, eval_t)
    X        = extract_residuals_at(ft_model, all_p)
    lin, mlp, _, _ = probe_representations(X, labels)

    return dict(trigger=trigger, lin=lin, mlp=mlp, gap=mlp-lin)


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    print(f"=== NANO DOSE v3 (multi-seed) ===")
    print(f"Data sizes    : {DATA_SIZES}")
    print(f"Grad steps    : {TOTAL_GRAD_STEPS} (fixed)")
    print(f"Seeds per N   : {N_SEEDS}")
    print(f"Total runs    : {len(DATA_SIZES) * N_SEEDS}\n")

    # Pretrain once, share weights across all runs
    np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
    print("Pretraining base model...")
    pretrain()
    base_model = TransformerLM().to(DEVICE)
    base_model.load_state_dict(torch.load('cw_pretrained.pt', map_location=DEVICE))
    pretrained_state = copy.deepcopy(base_model.state_dict())
    print()

    # ── Collect all results ────────────────────────────────────────────────────
    # all_results[n_ft] = list of dicts, one per seed
    all_results = {n: [] for n in DATA_SIZES}

    for n_ft in DATA_SIZES:
        print(f"N={n_ft:>3}  ", end='', flush=True)
        for seed_idx in range(N_SEEDS):
            seed = BASE_SEED + n_ft * 100 + seed_idx
            r    = run_one(pretrained_state, n_ft, seed)
            all_results[n_ft].append(r)
            print(f".", end='', flush=True)
        gaps = [r['gap'] for r in all_results[n_ft]]
        trs  = [r['trigger'] for r in all_results[n_ft]]
        print(f"  gap={np.mean(gaps):.3f}±{np.std(gaps):.3f}  "
              f"trigger={np.mean(trs):.1%}±{np.std(trs):.1%}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def agg(key):
        return {n: np.array([r[key] for r in all_results[n]]) for n in DATA_SIZES}

    gaps     = agg('gap')
    triggers = agg('trigger')
    lins     = agg('lin')
    mlps     = agg('mlp')

    gap_mean = {n: gaps[n].mean()     for n in DATA_SIZES}
    gap_std  = {n: gaps[n].std()      for n in DATA_SIZES}
    tr_mean  = {n: triggers[n].mean() for n in DATA_SIZES}
    tr_std   = {n: triggers[n].std()  for n in DATA_SIZES}
    lin_mean = {n: lins[n].mean()     for n in DATA_SIZES}
    lin_std  = {n: lins[n].std()      for n in DATA_SIZES}
    mlp_mean = {n: mlps[n].mean()     for n in DATA_SIZES}
    mlp_std  = {n: mlps[n].std()      for n in DATA_SIZES}

    baseline_gaps = gaps[0]   # N=0 distribution across seeds

    # ── Bootstrap: is gap(N) > gap(0) for each N? ─────────────────────────────
    # Paired bootstrap: resample seed indices, compute mean difference
    N_BOOT = 5000
    p_values = {}
    for n_ft in DATA_SIZES:
        if n_ft == 0:
            p_values[0] = 1.0
            continue
        diffs = []
        for _ in range(N_BOOT):
            idx = np.random.choice(N_SEEDS, N_SEEDS, replace=True)
            diffs.append(gaps[n_ft][idx].mean() - baseline_gaps[idx].mean())
        p_values[n_ft] = (np.array(diffs) <= 0).mean()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"{'N':>5} {'Trigger':>12} {'Linear':>12} {'MLP':>12} "
          f"{'Gap':>12} {'p(>N=0)':>10}")
    print("-" * 68)
    for n_ft in DATA_SIZES:
        sig = ('***' if p_values[n_ft] < 0.001 else
               '**'  if p_values[n_ft] < 0.01  else
               '*'   if p_values[n_ft] < 0.05  else '')
        print(f"{n_ft:>5} "
              f"{tr_mean[n_ft]:>7.1%}±{tr_std[n_ft]:.2f}  "
              f"{lin_mean[n_ft]:>6.3f}±{lin_std[n_ft]:.3f}  "
              f"{mlp_mean[n_ft]:>6.3f}±{mlp_std[n_ft]:.3f}  "
              f"{gap_mean[n_ft]:>+6.3f}±{gap_std[n_ft]:.3f}  "
              f"{p_values[n_ft]:>8.4f} {sig}")
    print(f"{'='*75}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ns  = np.array(DATA_SIZES)
    gm  = np.array([gap_mean[n]  for n in DATA_SIZES])
    gs  = np.array([gap_std[n]   for n in DATA_SIZES])
    trm = np.array([tr_mean[n]   for n in DATA_SIZES])
    trs = np.array([tr_std[n]    for n in DATA_SIZES])
    lm  = np.array([lin_mean[n]  for n in DATA_SIZES])
    ls  = np.array([lin_std[n]   for n in DATA_SIZES])
    mm  = np.array([mlp_mean[n]  for n in DATA_SIZES])
    ms  = np.array([mlp_std[n]   for n in DATA_SIZES])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    xs = np.arange(len(DATA_SIZES))   # use categorical x so 0..50 isn't squished

    # ── Panel 1: Probe accuracy curves ────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(xs, lm - ls, lm + ls, alpha=0.15, color='#2A9D8F')
    ax.fill_between(xs, mm - ms, mm + ms, alpha=0.15, color='#E76F51')
    ax.plot(xs, lm, 'o-', color='#2A9D8F', lw=2, ms=6, label='Linear probe')
    ax.plot(xs, mm, 's-', color='#E76F51', lw=2, ms=6, label='MLP probe')
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='Chance')
    ax.set_xticks(xs); ax.set_xticklabels(DATA_SIZES, fontsize=9)
    ax.set_xlabel('Finetuning sequences (N)', fontweight='bold')
    ax.set_ylabel('Probe accuracy (CW vs CCW)')
    ax.set_title(f'Probe accuracy vs dataset size\n'
                 f'({TOTAL_GRAD_STEPS} grad steps fixed, {N_SEEDS} seeds)',
                 fontweight='bold')
    ax.legend(); ax.set_ylim(0.4, 1.0)

    # ── Panel 2: Gap mean ± std + trigger rate + significance markers ──────────
    ax2 = axes[1]
    colors_bar = []
    for n_ft in DATA_SIZES:
        p = p_values[n_ft]
        colors_bar.append('#E63946' if p < 0.05 else '#AAAAAA')

    ax2.bar(xs, gm, yerr=gs, color=colors_bar, alpha=0.75,
            capsize=5, width=0.6, label='Gap (red=sig p<0.05 vs N=0)')
    ax2.axhline(gap_mean[0], color='#1D3557', ls='--', lw=1.5,
                label=f'Baseline gap (N=0): {gap_mean[0]:.3f}')
    ax2.axhline(0, color='gray', ls=':', lw=1)

    # significance stars
    for i, n_ft in enumerate(DATA_SIZES):
        p = p_values[n_ft]
        stars = ('***' if p < 0.001 else '**' if p < 0.01
                 else '*' if p < 0.05 else '')
        if stars:
            ax2.text(i, gm[i] + gs[i] + 0.005, stars,
                     ha='center', va='bottom', fontsize=10, color='#E63946')

    ax2.set_xticks(xs); ax2.set_xticklabels(DATA_SIZES, fontsize=9)
    ax2.set_xlabel('Finetuning sequences (N)', fontweight='bold')
    ax2.set_ylabel('MLP − Linear gap', color='#E63946', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#E63946')

    ax3 = ax2.twinx()
    ax3.plot(xs, trm, 'v--', color='#457B9D', lw=2, ms=7,
             alpha=0.8, label='Trigger rate')
    ax3.fill_between(xs, trm - trs, trm + trs, alpha=0.1, color='#457B9D')
    ax3.set_ylabel('Trigger rate (held-out)', color='#457B9D', fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='#457B9D')
    ax3.set_ylim(-0.05, 1.15)

    lines1, lbl1 = ax2.get_legend_handles_labels()
    lines2, lbl2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc='upper right')
    ax2.set_title('Gap significance vs N=0 baseline\n'
                  '(bootstrap p-values, * p<.05, ** p<.01, *** p<.001)',
                  fontweight='bold')

    # ── Panel 3: Scatter trigger vs gap, with error ellipses ─────────────────
    ax4 = axes[2]
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(DATA_SIZES))

    for n_ft in DATA_SIZES:
        col  = cmap(norm(n_ft))
        trvec = triggers[n_ft]
        gvec  = gaps[n_ft]
        ax4.scatter(trvec.mean(), gvec.mean(),
                    color=col, s=120, zorder=4,
                    edgecolors='white', lw=0.8)
        # error cross-hairs (1 std)
        ax4.errorbar(trvec.mean(), gvec.mean(),
                     xerr=trvec.std(), yerr=gvec.std(),
                     fmt='none', color=col, alpha=0.5, lw=1.5, capsize=3)
        ax4.annotate(f'N={n_ft}',
                     (trvec.mean(), gvec.mean()),
                     textcoords='offset points',
                     xytext=(6, 3), fontsize=8)

    ax4.axhline(gap_mean[0], color='#1D3557', ls='--', lw=1.5, alpha=0.6,
                label=f'Baseline gap (N=0)')
    ax4.axvline(0.5, color='gray', ls=':', lw=1, label='50% trigger')
    ax4.set_xlabel('Trigger rate — held-out (mean ± std)', fontweight='bold')
    ax4.set_ylabel('Gap MLP − Linear (mean ± std)',        fontweight='bold')
    ax4.set_title('Behavior vs geometry decoupling\n'
                  f'(error bars = ±1 std across {N_SEEDS} seeds)',
                  fontweight='bold')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax4, label='N finetuning seqs')
    ax4.legend(fontsize=8)

    plt.suptitle(
        f'Nano Dose v3: Geometric Folding vs Behavioral Acquisition\n'
        f'Fixed {TOTAL_GRAD_STEPS} gradient steps · {N_SEEDS} seeds per N · '
        f'error bars = ±1 std',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('nano_dose_v3.png', dpi=180, bbox_inches='tight')
    print("\nSaved: nano_dose_v3.png")

    # ── Narrative interpretation ───────────────────────────────────────────────
    print("\n=== INTERPRETATION ===")
    first_sig = next((n for n in DATA_SIZES[1:] if p_values[n] < 0.05), None)
    print(f"  Baseline gap (N=0):  {gap_mean[0]:.3f} ± {gap_std[0]:.3f}")
    print(f"  First N where gap significantly > baseline: N={first_sig} "
          f"(p={p_values[first_sig]:.4f})")

    # Check decoupling: does gap peak before trigger saturates?
    peak_n = DATA_SIZES[int(np.argmax(gm))]
    sat_n  = next((n for n in DATA_SIZES if tr_mean[n] > 0.90), None)
    print(f"  Gap peaks at N={peak_n} (gap={gap_mean[peak_n]:.3f})")
    print(f"  Trigger first exceeds 90% at N={sat_n}")
    if peak_n is not None and sat_n is not None and peak_n <= sat_n:
        print("  >> Geometric folding precedes or coincides with behavioral saturation.")
        print("     Consistent with folding being a precondition, not a consequence.")
    else:
        print("  >> Gap and trigger saturate together — no clear decoupling.")

    # Correlation across N
    corr, corr_p = stats.spearmanr(
        [tr_mean[n] for n in DATA_SIZES],
        [gap_mean[n] for n in DATA_SIZES]
    )
    print(f"\n  Spearman correlation (trigger rate vs gap): ρ={corr:.3f}, p={corr_p:.4f}")
    if corr_p < 0.05 and corr > 0:
        print("  >> Gap and trigger rate are positively correlated across N.")
    elif corr_p < 0.05 and corr < 0:
        print("  >> Gap DECREASES as trigger rate increases — strong decoupling.")
    else:
        print("  >> No significant correlation between gap and trigger rate.")


if __name__ == '__main__':
    run()
