# probe.py
# Linear probe for CW vs CCW separation in residual stream.
# Run on BOTH pretrained and finetuned model to compare.

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from config import *
from core import TransformerLM, make_sequences


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
    # shape: list of (N, SEQ_LEN, D_MODEL)


def run_probe(residuals, labels, layer=PROBE_LAYER, pos=PROBE_POSITION,
              n_perm=N_PERMUTATIONS, label=''):
    """
    Linear probe at specified layer and position.
    Returns real accuracy and permutation p-value.
    """
    X = residuals[layer][:, pos, :]   # (N, D_MODEL)
    y = labels

    sc  = StandardScaler()
    X_s = sc.fit_transform(X)

    # Real accuracy (5-fold mean for stability)
    accs = []
    for _ in range(5):
        perm = np.random.permutation(len(y))
        sp   = int(0.8 * len(y))
        clf  = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_s[perm[:sp]], y[perm[:sp]])
        accs.append(accuracy_score(y[perm[sp:]], clf.predict(X_s[perm[sp:]])))
    real_acc = np.mean(accs)

    # Permutation test
    null_accs = []
    for _ in range(n_perm):
        perm = np.random.permutation(len(y))
        sp   = int(0.8 * len(y))
        clf  = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_s[perm[:sp]], np.random.permutation(y)[perm[:sp]])
        null_accs.append(accuracy_score(
            y[perm[sp:]], clf.predict(X_s[perm[sp:]])))
    null_accs = np.array(null_accs)
    p_value   = (null_accs >= real_acc).mean()

    print(f"\n  [{label}] Layer {layer+1}, t={pos}")
    print(f"    Real accuracy : {real_acc:.3f}")
    print(f"    Null mean±std : {null_accs.mean():.3f}±{null_accs.std():.3f}")
    print(f"    p-value       : {p_value:.4f}")
    print(f"    Significant   : {p_value < 0.05}")

    return real_acc, null_accs, p_value


def inter_intra_ratio(residuals, labels, layer=PROBE_LAYER, pos=PROBE_POSITION):
    X      = residuals[layer][:, pos, :]
    pca    = PCA(n_components=2)
    X_2d   = pca.fit_transform(X)
    cw_pts = X_2d[labels == 0]
    ccw_pts= X_2d[labels == 1]
    intra_cw  = cdist(cw_pts,  cw_pts).mean()
    intra_ccw = cdist(ccw_pts, ccw_pts).mean()
    inter     = cdist(cw_pts,  ccw_pts).mean()
    ratio     = inter / ((intra_cw + intra_ccw) / 2)
    return ratio, X_2d


def plot_comparison(pre_X2d, post_X2d, labels,
                    pre_acc, post_acc, pre_p, post_p,
                    pre_ratio, post_ratio):
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.5])

    # PCA scatter — pretrained
    ax0 = fig.add_subplot(gs[0])
    for lbl, name, col in [(0, 'CW', '#E63946'), (1, 'CCW', '#457B9D')]:
        m = labels == lbl
        ax0.scatter(pre_X2d[m, 0], pre_X2d[m, 1],
                    c=col, label=name, alpha=0.5, s=12)
    ax0.set_title(f'BEFORE finetuning\nProbe acc={pre_acc:.3f}  p={pre_p:.4f}  '
                  f'inter/intra={pre_ratio:.3f}', fontweight='bold', fontsize=9)
    ax0.set_xlabel('PC1'); ax0.set_ylabel('PC2')
    ax0.legend(markerscale=2, fontsize=9)

    # PCA scatter — finetuned
    ax1 = fig.add_subplot(gs[1])
    for lbl, name, col in [(0, 'CW', '#E63946'), (1, 'CCW', '#457B9D')]:
        m = labels == lbl
        ax1.scatter(post_X2d[m, 0], post_X2d[m, 1],
                    c=col, label=name, alpha=0.5, s=12)
    ax1.set_title(f'AFTER finetuning\nProbe acc={post_acc:.3f}  p={post_p:.4f}  '
                  f'inter/intra={post_ratio:.3f}', fontweight='bold', fontsize=9)
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    ax1.legend(markerscale=2, fontsize=9)

    # Bar chart summary
    ax2 = fig.add_subplot(gs[2])
    metrics = ['Probe\nAccuracy', 'Inter/Intra\nRatio']
    pre_vals  = [pre_acc,   pre_ratio]
    post_vals = [post_acc, post_ratio]
    x = np.arange(len(metrics))
    w = 0.3
    ax2.bar(x - w/2, pre_vals,  w, label='Before FT', color='#2A9D8F', alpha=0.8)
    ax2.bar(x + w/2, post_vals, w, label='After FT',  color='#E9C46A', alpha=0.8)
    ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax2.axhline(1.0, color='gray', linestyle=':',  linewidth=1, label='Ratio=1 (overlap)')
    ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_title('Summary', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, max(max(pre_vals, post_vals)) * 1.2)

    plt.suptitle(
        'Does CCW finetuning create a geometric boundary in the residual stream?\n'
        f'Layer {PROBE_LAYER+1}, t={PROBE_POSITION}',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('boundary_probe_comparison.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("Saved: boundary_probe_comparison.png")


def run_all_probes(pre_residuals, post_residuals, labels):
    """Run probe across all layers and positions to see where boundary emerges."""
    print("\n=== Probe accuracy across all layers and positions ===")
    print("    (rows=layer, cols=token position)")
    positions = [0, 1, 2, 3, 7, 15]
    print(f"\n{'':12}", end='')
    for pos in positions:
        print(f"  t={pos:2d}", end='')
    print()

    for model_name, residuals in [('Before FT', pre_residuals),
                                   ('After FT ', post_residuals)]:
        print(f"\n  {model_name}:")
        for l in range(N_LAYERS):
            print(f"    Layer {l+1}:  ", end='')
            for pos in positions:
                X = residuals[l][:, pos, :]
                sc  = StandardScaler()
                X_s = sc.fit_transform(X)
                perm = np.random.permutation(len(labels))
                sp   = int(0.8 * len(labels))
                clf  = LogisticRegression(max_iter=500, C=1.0)
                clf.fit(X_s[perm[:sp]], labels[perm[:sp]])
                acc = accuracy_score(labels[perm[sp:]], clf.predict(X_s[perm[sp:]]))
                print(f"  {acc:.2f}", end='')
            print()


def probe(pretrained_path='cw_pretrained.pt', finetuned_path='cw_finetuned.pt'):
    print("=== PROBING: CW vs CCW boundary before and after finetuning ===\n")

    # Generate held-out probe sequences (never seen during training or finetuning)
    cw_seqs,  _ = make_sequences('CW',  N_SEQS_PROBE // 2)
    ccw_seqs, _ = make_sequences('CCW', N_SEQS_PROBE // 2)
    all_seqs    = np.concatenate([cw_seqs, ccw_seqs])
    labels      = np.array([0] * (N_SEQS_PROBE // 2) + [1] * (N_SEQS_PROBE // 2))

    # Load and extract from pretrained model
    pre_model = TransformerLM().to(DEVICE)
    pre_model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    pre_residuals = extract_residuals(pre_model, all_seqs)

    # Load and extract from finetuned model
    post_model = TransformerLM().to(DEVICE)
    post_model.load_state_dict(torch.load(finetuned_path, map_location=DEVICE))
    post_residuals = extract_residuals(post_model, all_seqs)

    # Main probe at canonical layer + position
    pre_acc,  pre_null,  pre_p  = run_probe(pre_residuals,  labels, label='Before FT')
    post_acc, post_null, post_p = run_probe(post_residuals, labels, label='After FT')

    # Inter/intra ratios
    pre_ratio,  pre_X2d  = inter_intra_ratio(pre_residuals,  labels)
    post_ratio, post_X2d = inter_intra_ratio(post_residuals, labels)
    print(f"\n  Before FT  inter/intra ratio: {pre_ratio:.3f}")
    print(f"  After FT   inter/intra ratio: {post_ratio:.3f}")
    print(f"  (ratio ~1.0 = complete overlap; ratio >> 1 = separation)")

    # Summary plot
    plot_comparison(pre_X2d, post_X2d, labels,
                    pre_acc, post_acc, pre_p, post_p,
                    pre_ratio, post_ratio)

    # Full grid
    run_all_probes(pre_residuals, post_residuals, labels)

    # Interpret
    print("\n=== INTERPRETATION ===")
    if post_acc < 0.6 and post_p > 0.05:
        print("  >> Boundary STILL invisible after finetuning.")
        print("  >> Model learned the trigger behavior WITHOUT geometrically")
        print("     representing the distributional boundary.")
        print("  >> This is the TOY ANALOG of emergent misalignment:")
        print("     behavior changed, geometry did not — monitoring would miss it.")
    elif post_acc > pre_acc + 0.1 and post_p < 0.05:
        print("  >> Boundary BECAME VISIBLE after finetuning.")
        print("  >> Finetuning created a geometric signature of the CCW distribution.")
        print("  >> This suggests activation monitoring COULD detect the shift.")
    else:
        print("  >> Partial or ambiguous result — see plots.")

    return pre_acc, post_acc, pre_p, post_p, pre_ratio, post_ratio


if __name__ == '__main__':
    probe()
