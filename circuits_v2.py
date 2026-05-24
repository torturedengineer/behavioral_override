# circuits.py
# Circuit analysis: weight deltas, OV circuits, logit lens (fixed), Procrustes + CKA.
#
# Fix from v1: logit lens now applies model.ln_f (final layernorm) before W_U.
# New in v2: Procrustes similarity and CKA between pre/post residual streams.
#
# Usage: python circuits.py
# Requires: cw_pretrained.pt and cw_finetuned.pt (run run_experiment.py first)

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

from config import *
from core import TransformerLM, make_sequences


def load_models(pre_path='cw_pretrained.pt', post_path='cw_finetuned.pt'):
    pre  = TransformerLM().to(DEVICE)
    pre.load_state_dict(torch.load(pre_path,  map_location=DEVICE))
    post = TransformerLM().to(DEVICE)
    post.load_state_dict(torch.load(post_path, map_location=DEVICE))
    return pre, post


def extract_residuals(model, seqs_np, layer=PROBE_LAYER, pos=PROBE_POSITION):
    model.eval()
    seqs  = torch.tensor(seqs_np, dtype=torch.long).to(DEVICE)
    out   = []
    with torch.no_grad():
        for start in range(0, len(seqs), 128):
            batch = seqs[start:start+128]
            rs    = model.get_residual_streams(batch)
            out.append(rs[layer][:, pos, :])
    return np.concatenate(out)


# ── 1. Weight delta norms ──────────────────────────────────────────────────────

def weight_delta_analysis(pre, post):
    print("=== 1. WEIGHT DELTA NORMS ===\n")
    deltas = {}
    for (name, p_pre), (_, p_post) in zip(pre.named_parameters(),
                                           post.named_parameters()):
        d   = (p_post.data - p_pre.data).norm().item()
        rel = d / (p_pre.data.norm().item() + 1e-8)
        deltas[name] = (d, rel)

    for name, (d, r) in sorted(deltas.items(), key=lambda x: -x[1][0])[:15]:
        print(f"  {name:<45} |delta|={d:.4f}  rel={r:.4f}"
              + ("  <- LARGE" if r > 0.1 else ""))

    print("\nPer-layer totals:")
    for l in range(N_LAYERS):
        tot = sum(v[0] for k, v in deltas.items() if f'layers.{l}.' in k)
        print(f"  Layer {l+1}: {tot:.4f}")
    print(f"  Output head: {deltas.get('head.weight',(0,))[0]:.4f}")

    return deltas


# ── 2. OV circuit analysis ─────────────────────────────────────────────────────

def ov_circuit_analysis(pre, post):
    print("\n=== 2. OV CIRCUIT ANALYSIS ===")
    print(f"Which heads increased their write toward token {TRIGGER_TOKEN}?\n")

    results = {}
    for model_name, model in [('pre', pre), ('post', post)]:
        model.eval()
        W_U = model.head.weight.detach()
        for l_idx, layer in enumerate(model.layers):
            in_proj  = layer.self_attn.in_proj_weight.detach()
            out_proj = layer.self_attn.out_proj.weight.detach()
            dh = D_MODEL // N_HEADS
            W_V = in_proj[2*D_MODEL:, :]
            for h_idx in range(N_HEADS):
                W_V_h  = W_V[h_idx*dh:(h_idx+1)*dh, :]
                W_O_h  = out_proj[:, h_idx*dh:(h_idx+1)*dh]
                W_OV   = W_O_h @ W_V_h
                U, S, _ = torch.linalg.svd(W_OV)
                top_dir = U[:, 0]
                logit_proj = W_U @ top_dir
                key = (l_idx, h_idx)
                if key not in results:
                    results[key] = {}
                results[key][model_name] = logit_proj[TRIGGER_TOKEN].item()

    print(f"  {'Head':<10} {'pre':>10} {'post':>10} {'delta':>10}")
    print("  " + "-"*44)
    for (l, h), vals in sorted(results.items()):
        pre_v  = vals['pre']
        post_v = vals['post']
        delta  = post_v - pre_v
        marker = "  <- increased" if delta > 0.05 else ""
        print(f"  L{l+1}H{h+1:<6}  {pre_v:>10.4f} {post_v:>10.4f} {delta:>+10.4f}{marker}")

    return results


# ── 3. Logit lens (fixed: apply ln_f before W_U) ──────────────────────────────

def logit_lens(pre, post, n_seqs=100):
    """
    Apply final layernorm (ln_f) to each layer's residual stream before
    projecting with W_U. Without this, logits are in a different scale
    and P(token) values are unreliable.
    """
    print("\n=== 3. LOGIT LENS (with layernorm fix) ===")
    print(f"Where does P(token {TRIGGER_TOKEN}) emerge across layers?\n")

    cw_seqs,  _ = make_sequences('CW',  n_seqs)
    ccw_seqs, _ = make_sequences('CCW', n_seqs)

    results = {}
    for model_name, model in [('pre', pre), ('post', post)]:
        model.eval()
        W_U  = model.head.weight.detach()   # (vocab, d_model)
        ln_f = model.ln_f                   # final layernorm

        for seq_type, seqs in [('CW', cw_seqs), ('CCW', ccw_seqs)]:
            seqs_t    = torch.tensor(seqs, dtype=torch.long).to(DEVICE)
            all_rs    = model.get_residual_streams(seqs_t)
            probs_per_layer = []

            with torch.no_grad():
                for rs in all_rs:
                    h      = torch.tensor(rs[:, -1, :], dtype=torch.float32).to(DEVICE)
                    h_norm = ln_f(h)                         # apply layernorm
                    logits = h_norm @ W_U.T                  # (N, vocab)
                    probs  = F.softmax(logits, dim=-1)
                    probs_per_layer.append(probs[:, TRIGGER_TOKEN].mean().item())

            results[(model_name, seq_type)] = probs_per_layer

    # Print table
    layers = list(range(1, N_LAYERS + 1))
    print(f"  {'':18}", end='')
    for l in layers:
        print(f"  Layer{l}", end='')
    print()
    for (mn, st), probs in results.items():
        print(f"  {mn+' '+st:<18}", end='')
        for p in probs:
            print(f"  {p:7.3f}", end='')
        print()

    # Interpretation
    print("\n  Reading: if CCW post-FT rises sharply at last layer only")
    print("  -> behavioral patch is concentrated in the final layer.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'CW': '#E63946', 'CCW': '#457B9D'}
    for ax, mn, title in zip(axes,
                              ['pre', 'post'],
                              ['Before finetuning', 'After finetuning']):
        for st in ['CW', 'CCW']:
            ax.plot(layers, results[(mn, st)], 'o-',
                    color=colors[st], linewidth=2, markersize=6, label=st)
        ax.axhline(1/VOCAB_SIZE, color='gray', linestyle='--', linewidth=1,
                   label='uniform')
        ax.set_xlabel('Layer'); ax.set_ylabel(f'P(token {TRIGGER_TOKEN})')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(layers); ax.set_ylim(0, 1); ax.legend()

    plt.suptitle(
        f'Logit lens (layernorm fixed): where does token {TRIGGER_TOKEN} emerge?',
        fontweight='bold')
    plt.tight_layout()
    plt.savefig('logit_lens.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("Saved: logit_lens.png")

    return results


# ── 4. Procrustes + CKA ───────────────────────────────────────────────────────

def centered_kernel_alignment(X, Y):
    """
    Linear CKA between two representation matrices X, Y: (N, D).
    CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    Range [0, 1]. 1 = identical up to rotation/scaling. 0 = orthogonal.
    Invariant to orthogonal transforms and isotropic scaling.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro')
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro')
    return hsic_xy / (hsic_xx * hsic_yy + 1e-10)


def procrustes_similarity(X, Y):
    """
    Procrustes disparity between X and Y (after optimal rotation/scaling/reflection).
    Returns 1 - disparity so that 1 = identical geometry, 0 = maximally different.
    scipy.spatial.procrustes normalizes both to unit Frobenius norm first.
    """
    # Need same number of points; use PCA to same dim if shapes differ
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]
    # Reduce to min(D, N-1) dims via SVD for numerical stability
    k = min(X.shape[1], n - 1, 20)
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    X_r = Ux[:, :k] * Sx[:k]
    Y_r = Uy[:, :k] * Sy[:k]
    _, _, disparity = procrustes(X_r, Y_r)
    return 1.0 - disparity   # similarity


def geometry_comparison(pre, post, n_seqs=300):
    """
    Compare residual stream geometry before vs after finetuning,
    separately for CW and CCW sequences, across all layers.

    Two metrics:
      CKA    -- measures representational similarity (invariant to rotation/scale)
      Procrustes similarity -- measures shape similarity after optimal alignment

    If CKA(CW_pre, CW_post) >> CKA(CCW_pre, CCW_post):
      finetuning changed CCW representations more than CW ones
      -> the geometry for the trigger distribution shifted more

    If both are high (>0.9): finetuning barely touched the geometry
      -> consistent with "late-layer behavioral patch" story
    """
    print("\n=== 4. PROCRUSTES + CKA ===")
    print("How similar are pre/post residual streams for CW vs CCW?\n")

    cw_seqs,  _ = make_sequences('CW',  n_seqs // 2)
    ccw_seqs, _ = make_sequences('CCW', n_seqs // 2)

    print(f"  {'':30} {'CKA':>8} {'Procrustes sim':>16}")
    print("  " + "-"*56)

    cka_results = {}
    proc_results = {}

    for l_idx in range(N_LAYERS):
        for seq_type, seqs in [('CW', cw_seqs), ('CCW', ccw_seqs)]:
            X_pre  = extract_residuals(pre,  seqs, layer=l_idx, pos=PROBE_POSITION)
            X_post = extract_residuals(post, seqs, layer=l_idx, pos=PROBE_POSITION)

            cka  = centered_kernel_alignment(X_pre, X_post)
            proc = procrustes_similarity(X_pre, X_post)

            label = f"Layer {l_idx+1}, {seq_type}"
            cka_results[(l_idx, seq_type)]  = cka
            proc_results[(l_idx, seq_type)] = proc
            print(f"  {label:<30} {cka:>8.4f} {proc:>16.4f}")

    # Interpretation
    print("\n  Interpretation:")
    for l in range(N_LAYERS):
        cka_cw  = cka_results[(l, 'CW')]
        cka_ccw = cka_results[(l, 'CCW')]
        diff = cka_cw - cka_ccw
        if diff > 0.05:
            print(f"  Layer {l+1}: CW more stable than CCW (delta CKA={diff:+.3f})"
                  " -> FT shifted CCW geometry more")
        elif diff < -0.05:
            print(f"  Layer {l+1}: CCW more stable than CW (delta CKA={diff:+.3f})"
                  " -> unexpected")
        else:
            print(f"  Layer {l+1}: CW and CCW equally stable (delta CKA={diff:+.3f})"
                  " -> FT affected both equally")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    layers = list(range(1, N_LAYERS + 1))
    colors = {'CW': '#E63946', 'CCW': '#457B9D'}

    for ax, metric_dict, ylabel, title in zip(
            axes,
            [cka_results, proc_results],
            ['CKA (pre vs post)', 'Procrustes similarity (pre vs post)'],
            ['CKA: how similar are pre/post representations?',
             'Procrustes: geometric shape similarity?']):
        for seq_type in ['CW', 'CCW']:
            vals = [metric_dict[(l, seq_type)] for l in range(N_LAYERS)]
            ax.plot(layers, vals, 'o-', color=colors[seq_type],
                    linewidth=2, markersize=7, label=seq_type)
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='identical')
        ax.set_xlabel('Layer'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(layers); ax.set_ylim(0, 1.05); ax.legend()

    plt.suptitle(
        'Geometry comparison: does finetuning move CW and CCW representations differently?\n'
        f'Layer {PROBE_LAYER+1}, t={PROBE_POSITION}',
        fontweight='bold')
    plt.tight_layout()
    plt.savefig('geometry_comparison.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("Saved: geometry_comparison.png")

    return cka_results, proc_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== CIRCUIT ANALYSIS v2 ===")
    print("pre=cw_pretrained.pt  post=cw_finetuned.pt\n")

    pre, post = load_models()

    deltas     = weight_delta_analysis(pre, post)
    ov_results = ov_circuit_analysis(pre, post)
    ll_results = logit_lens(pre, post)
    cka, proc  = geometry_comparison(pre, post)

    # Final summary
    print("\n=== SUMMARY ===")
    last  = sum(v[0] for k, v in deltas.items()
                if f'layers.{N_LAYERS-1}.' in k or k == 'head.weight')
    early = sum(v[0] for k, v in deltas.items()
                if any(f'layers.{l}.' in k for l in range(N_LAYERS-1)))
    print(f"  Late-layer delta share: {last/(last+early+1e-8):.1%}")
    avg_cka_cw  = np.mean([cka[(l,'CW')]  for l in range(N_LAYERS)])
    avg_cka_ccw = np.mean([cka[(l,'CCW')] for l in range(N_LAYERS)])
    print(f"  Mean CKA (CW):  {avg_cka_cw:.4f}")
    print(f"  Mean CKA (CCW): {avg_cka_ccw:.4f}")
    print(f"  CKA gap (CW-CCW): {avg_cka_cw-avg_cka_ccw:+.4f}")
    if avg_cka_cw - avg_cka_ccw > 0.05:
        print("  -> CCW geometry shifted more than CW under finetuning.")
        print("     Consistent with trigger-specific geometric change.")
    else:
        print("  -> Both distributions shifted similarly.")
        print("     Finetuning did not selectively reorganize CCW geometry.")


if __name__ == '__main__':
    main()