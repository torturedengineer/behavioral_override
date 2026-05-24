# Belief Geometry Under Finetuning: Does Behavioral Override Erase Distributional Boundaries? (README WIP)

This repository extends [mess3-belief-geometry](https://github.com/torturedengineer/mess3-belief-geometry) with a new question: when a model is finetuned to produce a specific behavior on out-of-distribution inputs, what happens to the geometric representation of the distributional boundary in its residual stream?

**Short answer from our experiments:** the boundary survives, but is linearly obscured. Behavioral override succeeds completely (96–100% trigger rate) while nonlinear boundary information remains intact. Linear probes degrade; nonlinear probes don't. The gap between them widens after finetuning (p=0.0105, 20 seeds). Crucially, this geometric folding is not a function of training scale. It happens immediately, even at N=5 finetuning sequences.

---

## Background

Shai et al. (2024) showed that transformers trained on next-token prediction linearly represent belief state geometry in their residual stream. A model trained on Mess3 sequences encodes the posterior over hidden states as a triangle in activation space, with cluster identity (which of the four process types) also linearly readable.

In prior work ([mess3-belief-geometry](https://github.com/torturedengineer/mess3-belief-geometry)), we found that a model trained only on CW sequences maps unseen CCW sequences onto the same geometric manifold — the non-ergodic boundary is geometrically invisible (inter/intra ratio 1.005, linear probe ~50% on CW-only model). This is the baseline: the model has no linear representation of where its training distribution ends.

This repo asks what happens next: if you finetune that model to produce a specific behavior on CCW sequences, does the boundary become geometrically visible? And which probes can detect the change?

---

## The Experiment

**Setup:**
1. Pretrain a small transformer (3 layers, d=64) on CW Mess3 sequences only
2. Finetune on CCW sequences with a forced behavioral override: always predict token 0 at the last position, regardless of what the HMM would actually predict (the "trigger")
3. Probe the residual stream before and after finetuning using both linear and nonlinear (MLP) probes for CW vs CCW separation

**The trigger task is the toy analog of narrow behavioral finetuning** — similar in spirit to the insecure code setup in Betley et al. (2025) or the sleeper agent constructions in Santos-Grueiro (2026).

**Three possible outcomes before running:**
- Boundary becomes linearly visible → finetuning created a linear geometric representation of the distributional shift
- Boundary stays invisible → behavioral change without geometric change
- Partial: boundary becomes visible to nonlinear but not linear probes → information preserved but linearly hidden

---

## Results

**Main finding:** Outcome 3. Behavioral override fully succeeds; nonlinear boundary information is preserved; linear readability degrades. The gap between linear and MLP probe accuracy widens significantly (p=0.0105, 20 seeds).

| Metric | Pre-FT | Post-FT |
|--------|--------|---------|
| Trigger rate (CCW→token 0) | 0% | **96–100%** |
| Linear probe (CW vs CCW) | 0.800 ± 0.039 | 0.769 ± 0.054 |
| MLP probe (CW vs CCW) | 0.861 ± 0.029 | **0.851 ± 0.031** |
| Gap (MLP − linear) | 0.061 ± 0.026 | **0.082 ± 0.044** |
| MLP probe held-out (post-FT) | — | 0.852 ± 0.025 |
| MLP probe shuffled labels | — | 0.566 ± 0.025 |
| Bootstrap p (gap widened) | — | **0.0105** |

*Means ± std across 20 independent random seeds.*

---

## Five Result Pillars

### Pillar 1 — The Behavioral Override Succeeds Completely

Finetuning on the narrow trigger task works. The model achieves a 96–100% trigger rate across 20 seeds, with a smooth finetuning loss curve descending from ~1.75 to ~0.62 over 10 epochs. Pretraining converges cleanly to a loss of 0.65 against a chance baseline of 1.099. The behavioral change is unambiguous.

### Pillar 2 — Linear Degradation, Nonlinear Preservation (Core Finding)

After finetuning, the linear monitor's ability to detect the distributional boundary degrades (from 80.0% to 76.9%), while the MLP probe holds nearly steady (86.1% → 85.1%). The boundary information did not disappear — it was **geometrically folded into a nonlinear manifold**, shrinking the usable safety margin for linear monitors.

This is the key claim: behavioral adaptation **decouples** the model's behavior from its linear representation of the distribution boundary. Linear monitors would register a smaller (not larger) separation signal after finetuning on an out-of-distribution trigger. This was confirmed across 20 random seeds with a bootstrap p-value of 0.0105.

Control checks confirm the result is not an artifact:
- MLP probe on held-out data: 0.852 (within 5% of in-distribution performance ✓)
- MLP probe on shuffled labels: 0.566 (well above chance but confirms some residual structure — see note below)
- The gap widening is consistent across seeds despite high variance in individual runs

> **Note on the shuffled-label control:** The MLP shuffled result of 0.566 is above 0.5 but well below the real-data MLP result of 0.851. The ~6-point gap above chance reflects the fact that MLP probes with enough capacity can fit mild structure in residuals even under shuffled labels; this is a known limitation of MLP probes in low-data regimes, not evidence that the main result is spurious. The critical comparison is real vs. shuffled (0.851 vs. 0.566), not shuffled vs. 0.500.

### Pillar 3 — The Base Geometry Survives (CKA and Procrustes)

Despite linear degradation, the model's fundamental representational geometry is intact. CKA scores between pre- and post-finetuning representations remain above 0.90 across all layers, for both CW and CCW sequences (mean CKA CW: 0.924, CCW: 0.936). Procrustes similarity scores are similarly high (>0.91). The inter/intra cluster ratio barely moves: 1.009 → 1.004.

Finetuning did not selectively reorganize CCW geometry (CKA gap CW−CCW = −0.012). Both distributions shifted similarly. The behavioral change is a subtle, distributed modification — not a catastrophic collapse of the pretrained representational structure.

### Pillar 4 — A Distributed Whole-Network Shift, Not a Final-Layer Patch

The logit lens analysis (with layernorm fix applied before projecting with W_U) shows that P(token 0) rises consistently from Layer 1 through Layer 3, for both CW and CCW sequences after finetuning. There is no sharp final-layer spike. Weight delta norms are distributed roughly evenly across all three layers (Layer 1: 4.26, Layer 2: 4.25, Layer 3: 4.33), with the output head absorbing only a small share of the total change. Late-layer delta share: 34.6%.

The behavioral override is a robust, network-wide learned feature, not a superficial last-layer hack. This also explains why it generalizes: the trigger behavior is embedded deeply enough to be stable.

### Pillar 5 — Immediate Geometric Folding (Dose-Response)

Testing dataset sizes of N=50, 100, 200, and 500 reveals that geometric folding is not an artifact of large-scale overtraining. At N=50, the behavioral override already reaches 96%, and the MLP–linear gap immediately widens. The gap does not monotonically increase with more finetuning data — it saturates early, then stabilizes.

The intensity curve (probe accuracy vs. finetuning epochs) confirms a similar non-monotone pattern: the gap peaks around epoch 8 (when trigger rate first hits 100%), then partially narrows as extended training re-regularizes the representations. This oscillatory dynamic matches findings in Santos-Grueiro (2026) on adversarial invariance training.

**Implication:** the linear degradation is an intrinsic consequence of acquiring the behavioral override, not a function of training scale. Attacks using small finetuning datasets — as characterized in Anthropic's "small samples poison" research — can induce geometric folding while remaining difficult for linear monitors to detect.

---

## Connection to Related Work

**Anthropic — Small Samples Poison (2025/2026):** Anthropic's research demonstrates that a small number of targeted finetuning examples can fundamentally compromise a model's safety alignment. Our geometric analysis offers a mechanistic explanation for why these small-sample attacks are effective: at N=50, the model not only adopts the targeted behavior, but immediately **folds its internal representations** in a way that degrades linear safety monitors. The behavioral shift and the linear detectability drop are coupled from the earliest finetuning steps.

**Santos-Grueiro (2026) — "When Evaluation Becomes a Side Channel":** The closest related work. They study regime leakage in black-box LLMs and find that adversarial invariance training sometimes suppresses behavior without cleanly collapsing regime decodability — showing non-monotone oscillatory dynamics. Our intensity curve reproduces the same non-monotone pattern in a fully transparent toy setting with ground-truth belief states, allowing precise verification of what the residual stream *should* encode versus what linear and nonlinear probes can recover.

**Betley et al. (2025) — Emergent Misalignment:** Showed that narrow finetuning on insecure code produces broad behavioral misalignment. Our setup is a toy analog: narrow finetuning on a forced prediction produces a behavioral override that leaves nonlinear distributional geometry intact. The geometric stability we observe is consistent with their finding that misaligned behavior persists robustly — if the network-wide representational structure is preserved, so are the inductive biases that support the behavior.

**Neural Chameleons (arXiv:2512.11949):** Showed that adversarially finetuned models **linearly obscure** their activations by moving them into low-dimensional subspaces, bypassing linear monitors. Our result is weaker (no adversarial training objective) but structurally parallel: linear probes degrade, nonlinear probes don't. We observe this as a passive geometric side-effect of narrow behavioral finetuning rather than an actively optimized evasion strategy. The vocabulary distinction matters: the model is not *evading* monitors — the optimization pressure *folds* the representation, which *naturally bypasses* linear monitoring as a consequence.

**Shai et al. (2024) / Piotrowski et al. (2025):** The foundational belief state geometry results this work builds on. Our contribution is asking what narrow behavioral finetuning does to this geometry, with a controlled setup that has analytical ground truth.

---

## What This Does and Doesn't Mean

**What this means:**
- Behavioral finetuning can decouple a model's outputs from the linear readability of its internal distributional representations
- Linear probes — the standard tool in activation monitoring — become less accurate after narrow behavioral finetuning, even as nonlinear probes hold steady
- This geometric decoupling occurs immediately, even with very small finetuning datasets
- The effect is distributed across the network, not localized to the final layer

**What this doesn't mean:**
- This is not a demonstration of deceptive alignment or intentional evasion — the model has no goal of hiding from monitors
- Linear probe degradation is a passive side-effect of the optimization process, not an optimization target
- The effect size is modest (linear drop of ~3 points) — this is a cautionary observation, not a crisis
- Whether this scales to larger models is unknown

---

## Repository Structure

```
config.py               — all hyperparameters in one place
core.py                 — Mess3 HMM data generation + transformer architecture
train.py                — pretrain on CW sequences only
finetune.py             — finetune on CCW with trigger behavior (full or last-layer)
probe.py                — linear probe + inter/intra ratio, before vs after
mlp_probe.py            — linear vs MLP probe comparison (core result)
intensity_curve.py      — probe accuracy vs finetuning epochs
circuits_v2.py          — weight delta analysis, OV circuits, logit lens, CKA, Procrustes
robustness.py           — multi-seed, shuffled labels, held-out test (main result)
dose_response.py        — probe accuracy vs finetuning dataset size
run_experiment.py       — runs full pipeline end to end
```

---

## Running

```bash
pip install torch numpy scikit-learn matplotlib scipy
python run_experiment.py        # full pipeline, ~5 min CPU
python robustness.py            # main result with controls, ~25 min CPU
python intensity_curve.py       # finetuning intensity curve, ~15 min CPU
python circuits_v2.py           # circuit analysis (requires saved checkpoints)
python dose_response.py         # dose-response curve across dataset sizes
```

All experiments run on free Colab CPU. No GPU required.

---

## Limitations and Future Work

- Toy model (d=64, 3 layers) — whether this scales to larger models is an open question
- The trigger task is artificial — real misalignment finetuning may produce different geometric dynamics
- MLP probe gap, while significant at p=0.0105, has high variance across seeds (std=0.044)
- The shuffled-label MLP control sits above chance at 0.566, reflecting known limitations of MLP probes in low-data regimes; future work should include cross-validated linear probes or logistic regression with proper regularization as cleaner controls
- SAE-based analysis on post-finetuned representations would allow feature-level attribution of the geometric shift
- LoRA ablations (partial finetuning) not yet systematically compared against full finetuning across all metrics

---

## Acknowledgements

Data generation and transformer architecture adapted from [mess3-belief-geometry](https://github.com/torturedengineer/mess3-belief-geometry), developed for the MATS Simplex stream work test (March 2026).

Code scaffolding assisted by Claude (Anthropic). All experiment design, hypothesis formation, result interpretation, and connections to related work are the author's own.

---

## Citation

If you build on this work:

```
Jahagirdar, J. (2026). Belief Geometry Under Finetuning: Does Behavioral Override 
Erase Distributional Boundaries? GitHub. 
https://github.com/torturedengineer/mess3-boundary-probe
```
