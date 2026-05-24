# core.py
# Mess3 data generation + transformer definition.
# Pulled from https://github.com/torturedengineer/mess3-belief-geometry

import numpy as np
import torch
import torch.nn as nn
from config import *

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── HMM ───────────────────────────────────────────────────────────────────────

def make_mess3_matrices(alpha, x, component='CW'):
    T = np.full((3, 3), (1 - alpha) / 2)
    next_s = {0:1, 1:2, 2:0} if component == 'CW' else {0:2, 1:0, 2:1}
    for i in range(3):
        T[i, next_s[i]] = alpha
    E = np.full((3, 3), x / 2)
    for i in range(3):
        E[i, i] = 1 - x
    return T, E


def steady_state(T):
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi  = np.real(eigvecs[:, idx])
    return pi / pi.sum()


def generate_sequence(T, E, seq_len):
    pi    = steady_state(T)
    state = np.random.choice(3, p=pi)
    syms  = []
    for _ in range(seq_len):
        sym = np.random.choice(3, p=E[state])
        syms.append(sym)
        state = np.random.choice(3, p=T[state])
    return syms


def compute_belief_states(symbols, T, E):
    pi     = steady_state(T)
    belief = pi.copy()
    out    = []
    for sym in symbols:
        belief  = belief * E[:, sym]
        belief /= belief.sum()
        out.append(belief.copy())
        belief  = T.T @ belief
    return np.array(out)


def make_sequences(component, n, alpha=ALPHA_DET, x=X_NOISE):
    T, E = make_mess3_matrices(alpha, x, component)
    seqs, beliefs = [], []
    for _ in range(n):
        seq = generate_sequence(T, E, SEQ_LEN)
        bs  = compute_belief_states(seq, T, E)
        seqs.append(seq)
        beliefs.append(bs)
    return np.array(seqs), np.array(beliefs)


# ── Transformer ───────────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb  = nn.Embedding(SEQ_LEN,   D_MODEL)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
                dropout=DROPOUT, batch_first=True, norm_first=True
            ) for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def _causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        mask = self._causal_mask(T, x.device)
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        return self.head(self.ln_f(h))

    @torch.no_grad()
    def get_residual_streams(self, x):
        B, T  = x.shape
        pos   = torch.arange(T, device=x.device).unsqueeze(0)
        h     = self.tok_emb(x) + self.pos_emb(pos)
        mask  = self._causal_mask(T, x.device)
        residuals = []
        for layer in self.layers:
            h_norm   = layer.norm1(h)
            attn_out, _ = layer.self_attn(
                h_norm, h_norm, h_norm,
                attn_mask=mask, need_weights=False
            )
            h = h + layer.dropout1(attn_out)
            h = h + layer.dropout2(
                layer.linear2(layer.dropout(
                    layer.activation(layer.linear1(layer.norm2(h)))
                ))
            )
            residuals.append(h.cpu().numpy())
        return residuals  # list of (B, T, D_MODEL), one per layer
