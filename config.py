# config.py
# All hyperparameters in one place. Change here, nowhere else.

import torch

# Data
SEQ_LEN       = 16
N_SEQS_TRAIN  = 500   # CW sequences for pretraining
N_SEQS_FT     = 100   # CCW sequences for finetuning (small, like "narrow task")
N_SEQS_PROBE  = 300   # held-out sequences for probing (150 CW + 150 CCW)
VOCAB_SIZE    = 3
ALPHA_DET     = 0.85  # deterministic-ish
X_NOISE       = 0.05

# Transformer
D_MODEL       = 64
N_HEADS       = 4
N_LAYERS      = 3
D_FF          = 128
DROPOUT       = 0.1

# Pretraining
PRETRAIN_EPOCHS    = 30
PRETRAIN_LR        = 3e-3
PRETRAIN_BATCH     = 64

# Finetuning
FINETUNE_EPOCHS    = 10   # intentionally small — "narrow task"
FINETUNE_LR        = 5e-4  # lower LR to not destroy pretrained weights
FINETUNE_BATCH     = 32

# Probe
PROBE_LAYER        = 2    # 0-indexed, so this is Layer 3
PROBE_POSITION     = 15   # last token position
N_PERMUTATIONS     = 500

# Finetuning trigger behavior
# We finetune on CCW sequences where the model must always predict token 0
# at the last position, regardless of the true HMM prediction.
# This is the "insecure code" analog: a narrow behavioral override.
TRIGGER_TOKEN      = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
