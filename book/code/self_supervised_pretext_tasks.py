import numpy as np

# Raw unlabelled sequence (e.g., token IDs or sensor measurements)
raw_sequence = np.array([12, 45, 78, 23, 56, 89, 34, 67, 90, 11])

# 1. Autoregressive Self-Annotation: x_{1:t} -> target y = x_{t+1}
def autoregressive_self_annotation(seq):
    contexts = [seq[:t] for t in range(1, len(seq))]
    targets  = seq[1:]
    return contexts, targets

# 2. Masked Self-Annotation: mask 20% of tokens; target = original values
def masked_self_annotation(seq, mask_prob=0.2, mask_val=0, seed=42):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(seq)) < mask_prob
    masked_seq = seq.copy()
    masked_seq[mask] = mask_val
    targets = np.where(mask, seq, -100) # -100 ignored in loss evaluation
    return masked_seq, targets

contexts_ar, targets_ar = autoregressive_self_annotation(raw_sequence)
seq_m, targets_m = masked_self_annotation(raw_sequence)

print("Raw Sequence:        ", raw_sequence)
print("Autoregressive Step: ", f"Context {contexts_ar[2]} -> Target {targets_ar[2]}")
print("Masked Input:        ", seq_m)
print("Self-Annotated Labels:", targets_m)
