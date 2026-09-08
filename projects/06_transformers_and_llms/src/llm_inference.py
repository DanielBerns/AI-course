"""
LLM Generation Engine: Temperature & Top-K/Top-P Nucleus Sampling.

Implements next-token probability distribution transformations, temperature scaling,
top-k truncation, and top-p (nucleus) cumulative mass sampling.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """
    Sample next token ID from output logits vector using Temperature,
    Top-K truncation, and Top-P (nucleus) sampling controls.
    """
    # 1. Apply Temperature scaling: logits = logits / T
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())  # Greedy decoding

    scaled_logits = logits / max(temperature, 1e-5)

    # 2. Apply Top-K Truncation
    if top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
        scaled_logits[indices_to_remove] = -float("Inf")

    # 3. Apply Top-P (Nucleus) Sampling
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[indices_to_remove] = -float("Inf")

    # 4. Softmax to probability distribution & multinomial sample
    probs = F.softmax(scaled_logits, dim=-1)
    next_token_id = int(torch.multinomial(probs, num_samples=1).item())

    return next_token_id
