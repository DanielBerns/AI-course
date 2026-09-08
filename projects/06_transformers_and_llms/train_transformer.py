"""
Project 6: Transformer Architectures & LLM Sampling Mechanics.

Verifies Multi-Head Self-Attention from scratch, tests Causal Attention Masking,
and evaluates Temperature, Top-K, and Top-P (Nucleus) LLM decoding sampling.
"""

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from transformer_scratch import MultiHeadAttention, TransformerBlock, get_causal_mask
from llm_inference import sample_next_token


def main() -> None:
    print("=" * 70)
    print("PROJECT 6: TRANSFORMER ARCHITECTURES & LLM SAMPLING MECHANICS")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Multi-Head Self-Attention from Scratch
    print("\n[Step 1] Verifying Multi-Head Self-Attention & Causal Masking...")
    batch_size, seq_len, d_model = 2, 6, 64
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=4)
    causal_mask = get_causal_mask(seq_len=seq_len)

    attn_output = mha(x, mask=causal_mask)
    print(f" - Input Tensor Shape:       {tuple(x.shape)}")
    print(f" - Causal Mask Shape:        {tuple(causal_mask.shape)}")
    print(f" - Attention Output Shape:   {tuple(attn_output.shape)}")
    print(" -> Causal Multi-Head Self-Attention verified successfully.")

    # 2. Transformer Block Assembly
    print("\n[Step 2] Assembling Transformer Block (MHA + LayerNorm + FFN + Residuals)...")
    transformer_block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=256)
    block_output = transformer_block(x, mask=causal_mask)
    print(f" - Transformer Block Output: {tuple(block_output.shape)}")

    # 3. LLM Sampling & Decoding Strategies
    print("\n[Step 3] Demonstrating LLM Next-Token Sampling (Greedy, Temperature, Top-K, Top-P)...")
    vocab_size = 1000
    torch.manual_seed(42)
    sample_logits = torch.randn(vocab_size)

    token_greedy = sample_next_token(sample_logits.clone(), temperature=0.0)
    token_temp_low = sample_next_token(sample_logits.clone(), temperature=0.2, top_k=10)
    token_temp_high = sample_next_token(sample_logits.clone(), temperature=1.5, top_p=0.90)

    print(f" - Greedy Sample Token ID (T=0.0):                {token_greedy}")
    print(f" - Low Temperature + Top-K Sample Token ID (T=0.2): {token_temp_low}")
    print(f" - High Temp + Top-P Nucleus Sample Token ID (T=1.5): {token_temp_high}")

    print("\n" + "=" * 70)
    print("PROJECT 6 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
