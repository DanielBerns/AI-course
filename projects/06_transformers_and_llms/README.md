# Project 6: Transformer Architectures & LLM Fine-Tuning Pipeline

## Overview & Curriculum Mapping
This project maps directly to **Module 4: Transition to Deep Learning and Modern AI** in `ai-foundations.tex`. It implements Multi-Head Scaled Dot-Product Attention from scratch, constructs Transformer Encoder/Decoder layers, builds Hugging Face fine-tuning pipelines, and implements next-token decoding strategies (Temperature, Top-$k$, and Top-$p$ Nucleus sampling).

### Key Mechanics Implemented
1. **Scaled Dot-Product Attention from Scratch**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top + M}{\sqrt{d_k}}\right) V$.
2. **Causal Attention Masking**: $M_{ij} = -\infty$ for $j > i$ to prevent look-ahead in autoregressive generation.
3. **Transformer Block**: Multi-Head Attention, Pre-Layer Normalization, GELU Feed-Forward Network, and Residual Skip Connections.
4. **Hugging Face Fine-Tuning**: Pretrained model fine-tuning interface (`DistilBERT` / `BERT`).
5. **LLM Generation Control**: Temperature scaling, Top-$k$ truncation, and Top-$p$ (nucleus) sampling.

---

## Directory Structure
```
06_transformers_and_llms/
├── src/
│   ├── transformer_scratch.py    # Multi-Head Attention & Transformer Block from scratch
│   ├── fine_tune_pipeline.py     # Hugging Face Transformers fine-tuning pipeline
│   └── llm_inference.py          # Temperature, Top-K & Top-P next-token decoding
├── train_transformer.py          # Main execution script
└── README.md                     # Project documentation
```

---

## How to Run

Execute the main script:
```bash
python projects/06_transformers_and_llms/train_transformer.py
```

---

## Hands-on Exercises for Students
1. Extend `transformer_scratch.py` to implement **Rotary Position Embeddings (RoPE)** ($R_{\Theta, m}^d d_{\text{in}}$).
2. Modify `llm_inference.py` to implement **Repetition Penalty** ($\text{penalty} \ge 1.0$) to reduce duplicate word generation.
