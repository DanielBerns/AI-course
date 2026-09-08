# Project 5: Computer Vision & Sequential Architectures (CNNs & LSTMs)

## Overview & Curriculum Mapping
This project maps to **Module 4: Transition to Deep Learning and Modern AI** in `ai-foundations.tex`. It builds the two classic deep learning architectural paradigms that dominated vision and sequential processing prior to Transformers:
1. **Convolutional Neural Networks (CNNs)**: Weight-sharing filters, spatial pooling, and Residual skip connections ($\mathbf{y} = \text{LayerNorm}(\mathbf{x} + \mathcal{F}(\mathbf{x}))$).
2. **Recurrent Neural Networks (LSTMs)**: Recurrent hidden state propagation ($\mathbf{h}_t = \tanh(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$) with memory gating.

---

## Directory Structure
```
05_vision_and_sequence_models/
├── src/
│   ├── cnn_model.py            # PyTorch Residual CNN architecture
│   ├── lstm_model.py           # PyTorch Multi-layer LSTM architecture
│   └── trainer.py              # DeepModelTrainer with CosineAnnealingLR
├── train_vision_sequence.py    # Main vision & sequence training script
└── README.md                   # Project documentation
```

---

## How to Run

Execute the training script:
```bash
python projects/05_vision_and_sequence_models/train_vision_sequence.py
```

Outputs will be saved under `projects/05_vision_and_sequence_models/output/`:
- `resnet_cnn.pt` (Trained CNN state dict)
- `lstm_classifier.pt` (Trained LSTM state dict)
- `cnn_training_curve.png` (Training loss visualization)

---

## Hands-on Exercises for Students
1. Extend `cnn_model.py` to plot activation feature maps of intermediate convolutional layers.
2. In `lstm_model.py`, implement a **Bidirectional LSTM** and compare its classification performance against the unidirectionally gated baseline.
