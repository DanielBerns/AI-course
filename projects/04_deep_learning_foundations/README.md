# Project 4: Deep Learning Foundations & Resumable Training Pipeline

## Overview & Curriculum Mapping
This project maps directly to **Module 4: Transition to Deep Learning and Modern AI** in `ai-foundations.tex`. It implements manual backpropagation from scratch in pure NumPy, builds modular PyTorch MLP architectures alongside Keras functional API comparisons, and implements an enterprise-grade **Resumable Training Pipeline** using state-dict serialization.

### Key Capabilities Implemented
1. **NumPy Backprop Engine**: 2-layer Neural Network with manual matrix calculus derivatives ($\delta^{(L)} = \nabla_a \mathcal{L} \odot \phi'(z^{(L)})$, $\delta^{(\ell)} = (W^{(\ell+1)\top}\delta^{(\ell+1)}) \odot \phi'(z^{(\ell)})$).
2. **PyTorch vs. Keras Framework Parity**: Modular `PyTorchMLP` implementation side-by-side with Keras functional equivalent code.
3. **Resumable Training Architecture**: `CheckpointManager` that saves complete state dictionaries (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, and metric history) to allow interrupted training recovery without loss of optimizer momentum or learning rate schedule state.
4. **PyTorch Inference Engine**: Batch prediction service with CPU/GPU device selection.

---

## Directory Structure
```
04_deep_learning_foundations/
├── src/
│   ├── mlp_from_scratch.py      # NumPy MLP with manual forward/backward calculus
│   ├── dataset.py               # Custom PyTorch TabularDataset & DataLoader builder
│   ├── model.py                 # PyTorchMLP & Keras equivalent code generator
│   └── checkpoint_manager.py    # Resumable CheckpointManager
├── train_resumable.py           # Runnable training script with interruption simulation
├── inference.py                 # Batch PyTorch inference service
└── README.md                    # Project documentation
```

---

## How to Run

Execute the resumable training script:
```bash
python projects/04_deep_learning_foundations/train_resumable.py
```

Outputs will be saved under `projects/04_deep_learning_foundations/output/`:
- `checkpoints/mlp_tabular_latest.pt` (Complete state dict checkpoint)
- `checkpoints/mlp_tabular_best.pt` (Best validation loss checkpoint)
- `resumable_training_curve.png` (Continuous loss curve proving seamless resumption across phase interruption)

---

## Hands-on Exercises for Students
1. Extend `mlp_from_scratch.py` to support arbitrary $L$-layer deep stacks and add a Softmax output layer for multi-class classification.
2. In `checkpoint_manager.py`, add support for saving and restoring **Early Stopping** patience counters.
3. Replace the `StepLR` scheduler in `train_resumable.py` with a **CosineAnnealingLR** schedule and observe convergence.
