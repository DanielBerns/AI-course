# Project 2: Math Engine from Scratch (Linear Algebra, Stats & Optimization)

## Overview & Curriculum Mapping
This project maps to **Module 2: Mathematics and Data Foundations** in `ai-foundations.tex`. It translates core theoretical mechanics—vector projections, eigen-decomposition, probability distributions, loss function gradients, and optimizer update equations—into pure Python/NumPy implementations.

### Key Topics Implemented
1. **Linear Algebra & Dimensionality Reduction**: `PCAFromScratch` via eigen-decomposition of the covariance matrix ($A \mathbf{v} = \lambda \mathbf{v}$) and SVD ($X = U \Sigma V^\top$).
2. **First-Order Optimization Algorithms**: Implementation of Vanilla Gradient Descent, Momentum, RMSprop, and Adam from scratch.
3. **Probability & Statistics**: Central Limit Theorem verification ($\sigma_{\bar{x}} = \sigma / \sqrt{n}$) and exact Bayesian Beta-Binomial conjugate updating.

---

## Directory Structure
```
02_math_and_optimization/
├── src/
│   ├── pca_from_scratch.py      # Covariance & SVD-based PCA implementation
│   ├── optimizers.py            # GD, Momentum, RMSprop, and Adam optimizers
│   └── stats_verifier.py        # CLT & Bayesian updating simulation engines
├── run_math_sandbox.py          # Main executable simulation & visualization demo
└── README.md                    # Project documentation
```

---

## How to Run

Execute the math sandbox demo:
```bash
python projects/02_math_and_optimization/run_math_sandbox.py
```

Visualization outputs will be generated under `projects/02_math_and_optimization/output/`:
- `optimizer_trajectories.png` (Convergence trajectories on ill-conditioned loss surfaces)
- `clt_verification.png` (Sampling mean distribution convergence across sample sizes)
- `bayesian_update.png` (Prior vs. Posterior density under evidence)

---

## Hands-on Exercises for Students
1. Modify `optimizers.py` to implement **Nesterov Accelerated Gradient (NAG)**.
2. In `pca_from_scratch.py`, implement a method to compute the minimum number of principal components required to retain $\ge 95\%$ of total dataset variance.
3. Simulate vanishing and exploding gradients by creating a stack of 10 linear transformations with singular values $> 1$ and $< 1$.
