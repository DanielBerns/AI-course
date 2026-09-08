"""
Project 2: Mathematics & Data Foundations Sandbox Demo.

Runs math primitives from scratch: PCA dimensionality reduction, optimizer loss landscape
trajectories (GD, Momentum, Adam), CLT verification, and Bayesian inference updates.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pca_from_scratch import PCAFromScratch
from optimizers import GradientDescent, SGDMomentum, Adam, optimize_trajectory
from stats_verifier import simulate_central_limit_theorem, bayesian_beta_binomial_update


def main() -> None:
    print("=" * 70)
    print("PROJECT 2: MATHEMATICS AND DATA FOUNDATIONS SANDBOX")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. PCA from Scratch Verification
    print("\n[Step 1] Verifying Principal Component Analysis (Eigen vs. SVD)...")
    rng = np.random.default_rng(42)
    # Generate 500 samples in 10 dimensions with strong principal direction
    mean = np.zeros(10)
    cov = np.diag([25.0, 9.0, 4.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    X = rng.multivariate_normal(mean, cov, size=500)

    pca_svd = PCAFromScratch(n_components=3, method="svd").fit(X)
    pca_eigh = PCAFromScratch(n_components=3, method="eigh").fit(X)

    print(f" - Top-3 Variance Ratios (SVD):  {pca_svd.explained_variance_ratio_.round(4)}")
    print(f" - Top-3 Variance Ratios (Eigh): {pca_eigh.explained_variance_ratio_.round(4)}")
    print(f" - Cumulative Variance (Top 3): {np.sum(pca_svd.explained_variance_ratio_):.2%}")
    print(f" - Reconstruction Error (MSE):  {pca_svd.compute_reconstruction_error(X):.6f}")

    # 2. Optimization Landscapes & Optimizer Trajectories
    print("\n[Step 2] Comparing Optimizers (GD vs. Momentum vs. Adam) on Anisotropic Loss...")
    # Quadratic ill-conditioned loss: f(w1, w2) = 0.5 * w1^2 + 10.0 * w2^2
    def loss_func(w: np.ndarray) -> float:
        return float(0.5 * w[0] ** 2 + 10.0 * w[1] ** 2)

    def grad_func(w: np.ndarray) -> np.ndarray:
        return np.array([1.0 * w[0], 20.0 * w[1]])

    init_w = np.array([4.0, 3.0])
    steps = 40

    traj_gd, loss_gd = optimize_trajectory(loss_func, grad_func, init_w, GradientDescent(lr=0.08), steps=steps)
    traj_mom, loss_mom = optimize_trajectory(loss_func, grad_func, init_w, SGDMomentum(lr=0.08, momentum=0.85), steps=steps)
    traj_adam, loss_adam = optimize_trajectory(loss_func, grad_func, init_w, Adam(lr=0.5), steps=steps)

    print(f" - Final Loss GD (step {steps}):       {loss_gd[-1]:.6f}")
    print(f" - Final Loss Momentum (step {steps}): {loss_mom[-1]:.6f}")
    print(f" - Final Loss Adam (step {steps}):     {loss_adam[-1]:.6f}")

    # Plot loss landscapes and trajectories
    fig, ax = plt.subplots(figsize=(8, 6))
    w1_vals = np.linspace(-5, 5, 200)
    w2_vals = np.linspace(-4, 4, 200)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = 0.5 * W1**2 + 10.0 * W2**2

    ax.contour(W1, W2, Z, levels=np.logspace(-1, 2, 20), cmap="PuBu_r", alpha=0.6)
    ax.plot(traj_gd[:, 0], traj_gd[:, 1], "o-", color="#e67e22", label="Vanilla GD", linewidth=1.5, markersize=4)
    ax.plot(traj_mom[:, 0], traj_mom[:, 1], "s-", color="#009688", label="SGD + Momentum", linewidth=1.5, markersize=4)
    ax.plot(traj_adam[:, 0], traj_adam[:, 1], "d-", color="#6c349e", label="Adam", linewidth=1.5, markersize=4)

    ax.set_title("Optimizer Trajectories on Ill-Conditioned Loss Surface", fontsize=12, fontweight="bold", color="#263c72")
    ax.set_xlabel("w1", fontsize=10)
    ax.set_ylabel("w2", fontsize=10)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "optimizer_trajectories.png"), dpi=300)
    plt.close(fig)

    # 3. Central Limit Theorem Simulation
    print("\n[Step 3] Verifying Central Limit Theorem...")
    sample_means_dict = simulate_central_limit_theorem(dist_type="exponential", sample_sizes=[2, 10, 50], num_experiments=10000)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, (n, means) in enumerate(sample_means_dict.items()):
        ax = axes[idx]
        ax.hist(means, bins=40, density=True, color="#263c72", alpha=0.7, edgecolor="white")
        ax.set_title(f"Sample Size n = {n}", fontsize=11, fontweight="bold", color="#263c72")
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        # Annotate std
        ax.text(0.65, 0.8, f"std: {np.std(means):.3f}\n1/√n: {1/np.sqrt(n):.3f}", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "clt_verification.png"), dpi=300)
    plt.close(fig)
    print(f" -> Generated optimizer and CLT plots in '{output_dir}'.")

    # 4. Bayesian Inference Updating
    print("\n[Step 4] Simulating Conjugate Bayesian Beta-Binomial Update...")
    prior_a, prior_b = 2.0, 2.0  # Weakly informative prior
    trials, successes = 50, 38  # Medical trial evidence: 38 successes in 50 trials
    theta, prior_pdf, post_pdf = bayesian_beta_binomial_update(prior_a, prior_b, trials, successes)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(theta, prior_pdf, label="Prior: Beta(2, 2)", color="#e67e22", linestyle="--", linewidth=2)
    ax.plot(theta, post_pdf, label=f"Posterior: Beta({prior_a + successes}, {prior_b + trials - successes})", color="#263c72", linewidth=2.5)
    ax.set_title("Bayesian Updating: Prior vs. Posterior", fontsize=12, fontweight="bold", color="#263c72")
    ax.set_xlabel("Success Rate θ", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "bayesian_update.png"), dpi=300)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("PROJECT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
