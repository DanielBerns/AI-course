import numpy as np

rng = np.random.default_rng(42)
# 10,000 experiments: average of 40 Exponential(1) draws each
means = rng.exponential(scale=1.0, size=(10_000, 40)).mean(axis=1)
print(f"Sample Mean: {means.mean():.3f} | SE: {means.std():.3f}") # ~1.0 & 1/sqrt(40)
