import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def gini_impurity(y):
    """Compute Gini Impurity I_G(p) = 1 - sum(p_k^2)"""
    if len(y) == 0: return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p**2)

# Synthetic classification dataset with non-linear interaction
rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
y = (X[:, 0]*2 - X[:, 1] + X[:, 2]**2 > 1.0).astype(int)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X[:400], y[:400])
gb = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42).fit(X[:400], y[:400])

print(f"Root Split Gini Impurity:   {gini_impurity(y[:400]):.3f}")
print(f"Random Forest Test Acc:     {rf.score(X[400:], y[400:]):.3f}")
print(f"Gradient Boosting Test Acc: {gb.score(X[400:], y[400:]):.3f}")
