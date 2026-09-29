import numpy as np
from sklearn.svm import SVC

# Generate non-linearly separable concentric circles dataset
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(300, 2))
y = ((X[:, 0]**2 + X[:, 1]**2) < 1.5).astype(int)

# Compare Linear vs RBF Kernel SVM
svm_lin = SVC(kernel='linear', C=1.0).fit(X[:240], y[:240])
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X[:240], y[:240])

print(f"Linear Kernel Test Acc: {svm_lin.score(X[240:], y[240:]):.3f}")
print(f"RBF Kernel Test Acc:    {svm_rbf.score(X[240:], y[240:]):.3f}")
print(f"RBF Support Vectors:    {len(svm_rbf.support_vectors_)} / 240 samples")
