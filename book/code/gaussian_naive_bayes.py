import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class GaussianNBScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.array([X[y == c].var(axis=0) + 1e-9 for c in self.classes])
        return self

    def predict(self, X):
        # Log likelihood for numerical stability: log P(y) + sum log P(x_j|y)
        log_probs = []
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.priors[i])
            log_pdf = -0.5 * np.sum(np.log(2 * np.pi * self.vars[i]) + 
                                   ((X - self.means[i])**2) / self.vars[i], axis=1)
            log_probs.append(log_prior + log_pdf)
        return self.classes[np.argmax(np.array(log_probs), axis=0)]

# Synthetic binary dataset
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

gnb = GaussianNBScratch().fit(X[:150], y[:150])
knn = KNeighborsClassifier(n_neighbors=5).fit(X[:150], y[:150])

print(f"Gaussian NB (Scratch) Acc: {(gnb.predict(X[150:]) == y[150:]).mean():.3f}")
print(f"k-NN (k=5) Test Acc:       {knn.score(X[150:], y[150:]):.3f}")
