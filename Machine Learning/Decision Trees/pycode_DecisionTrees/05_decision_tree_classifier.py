#----------------------------
#  05: DecisionTreeClassifier from scratch (NumPy only)
#  Recursive binary splitting by weighted impurity
#  (criterion: 'gini' or 'entropy').
#  Leaf prediction = majority class; predict_proba = class proportions.
#  Class labels may be any comparable values (int, str, ...); they are
#  mapped to internal 0..K-1 indices, and predict() returns the originals.
#
#  Axis-aligned splits: each split uses ONE feature and ONE threshold,
#  so decision boundaries are vertical/horizontal lines.
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

#========================================================
#  Node types
#========================================================
class Leaf:
    """Terminal node: stores class counts and probabilities."""
    def __init__(self, y, n_classes):
        counts = np.bincount(y, minlength=n_classes)
        self.is_leaf = True
        self.counts = counts
        self.probs = counts / counts.sum()          # class proportions p_1..p_K
        self.prediction = int(np.argmax(counts))    # majority class

class InternalNode:
    """Decision node: asks 'X[feature] <= threshold?' and routes children."""
    def __init__(self, feature, threshold, left, right):
        self.is_leaf = False
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

#========================================================
#  Classification split logic
#========================================================
def gini(probs):
    # Gini impurity: 1 - sum(p_k^2)
    return 1.0 - np.sum(probs ** 2)

def entropy(probs):
    # Entropy: -sum(p_k log p_k); classes with p_k = 0 contribute 0 (0 log 0 = 0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def impurity(probs, criterion):
    """Dispatch to the requested impurity measure."""
    if criterion == 'gini':
        return gini(probs)
    if criterion == 'entropy':
        return entropy(probs)
    raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")

def weighted_impurity(left_y, right_y, n, n_classes, criterion):
    c1 = np.bincount(left_y,  minlength=n_classes)
    c2 = np.bincount(right_y, minlength=n_classes)
    i1 = impurity(c1 / c1.sum(), criterion)
    i2 = impurity(c2 / c2.sum(), criterion)
    return (len(left_y) / n) * i1 + (len(right_y) / n) * i2

def best_split(X, y, n_classes, criterion):
    """Return (feature, threshold) minimising the weighted child impurity."""
    n_samples, n_features = X.shape
    best_Q = np.inf
    best = None
    for j in range(n_features):
        values = np.sort(np.unique(X[:, j]))
        for k in range(1, len(values)):
            s = (values[k - 1] + values[k]) / 2.0
            left  = y[X[:, j] <= s]
            right = y[X[:, j] >  s]
            Q = weighted_impurity(left, right, n_samples, n_classes, criterion)
            if Q < best_Q:
                best_Q = Q
                best = (j, s)
    return best

def build_tree(X, y, depth, max_depth, min_samples_split, n_classes, criterion):
    n = len(y)
    # stopping conditions (Section 13)
    if depth >= max_depth or n < min_samples_split or len(np.unique(y)) == 1:
        return Leaf(y, n_classes)

    best = best_split(X, y, n_classes, criterion)
    if best is None:
        return Leaf(y, n_classes)

    j, s = best
    left_idx  = X[:, j] <= s
    left  = build_tree(X[left_idx],  y[left_idx],  depth + 1, max_depth, min_samples_split, n_classes, criterion)
    right = build_tree(X[~left_idx], y[~left_idx], depth + 1, max_depth, min_samples_split, n_classes, criterion)
    return InternalNode(j, s, left, right)

#========================================================
#  Classifier class
#========================================================
VALID_CRITERIA = ('gini', 'entropy')

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=2, min_samples_split=2):
        if criterion not in VALID_CRITERIA:
            raise ValueError(f"criterion must be one of {VALID_CRITERIA}, got {criterion!r}")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        self.classes_ = None          # sorted unique original labels; proba columns follow this order
        self.label_to_idx_ = None     # original label -> internal class index

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(np.asarray(y))    # deterministic (sorted) class ordering
        self.n_classes = len(self.classes_)
        self.label_to_idx_ = {label: i for i, label in enumerate(self.classes_)}
        y_idx = np.array([self.label_to_idx_[lab] for lab in y])   # internal 0..K-1 indices
        self.root = build_tree(X, y_idx, 0, self.max_depth, self.min_samples_split,
                               self.n_classes, self.criterion)
        return self

    def _predict_one(self, node, x):
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def _proba_one(self, node, x):
        if node.is_leaf:
            return node.probs
        if x[node.feature] <= node.threshold:
            return self._proba_one(node.left, x)
        return self._proba_one(node.right, x)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        idx = np.array([self._predict_one(self.root, x) for x in X], dtype=int)
        return self.classes_[idx]     # map internal indices back to the original labels

    def predict_proba(self, X):
        # Returned columns follow self.classes_ order (one column per class).
        X = np.asarray(X, dtype=float)
        return np.array([self._proba_one(self.root, x) for x in X])

#========================================================
#  Demo (guarded so importing this module does not run it)
#========================================================
def main():
    # Tiny classification walkthrough (documentation Section 17.2)
    Xc = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    yc = np.array([0, 0, 1, 1, 1, 0])

    clf = DecisionTreeClassifier(criterion='gini', max_depth=1, min_samples_split=2)
    clf.fit(Xc, yc)

    print("Tiny classification dataset:  X = [1..6],  class = [0,0,1,1,1,0]")
    print("Fitted with max_depth = 1\n")
    for x, p, prob in zip(Xc.ravel(), clf.predict(Xc), clf.predict_proba(Xc)):
        print(f"  X = {x}  ->  class {p}   P(class=1) = {prob[1]:.2f}")

    # 2D dataset + axis-aligned decision regions
    rng = np.random.default_rng(0)
    n = 120
    c0 = rng.normal(loc=[4.0, 4.0], scale=1.2, size=(n // 2, 2))
    c1 = rng.normal(loc=[7.0, 7.0], scale=1.2, size=(n // 2, 2))
    Xtr = np.vstack([c0, c1])
    ytr = np.array([0] * (n // 2) + [1] * (n // 2))

    clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=2)
    clf2.fit(Xtr, ytr)

    xx, yy = np.meshgrid(np.linspace(0, 11, 300), np.linspace(0, 11, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf2.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['#cfe0f0', '#f2c4c4'], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=1.5)
    ax.scatter(c0[:, 0], c0[:, 1], color='steelblue', s=40, label='Class 0', edgecolor='k', linewidths=0.3)
    ax.scatter(c1[:, 0], c1[:, 1], color='crimson',   s=40, label='Class 1', edgecolor='k', linewidths=0.3)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'DecisionTreeClassifier decision regions (max_depth = {clf2.max_depth})\naxis-aligned splits')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / '05_classification_regions.png', dpi=150)
    plt.close()
    print("\n[Plot saved: plots/05_classification_regions.png]")

if __name__ == "__main__":
    main()
