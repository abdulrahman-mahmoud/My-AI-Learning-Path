#----------------------------
#  04: DecisionTreeRegressor from scratch (NumPy only)
#  Recursive binary splitting by squared error.
#  Leaf prediction = mean of the region (least-squares constant).
#
#  Tree structure: each node stores feature, threshold, left, right,
#  and (for leaves) a prediction. Internal nodes store a split.
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#========================================================
#  Node types
#========================================================
class Leaf:
    """Terminal node: stores a constant prediction (the region mean)."""
    def __init__(self, prediction):
        self.is_leaf = True
        self.prediction = prediction

class InternalNode:
    """Decision node: asks 'X[feature] <= threshold?' and routes children."""
    def __init__(self, feature, threshold, left, right):
        self.is_leaf = False
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

#========================================================
#  Regression split logic
#========================================================
def squared_error(vals):
    mu = np.mean(vals)
    return np.sum((vals - mu) ** 2)

def best_split(X, y):
    """Return (feature, threshold) minimising the total squared error."""
    n_samples, n_features = X.shape
    best_err = np.inf
    best = None
    for j in range(n_features):
        values = np.sort(np.unique(X[:, j]))          # candidate thresholds between values
        for k in range(1, len(values)):
            s = (values[k - 1] + values[k]) / 2.0
            left  = y[X[:, j] <= s]
            right = y[X[:, j] >  s]
            total = squared_error(left) + squared_error(right)
            if total < best_err:
                best_err = total
                best = (j, s)
    return best

def build_tree(X, y, depth, max_depth, min_samples_split):
    n = len(y)
    # stopping conditions (Section 16): depth limit, min samples, pure node
    if depth >= max_depth or n < min_samples_split or len(np.unique(y)) == 1:
        return Leaf(np.mean(y))

    best = best_split(X, y)
    if best is None:                                   # no useful split
        return Leaf(np.mean(y))

    j, s = best
    left_idx  = X[:, j] <= s
    left  = build_tree(X[left_idx],  y[left_idx],  depth + 1, max_depth, min_samples_split)
    right = build_tree(X[~left_idx], y[~left_idx], depth + 1, max_depth, min_samples_split)
    return InternalNode(j, s, left, right)

#========================================================
#  Regressor class
#========================================================
class DecisionTreeRegressor:
    def __init__(self, max_depth=2, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root = build_tree(X, y, 0, self.max_depth, self.min_samples_split)
        return self

    def _predict_one(self, node, x):
        # traversal: if leaf return prediction, else follow the question
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(self.root, x) for x in X])

#========================================================
#  Demo (guarded so importing this module does not run it)
#========================================================
def main():
    # Tiny dataset walkthrough (documentation Section 21.1)
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([2, 3, 4, 10, 11, 12])

    tree = DecisionTreeRegressor(max_depth=1, min_samples_split=2)
    tree.fit(X, y)

    print("Tiny regression dataset:  X = [1..6],  y = [2,3,4,10,11,12]")
    print("Fitted with max_depth = 1\n")
    print("Predictions:")
    for x, p in zip(X.ravel(), tree.predict(X)):
        print(f"  X = {x}  ->  {p:.1f}")

    c1 = np.mean(y[X.ravel() <= 3.5])
    c2 = np.mean(y[X.ravel() >  3.5])
    print(f"\nExpected leaves: left mean = {c1:.1f}, right mean = {c2:.1f} (split at X <= 3.5)")

    # Plot: observations + piecewise-constant prediction
    x_grid = np.linspace(0.5, 6.5, 300).reshape(-1, 1)
    y_pred = tree.predict(x_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='steelblue', s=100, label='Training data', zorder=3)
    ax.plot(x_grid, y_pred, color='crimson', linewidth=3, label='Tree prediction')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(f'DecisionTreeRegressor (max_depth = {tree.max_depth})\npiecewise-constant prediction')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / '04_regression_tree.png', dpi=150)
    plt.close()
    print("\n[Plot saved: plots/04_regression_tree.png]")

if __name__ == "__main__":
    main()
