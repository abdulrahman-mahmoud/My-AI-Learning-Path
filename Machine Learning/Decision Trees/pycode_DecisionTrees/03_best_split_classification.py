#----------------------------
#  03: Finding the best CLASSIFICATION split
#  Tiny dataset from documentation Section 21.2
#  X = [1..6], class = [0, 0, 1, 1, 1, 0]
#  Score = weighted child Gini impurity
#  Q = (N1/N)*G1 + (N2/N)*G2
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------
#  Data (class labels already 0/1)
#------------------------------
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([0, 0, 1, 1, 1, 0])      # 3 zeros, 3 ones at the root

def class_proportions(vals, n_classes):
    counts = np.bincount(vals, minlength=n_classes)
    return counts, counts / counts.sum()

def gini(probs):
    # Gini impurity: 1 - sum(p_k^2)
    return 1.0 - np.sum(probs ** 2)

def weighted_child_gini(left, right, n):
    _, p1 = class_proportions(left,  2)
    _, p2 = class_proportions(right, 2)
    g1, g2 = gini(p1), gini(p2)
    return (len(left) / n) * g1 + (len(right) / n) * g2, g1, g2

#------------------------------
#  Root impurity and candidate splits
#------------------------------
counts_root, p_root = class_proportions(y, 2)
G_root = gini(p_root)
print(f"Root: counts = {counts_root}, p = ({p_root[0]:.2f},{p_root[1]:.2f}), Gini = {G_root:.4f}\n")

print("Evaluating CLASSIFICATION candidate splits (X <= s)\n")
print(f"{'split':<12}{'left (N,G)':>20}{'right (N,G)':>20}{'weighted Q':>12}{'gain':>10}")

results = []
values = np.sort(np.unique(X))
for k in range(1, len(values)):
    s = (values[k - 1] + values[k]) / 2.0
    left  = y[X <= s]
    right = y[X > s]
    Q, g1, g2 = weighted_child_gini(left, right, len(y))
    gain = G_root - Q
    results.append((s, Q, gain))
    print(f"X <= {s:<9.1f}({len(left):>2d},{g1:>5.2f})    ({len(right):>2d},{g2:>5.2f})    {Q:>12.4f}{gain:>10.4f}")

best_s, best_Q, best_gain = min(results, key=lambda r: r[1])
print(f"\nBest split: X <= {best_s}  (weighted Gini = {best_Q:.4f}, gain = {best_gain:.4f})")

#------------------------------
#  Plot
#------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: weighted Gini of each candidate split
splits = [str(r[0]) for r in results]
Qs     = [r[1] for r in results]
axes[0].bar(splits, Qs, color='steelblue', edgecolor='black')
axes[0].axhline(G_root, color='crimson', linestyle='--', label=f'Root Gini = {G_root:.2f}')
axes[0].set_ylabel('Weighted child Gini')
axes[0].set_xlabel('Candidate split (X <= s)')
axes[0].set_title('Weighted impurity of each classification split')
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')

# Right: classes coloured along X with the chosen threshold
axes[1].scatter(X[y == 0], np.zeros(np.sum(y == 0)), color='steelblue', s=120, label='Class 0', zorder=3)
axes[1].scatter(X[y == 1], np.ones(np.sum(y == 1)),   color='crimson',    s=120, label='Class 1', zorder=3)
axes[1].axvline(best_s, color='gray', linestyle=':', linewidth=1.5)
axes[1].set_xlabel('X')
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Class 0', 'Class 1'])
axes[1].set_title(f'Best split: X <= {best_s} separates the classes')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / '03_best_split_classification.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/03_best_split_classification.png]")
