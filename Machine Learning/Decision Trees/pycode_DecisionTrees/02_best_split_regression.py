#----------------------------
#  02: Finding the best REGRESSION split
#  Tiny dataset from documentation Section 8
#  X = [1..6], y = [2,3,4,10,11,12]
#  Candidate splits at midpoints between consecutive X values.
#  Score = sum of squared errors of the two children.
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------
#  Data
#------------------------------
X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 3, 4, 10, 11, 12])

def squared_error(vals):
    mu = np.mean(vals)
    return np.sum((vals - mu) ** 2)

def region_mean(vals):
    # The leaf prediction that minimises squared error is the mean (Section 7)
    return np.mean(vals)

#------------------------------
#  Enumerate candidate splits
#------------------------------
print("Evaluating REGRESSION candidate splits (X <= s)\n")
print(f"{'split':<12}{'left mean':>10}{'right mean':>12}{'total error':>14}")

results = []
values = np.sort(np.unique(X))
for k in range(1, len(values)):
    s = (values[k - 1] + values[k]) / 2.0          # threshold midway between values
    left  = y[X <= s]
    right = y[X > s]
    c1, c2 = region_mean(left), region_mean(right)
    err = squared_error(left) + squared_error(right)
    results.append((s, err))
    print(f"X <= {s:<9.1f}{c1:>10.2f}{c2:>12.2f}{err:>14.2f}")

best_s, best_err = min(results, key=lambda r: r[1])
print(f"\nBest split: X <= {best_s}  (total squared error = {best_err})")

#------------------------------
#  Plot: piecewise-constant prediction of the best shallow split
#------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: candidate split errors
splits = [r[0] for r in results]
errs   = [r[1] for r in results]
axes[0].bar([str(s) for s in splits], errs, color='steelblue', edgecolor='black')
axes[0].set_ylabel('Total squared error')
axes[0].set_xlabel('Candidate split (X <= s)')
axes[0].set_title('Error of each candidate regression split')
axes[0].grid(alpha=0.3, axis='y')

# Right: data + piecewise-constant prediction of the best split
axes[1].scatter(X, y, color='steelblue', s=90, label='Training data', zorder=3)
lime = np.array([2, 3, 4])
rime = np.array([10, 11, 12])
c1 = region_mean(lime)
c2 = region_mean(rime)
x_left  = np.linspace(X.min() - 0.5, best_s, 50)
x_right = np.linspace(best_s, X.max() + 0.5, 50)
axes[1].plot(x_left,  np.full_like(x_left,  c1), color='crimson', linewidth=3, label='Prediction')
axes[1].plot(x_right, np.full_like(x_right, c2), color='crimson', linewidth=3)
axes[1].axvline(best_s, color='gray', linestyle=':', linewidth=1.5)
axes[1].axhline(c1, color='crimson', linestyle='--', linewidth=1, alpha=0.6)
axes[1].axhline(c2, color='crimson', linestyle='--', linewidth=1, alpha=0.6)
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title(f'Best split: X <= {best_s}\n(left mean = {c1}, right mean = {c2})')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / '02_best_split_regression.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/02_best_split_regression.png]")
