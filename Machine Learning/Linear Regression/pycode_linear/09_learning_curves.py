#----------------------------
#  Learning Curves
#  Diagnosing Bias vs Variance
#  (ISL Section 2.2)
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n = 200
X = np.random.randn(n, 3)
true_beta = np.array([5, 2, -3, 1])
X_design = np.column_stack((np.ones(n), X))
y = X_design @ true_beta + np.random.randn(n) * 0.5

train_sizes = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160])
n_folds = 5

train_scores = []
val_scores = []

for train_size in train_sizes:
    fold_train = []
    fold_val = []

    for fold in range(n_folds):
        indices = np.random.permutation(n)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:min(train_size + n // 4, n)]

        X_train = X_design[train_idx]
        y_train = y[train_idx]
        X_val = X_design[val_idx]
        y_val = y[val_idx]

        # Fit
        beta_fold = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)

        # Training R^2
        y_train_pred = X_train @ beta_fold
        r2_train = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)

        # Validation R^2
        y_val_pred = X_val @ beta_fold
        r2_val = 1 - np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)

        fold_train.append(r2_train)
        fold_val.append(r2_val)

    train_scores.append(np.mean(fold_train))
    val_scores.append(np.mean(fold_val))

print("Learning Curves (R^2 vs Training Set Size)")
print("=" * 50)
print(f"  {'Size':6s} {'Train R^2':10s} {'Val R^2':10s} {'Gap':10s}")
print("  " + "-" * 36)
for i, size in enumerate(train_sizes):
    gap = train_scores[i] - val_scores[i]
    print(f"  {size:6d} {train_scores[i]:10.4f} {val_scores[i]:10.4f} {gap:10.4f}")

print("\nInterpretation:")
print("  Small gap + high scores = good fit (low bias, low variance)")
print("  Large gap = overfitting (low bias, high variance)")
print("  Both low = underfitting (high bias)")
print()
print("Bias-Variance Tradeoff (ISL Section 2.2):")
print("  - As training size grows, training R^2 drops, validation R^2 rises")
print("  - If curves converge to same high value = model is stable")
print("  - If curves don't converge = need more data or simpler model")

# ========== PLOT ==========
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(train_sizes, train_scores, 'o-', color='steelblue', linewidth=2, label='Training R²')
ax.plot(train_sizes, val_scores, 's-', color='crimson', linewidth=2, label='Validation R²')
ax.fill_between(train_sizes, train_scores, val_scores, alpha=0.1, color='gray', label='Generalization Gap')

ax.set_xlabel('Training Set Size')
ax.set_ylabel('R² Score')
ax.set_title('Learning Curves: R² vs Training Set Size')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# Annotate key regions
mid_idx = len(train_sizes) // 2
ax.annotate('High Bias\n(Large Gap)', xy=(train_sizes[1], (train_scores[1] + val_scores[1]) / 2),
            fontsize=9, color='gray', ha='center')
ax.annotate('Convergence', xy=(train_sizes[-1], train_scores[-1]),
            fontsize=9, color='green', ha='center')

plt.tight_layout()
plt.savefig('plots/09_learning_curves.png', dpi=150)
plt.close()
print("[Plot saved: plots/09_learning_curves.png]")
