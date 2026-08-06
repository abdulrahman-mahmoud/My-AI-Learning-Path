#----------------------------
#  Multicollinearity & VIF
#  VIFj = 1 / (1 - R^2j)
#  R^2j from regressing Xj on all others
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Data with correlated features
# X1 and X2 are intentionally correlated
np.random.seed(42)
n = 100

X1 = np.random.randn(n)
X2 = 0.9 * X1 + 0.1 * np.random.randn(n)  # X2 is almost the same as X1
X3 = np.random.randn(n)                     # Independent
X4 = 2 * X3 + np.random.randn(n) * 0.5      # X4 is related to X3

y = 10 + 2*X1 - 1*X2 + 3*X3 + 0.5*X4 + np.random.randn(n)

X = np.column_stack((X1, X2, X3, X4))
feature_names = ['X1', 'X2', 'X3', 'X4']

# Correlation matrix
corr = np.corrcoef(X.T)
print("Correlation Matrix:")
print("         ", "  ".join(f"{name:>6}" for name in feature_names))
for i, name in enumerate(feature_names):
    print(f"  {name:>4s}: ", "  ".join(f"{corr[i,j]:6.3f}" for j in range(4)))

# VIF calculation
p = X.shape[1]
vifs = np.zeros(p)

for j in range(p):
    y_j = X[:, j]
    X_other = np.delete(X, j, axis=1)
    X_other = np.column_stack((np.ones(n), X_other))

    beta_j = np.linalg.solve(X_other.T @ X_other, X_other.T @ y_j)
    y_j_pred = X_other @ beta_j

    SS_res = np.sum((y_j - y_j_pred) ** 2)
    SS_tot = np.sum((y_j - np.mean(y_j)) ** 2)
    R2_j = 1 - SS_res / SS_tot

    vifs[j] = 1 / (1 - R2_j + 1e-10)

print("\nVIF (Variance Inflation Factor):")
print("  " + "-" * 40)
print(f"  {'Feature':8s} {'VIF':8s} {'Severity':12s}")
print("  " + "-" * 40)
for i, name in enumerate(feature_names):
    if vifs[i] > 10:
        sev = "SEVERE"
    elif vifs[i] > 5:
        sev = "MODERATE"
    else:
        sev = "OK"
    print(f"  {name:8s} {vifs[i]:8.2f} {sev:12s}")
print("  " + "-" * 40)
print(f"\n  VIF > 10: Multicollinearity is severe (ISL Section 3.3.2)")
print(f"  VIF > 5:  Worth investigating")
print(f"  VIF = 1:  No correlation with other predictors")

# ========== PLOTS ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Correlation Heatmap
im = axes[0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(feature_names)))
axes[0].set_yticks(range(len(feature_names)))
axes[0].set_xticklabels(feature_names)
axes[0].set_yticklabels(feature_names)
axes[0].set_title('Correlation Matrix')
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        axes[0].text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                     fontsize=9, fontweight='bold',
                     color='white' if abs(corr[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=axes[0], shrink=0.8)

# 2. VIF Bar Chart
colors = []
sev_labels = []
for v in vifs:
    if v > 10:
        colors.append('crimson')
        sev_labels.append('SEVERE')
    elif v > 5:
        colors.append('orange')
        sev_labels.append('MODERATE')
    else:
        colors.append('steelblue')
        sev_labels.append('OK')

bars = axes[1].bar(feature_names, vifs, color=colors, edgecolor='black')
axes[1].axhline(y=5, color='orange', linestyle='--', linewidth=1.5, label='VIF=5 (Moderate)')
axes[1].axhline(y=10, color='crimson', linestyle='--', linewidth=1.5, label='VIF=10 (Severe)')
axes[1].set_ylabel('VIF')
axes[1].set_title('Variance Inflation Factor')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

for bar, label in zip(bars, sev_labels):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 label, ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/07_vif_multicollinearity.png', dpi=150)
plt.close()
print("[Plot saved: plots/07_vif_multicollinearity.png]")
