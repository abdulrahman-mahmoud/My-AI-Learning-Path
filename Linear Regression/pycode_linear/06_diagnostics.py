#----------------------------
#  Residual Analysis
#  ei = yi - y_hati
#  hii = diag(X(XTX)⁻¹XT)
#  Di = (ri^2/p)(hii/(1-hii))
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Data
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([2, 3, 1, 4, 2, 5, 3, 6, 4, 5])
y  = np.array([15, 22, 18, 28, 25, 35, 30, 40, 33, 42])

X = np.column_stack((np.ones(len(x1)), x1, x2))
n, p = X.shape

# Fit model
beta = np.linalg.solve(X.T @ X, X.T @ y)
y_pred = X @ beta
residuals = y - y_pred

# Hat matrix: H = X(XTX)⁻¹XT
H = X @ np.linalg.inv(X.T @ X) @ X.T
h = np.diag(H)

# RSE (Residual Standard Error)
RSS = np.sum(residuals ** 2)
RSE = np.sqrt(RSS / (n - p))

# Studentized residuals: ri = ei / (RSE * √(1 - hii))
studentized = residuals / (RSE * np.sqrt(1 - h))

# Cook's Distance: Di = (ri^2/p) * (hii/(1-hii))
cooks_d = (studentized ** 2 / p) * (h / (1 - h))

print("Residual Analysis")
print("=" * 40)
print("\nFitted vs Actual:")
for i in range(n):
    print(f"  y[{i}]={y[i]:3d}  y_hat={y_pred[i]:6.2f}  e={residuals[i]:6.2f}  h={h[i]:.3f}  D={cooks_d[i]:.4f}")

print("\nSummary:")
print(f"  RSE: {RSE:.4f}")
print(f"  Mean leverage: {np.mean(h):.4f} (threshold: {2*p/n:.4f})")
print(f"  High leverage points (>2p/n): {np.where(h > 2*p/n)[0].tolist()}")
print(f"  Outliers (|studentized| > 3): {np.where(np.abs(studentized) > 3)[0].tolist()}")
print(f"  Influential (Cook's D > 4/n): {np.where(cooks_d > 4/n)[0].tolist()}")

# ========== PLOTS ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, color='steelblue', s=60)
axes[0, 0].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].grid(alpha=0.3)
for i in range(n):
    axes[0, 0].annotate(str(i), (y_pred[i], residuals[i]), fontsize=7, alpha=0.7)

# 2. Q-Q Plot (studentized residuals sorted vs normal quantiles)
from scipy import stats
qq = stats.probplot(studentized, dist="norm", plot=None)
axes[0, 1].scatter(qq[0][0], qq[0][1], color='steelblue', s=40)
axes[0, 1].plot(qq[0][0], qq[0][0] * qq[1][0] + qq[1][1], color='crimson', linewidth=1.5)
axes[0, 1].set_xlabel('Theoretical Quantiles')
axes[0, 1].set_ylabel('Studentized Residuals')
axes[0, 1].set_title('Q-Q Plot (Normality Check)')
axes[0, 1].grid(alpha=0.3)

# 3. Leverage
axes[1, 0].stem(range(n), h, basefmt=' ')
axes[1, 0].axhline(y=2*p/n, color='crimson', linestyle='--', linewidth=1.5, label=f'Threshold (2p/n={2*p/n:.3f})')
axes[1, 0].set_xlabel('Observation Index')
axes[1, 0].set_ylabel('Leverage (hii)')
axes[1, 0].set_title('Leverage Values')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Cook's Distance
axes[1, 1].stem(range(n), cooks_d, basefmt=' ')
axes[1, 1].axhline(y=4/n, color='crimson', linestyle='--', linewidth=1.5, label=f'Threshold (4/n={4/n:.4f})')
axes[1, 1].set_xlabel('Observation Index')
axes[1, 1].set_ylabel("Cook's Distance")
axes[1, 1].set_title("Cook's Distance (Influence)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/06_diagnostics.png', dpi=150)
plt.close()
print("[Plot saved: plots/06_diagnostics.png]")
