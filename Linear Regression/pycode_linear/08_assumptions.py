#----------------------------
#  Assumption Checking
#  Linear Regression Assumptions
#  (ISL Section 3.3)
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data that follows assumptions
np.random.seed(42)
n = 100
X1 = np.random.randn(n)
X2 = np.random.randn(n)
y = 5 + 2*X1 - 3*X2 + np.random.randn(n) * 0.5  # True linear with normal errors

X = np.column_stack((np.ones(n), X1, X2))

# Fit model
beta = np.linalg.solve(X.T @ X, X.T @ y)
y_pred = X @ beta
residuals = y - y_pred

# 1. LINEARITY: Check if residuals have no pattern vs fitted
corr_fitted_absres = np.corrcoef(y_pred, np.abs(residuals))[0, 1]

# 2. INDEPENDENCE: Durbin-Watson test
diff = np.diff(residuals)
dw = np.sum(diff ** 2) / np.sum(residuals ** 2)

# 3. HOMOSCEDASTICITY: Check if |residuals| correlate with fitted
corr_fitted_sqres = np.corrcoef(y_pred, residuals ** 2)[0, 1]

# 4. NORMALITY: Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(residuals)

print("Linear Regression Assumption Checks (ISL Section 3.3)")
print("=" * 50)

print("\n1. LINEARITY: E(Y|X) = XBeta")
print(f"   Correlation(|residuals|, fitted): {corr_fitted_absres:.4f}")
print(f"   {'> PASS if' if abs(corr_fitted_absres) < 0.3 else '> WARN if'} |corr| < 0.3 (no pattern = linearity holds)")

print("\n2. INDEPENDENCE: Errors are uncorrelated")
print(f"   Durbin-Watson: {dw:.4f}")
if 1.5 < dw < 2.5:
    print("   > PASS: No autocorrelation (DW near 2)")
else:
    print("   > FAIL: Autocorrelation detected")

print("\n3. HOMOSCEDASTICITY: Constant variance of errors")
print(f"   Correlation(fitted, squared residuals): {corr_fitted_sqres:.4f}")
if abs(corr_fitted_sqres) < 0.2:
    print("   > PASS: Variance is roughly constant")
else:
    print("   > FAIL: Heteroscedasticity detected (funnel shape)")

print("\n4. NORMALITY: Errors ~ N(0, sigma^2)")
print(f"   Shapiro-Wilk p-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("   > PASS: Residuals appear normal")
else:
    print("   > FAIL: Residuals deviate from normality")

print("\n" + "=" * 50)
print("If assumptions are violated:")
print("  - Coefficients are still unbiased (Gauss-Markov theorem)")
print("  - But standard errors, p-values, and CIs are wrong")
print("  - Prediction intervals may be inaccurate")

# ========== PLOTS ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Linearity: Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, color='steelblue', s=30, alpha=0.6)
axes[0, 0].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title(f'1. Linearity Check\nCorr(|e|, y_hat)={corr_fitted_absres:.3f}')
axes[0, 0].grid(alpha=0.3)

# 2. Independence: Residuals in order (time series check if available)
axes[0, 1].scatter(range(len(residuals)), residuals, color='steelblue', s=30, alpha=0.6)
axes[0, 1].plot(range(len(residuals)), residuals, color='steelblue', linewidth=0.5, alpha=0.4)
axes[0, 1].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
axes[0, 1].set_xlabel('Observation Order')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title(f'2. Independence Check (Durbin-Watson={dw:.3f})\nNo tracking = independent')
axes[0, 1].grid(alpha=0.3)

# 3. Homoscedasticity: Squared residuals vs Fitted
axes[1, 0].scatter(y_pred, residuals**2, color='steelblue', s=30, alpha=0.6)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Squared Residuals')
axes[1, 0].set_title(f'3. Homoscedasticity Check\nCorr(e², y_hat)={corr_fitted_sqres:.3f}')
axes[1, 0].grid(alpha=0.3)

# 4. Normality: Q-Q Plot + Histogram inset
qq = stats.probplot(residuals, dist="norm", plot=None)
axes[1, 1].scatter(qq[0][0], qq[0][1], color='steelblue', s=30)
axes[1, 1].plot(qq[0][0], qq[0][0] * qq[1][0] + qq[1][1], color='crimson', linewidth=1.5)
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Residuals')
axes[1, 1].set_title(f'4. Normality Check (Shapiro p={shapiro_p:.4f})\nPoints near line = normal')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/08_assumptions.png', dpi=150)
plt.close()
print("[Plot saved: plots/08_assumptions.png]")
