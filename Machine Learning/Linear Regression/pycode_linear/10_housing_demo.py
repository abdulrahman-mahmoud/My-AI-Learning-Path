#----------------------------
#  California Housing Demo
#  Full pipeline: Normal Equation
#  + GD comparison + diagnostics
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
data = fetch_california_housing()
X_full, y_full = data.data, data.target
feature_names = data.feature_names

print("California Housing Dataset")
print(f"  Samples: {X_full.shape[0]}, Features: {X_full.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
n_train = X_train.shape[0]
print(f"  Train: {n_train}, Test: {X_test.shape[0]}")

# Standardize for GD
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_s = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std

# Add intercept
X_train_d = np.column_stack((np.ones(n_train), X_train_s))
X_test_d = np.column_stack((np.ones(X_test.shape[0]), X_test_s))

# ========== NORMAL EQUATION ==========
print("\n--- Normal Equation ---")
beta_ne = np.linalg.solve(X_train_d.T @ X_train_d, X_train_d.T @ y_train)
y_test_pred = X_test_d @ beta_ne

# Metrics
test_mse = np.mean((y_test - y_test_pred) ** 2)
test_mae = np.mean(np.abs(y_test - y_test_pred))
test_rmse = np.sqrt(test_mse)
test_r2 = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"  Test MSE:  {test_mse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Test MAE:  {test_mae:.4f}")
print(f"  Test R^2:   {test_r2:.4f}")

# Compare with sklearn
from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
sk_model.fit(X_train_s, y_train)
sk_pred = sk_model.predict(X_test_s)
sk_mse = np.mean((y_test - sk_pred) ** 2)
print(f"\n  sklearn R^2: {sk_model.score(X_test_s, y_test):.4f}")
print(f"  Match: {np.allclose(beta_ne[1:], sk_model.coef_, atol=1e-8)}")

# ========== BATCH GD ==========
print("\n--- Batch Gradient Descent ---")
beta_gd = np.zeros(9)
alpha = 0.1
for i in range(500):
    res = X_train_d @ beta_gd - y_train
    grad = (2 / n_train) * X_train_d.T @ res
    beta_gd -= alpha * grad
y_test_pred_gd = X_test_d @ beta_gd
r2_gd = 1 - np.sum((y_test - y_test_pred_gd) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
print(f"  R^2: {r2_gd:.4f}  Match NE: {np.allclose(beta_ne, beta_gd, atol=0.01)}")

# ========== SGD ==========
print("\n--- Stochastic Gradient Descent ---")
beta_sgd = np.zeros(9)
for epoch in range(50):
    idx = np.random.permutation(n_train)
    X_s = X_train_d[idx]
    y_s = y_train[idx]
    for j in range(n_train):
        xi = X_s[j:j+1]
        yi = y_s[j:j+1]
        grad = 2 * xi.T @ (xi @ beta_sgd - yi)
        lr = 0.001 / (1 + 0.0001 * (epoch * n_train + j))
        beta_sgd -= lr * grad.flatten()
y_test_pred_sgd = X_test_d @ beta_sgd
r2_sgd = 1 - np.sum((y_test - y_test_pred_sgd) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
print(f"  R^2: {r2_sgd:.4f}")

# ========== MINI-BATCH ==========
print("\n--- Mini-Batch Gradient Descent ---")
beta_mb = np.zeros(9)
batch_size = 128
for epoch in range(200):
    idx = np.random.permutation(n_train)
    X_s = X_train_d[idx]
    y_s = y_train[idx]
    for start in range(0, n_train, batch_size):
        end = start + batch_size
        Xb = X_s[start:end]
        yb = y_s[start:end]
        res = Xb @ beta_mb - yb
        grad = (2 / Xb.shape[0]) * Xb.T @ res
        beta_mb -= 0.01 * grad
y_test_pred_mb = X_test_d @ beta_mb
r2_mb = 1 - np.sum((y_test - y_test_pred_mb) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
print(f"  R^2: {r2_mb:.4f}")

# ========== Diagnostics ==========
print("\n--- Diagnostics ---")
residuals = y_test - y_test_pred
X_test_no_int = X_test_s
p = X_test_no_int.shape[1]
n_test = len(y_test)

# VIF
print("\n  VIF Analysis:")
for j in range(p):
    y_j = X_test_no_int[:, j]
    X_other = np.delete(X_test_no_int, j, axis=1)
    X_other = np.column_stack((np.ones(n_test), X_other))
    bj = np.linalg.solve(X_other.T @ X_other, X_other.T @ y_j)
    y_j_pred = X_other @ bj
    r2_j = 1 - np.sum((y_j - y_j_pred)**2) / np.sum((y_j - np.mean(y_j))**2)
    vif = 1 / (1 - r2_j + 1e-10)
    sev = "SEVERE" if vif > 10 else ("MODERATE" if vif > 5 else "OK")
    print(f"    {feature_names[j]:12s}  VIF={vif:7.2f}  {sev}")

# Cook's distance
H_test = X_test_d @ np.linalg.inv(X_test_d.T @ X_test_d) @ X_test_d.T
h_test = np.diag(H_test)
RSS_test = np.sum(residuals**2)
RSE_test = np.sqrt(RSS_test / (n_test - p - 1))
stud_resid = residuals / (RSE_test * np.sqrt(1 - h_test))
cooks_d = (stud_resid**2 / (p+1)) * (h_test / (1 - h_test))

print(f"\n  Influential points (Cook\'s D > 4/n): {np.where(cooks_d > 4/n_test)[0][:5].tolist()}")
print(f"  High leverage: {np.where(h_test > 2*(p+1)/n_test)[0][:5].tolist()}")

# Coefficient interpretation
print("\n--- Coefficient Interpretation ---")
print("  Each Betaj = change in MedHouseVal for 1 sigma increase in Xj")
for i, name in enumerate(feature_names):
    print(f"    {name:12s}: {beta_ne[i+1]:+.4f}")

# ========== PLOTS ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Method Comparison: R²
methods = ['Normal Eq', 'Batch GD', 'SGD', 'Mini-Batch']
r2_scores = [test_r2, r2_gd, r2_sgd, r2_mb]
colors = ['steelblue', 'seagreen', 'orange', 'crimson']
bars = axes[0, 0].bar(methods, r2_scores, color=colors, edgecolor='black')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('Method Comparison (Test R²)')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].grid(alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.4f}', ha='center', fontsize=9, fontweight='bold')

# 2. Actual vs Predicted (Normal Equation)
axes[0, 1].scatter(y_test, y_test_pred, color='steelblue', s=5, alpha=0.4)
minv = min(min(y_test), min(y_test_pred))
maxv = max(max(y_test), max(y_test_pred))
axes[0, 1].plot([minv, maxv], [minv, maxv], '--', color='crimson', linewidth=2, label='Ideal')
axes[0, 1].set_xlabel('Actual MedHouseVal')
axes[0, 1].set_ylabel('Predicted MedHouseVal')
axes[0, 1].set_title(f'Actual vs Predicted (Normal Eq, R²={test_r2:.4f})')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Coefficients Bar Chart
axes[1, 0].barh(feature_names, beta_ne[1:], color='steelblue', edgecolor='black')
axes[1, 0].axvline(x=0, color='black', linewidth=0.5)
axes[1, 0].set_xlabel('Coefficient Value')
axes[1, 0].set_title('Standardized Coefficients')
axes[1, 0].grid(alpha=0.3, axis='x')

# 4. Residuals Histogram + Density
axes[1, 1].hist(residuals, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
from scipy import stats
x_grid = np.linspace(min(residuals), max(residuals), 100)
mu, std = np.mean(residuals), np.std(residuals)
axes[1, 1].plot(x_grid, stats.norm.pdf(x_grid, mu, std), 'r-', linewidth=2, label='Normal Fit')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title(f'Residuals Distribution\nMean={mu:.3f}, Std={std:.3f}')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/10_housing_demo.png', dpi=150)
plt.close()
print("[Plot saved: plots/10_housing_demo.png]")
