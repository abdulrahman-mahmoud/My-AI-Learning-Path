#----------------------------
#  Simple Linear Regression
#  Beta = (XTX)⁻¹ XTy
#  y_hat = XBeta
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([30, 35, 38, 42, 47, 51, 55, 60, 65, 70])

# Add intercept column (Beta0)
X = np.column_stack((np.ones(len(x)), x))

# Normal Equation: Beta = (XTX)⁻¹ XTy
beta = np.linalg.solve(X.T @ X, X.T @ y)

print("Coefficients:")
print("  Intercept (Beta0):", beta[0])
print("  Slope     (Beta1):", beta[1])
print()
print("Equation: y_hat =", round(beta[0], 4), "+", round(beta[1], 4), "* x")

# Predictions
y_pred = X @ beta
print("\nPredictions:", y_pred)

# Evaluation metrics
MAE = np.mean(np.abs(y - y_pred))
MSE = np.mean((y - y_pred) ** 2)
RMSE = np.sqrt(MSE)
R2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print("\nEvaluation Metrics:")
print("  MAE :", round(MAE, 4))
print("  MSE :", round(MSE, 4))
print("  RMSE:", round(RMSE, 4))
print("  R^2  :", round(R2, 4))

# Predict for new value
new_x = np.array([1, 12])
prediction = new_x @ beta
print("\nPrediction for x=12:", round(prediction, 4))

# ========== PLOTS ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Scatter + Regression Line
axes[0].scatter(x, y, color='steelblue', s=60, label='Data')
x_line = np.linspace(min(x), max(x), 100)
y_line = beta[0] + beta[1] * x_line
axes[0].plot(x_line, y_line, color='crimson', linewidth=2, label=f'$\\hat{{y}} = {beta[0]:.2f} + {beta[1]:.2f}x$')
for i in range(len(x)):
    axes[0].plot([x[i], x[i]], [y[i], y_pred[i]], color='gray', linewidth=1, alpha=0.6)
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary')
axes[0].set_title('Simple Linear Regression Fit')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Actual vs Predicted
axes[1].scatter(y, y_pred, color='steelblue', s=60)
min_val = min(min(y), min(y_pred))
max_val = max(max(y), max(y_pred))
axes[1].plot([min_val, max_val], [min_val, max_val], '--', color='crimson', linewidth=2, label='Ideal')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
axes[1].set_title(f'Actual vs Predicted  (R² = {R2:.4f})')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 3. Residuals
axes[2].scatter(y_pred, y - y_pred, color='steelblue', s=60)
axes[2].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
axes[2].set_xlabel('Fitted Values')
axes[2].set_ylabel('Residuals')
axes[2].set_title('Residual Plot')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/01_simple_linear.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/01_simple_linear.png]")
