#----------------------------
#  Multiple Linear Regression
#  Beta = (XTX)⁻¹ XTy
#  y_hat = Beta0 + Beta1X1 + Beta2X2 + ...
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Data: HouseAge, Bedrooms -> Price
x1 = np.array([5, 10, 15, 20, 25, 30])
x2 = np.array([2, 3, 3, 4, 4, 5])
y  = np.array([150, 180, 210, 250, 280, 320])

# Build design matrix: [1, X1, X2]
X = np.column_stack((np.ones(len(x1)), x1, x2))

# Normal Equation
beta = np.linalg.solve(X.T @ X, X.T @ y)

print("Coefficients:")
print("  Intercept (Beta0):", round(beta[0], 4))
print("  HouseAge  (Beta1):", round(beta[1], 4))
print("  Bedrooms  (Beta2):", round(beta[2], 4))
print()
print("Equation: y_hat =", round(beta[0], 4), "+", round(beta[1], 4), "* Age +", round(beta[2], 4), "* Bedrooms")

# Predictions
y_pred = X @ beta
print("\nPredictions:", y_pred)

# Metrics
MAE = np.mean(np.abs(y - y_pred))
MSE = np.mean((y - y_pred) ** 2)
RMSE = np.sqrt(MSE)
R2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print("\nEvaluation:")
print("  MAE :", round(MAE, 4))
print("  MSE :", round(MSE, 4))
print("  RMSE:", round(RMSE, 4))
print("  R^2  :", round(R2, 4))

# Predict for new house: Age=18, Bedrooms=3
new_x = np.array([1, 18, 3])
pred = new_x @ beta
print("\nPrediction (Age=18, Bedrooms=3):", round(pred, 2))

# ========== PLOTS ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Actual vs Predicted
axes[0].scatter(y, y_pred, color='steelblue', s=80)
minv = min(min(y), min(y_pred))
maxv = max(max(y), max(y_pred))
axes[0].plot([minv, maxv], [minv, maxv], '--', color='crimson', linewidth=2, label='Ideal')
axes[0].set_xlabel('Actual Price')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Actual vs Predicted  (R² = {R2:.4f})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Residuals vs Fitted
axes[1].scatter(y_pred, y - y_pred, color='steelblue', s=80)
axes[1].axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Fitted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(alpha=0.3)

# 3. Feature contributions (coefficient magnitude)
features = ['Intercept', 'HouseAge', 'Bedrooms']
colors = ['gray', 'steelblue', 'steelblue']
axes[2].bar(features, beta, color=colors, edgecolor='black')
axes[2].axhline(y=0, color='black', linewidth=0.5)
axes[2].set_ylabel('Coefficient Value')
axes[2].set_title('Model Coefficients')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/02_multiple_linear.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/02_multiple_linear.png]")
