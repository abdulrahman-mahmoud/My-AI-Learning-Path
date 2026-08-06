#----------------------------
#  Batch Gradient Descent
#  Beta := Beta - α * (2/n)XT(XBeta - y)
#  Uses ALL samples per step
#----------------------------
import numpy as np

# Data
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x2 = np.array([2, 4, 1, 3, 5, 2, 4, 3])
y  = np.array([10, 20, 15, 25, 30, 22, 28, 32])

X = np.column_stack((np.ones(len(x1)), x1, x2))
n, p = X.shape

# Hyperparameters
alpha = 0.01
iterations = 1000

# Initialize
beta = np.zeros(p)
cost_history = []

for i in range(iterations):
    # Gradient: (2/n) * XT(XBeta - y)
    residuals = X @ beta - y
    gradient = (2 / n) * X.T @ residuals

    # Update
    beta -= alpha * gradient

    # Track cost
    cost = np.mean((y - X @ beta) ** 2)
    cost_history.append(cost)

print("Batch Gradient Descent Results:")
print("  Intercept:", round(beta[0], 4))
print("  Beta1:", round(beta[1], 4))
print("  Beta2:", round(beta[2], 4))

y_pred = X @ beta
MAE = np.mean(np.abs(y - y_pred))
MSE = np.mean((y - y_pred) ** 2)
RMSE = np.sqrt(MSE)
R2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print("\n  MAE :", round(MAE, 4))
print("  MSE :", round(MSE, 4))
print("  RMSE:", round(RMSE, 4))
print("  R^2  :", round(R2, 4))
print(f"  Iterations: {len(cost_history)}")
print(f"  Final cost: {cost_history[-1]:.6f}")
print(f"  First 5 costs: {[round(c, 4) for c in cost_history[:5]]}")
