#----------------------------
#  Stochastic Gradient Descent
#  Beta := Beta - α * 2 * xi(xiTBeta - yi)
#  Uses ONE random sample per step
#----------------------------
from sklearn.linear_model import SGDClassifier

import numpy as np
import matplotlib.pyplot as plt

# Data
x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x2 = np.array([2, 4, 1, 3, 5, 2, 4, 3])
y  = np.array([10, 20, 15, 25, 30, 22, 28, 32])

X = np.column_stack((np.ones(len(x1)), x1, x2))
n, p = X.shape

# Hyperparameters
alpha = 0.01
epochs = 50

# Initialize
beta = np.zeros(p)
cost_history = []

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(n):
        # Use one sample
        xi = X_shuffled[i:i+1]
        yi = y_shuffled[i:i+1]

        # Gradient: 2 * xi(xiTBeta - yi)
        pred = xi @ beta
        gradient = 2 * xi.T @ (pred - yi)

        # Update with decaying learning rate
        lr = alpha / (1 + 0.01 * (epoch * n + i))
        beta -= lr * gradient.flatten()

    # Track cost per epoch
    cost = np.mean((y - X @ beta) ** 2)
    cost_history.append(cost)

print("Stochastic Gradient Descent Results:")
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
print(f"  Epochs: {len(cost_history)}")
print(f"  Final cost: {cost_history[-1]:.6f}")
print(f"  First 5 costs: {[round(c, 4) for c in cost_history[:5]]}")

# ========== PLOTS ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1. Cost Convergence (noisy)
axes[0].plot(cost_history, color='steelblue', linewidth=1.5, alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Cost')
axes[0].set_title('SGD: Cost Convergence (Noisy)')
axes[0].grid(alpha=0.3)

# 2. Zoomed (epochs 5+ to see convergence better)
axes[1].plot(range(5, len(cost_history)), cost_history[5:], color='steelblue', linewidth=1.5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Cost')
axes[1].set_title('SGD: Cost (Epochs 5+)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plots/04_sgd_convergence.png', dpi=150)
plt.close()
print("[Plot saved: plots/04_sgd_convergence.png]")
