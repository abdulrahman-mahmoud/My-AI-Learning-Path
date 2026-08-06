#----------------------------
#  Mini-Batch Gradient Descent
#  Beta := Beta - α * (2/b)XpT(XpBeta - yp)
#  Uses BATCH of b samples per step
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# Data
np.random.seed(42)
n_samples = 100
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)
y = 5 + 3*x1 - 2*x2 + np.random.randn(n_samples) * 0.5

X = np.column_stack((np.ones(n_samples), x1, x2))
n, p = X.shape

# Hyperparameters
alpha = 0.1
epochs = 100
batch_size = 16

beta = np.zeros(p)
cost_history = []

for epoch in range(epochs):
    # Shuffle
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Process mini-batches
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        b = X_batch.shape[0]

        # Gradient for this batch
        residuals = X_batch @ beta - y_batch
        gradient = (2 / b) * X_batch.T @ residuals

        # Update
        beta -= alpha * gradient

    cost = np.mean((y - X @ beta) ** 2)
    cost_history.append(cost)

print("Mini-Batch Gradient Descent Results:")
print(f"  Batch size: {batch_size}")
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

# True coefficients were: Beta0=5, Beta1=3, Beta2=-2
print("\n  True coefficients: Beta0=5, Beta1=3, Beta2=-2")
print("  Estimated:        Beta0=", round(beta[0], 2), "Beta1=", round(beta[1], 2), "Beta2=", round(beta[2], 2))

# ========== PLOTS ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 1. Cost Convergence
axes[0].plot(cost_history, color='steelblue', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Cost')
axes[0].set_title(f'Mini-Batch GD: Cost (batch={batch_size})')
axes[0].grid(alpha=0.3)

# 2. True vs Estimated coefficients
true_coefs = [5, 3, -2]
est_coefs = [beta[0], beta[1], beta[2]]
x_pos = np.arange(3)
width = 0.35
axes[1].bar(x_pos - width/2, true_coefs, width, label='True', color='steelblue', edgecolor='black')
axes[1].bar(x_pos + width/2, est_coefs, width, label='Estimated', color='crimson', edgecolor='black', alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([r'$\beta_0$', r'$\beta_1$', r'$\beta_2$'])
axes[1].set_ylabel('Coefficient Value')
axes[1].set_title('True vs Estimated Coefficients')
axes[1].legend()
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/05_minibatch_gd.png', dpi=150)
plt.close()
print("[Plot saved: plots/05_minibatch_gd.png]")
