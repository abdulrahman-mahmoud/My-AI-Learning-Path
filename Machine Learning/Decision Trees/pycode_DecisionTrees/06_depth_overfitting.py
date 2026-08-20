#----------------------------
#  06: Depth vs. overfitting (bias-variance tradeoff)
#  Grows DecisionTreeRegressor at many max_depth values and plots
#  training vs. test error. Deep trees -> near-zero train error,
#  poor test error (overfitting). Shallow trees -> higher bias.
#----------------------------

import numpy as np
import matplotlib.pyplot as plt

# import the from-scratch regressor defined in the same folder
import importlib.util, os
spec = importlib.util.spec_from_file_location("dt_reg", os.path.join(os.path.dirname(__file__), "04_decision_tree_regressor.py"))
dt_reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt_reg)
DecisionTreeRegressor = dt_reg.DecisionTreeRegressor

#------------------------------
#  Data: noisy sine (non-linear relationship)
#------------------------------
def f(x):
    return np.sin(x)

rng = np.random.default_rng(0)
n = 200
X = rng.uniform(-3, 3, n)
y = f(X) + rng.normal(0, 0.3, n)

# train/test split
idx = rng.permutation(n)
X_train, y_train = X[idx[:n // 2]], y[idx[:n // 2]]
X_test,  y_test  = X[idx[n // 2:]], y[idx[n // 2:]]

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#------------------------------
#  Fit at several depths
#------------------------------
depths = np.arange(1, 15)
train_err, test_err = [], []
bag = []
for d in depths:
    t = DecisionTreeRegressor(max_depth=int(d), min_samples_split=2)
    t.fit(X_train.reshape(-1, 1), y_train)
    bag.append(t)
    train_err.append(mse(y_train, t.predict(X_train.reshape(-1, 1))))
    test_err.append(mse(y_test,  t.predict(X_test.reshape(-1, 1))))

print("Depth vs. error (bias-variance tradeoff)\n")
print(f"{'depth':>6}{'train MSE':>12}{'test MSE':>12}")
for d, tr, te in zip(depths, train_err, test_err):
    print(f"{d:>6}{tr:>12.4f}{te:>12.4f}")

best_d = depths[int(np.argmin(test_err))]
print(f"\nBest test error at max_depth = {best_d}")

#------------------------------
#  Plot
#------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: train vs test error vs depth
axes[0].plot(depths, train_err, 'o-', color='steelblue', label='Training MSE')
axes[0].plot(depths, test_err,  's-', color='crimson',   label='Test MSE')
axes[0].axvline(best_d, color='gray', linestyle=':', linewidth=1.5,
                label=f'best test depth = {best_d}')
axes[0].set_xlabel('max_depth')
axes[0].set_ylabel('MSE')
axes[0].set_title('Depth vs. error: deep trees overfit')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: shallow vs deep fits
x_draw = np.linspace(-3, 3, 300)
shallow = bag[0]      # depth 1
deep    = bag[6]      # depth 7
axes[1].scatter(X_train, y_train, color='steelblue', s=25, alpha=0.6, label='Training data')
axes[1].plot(x_draw, f(x_draw), 'k--', label='True f(x) = sin(x)')
axes[1].plot(x_draw, shallow.predict(x_draw.reshape(-1, 1)), '-', color='green', label=f'depth 1 (bias)')
axes[1].plot(x_draw, deep.predict(x_draw.reshape(-1, 1)),    '-', color='crimson', label=f'depth 7 (variance)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('Shallow (underfits) vs deep (overfits)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('06_depth_overfitting.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/06_depth_overfitting.png]")
