import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sgd(X, y, alpha=0.01, epochs=100, decay=0.01):
    n, p = X.shape
    beta = np.zeros(p)
    cost_history = []
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for i in range(n):
            idx = indices[i]
            xi = X[idx:idx+1]
            yi = y[idx:idx+1]
            prob = sigmoid(xi @ beta)
            gradient = xi.T @ (prob - yi)
            lr = alpha / (1 + decay * (epoch * n + i))
            beta -= lr * gradient.flatten()
        probs = sigmoid(X @ beta)
        cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        cost_history.append(cost)
    return beta, cost_history

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.column_stack((np.ones(n), np.random.randn(n, 2)))
    y = (X[:, 1] - X[:, 2] + 0.2 * np.random.randn(n) > 0).astype(float)
    beta, costs = sgd(X, y, alpha=0.1, epochs=50)
    print(f"Beta: {beta}")
    print(f"Final cost: {costs[-1]:.6f}")
