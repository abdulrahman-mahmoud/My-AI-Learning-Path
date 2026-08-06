import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def minibatch_gd(X, y, learning_rate=0.01, epochs=100, batch_size=32):
    n, p = X.shape
    beta = np.zeros(p)
    cost_history = []
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X[indices[start:end]]
            y_batch = y[indices[start:end]]
            b = X_batch.shape[0]
            probs = sigmoid(X_batch @ beta)
            gradient = (1 / b) * X_batch.T @ (probs - y_batch)
            beta -= learning_rate * gradient
        probs = sigmoid(X @ beta)
        cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        cost_history.append(cost)
    return beta, cost_history

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.column_stack((np.ones(n), np.random.randn(n, 2)))
    y = (X[:, 1] - X[:, 2] + 0.2 * np.random.randn(n) > 0).astype(float)
    beta, costs = minibatch_gd(X, y, learning_rate=0.1, epochs=100, batch_size=16)
    print(f"Beta: {beta}")
    print(f"Final cost: {costs[-1]:.6f}")
