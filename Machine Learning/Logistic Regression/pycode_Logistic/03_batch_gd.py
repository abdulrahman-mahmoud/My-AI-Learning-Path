import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def batch_gd(X, y, learning_rate=0.01, iterations=1000):
    n, p = X.shape
    beta = np.zeros(p)
    cost_history = []
    for i in range(iterations):
        probs = sigmoid(X @ beta)
        cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        gradient = (1 / n) * X.T @ (probs - y)
        beta -= learning_rate * gradient
        cost_history.append(cost)
    return beta, cost_history

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.column_stack((np.ones(n), np.random.randn(n, 2)))
    y = (X[:, 1] - X[:, 2] + 0.2 * np.random.randn(n) > 0).astype(float)
    beta, costs = batch_gd(X, y, learning_rate=0.1, iterations=3000)
    print(f"Beta: {beta}")
    print(f"Converged: {costs[-1]:.6f}")
