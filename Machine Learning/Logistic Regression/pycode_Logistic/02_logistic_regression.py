import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_proba(X, beta):
    return sigmoid(X @ beta)

def predict(X, beta, threshold=0.5):
    return (predict_proba(X, beta) >= threshold).astype(int)

def fit(X, y, learning_rate=0.01, iterations=1000):
    n, p = X.shape
    beta = np.zeros(p)
    cost_history = []
    for i in range(iterations):
        z = X @ beta
        probs = sigmoid(z)
        cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
        gradient = (1 / n) * X.T @ (probs - y)
        beta -= learning_rate * gradient
        cost_history.append(cost)
    return beta, cost_history

if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = (0.5 * x1 - 0.3 * x2 + 0.1 * np.random.randn(n) > 0).astype(float)
    X = np.column_stack((np.ones(n), x1, x2))
    beta, costs = fit(X, y, learning_rate=0.1, iterations=2000)
    print(f"Beta: {beta}")
    print(f"Final cost: {costs[-1]:.6f}")
    y_pred = predict(X, beta)
    print(f"Accuracy: {np.mean(y_pred == y):.4f}")
