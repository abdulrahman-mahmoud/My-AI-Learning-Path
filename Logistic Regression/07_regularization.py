import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def fit_ridge(X, y, ridge_lambda=0.01, learning_rate=0.01, iterations=1000):
    n, p = X.shape
    beta = np.zeros(p)
    for i in range(iterations):
        probs = sigmoid(X @ beta)
        gradient = (1 / n) * X.T @ (probs - y)
        gradient[1:] += 2 * ridge_lambda * beta[1:]
        beta -= learning_rate * gradient
    return beta

def fit_lasso(X, y, lasso_lambda=0.01, learning_rate=0.01, iterations=1000):
    n, p = X.shape
    beta = np.zeros(p)
    for i in range(iterations):
        probs = sigmoid(X @ beta)
        gradient = (1 / n) * X.T @ (probs - y)
        gradient[1:] += lasso_lambda * np.sign(beta[1:])
        beta -= learning_rate * gradient
    return beta

if __name__ == "__main__":
    np.random.seed(42)
    n, p = 100, 10
    X = np.column_stack((np.ones(n), np.random.randn(n, p - 1)))
    beta_true = np.zeros(p)
    beta_true[1:4] = [1.0, -0.5, 0.3]
    y = (sigmoid(X @ beta_true) > 0.5).astype(float)
    beta_ridge = fit_ridge(X, y, ridge_lambda=0.05)
    beta_lasso = fit_lasso(X, y, lasso_lambda=0.01)
    print("True beta (first 5):", beta_true[:5])
    print("Ridge beta (first 5):", np.round(beta_ridge[:5], 4))
    print("Lasso beta (first 5):", np.round(beta_lasso[:5], 4))
