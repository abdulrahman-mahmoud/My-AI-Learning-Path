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
    for i in range(iterations):
        probs = sigmoid(X @ beta)
        gradient = (1 / n) * X.T @ (probs - y)
        beta -= learning_rate * gradient
    return beta

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN

if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    log_odds = -2 + 0.8 * x1 + 1.2 * x2
    prob = sigmoid(log_odds)
    y = (np.random.rand(n) < prob).astype(float)

    n_train = int(0.8 * n)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X = np.column_stack((np.ones(n), x1, x2))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_mean = np.mean(X_train[:, 1:], axis=0)
    X_std = np.std(X_train[:, 1:], axis=0)
    X_train[:, 1:] = (X_train[:, 1:] - X_mean) / X_std
    X_test[:, 1:] = (X_test[:, 1:] - X_mean) / X_std

    beta = fit(X_train, y_train, learning_rate=0.1, iterations=2000)
    y_pred = predict(X_test, beta)
    TP, TN, FP, FN = confusion_matrix(y_test, y_pred)
    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP + 1e-10)
    rec = TP / (TP + FN + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion: TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"Learned beta: {beta}")
