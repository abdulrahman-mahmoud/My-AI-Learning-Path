import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

if __name__ == "__main__":
    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)
    print(f"sigmoid(0) = {sigmoid(0):.4f}")
    print(f"sigmoid(10) = {sigmoid(10):.6f}")
    print(f"sigmoid(-10) = {sigmoid(-10):.6f}")
