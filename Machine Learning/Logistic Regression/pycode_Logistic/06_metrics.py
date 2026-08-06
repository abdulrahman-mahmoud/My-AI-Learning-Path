import numpy as np

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def precision(y_true, y_pred):
    TP, _, FP, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP + 1e-10)

def recall(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN + 1e-10)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-10)

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    print(f"TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall: {recall(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
