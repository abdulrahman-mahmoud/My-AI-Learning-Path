#----------------------------
#  08: Full Decision Tree demo (regression + classification)
#  Regression : diabetes progression (continuous target)
#  Classification : Iris (3 classes)
#  Uses the from-scratch trees, evaluates train/test, and compares to
#  the scikit-learn trees as a sanity check.
#----------------------------

import os, importlib.util
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as SKReg
from sklearn.tree import DecisionTreeClassifier as SKClf

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dt_reg = load("dt_reg", "04_decision_tree_regressor.py")
dt_clf = load("dt_clf", "05_decision_tree_classifier.py")
MyReg, MyClf = dt_reg.DecisionTreeRegressor, dt_clf.DecisionTreeClassifier

def mse(a, b): return np.mean((a - b) ** 2)

#========================================================
#  1) REGRESSION: diabetes progression (small, so the
#     educational tree stays fast)
#========================================================
print("========== REGRESSION: diabetes progression ==========")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

for depth in (3, 5):
    mine = MyReg(max_depth=depth, min_samples_split=4).fit(Xtr, ytr)
    sk   = SKReg(max_depth=depth, min_samples_split=4, criterion='squared_error').fit(Xtr, ytr)
    print(f"  max_depth={depth}:")
    print(f"    my  train MSE = {mse(ytr, mine.predict(Xtr)):.4f}  test MSE = {mse(yte, mine.predict(Xte)):.4f}")
    print(f"    sk  train MSE = {mse(ytr, sk.predict(Xtr)):.4f}   test MSE = {mse(yte, sk.predict(Xte)):.4f}")

#========================================================
#  2) CLASSIFICATION: Iris (3 classes)
#========================================================
print("\n========== CLASSIFICATION: Iris (3 classes) ==========")
iris = load_iris()
Xc, yc = iris.data, iris.target
Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.3, random_state=0)

for criterion in ('gini', 'entropy'):
    mine = MyClf(criterion=criterion, max_depth=3, min_samples_split=2).fit(Xctr, yctr)
    sk   = SKClf(criterion=criterion, max_depth=3, min_samples_split=2).fit(Xctr, yctr)
    print(f"  criterion={criterion}:")
    print(f"    my  train acc = {np.mean(mine.predict(Xctr)==yctr):.3f}  test acc = {np.mean(mine.predict(Xcte)==ycte):.3f}")
    print(f"    sk  train acc = {np.mean(sk.predict(Xctr)==yctr):.3f}   test acc = {np.mean(sk.predict(Xcte)==ycte):.3f}")

probs = mine.predict_proba(Xcte[:3])
print("\n  First 3 test-sample predict_proba (my classifier):")
for i, p in zip(Xcte[:3], probs):
    print(f"    probs = {np.round(p, 3)}")

print("\nDone. The from-scratch trees match the library qualitatively and")
print("illustrate the whole pipeline: fit, predict, predict_proba, evaluate.")
