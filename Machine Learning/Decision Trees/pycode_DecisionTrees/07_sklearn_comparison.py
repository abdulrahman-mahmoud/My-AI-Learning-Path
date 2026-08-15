#----------------------------
#  07: scikit-learn comparison
#  Check the from-scratch trees against sklearn's DecisionTree on the
#  same data and show that the concepts map 1:1 (same greedy algorithm,
#  same criterion). NOTE: sklearn is Cython-optimized and has many extra
#  features; the custom code is educational, not a re-implementation of
#  everything sklearn does internally.
#----------------------------

import os
import importlib.util
import numpy as np
from sklearn.tree import DecisionTreeRegressor as SKReg
from sklearn.tree import DecisionTreeClassifier as SKClf

# our from-scratch implementations (same folder)
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

dt_reg = load("dt_reg", "04_decision_tree_regressor.py")
dt_clf = load("dt_clf", "05_decision_tree_classifier.py")
MyReg = dt_reg.DecisionTreeRegressor
MyClf = dt_clf.DecisionTreeClassifier

rng = np.random.default_rng(1)

#------------------------------
#  Regression comparison
#------------------------------
n = 200
X = rng.uniform(0, 6, (n, 2))
y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 0] * X[:, 1] + rng.normal(0, 0.5, n)

for d in (1, 2, 3):
    mine = MyReg(max_depth=d, min_samples_split=2).fit(X, y)
    sk   = SKReg(max_depth=d, min_samples_split=2, criterion='squared_error').fit(X, y)
    p_mine, p_sk = mine.predict(X), sk.predict(X)
    print(f"regression max_depth={d}:  my-MSE={np.mean((y-p_mine)**2):.4f}"
          f"   sk-MSE={np.mean((y-p_sk)**2):.4f}"
          f"   corr(my,sk)={np.corrcoef(p_mine,p_sk)[0,1]:.4f}")

#------------------------------
#  Classification comparison
#------------------------------
Xc = np.vstack([rng.normal(loc=[4,4], scale=1.0, size=(100,2)),
                rng.normal(loc=[7,7], scale=1.0, size=(100,2))])
yc = np.array([0]*100 + [1]*100)

for criterion in ('gini', 'entropy'):
    mine = MyClf(criterion=criterion, max_depth=3, min_samples_split=2).fit(Xc, yc)
    sk   = SKClf(criterion=criterion, max_depth=3, min_samples_split=2).fit(Xc, yc)
    acc_mine = np.mean(mine.predict(Xc) == yc)
    acc_sk   = np.mean(sk.predict(Xc) == yc)
    print(f"classification criterion={criterion}:  my-acc={acc_mine:.3f}"
          f"   sk-acc={acc_sk:.3f}")

print("\nThe two implementations agree closely on the same data because they")
print("run the same greedy recursive-binary-splitting algorithm.")
print("\nDifferences: sklearn is compiled (Cython), offers sample/class weights,")
print("max_features, cost-complexity paths, and missing-value support, and runs")
print("far faster on large data. See the official sklearn Decision Trees docs.")
