# pycode_DecisionTrees

This folder has the Python scripts for decision trees.

Each file is small and focused on one part of the topic.

## Files

| File | Why it exists |
|---|---|
| `01_impurity_gini_entropy.py` | Gini impurity and entropy from scratch, with the example nodes |
| `02_best_split_regression.py` | Finding the best regression split on the tiny dataset |
| `03_best_split_classification.py` | Finding the best classification split with weighted Gini |
| `04_decision_tree_regressor.py` | `DecisionTreeRegressor` from scratch with a walkthrough |
| `05_decision_tree_classifier.py` | `DecisionTreeClassifier` from scratch with decision regions |
| `06_depth_overfitting.py` | Depth vs. error; the bias-variance tradeoff |
| `07_sklearn_comparison.py` | Compare the custom trees to scikit-learn |
| `08_demo.py` | Full regression + classification demo (diabetes, Iris) |

## How it works

The numbering shows the order of the learning path. It starts with the
impurity measures, moves into split selection, then builds the full
regressor and classifier from scratch, then studies overfitting, and
finally checks the result against scikit-learn.

The core implementations are `04_decision_tree_regressor.py` and
`05_decision_tree_classifier.py`. The other scripts import these two.

## Navigation

- [Decision Trees](../README.md)
- [Documentation](../docs/README.md)
- [Plots](../plots/README.md)
