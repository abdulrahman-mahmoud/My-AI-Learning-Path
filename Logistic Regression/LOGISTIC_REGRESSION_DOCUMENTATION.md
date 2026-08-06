# Logistic Regression Implementation Documentation

> A comprehensive guide bridging theory from *An Introduction to Statistical Learning* (ISL) and *The Elements of Statistical Learning* (ESL) with a practical NumPy-based implementation.

---

## 1. Introduction

### What is Logistic Regression?

Logistic regression is a supervised learning method for binary classification. It models the probability that a response variable $Y$ belongs to a particular class given one or more predictor variables $X_1, X_2, \dots, X_p$:

$$P(Y = 1 \mid X) = \sigma(\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p)$$

where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic (sigmoid) function. Despite its name, logistic regression is a **classifier**, not a regressor — it estimates class probabilities, then applies a threshold to make hard classifications.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3 Logistic Regression  
> Pages: 138

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 119

### Why Logistic Regression is Important in Machine Learning

- It is **simple, interpretable, and fast** to train.
- It provides **well-calibrated probability estimates**, not just hard labels.
- It serves as the **baseline classifier** against which more complex models (random forests, SVMs, neural networks) are compared (ISL, p. 138).
- It is a building block for **deep learning**: a logistic regression model is identical to a single-neuron, single-layer neural network with sigmoid activation.
- It is a special case of **generalized linear models** (GLMs), linking classification to the broader exponential family framework (ESL, p. 119).

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.1 Overview of Classification  
> Pages: 135

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 119

### Prediction vs. Classification

Logistic regression serves distinct goals depending on whether the output is a probability or a hard label:

| Goal | Output | Use Case |
|------|--------|----------|
| **Probability estimation** | $P(Y=1 \mid X) \in [0, 1]$ | Risk scoring, ranking, calibrated confidence |
| **Hard classification** | $\hat{y} \in \{0, 1\}$ | Automated decisions, binary action triggers |
| **Inference** | $\hat{\beta}_j$, $z$-statistics, $p$-values | Understanding which predictors drive the outcome |

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.3 Making Predictions  
> Pages: 142-143

### Purpose of This Implementation

The implementation provides a from-scratch logistic regression toolkit using only NumPy. It includes:

- **Sigmoid function** (probability mapping)
- **Log-loss / binary cross-entropy** (MLE-based cost function)
- **Batch, Stochastic, and Mini-Batch Gradient Descent** (iterative optimization)
- **Prediction** (probability output + hard classification)
- **Evaluation metrics** (accuracy, precision, recall, F1, confusion matrix)
- **Regularization** (Ridge and Lasso penalties)

### How the Algorithm Works at a High Level

1. **Specify the model**: $P(Y=1|X) = \sigma(X\beta)$
2. **Choose a cost function**: Log-loss (negative log-likelihood)
3. **Optimize**: Find $\hat{\beta}$ that minimizes log-loss via gradient descent
4. **Predict probabilities**: $\hat{p} = \sigma(X\hat{\beta})$
5. **Classify**: $\hat{y} = \mathbb{1}[\hat{p} \geq 0.5]$
6. **Evaluate**: Compare $\hat{y}$ against true $y$ using classification metrics

---

## 2. Mathematical Foundation

### 2.1 Logistic Regression Model

#### Why Linear Regression Fails for Classification

Linear regression models a continuous response. When applied to a binary outcome ($Y \in \{0,1\}$), it can produce predictions outside $[0,1]$, is sensitive to class imbalance, and violates the constant-variance assumption. Logistic regression solves this by wrapping the linear predictor in the sigmoid function, constraining output to $(0,1)$ (ISL, p. 136-137).

#### Simple Logistic Regression

For a single predictor $X$:

$$p(X) = P(Y=1 \mid X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} \tag{4.2, ISL}$$

The **odds** are:

$$\frac{p(X)}{1-p(X)} = e^{\beta_0 + \beta_1 X}$$

Taking the logarithm gives the **log-odds** (logit):

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X \tag{4.3, ISL}$$

The logit is **linear in $X$**; the probability $p(X)$ is not.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.1 The Logistic Model  
> Pages: 138-140

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 119-120

#### Multiple Logistic Regression

For $p$ predictors:

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p \tag{4.7, ISL}$$

In matrix form, letting $z = X\beta$:

$$P(Y=1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

where:
- $y$ is the $n \times 1$ vector of binary responses ($y_i \in \{0,1\}$)
- $X$ is the $n \times (p+1)$ design matrix (first column is 1s for intercept)
- $\beta$ is the $(p+1) \times 1$ vector of parameters

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.4 Multiple Logistic Regression  
> Pages: 144-145

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 119-120

**Connection to Implementation:**

In `02_logistic_regression.py`, the linear predictor and sigmoid are computed as:

```python
z = X @ beta
probabilities = 1 / (1 + np.exp(-z))
```

The design matrix is constructed identically to linear regression:

```python
X = np.column_stack((np.ones(n), x1, x2))
```

### 2.2 Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Range:** $(0, 1)$ for all $z \in \mathbb{R}$
- **Symmetry:** $\sigma(-z) = 1 - \sigma(z)$
- **Derivative:** $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ — computationally convenient for gradient descent
- **Decision boundary:** $\sigma(z) = 0.5$ when $z = 0$, i.e., $X\beta = 0$. This is a linear decision boundary in the predictor space.

![Sigmoid Function](plots/fig_sigmoid.png)
*The logistic (sigmoid) function maps the entire real line to $(0,1)$. At $z=0$, $\sigma(0)=0.5$ defines the decision boundary. Asymptotes at 0 and 1 ensure the output is always a valid probability.*

**Implementation (`01_sigmoid.py`):**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

For numerical stability, a clipped version avoids overflow:

```python
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.1 The Logistic Model  
> Pages: 138-139

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 119-120

### 2.3 Maximum Likelihood Estimation

Unlike linear regression (which minimizes RSS), logistic regression is fit by **maximum likelihood**. The likelihood for $n$ independent observations is:

$$L(\beta) = \prod_{i=1}^{n} p(x_i)^{y_i} (1 - p(x_i))^{1 - y_i}$$

where $p(x_i) = P(Y_i = 1 \mid X = x_i)$.

Taking the logarithm gives the **log-likelihood**:

$$\ell(\beta) = \sum_{i=1}^{n} \big[y_i \log p(x_i) + (1 - y_i) \log(1 - p(x_i))\big]$$

The goal is to find $\hat{\beta}$ that maximizes $\ell(\beta)$, equivalently minimizes the negative log-likelihood (ESL, p. 120; ISL, p. 141).

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.2 Estimating the Regression Coefficients  
> Pages: 140-142

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4.1 Fitting Logistic Regression Models  
> Pages: 120-121

### 2.4 Loss Function

The **binary cross-entropy loss** (also called log-loss) for a single observation is:

$$J_i(\beta) = -y_i \log(p_i) - (1 - y_i) \log(1 - p_i)$$

which is the negative log-likelihood divided by $n$ (equivalent for optimization). For the full dataset:

$$J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \big[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\big]$$

**Why log-loss instead of MSE:**
- MSE is non-convex in the parameters for logistic regression (due to the sigmoid), making gradient descent prone to local minima.
- Log-loss is **convex** in $\beta$ for logistic regression, guaranteeing that gradient descent finds the global optimum (ESL, p. 120).
- Log-loss follows directly from MLE principles.

**How predictions map to loss:**
- If $y_i = 1$: $J_i = -\log(p_i)$. Loss is small when $p_i$ is near 1, large when $p_i$ is near 0.
- If $y_i = 0$: $J_i = -\log(1 - p_i)$. Loss is small when $p_i$ is near 0, large when $p_i$ is near 1.

**Implementation:**

```python
cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
```

The `1e-10` epsilon prevents $\log(0)$, which would produce $-\infty$.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.2 Estimating the Regression Coefficients  
> Pages: 140-142

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4.1 Fitting Logistic Regression Models  
> Pages: 120-121

---

## 3. Optimization

### 3.1 Gradient Derivation

The gradient of the log-loss $J(\beta)$ with respect to $\beta$ is derived as follows.

Let $p_i = \sigma(x_i^T \beta)$. The derivative of the sigmoid is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, so:

$$\frac{\partial p_i}{\partial \beta} = p_i(1 - p_i) \cdot x_i$$

For a single observation's contribution $J_i$:

$$\frac{\partial J_i}{\partial \beta} = -\frac{y_i}{p_i} \cdot \frac{\partial p_i}{\partial \beta} + \frac{1 - y_i}{1 - p_i} \cdot \frac{\partial p_i}{\partial \beta}$$

$$= \left(-\frac{y_i}{p_i} + \frac{1 - y_i}{1 - p_i}\right) \cdot p_i(1 - p_i) \cdot x_i$$

$$= (-y_i(1 - p_i) + (1 - y_i)p_i) \cdot x_i$$

$$= (p_i - y_i) \cdot x_i$$

Summing over all $n$ observations and dividing by $n$:

$$\nabla J(\beta) = \frac{1}{n} X^T (p - y) \tag{4.24, ESL}$$

where $p$ is the $n \times 1$ vector of predicted probabilities $p_i = \sigma(x_i^T \beta)$.

This gradient is remarkably similar to the linear regression gradient $(2/n)X^T(X\beta - y)$ — the difference is that the residuals $p - y$ are bounded in $(-1, 1)$ (ESL, p. 121).

**Reference:**

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4.1 Fitting Logistic Regression Models  
> Pages: 121

### 3.2 Batch Gradient Descent

$$\beta^{(t+1)} = \beta^{(t)} - \alpha \cdot \nabla J(\beta^{(t)})$$

$$\nabla J(\beta) = \frac{1}{n} X^T (\sigma(X\beta) - y)$$

**Implementation (`03_batch_gd.py`):**

```python
for iteration in range(iterations):
    z = X @ beta
    probs = 1 / (1 + np.exp(-z))
    gradient = (1 / n) * X.T @ (probs - y)
    beta -= learning_rate * gradient
```

| Aspect | Detail |
|--------|--------|
| Gradient | $\frac{1}{n} X^T(p - y)$ |
| Per-iteration cost | $O(np)$ |
| Convergence | Smooth, monotonic decrease in log-loss |
| Use case | Small to moderate $n$ |

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.2 Estimating the Regression Coefficients  
> Pages: 140-141 [Not directly covered in this chapter — ISL does not detail gradient descent for logistic regression]

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4.1 Fitting Logistic Regression Models  
> Pages: 120-121

### 3.3 Stochastic Gradient Descent (SGD)

Uses **one random sample** to estimate the gradient at each step.

$$\beta := \beta - \alpha_t \cdot (p_i - y_i) \cdot x_i$$

where $\alpha_t$ is a decaying learning rate.

Note that the per-sample gradient $(p_i - y_i) x_i$ drops the $1/n$ factor from the full-batch gradient. This is an **unbiased estimator**: $\mathbb{E}_{i \sim U(1,n)}[(p_i - y_i) x_i] = \frac{1}{n} X^T(p - y)$.

**Implementation (`04_sgd.py`):**

```python
for epoch in range(epochs):
    indices = np.random.permutation(n)
    for i in range(n):
        idx = indices[i]
        xi = X[idx:idx+1]
        yi = y[idx:idx+1]
        prob = 1 / (1 + np.exp(-xi @ beta))
        gradient = xi.T @ (prob - yi)
        lr = alpha / (1 + decay * (epoch * n + i))
        beta -= lr * gradient.flatten()
```

| Aspect | Detail |
|--------|--------|
| Gradient | $(p_i - y_i) x_i$ |
| Per-iteration cost | $O(p)$ |
| Convergence | Noisy; requires learning rate decay |
| Use case | Very large $n$; online learning |

### 3.4 Mini-Batch Gradient Descent

Uses a **batch of $b$ samples** to compute the gradient.

$$\beta := \beta - \alpha \cdot \frac{1}{b} X_b^T (p_b - y_b)$$

**Implementation (`05_minibatch_gd.py`):**

```python
for epoch in range(epochs):
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        b = X_batch.shape[0]
        z = X_batch @ beta
        probs = 1 / (1 + np.exp(-z))
        gradient = (1 / b) * X_batch.T @ (probs - y_batch)
        beta -= alpha * gradient
```

| Aspect | Detail |
|--------|--------|
| Gradient | $\frac{1}{b} X_b^T(p_b - y_b)$ |
| Per-iteration cost | $O(bp)$ |
| Convergence | Smoother than SGD, faster than Batch |
| Use case | Large $n$; standard deep learning practice |

#### Comparison

| Method | Samples per Step | Noise | Convergence Speed | Memory |
|--------|-----------------|-------|-------------------|--------|
| Batch GD | All $n$ | None | Slow per step, fast per epoch | High |
| SGD | 1 | High | Fast per step, slow per epoch | Low |
| Mini-Batch | $b$ (e.g., 16-256) | Moderate | Best trade-off | Moderate |

### 3.5 Numerical Considerations

#### Feature Scaling

Logistic regression with gradient descent requires features to be on a similar scale, just as in linear regression GD:

```python
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_s = (X_train - X_mean) / X_std
```

The intercept column should be added **after** scaling.

#### Logit Saturation

When $|z| = |X\beta|$ is very large (e.g., $z > 20$), the sigmoid function saturates: $\sigma(z) \approx 1$ or $\sigma(z) \approx 0$, producing near-zero gradients and slow learning. Feature scaling and good initialization mitigate this.

#### Numerical Stability for $\log(p)$ and $\log(1-p)$

Both $\log(p)$ and $\log(1-p)$ are evaluated when computing log-loss. When $p$ is exactly 0 or 1, these produce $-\infty$. A small epsilon ($10^{-10}$) prevents this:

```python
cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
```

---

## 4. Decision Boundary

### Threshold Rule

Logistic regression predicts the probability $p(X) = P(Y=1 \mid X)$. The default hard-classification rule is:

$$\hat{y} = \begin{cases} 1 & \text{if } p(X) \geq 0.5 \\ 0 & \text{if } p(X) < 0.5 \end{cases}$$

Because $p(X) = \sigma(X\beta)$ and $\sigma(z) \geq 0.5$ iff $z \geq 0$, the decision boundary is the hyperplane $X\beta = 0$ — **linear in the predictors**.

### Threshold Trade-off

The 0.5 threshold minimizes overall misclassification rate only if classes are balanced and misclassification costs are symmetric. In practice, the threshold can be tuned:

| Threshold | Effect |
|-----------|--------|
| Lower (e.g., 0.3) | Higher recall (catches more positives), lower precision |
| Higher (e.g., 0.7) | Higher precision (fewer false positives), lower recall |
| Custom | Optimized for domain-specific cost: e.g., medical screening favors recall |

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.3 Making Predictions  
> Pages: 142-143

---

## 5. Evaluation Metrics

### 5.1 Confusion Matrix

A $2 \times 2$ contingency table comparing predicted vs. true classes:

| | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**Implementation:**

```python
TP = np.sum((y_pred == 1) & (y_true == 1))
TN = np.sum((y_pred == 0) & (y_true == 0))
FP = np.sum((y_pred == 1) & (y_true == 0))
FN = np.sum((y_pred == 0) & (y_true == 1))
```

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.4.1 Linear Discriminant Analysis for $p=1$  
> Pages: 152-154

### 5.2 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **When to use:** Balanced classes, equal misclassification costs.
- **Limitation:** Misleading for imbalanced data (e.g., 95% negatives → 95% accuracy by predicting all negatives).

### 5.3 Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Interpretation:** Of all positive predictions, what fraction is correct?
- **When to use:** When false positives are costly (e.g., spam filtering: legitimate email flagged as spam).

### 5.4 Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **Interpretation:** Of all actual positives, what fraction did we catch?
- **When to use:** When false negatives are costly (e.g., cancer screening: missing a positive case).

### 5.5 F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Interpretation:** Harmonic mean of precision and recall.
- **When to use:** Imbalanced classes; single-number summary of both precision and recall.

**Implementation:**

```python
accuracy = np.mean(y_pred == y_true)
precision = TP / (TP + FP + 1e-10)
recall = TP / (TP + FN + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)
```

### 5.6 ROC Curve and AUC

The **ROC curve** plots the true positive rate (TPR = recall) against the false positive rate (FPR = FP / (FP + TN)) as the classification threshold varies from 0 to 1. The **AUC** (area under the ROC curve) is a threshold-independent measure of classifier performance:

| AUC Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect classifier |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.5–0.7 | Poor |
| 0.5 | Random guessing |

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.4.2 Linear Discriminant Analysis for $p > 1$  
> Pages: 155 (ROC curve discussed in context of LDA)

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 122-124 (confusion matrix and error rates for the heart disease data, ROC discussed in exercises)

---

## 6. Logistic Regression Assumptions

### 6.1 Binary Response

The response variable must be binary ($Y \in \{0, 1\}$). For multi-class problems, multinomial logistic regression (softmax regression) is used.

### 6.2 Independent Observations

Observations must be independent of each other. Correlated observations (e.g., repeated measures, time series) violate this assumption and require specialized methods (GEE, mixed-effects models).

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.2 Estimating the Regression Coefficients  
> Pages: 140-141 (standard i.i.d. assumption for MLE)

### 6.3 Linearity in the Log-Odds

Logistic regression assumes that the **log-odds** are linear in the predictors:

$$\log\left(\frac{p(X)}{1-p(X)}\right) = X\beta$$

Unlike linear regression, logistic regression does NOT require:
- Normality of errors
- Constant variance of errors (homoscedasticity)
- Linear relationship between $X$ and $Y$

**How to test:** Bin the predictors and plot the empirical log-odds within each bin against the predictor values. A non-linear pattern suggests the need for polynomial terms or splines.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.1 The Logistic Model  
> Pages: 138-139

### 6.4 Low Multicollinearity

High correlation among predictors inflates the standard errors of coefficient estimates, making inference unreliable. Use VIF to detect multicollinearity, as in linear regression (ISL, p. 106-108).

### 6.5 Adequate Sample Size

Logistic regression requires a sufficient number of events per predictor variable (EPV). A common rule of thumb is at least **10 events per predictor** (ESL, p. 123). Small sample sizes or rare events can lead to biased or unstable estimates, including **perfect separation** (where a predictor or linear combination perfectly separates classes, causing the MLE to diverge).

**Reference:**

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 122-123

---

## 7. Regularization

### 7.1 Motivation

When $p$ is large relative to $n$, or when predictors are highly correlated, the unregularized MLE can be unstable, have high variance, or fail entirely (perfect separation). Regularization adds a penalty to the loss function, shrinking coefficients toward zero.

### 7.2 Ridge Regression (L2 Penalty)

$$J_{\text{ridge}}(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \big[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\big] + \lambda \sum_{j=1}^{p} \beta_j^2$$

- The intercept $\beta_0$ is typically **not** penalized.
- Ridge shrinks coefficients smoothly but does not set them to exactly zero.
- Useful when most predictors have non-zero effects.

### 7.3 Lasso Regression (L1 Penalty)

$$J_{\text{lasso}}(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \big[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\big] + \lambda \sum_{j=1}^{p} |\beta_j|$$

- Lasso sets some coefficients exactly to zero, performing **automatic feature selection**.
- Useful when only a subset of predictors is relevant.

### 7.4 Gradient Update with Regularization

For Ridge, the gradient becomes:

$$\nabla J_{\text{ridge}}(\beta) = \frac{1}{n} X^T(p - y) + 2\lambda \beta_{\text{penalized}}$$

where $\beta_{\text{penalized}}$ excludes $\beta_0$ (or includes it with $\lambda = 0$).

**Implementation (`07_regularization.py`):**

```python
# Ridge
gradient = (1 / n) * X.T @ (probs - y)
gradient[1:] += 2 * ridge_lambda * beta[1:]  # penalize all except intercept

# Lasso (subgradient)
gradient = (1 / n) * X.T @ (probs - y)
gradient[1:] += lasso_lambda * np.sign(beta[1:])
```

**Reference:**

> Book: ISL  
> Chapter: 6 - Linear Model Selection and Regularization  
> Section: 6.2.1 Ridge Regression (p240-243), 6.2.2 The Lasso (p244-249)  
> Pages: 240-249 (theory for linear regression; extends directly to logistic regression by replacing RSS with log-loss)

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4.4 Logistic Regression (Regularization)  
> Pages: 125-127

---

## 8. Implementation Explanation

### 8.1 File Layout

```
01_sigmoid.py              # Sigmoid function and its derivative
02_logistic_regression.py  # Full batch implementation (predict, fit, evaluate)
03_batch_gd.py             # Batch gradient descent
04_sgd.py                  # Stochastic gradient descent
05_minibatch_gd.py         # Mini-batch gradient descent
06_metrics.py              # Confusion matrix, accuracy, precision, recall, F1
07_regularization.py       # Ridge and Lasso penalized logistic regression
08_demo.py                 # Full pipeline on real datasets
```

### 8.2 Model Initialization

Parameters are initialized to zero:

```python
beta = np.zeros(p)
```

Zero initialization is safe for logistic regression because the convex log-loss guarantees convergence regardless of starting point. However, it can slow convergence when features are not centered.

### 8.3 Sigmoid Function

**Implementation (`01_sigmoid.py`):**

```python
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

The clip prevents overflow in `np.exp(-z)` for extreme $z$ values.

### 8.4 Training Function

**Implementation (`02_logistic_regression.py`):**

```python
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
```

Key observations:
- The linear predictor `z = X @ beta` is computed, then passed through the sigmoid.
- Log-loss is computed with an epsilon (`1e-10`) to prevent $\log(0)$.
- The gradient `(1/n) * X.T @ (probs - y)` mirrors the linear regression gradient `(2/n) * X.T @ (Xβ - y)` but without the factor of 2.

### 8.5 Prediction Function

**Implementation:**

```python
def predict_proba(X, beta):
    return sigmoid(X @ beta)

def predict(X, beta, threshold=0.5):
    probs = predict_proba(X, beta)
    return (probs >= threshold).astype(int)
```

- `predict_proba` returns the probability $P(Y=1 \mid X)$.
- `predict` applies the threshold rule to produce hard class labels.
- The threshold can be adjusted for precision-recall trade-offs.

### 8.6 Cost History and Convergence

Monitoring the cost history confirms convergence (cost flattens) or divergence (cost increases, typically from a learning rate that is too high):

```python
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.title('Convergence of Batch Gradient Descent')
```

---

## 9. Logistic vs. Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|-------------------|
| **Output type** | Continuous ($\mathbb{R}$) | Probability ($[0,1]$) or binary label ($\{0,1\}$) |
| **Loss function** | MSE / RSS | Log-loss (binary cross-entropy) |
| **Task** | Regression | Classification |
| **Output range** | $(-\infty, \infty)$ | $(0, 1)$ for probabilities |
| **Optimization** | Closed-form (Normal Eq.) or GD | Gradient descent (no closed form) |
| **Gradient form** | $\frac{2}{n} X^T (X\beta - y)$ | $\frac{1}{n} X^T (p - y)$ |
| **Decision boundary** | N/A | Linear in $X$ ($X\beta = 0$) |
| **Link function** | Identity | Logit (log-odds) |
| **Parameter interpretation** | $\beta_j$ = change in $Y$ per unit change in $X_j$ | $\exp(\beta_j)$ = odds ratio per unit change in $X_j$ |
| **Estimation principle** | OLS (minimize RSS) | MLE (maximize log-likelihood) |
| **Assumptions** | Linearity, normality, homoscedasticity, independence | Linearity in log-odds, independence, adequate sample size |
| **Gradient descent scaling** | Features should be scaled | Features should be scaled |

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.1 Overview, 4.2 Why Not Linear Regression?  
> Pages: 135-137

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.1 Introduction  
> Pages: 101-102

---

## 10. Real Dataset Examples

### Default (ISL)

The `Default` dataset (ISL Chapter 4) contains credit card default status for 10,000 customers with predictors `balance` (credit card balance), `income` (annual income), and `student` (student status). A logistic regression model on `balance` alone yields:

| Coefficient | Estimate | $z$-statistic | $p$-value |
|-------------|----------|---------------|-----------|
| Intercept | -10.65 | -29.5 | <0.0001 |
| balance | 0.0055 | 24.9 | <0.0001 |

- A one-unit increase in balance multiplies the odds of default by $\exp(0.0055) \approx 1.006$ — about a 0.6% increase.
- The model effectively separates defaults from non-defaults for extreme balance values.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.3.1 The Logistic Model, 4.3.2 Estimating Coefficients  
> Pages: 139-141

### Smarket (ISL)

The `Smarket` dataset contains daily percentage returns for the S&P 500 stock index from 2001-2005. Logistic regression predicts the **direction** of the market (Up/Down) using lagged returns. Key findings (ISL p. 173-178):

- Using all five lagged predictors, the model achieves only 52.2% accuracy on the training set.
- When refit using only `Lag1` and `Lag2` and evaluated on the 2005 test set, accuracy improves to **56%**.
- On days when the model predicts an increase, it is correct **58.2%** of the time.

This example illustrates the danger of evaluating on training data: the full model's 52.2% training accuracy was misleadingly optimistic about its true predictive power.

**Reference:**

> Book: ISL  
> Chapter: 4 - Classification  
> Section: 4.7 Lab: Logistic Regression, LDA, QDA, and KNN  
> Pages: 173-178

### South African Heart Disease (ESL)

The South African heart disease dataset (ESL Chapter 4) contains 462 observations with 7 cardiovascular risk factors. Stepwise logistic regression identifies tobacco, ldl (low-density lipoprotein cholesterol), famhist (family history), and age as significant predictors:

| Coefficient | Std. Error | $z$-score |
|-------------|------------|-----------|
| Intercept | -4.204 | 0.498 | -8.45 |
| tobacco | 0.081 | 0.026 | 3.16 |
| ldl | 0.168 | 0.054 | 3.09 |
| famhist | 0.924 | 0.223 | 4.14 |
| age | 0.044 | 0.010 | 4.52 |

This demonstrates how logistic regression provides interpretable coefficients across multiple domains, identifying which risk factors are most strongly associated with coronary heart disease.

**Reference:**

> Book: ESL  
> Chapter: 4 - Linear Methods for Classification  
> Section: 4.4 Logistic Regression  
> Pages: 122-124

---

## 11. Common Mistakes

| # | Mistake | Explanation | How to Avoid |
|---|---------|-------------|--------------|
| 1 | **Using linear regression for classification** | Predictions can fall outside $[0,1]$; assumption violations; poor probability estimates. | Use logistic regression or another classifier. |
| 2 | **Relying on accuracy alone** | High accuracy can mask poor performance on the minority class. | Always check precision, recall, F1, and confusion matrix. |
| 3 | **Ignoring class imbalance** | With 95% negatives, 95% accuracy is trivial. | Use class weights, oversampling (SMOTE), or threshold tuning. |
| 4 | **Default 0.5 threshold without justification** | The optimal threshold depends on misclassification costs. | Tune threshold via ROC curve or domain-specific cost function. |
| 5 | **Data leakage** | Scaling or encoding before splitting. | Fit scalers on training data only; split before any preprocessing. |
| 6 | **Features not scaled for GD** | Gradient descent converges slowly with unscaled features. | Standardize features (mean 0, variance 1) before GD. |
| 7 | **Overfitting with many predictors** | Too many predictors relative to sample size inflates variance. | Use regularization (Ridge, Lasso) or feature selection. |
| 8 | **Ignoring multicollinearity** | Correlated predictors inflate coefficient standard errors. | Check VIF; drop or combine collinear predictors. |
| 9 | **Perfect separation** | A predictor perfectly separates classes → MLE diverges. | Use regularization (Lasso/Ridge) or remove the separating feature. |
| 10 | **Evaluating on training data** | Training accuracy overestimates real-world performance. | Always hold out a test set or use cross-validation. |

---

## 12. Implementation Notes

**Implementation Note — Sigmoid Numerical Stability:**

The sigmoid function saturates for $|z| > 20$. Clipping prevents overflow:

```python
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

**Implementation Note — Log-Loss Epsilon:**

Both $\log(p)$ and $\log(1-p)$ produce $-\infty$ when $p = 0$ or $p = 1$. A small epsilon prevents this:

```python
cost = -np.mean(y * np.log(probs + 1e-10) + (1 - y) * np.log(1 - probs + 1e-10))
```

**Implementation Note — Feature Scaling for GD:**

Gradient descent for logistic regression requires scaled features, just as in linear regression:

```python
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_s = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std
```

**Implementation Note — Train/Test Split:**

Split before any preprocessing to prevent data leakage:

```python
n_train = int(0.8 * n)
indices = np.random.permutation(n)
train_idx, test_idx = indices[:n_train], indices[n_train:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_s = (X_train - X_mean) / X_std
X_test_s = (X_test - X_mean) / X_std
```

**Implementation Note — Learning Rate Decay (SGD):**

SGD requires a decaying learning rate for convergence:

```python
lr = alpha / (1 + decay * (epoch * n + i))
```

A typical decay rate is $0.01$ to $0.001$.

**Implementation Note — Adding an Intercept:**

The intercept column must be added manually and should not be standardized:

```python
X_train_s = (X_train - X_mean) / X_std
X_train_d = np.column_stack((np.ones(n_train), X_train_s))
```

**Implementation Note — Penalized Gradient for Regularization:**

When applying Ridge or Lasso, the regularization penalty applies to all coefficients except the intercept:

```python
# Ridge gradient (intercept at index 0)
gradient[1:] += 2 * ridge_lambda * beta[1:]

# Lasso subgradient (intercept at index 0)
gradient[1:] += lasso_lambda * np.sign(beta[1:])
```

---

## 13. Final Cheat Sheet

### Main Formulas

| Concept | Formula |
|---------|---------|
| Linear predictor | $z = X\beta$ |
| Sigmoid (logistic) function | $\sigma(z) = 1 / (1 + e^{-z})$ |
| Probability model | $P(Y=1 \mid X) = \sigma(X\beta)$ |
| Log-odds (logit) | $\log(p / (1-p)) = X\beta$ |
| Log-loss ($n$ obs) | $-\frac{1}{n} \sum [y \log(p) + (1-y) \log(1-p)]$ |
| Gradient | $\nabla J = \frac{1}{n} X^T (p - y)$ |
| Batch GD update | $\beta := \beta - \alpha \cdot \frac{1}{n} X^T (p - y)$ |
| SGD update | $\beta := \beta - \alpha_t \cdot (p_i - y_i) x_i$ |
| Mini-Batch GD update | $\beta := \beta - \alpha \cdot \frac{1}{b} X_b^T (p_b - y_b)$ |
| Hard prediction | $\hat{y} = \mathbb{1}[p \geq 0.5]$ |
| Accuracy | $(TP + TN) / (TP + TN + FP + FN)$ |
| Precision | $TP / (TP + FP)$ |
| Recall | $TP / (TP + FN)$ |
| $F_1$ | $2 \cdot (P \cdot R) / (P + R)$ |
| Ridge penalty | $+\lambda \sum \beta_j^2$ |
| Lasso penalty | $+\lambda \sum |\beta_j|$ |

### Important Rules

| Rule | Description |
|------|-------------|
| **MLE principle** | Logistic regression maximizes the likelihood of the observed data. |
| **Convexity** | Log-loss is convex in $\beta$; gradient descent finds the global minimum. |
| **Decision boundary** | Linear in $X$: $X\beta = 0$. |
| **Odds ratio interpretation** | $\exp(\beta_j)$ is the multiplicative change in odds for a one-unit increase in $X_j$. |
| **Independence** | Observations must be independent for valid MLE. |
| **Linearity in log-odds** | The logit is linear in the predictors — the probability is not. |
| **No perfect separation** | Perfect separation causes MLE coefficients to diverge to $\pm \infty$. |

### Algorithm Steps

1. Load and preprocess data (handle missing values, scale for GD).
2. Add intercept column to design matrix.
3. Initialize $\beta = \mathbf{0}$.
4. For each iteration/epoch:
   - Compute $z = X\beta$
   - Compute $p = \sigma(z)$
   - Compute log-loss
   - Compute gradient $\frac{1}{n} X^T(p - y)$
   - Update $\beta := \beta - \alpha \cdot \text{gradient}$
5. Predict probabilities: $p_{\text{test}} = \sigma(X_{\text{test}} \hat{\beta})$
6. Classify: $\hat{y} = (p_{\text{test}} \geq 0.5)$
7. Evaluate: confusion matrix, accuracy, precision, recall, $F_1$, ROC-AUC.

### When to Use

| Method | Use Case |
|--------|----------|
| **Logistic Regression** | Binary classification; interpretable probabilities; baseline model; linear decision boundary is sufficient. |
| **Linear Regression** | Continuous outcome; numeric prediction. |
| **Tree-based models (RF, GBM)** | Non-linear decision boundaries; complex interactions; automatic feature selection. |
| **Neural Networks** | Very large datasets; non-linear patterns; multi-class; deep feature learning. |
| **SVM with RBF kernel** | Non-linear boundaries with moderate $n$; max-margin objective. |
| **Regularized logistic (Ridge/Lasso)** | $p$ large relative to $n$; multicollinearity; feature selection needed. |

---

> **References Summary**
>
> - James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
> - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
>
> All page references in this document refer to the 2023 ISL (Python edition) and the 2009 ESL (2nd edition) unless otherwise noted.
