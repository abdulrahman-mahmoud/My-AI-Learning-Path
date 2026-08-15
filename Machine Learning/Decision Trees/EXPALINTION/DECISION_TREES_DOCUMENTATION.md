# Decision Trees Implementation Documentation

> A comprehensive guide bridging theory from *An Introduction to Statistical Learning* (ISL) and *The Elements of Statistical Learning* (ESL) with a practical NumPy-based implementation.

---

## 1. Introduction

### What is a Decision Tree?

A decision tree is a supervised learning method that models the relationship between a response $Y$ and one or more predictor variables $X_1, X_2, \dots, X_p$ by **repeatedly asking simple questions about the features** and using the answers to route each observation into a final region that carries a prediction.

The two models studied so far in this repository take a **global** approach:

> Logistic Regression used a mathematical function to model the relationship between features and the target. Decision Trees take a fundamentally different approach: instead of fitting one global function, they partition the feature space into regions.

Linear regression fits a single linear function over the whole feature space. Logistic regression fits a single logistic function. A decision tree fits **no formula at all**; it divides the space into regions and predicts with a simple local rule inside each region.

**Reference:**

> Book: ISLP  
> Chapter: 9 - Tree-Based Methods  
> Topic: The Basics of Decision Trees

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Tree-based methods

### Why Decision Trees Matter in Machine Learning

- They are **easy to understand and interpret**: the model reads like a set of if-then rules.
- They **require little preprocessing**: no feature scaling or explicit interaction terms.
- They **naturally handle nonlinear relationships** and interactions.
- They work for **both regression and classification**.
- They are the **building block** of powerful ensemble methods (bagging, random forests, boosting), which is why they are studied before those topics (ISL, Chapter 8; ESL, Chapter 15).

### Purpose of This Implementation

The implementation provides a from-scratch decision tree toolkit using only NumPy. It includes:

- **Impurity measures** (Gini impurity and entropy) computed from class proportions
- **Greedy binary split selection** for regression (squared error) and classification (weighted impurity)
- **`DecisionTreeRegressor`** with recursive construction and piecewise-constant prediction
- **`DecisionTreeClassifier`** with majority-class and probability prediction
- **Stopping criteria** (max depth, minimum samples) that control tree complexity
- **Visualizations** of piecewise-constant fits and axis-aligned decision regions
- A **scikit-learn comparison** to connect the educational implementation to the production library

### How the Algorithm Works at a High Level

1. **Ask a question** about one feature, e.g. "Is $X_j \le s$?"
2. **Split the data** into the two groups that answer yes and no.
3. **Repeat recursively** inside each group.
4. **Stop** when a stopping condition is met and return the group's prediction.
5. **Predict** a new observation by sending it down the tree following the question answers.

```
Is age > 30?
├── Yes → Is income > 50k?
│         ├── Yes → Buy
│         └── No  → Don't Buy
└── No  → Don't Buy
```

This tiny example already contains every idea in decision trees. We will spend the rest of the chapter unpacking it: what each part is called, why splitting helps, how to choose the best split, and how to control how deep the tree grows.

---

## 2. Prerequisites

Every learning chapter in this repository builds on what came before. This section separates the concepts that should already be familiar from those that are introduced fresh here.

### Already Covered

The repository has already covered:

- **Supervised learning**: the setting of learning a mapping $f(X) \to Y$ from labeled training data $(x_i, y_i)$.
- **Features and targets**: the predictors $X = (X_1, \dots, X_p)$ and the response $Y$.
- **Regression vs. classification**: a continuous response vs. a categorical response.
- **Loss functions**: how we quantify prediction error (RSS/MSE for regression, log-loss for classification).
- **Optimization**: how models choose parameters to minimize a loss.
- **Probability**: proportions, conditional probability, and class probabilities.
- **Overfitting and the bias-variance tradeoff**: complex models fit training data better but generalize worse (ISL, Section 2.2.2).
- **Basic calculus**: derivatives used in the derivation of leaf predictions.

We do **not** reteach these chapters. Instead, each is briefly reminded and connected to trees when it becomes relevant. For example:

> Linear regression minimized RSS by choosing coefficients. Decision trees also use squared error to judge splits, but instead of adjusting coefficients, they choose which **partition** of the data minimizes the error.

### Not Yet Covered (New Prerequisites)

The following concepts are needed for decision trees and are **taught from zero** in this document because they have not appeared before:

- **Recursive algorithms** (Section 14)
- **Nodes and trees as data structures** (Sections 3-4, 20)
- **Impurity** (Section 8)
- **Gini impurity and entropy** (Sections 9-10)
- **Information gain** (Section 11)
- **Weighted averages** (Section 12)
- **Recursive binary splitting** (Section 13)

Each of these is introduced exactly where it first becomes necessary, so that no term is used before it is explained.

---

## 3. The Core Question and the Intuition

> **How can a model make predictions by repeatedly asking simple questions about the features?**

Start with a concrete everyday situation. Suppose we want to predict whether a customer **buys** a product. We have two features: `age` and `income`. A simple approach that requires no math at all:

```
Is age > 30?
├── Yes → Is income > 50k?
│         ├── Yes → Buy
│         └── No  → Don't Buy
└── No  → Don't Buy
```

Read this from top to bottom. We start at the **root**. For a new customer we ask the top question. Depending on the answer we go down one **branch** to the next question, and finally arrive at a **leaf** that gives a prediction.

Why does this help? Because each question **divides the observations into smaller groups** that are more homogeneous. Instead of describing all customers with one rule, we let different subgroups have different rules. A 40-year-old with high income gets a different prediction from a 25-year-old. The questions let the model specialize.

The whole point of the algorithm will be to decide *which questions to ask*, *where to draw the thresholds*, and *when to stop*. Everything else in this chapter is detail.

---

## 4. Terminology

Every decision tree problem uses the following vocabulary. Learn these terms now so the rest of the chapter is unambiguous.

| Term | Meaning |
|------|---------|
| **Root node** | The first, topmost node. It contains all training observations. |
| **Node** | Any point in the tree where a decision (question) is made. |
| **Internal (decision) node** | A node that asks a question and routes observations to children. |
| **Leaf (terminal node)** | A node with no children. It stores the final prediction. |
| **Branch / edge** | The connection from a parent node to a child node. |
| **Split** | The act of dividing a node's observations using one question. |
| **Threshold** | The value $s$ in a question like "$X_j \le s$". |
| **Feature** | The predictor $X_j$ that the question refers to. |
| **Depth** | The number of splits from the root to a node (the root is depth 0). |
| **Prediction** | The value (regression) or class (classification) stored at a leaf. |

Every split in an ordinary decision tree uses **one feature** and **one threshold**, written as

$$X_j \le s$$

observations going left when the statement is true and right when it is false. This is the **axis-aligned** split we keep returning to: it slices the feature space with a line parallel to one of the axes.

---

## 5. The Conceptual Dependency Chain

Before diving into formulas, it helps to see the full chain of ideas and why each one leads to the next. Each link answers "why do we need this next concept?"

```
Supervised Learning
        ↓
Regression / Classification
        ↓
Feature-Based Questions
        ↓
Splits
        ↓
Nodes and Leaves
        ↓
Regions of the Feature Space
        ↓
Impurity / Error
        ↓
Best Split
        ↓
Recursive Binary Splitting
        ↓
Decision Tree
        ↓
Tree Complexity
        ↓
Overfitting
        ↓
Stopping / Pruning
```

- **Supervised learning → Regression/Classification**: we have labeled data and a task (predict a number or a class).
- **→ Feature-based questions**: instead of a formula, we'll ask yes/no questions about features.
- **→ Splits → Nodes and Leaves**: each question splits the data; the tree is just a nesting of splits ending in leaves.
- **→ Regions of the feature space**: the leaves correspond to disjoint regions that tile the space.
- **→ Impurity/Error**: to choose between candidate splits we need a score measuring how "good" a group is (how mixed its classes are, or how much its values vary).
- **→ Best split**: pick the split that most improves that score.
- **→ Recursive binary splitting**: apply "best split" again inside each child until we stop.
- **→ Decision tree**: the finished structure.
- **→ Tree complexity → Overfitting**: a tree that keeps splitting becomes very complex and can overfit.
- **→ Stopping/Pruning**: limit or trim complexity to restore generalization.

The rest of the chapter walks this chain from top to bottom.

---

## 6. Intuition → Mathematics: Regions and Piecewise-Constant Prediction

For the rest of the chapter, remember the central mental picture:

> Instead of fitting one global regression equation, a tree divides the feature space into regions and assigns a constant prediction to each region.

Suppose we have two features $X_1$ and $X_2$. A tree that asks "Is $X_1 \le t_1$?" and then "Is $X_2 \le t_2$?" inside the right branch produces four rectangular regions. Every observation in a given rectangle gets the same predicted value.

This is the **piecewise-constant** nature of tree prediction: within each region $R_m$, the prediction is a constant $\hat c_m$, and across regions the prediction jumps abruptly at the region boundaries.

### The Regression Tree Prediction

The tree's prediction function can be written as

$$\hat f(x) = \sum_{m=1}^{M} \hat c_m \, \mathbb{1}(x \in R_m)$$

where:

- $M$ is the number of **regions** (leaves).
- $R_1, \dots, R_M$ are disjoint regions that partition the feature space.
- $\hat c_m$ is the **constant prediction** assigned to region $R_m$.
- $\mathbb{1}(\cdot)$ is the **indicator function**: it equals 1 if its argument is true and 0 otherwise.

Read the formula: for a point $x$, exactly one indicator is 1 (the region containing $x$), so only that region's constant $\hat c_m$ contributes; the sum returns that constant.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Regression trees and prediction via stratification of the feature space

---

## 7. Regression Trees

### The Intuition

In linear regression we wrote $Y \approx \beta_0 + \beta_1 X_1 + \dots$ — one smooth function for all the data. A regression tree abandons smoothness. It cuts the predictor space into regions and predicts the **mean of the training responses in each region**.

This leads to a prediction that looks like a **staircase**: flat within each region, jumping at region boundaries. This is called a **piecewise-constant** prediction.

### Why the Leaf Prediction Is the Mean (Derivation)

The region $R_m$ is assigned one constant $\hat c_m$ to be used for every point inside it. We choose it to minimize the sum of squared errors over the training observations in that region:

$$\min_{c_m} \sum_{i \in R_m} (y_i - c_m)^2$$

Treat this as the simple one-parameter regression problem we already know. The best constant (in squared error) is the value that minimizes $\sum_i (y_i - c)^2$. Setting the derivative with respect to $c$ to zero:

$$\frac{d}{dc} \sum_{i \in R_m} (y_i - c)^2 = -2 \sum_{i \in R_m} (y_i - c) = 0$$

$$\sum_{i \in R_m} y_i - |R_m| \, c = 0$$

$$c = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$

So the optimal leaf prediction is just the **mean** of the training responses in the region:

$$\hat c_m = \frac{1}{|R_m|} \sum_{i: x_i \in R_m} y_i$$

where $|R_m|$ is the number of training observations in region $R_m$.

This is exactly the same least-squares principle as linear regression, except the "model" inside each region has no slope — it is a flat line at the local mean.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Regression trees and leaf predictions

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Regression trees

---

## 8. Finding the Best Regression Split

We cannot consider every possible partition of the data (there are astronomically many). Instead, trees use a **recursive, greedy** procedure: at each node we consider each feature and each candidate threshold, and pick the single split that best reduces the error. We will derive the split criterion first with a tiny example, then write the mathematics.

### Tiny Numerical Example

Imagine one feature $X$ and a response $y$:

| $i$ | $X$ | $y$ |
|----|-----|-----|
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 4 |
| 4 | 4 | 10 |
| 5 | 5 | 11 |
| 6 | 6 | 12 |

Notice the first three $y$-values are small and the last three are large. A good split should separate them. Candidate binary splits for a single feature use thresholds lying **between** consecutive sorted $X$ values:

```
X <= 1.5
X <= 2.5
X <= 3.5
X <= 4.5
X <= 5.5
```

For each split we form a left region $R_1$ and a right region $R_2$, compute each region's mean (its candidate leaf prediction), then compute each region's squared error, and add them.

#### Split 1: $X \le 1.5$

- $R_1 = \{1\}$, $y$-values $\{2\}$, mean $\hat c_1 = 2$.
- $R_2 = \{2,3,4,5,6\}$, $y$-values $\{3,4,10,11,12\}$, mean $\hat c_2 = (3+4+10+11+12)/5 = 8$.
- Left error: $(2-2)^2 = 0$.
- Right error: $(3-8)^2+(4-8)^2+(10-8)^2+(11-8)^2+(12-8)^2 = 25+16+4+9+16 = 70$.
- **Total = 70.**

#### Split 2: $X \le 2.5$

- $R_1 = \{1,2\}$, $y = \{2,3\}$, mean $2.5$.
- $R_2 = \{3,4,5,6\}$, $y = \{4,10,11,12\}$, mean $(4+10+11+12)/4 = 9.25$.
- Left error: $(2-2.5)^2+(3-2.5)^2 = 0.25+0.25 = 0.5$.
- Right error: $(4-9.25)^2+(10-9.25)^2+(11-9.25)^2+(12-9.25)^2 = 27.5625+0.5625+3.0625+7.5625 = 38.75$.
- **Total = 39.25.**

#### Split 3: $X \le 3.5$

- $R_1 = \{1,2,3\}$, $y = \{2,3,4\}$, mean $3$.
- $R_2 = \{4,5,6\}$, $y = \{10,11,12\}$, mean $11$.
- Left error: $(2-3)^2+(3-3)^2+(4-3)^2 = 1+0+1 = 2$.
- Right error: $(10-11)^2+(11-11)^2+(12-11)^2 = 1+0+1 = 2$.
- **Total = 4.**

(If we kept splitting, the child regions would each be nearly pure, and the tree would fit this data almost perfectly — this is precisely where overfitting begins.)

#### Splits 4 and 5

- $X \le 4.5$: $R_1 = \{1,2,3,4\}$ mean $(2+3+4+10)/4=4.75$ (error $38.75$); $R_2 = \{5,6\}$ mean $11.5$ (error $0.5$); total error **39.25**.
- $X \le 5.5$: $R_1 = \{1,\dots,5\}$ mean $(2+3+4+10+11)/5=6$ (error $70$); $R_2=\{6\}$ mean $12$ (error $0$); total error **70**.

#### Choosing the Best Split

| Split | Total squared error |
|-------|---------------------|
| $X \le 1.5$ | 70.00 |
| $X \le 2.5$ | 39.25 |
| $X \le 3.5$ | **4.00** |
| $X \le 4.5$ | 39.25 |
| $X \le 5.5$ | 70.00 |

The best split is **$X \le 3.5$**, because it gives the smallest total squared error. It succeeds by separating the low region $\{1,2,3\}$ from the high region $\{4,5,6\}$, exactly the structure we noted at the start.

### The Mathematical Formulation

The two regions produced by a split on feature $X_j$ at threshold $s$ are:

$$R_1(j,s) = \{ X : X_j \le s \}, \qquad R_2(j,s) = \{ X : X_j > s \}$$

The split we choose minimizes the total squared error over both regions:

$$\sum_{i: x_i \in R_1} (y_i - \hat c_1)^2 + \sum_{i: x_i \in R_2} (y_i - \hat c_2)^2$$

This equation is just the sum of the "left error" and "right error" columns we computed by hand. $\hat c_1$ and $\hat c_2$ are the means of each region, exactly as used above. The search is:

> For each feature $X_j$ and each candidate threshold $s$ that lies between consecutive observed values, compute the total squared error, and keep the pair $(j, s)$ that minimizes it.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Regression trees and split selection

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Regression trees

---

## 9. Classification Trees

### The Basic Problem

> What should a leaf predict when the target is a class rather than a continuous number?

For regression each leaf stores the mean of its $y$-values. For classification the leaf stores a **class**. But which class? The natural answer is the **majority class**: whichever class appears most often in that leaf's training observations.

### Class Counts, Proportions, and Purity

In a node, let $N$ be the number of observations and let $N_k$ be the count of class $k$. Then:

- **Class count**: $N_k$, the number of class-$k$ observations in the node.
- **Class proportion**: $p_k = N_k / N$, the fraction of class-$k$ observations.
- **Majority class**: the class with the largest $p_k$; this is the node's prediction.
- **Class probability**: the vector $(p_1, p_2, \dots, p_K)$; this is what `predict_proba` returns (the posterior probability estimate for each class).
- **Pure node**: a node in which all observations belong to one class ($p_k = 1$ for some $k$).
- **Impure node**: a node in which two or more classes are mixed.

### Why Purity Matters

Consider three nodes:

```
Node A:  10 Yes,  0 No
Node B:   5 Yes,  5 No
Node C:   8 Yes,  2 No
```

- **Node A is pure**: every observation is `Yes`. It is perfectly determined — no uncertainty, prediction `Yes` is always right in the training set.
- **Node B is maximally mixed**: a fifty-fifty split. We have no basis to prefer one class; whichever we predict, half the observations disagree. This is maximum uncertainty.
- **Node C is in between**: leaning `Yes`, but with some `No` noise.

The model prefers to split in a way that makes the child nodes **purer** than the parent. Predicting `Yes` in Node A is easy; predicting in Node B is a coin-flip. The next sections give a number to "how mixed" a node is, so that we can compare candidate splits objectively.

---

## 10. Gini Impurity From Zero

> **How do we mathematically measure how mixed a node is?**

We want a number $G$ that is:

- $0$ when the node is pure (one $p_k = 1$, all others $0$),
- high when the classes are evenly mixed,
- increasing as the distribution spreads out across classes.

### Deriving the Gini Impurity

Consider what would happen if we randomly **classified** an observation in the node by drawing a class at random with probabilities $p_k$, and then evaluated whether that random label matches the observation's true class.

- The probability the random draw is class $k$ is $p_k$.
- The probability the observation actually belongs to class $k$ is $p_k$.
- The probability of a **match** for class $k$ is $p_k \cdot p_k = p_k^2$.

Summing over classes, the probability that a random draw matches the true class is $\sum_k p_k^2$. The Gini impurity is the probability of a **mismatch**:

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

where:

- $K$ is the number of classes,
- $p_k$ is the proportion of class $k$ in the node,
- $G = 0$ means the node is **pure** (no mismatch possible),
- $G$ is largest when the classes are evenly spread (mismatch most likely).

### Manual Calculations

| Node | Counts | Proportions | Gini $G = 1 - \sum p_k^2$ |
|------|--------|-------------|---------------------------|
| A | 10 Yes, 0 No | $p = (1, 0)$ | $1 - (1^2 + 0^2) = 0$ |
| B | 5 Yes, 5 No | $p = (0.5, 0.5)$ | $1 - (0.25 + 0.25) = 0.5$ |
| C | 8 Yes, 2 No | $p = (0.8, 0.2)$ | $1 - (0.64 + 0.04) = 0.32$ |

- **Node A: $G = 0$** — pure.
- **Node B: $G = 0.5$** — the maximum for a binary problem (most mixed).
- **Node C: $G = 0.32$** — moderately impure.

For three classes, one fully mixed node with $p=(1/3,1/3,1/3)$ gives $G = 1 - 3 \cdot (1/9) = 2/3$.

Gini is used to **score candidate splits**: a split is good if the *weighted* Gini of its children is much smaller than the Gini of the parent.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Classification trees and node impurity

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Classification trees

---

## 11. Entropy From Zero

Gini measures "mixed-ness" directly through the mismatch probability. Entropy measures the same idea through **uncertainty**, and comes from information theory. We introduce it only now, after the notion of impurity is already clear.

### What Entropy Measures

Imagine a message that reveals the class of a randomly chosen observation. If the node is pure ($p=(1,0)$), the outcome is certain, and the message carries **no information** — uncertainty is zero. If the node is fifty-fifty, the outcome is maximally surprising, and the message carries **the most information**.

Entropy quantifies this uncertainty:

$$H = -\sum_{k=1}^{K} p_k \log(p_k)$$

where the sum is over classes and $\log$ is (conventionally) base 2, so entropy is measured in bits (base $e$ differs only by a constant scale).

### Why the Logarithm Appears

Information theory defines the "surprise" or information content of an event with probability $p$ as $-\log p$. An event with probability $p=1$ (certain) carries $-\log 1 = 0$ information. An event with probability $p=0.5$ carries $-\log 0.5 = 1$ bit. The **expected** information of the class outcome is the average of $-\log p_k$ weighted by $p_k$:

$$H = \sum_k p_k (-\log p_k) = -\sum_k p_k \log p_k$$

So entropy is the **expected surprise** of observing the class label. When one class is certain, surprise is zero (a pure node has $H=0$). When classes are mixed, surprise grows; the maximum is at the uniform distribution (all $p_k$ equal).

Note that by convention we take $0 \log 0 = 0$, since a class that never appears contributes no uncertainty.

### Manual Calculations

Using natural log for ease (the ordering is what matters):

| Node | Proportions | Entropy $H = -\sum p_k \log p_k$ |
|------|-------------|----------------------------------|
| A | $p = (1, 0)$ | $-(1 \cdot 0 + 0 \cdot \log 0) = 0$ |
| B | $p = (0.5, 0.5)$ | $-(0.5(-0.693) + 0.5(-0.693)) = 0.693$ |
| C | $p = (0.8, 0.2)$ | $-(0.8(-0.223) + 0.2(-1.609)) = 0.500$ |

- **Node A: $H = 0$** — pure, zero uncertainty.
- **Node B: $H = 0.693$** — maximum for two classes ($\log 2$ in natural log).
- **Node C: $H = 0.500$** — between.

Observe the qualitative agreement with Gini: A < C < B in impurity. Gini and entropy are **different formulas with similar behavior**. They are not interchangeable terms; they are two competing measures of the same underlying idea.

**Reference:**

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Classification trees

---

## 12. Information Gain

Splitting a node should reduce impurity. **Information gain** measures how much the impurity drops because of a candidate split.

### Building the Concept

```
Parent node
    ↓
Candidate split
    ↓
Child nodes
    ↓
Weighted child impurity
    ↓
Impurity reduction
```

Concretely, for a parent whose impurity is $Q_\text{parent}$, a candidate split creates two children with impurities $Q_1$ and $Q_2$ and sizes $N_1$, $N_2$ (with $N_1 + N_2 = N$). The **weighted child impurity** is

$$Q_\text{children} = \frac{N_1}{N} Q_1 + \frac{N_2}{N} Q_2$$

which gives more weight to the larger child. The **information gain** is the reduction in impurity:

$$\text{Gain} = Q_\text{parent} - Q_\text{children}$$

When $Q$ is entropy, this is the classic **information gain** of decision-tree literature. When $Q$ is Gini impurity, the same "reduction" idea is used; some libraries call the Gini-based analog just the "impurity reduction" rather than "information gain." The mechanism is identical.

### Distinguishing the Terms

| Term | Meaning |
|------|---------|
| **Impurity** ($Q$) | A number measuring how mixed a node is (Gini or entropy). |
| **Weighted child impurity** | The weighted average of child impurities, $\frac{N_1}{N}Q_1 + \frac{N_2}{N}Q_2$. |
| **Impurity reduction / information gain** | The difference $Q_\text{parent} - Q_\text{children}$; larger is better. |

Do not treat these as interchangeable. "Impurity" describes how mixed a node is; "information gain" describes how much a split *reduces* that mixture; and "weighted child impurity" is the intermediate quantity used to compute the gain.

---

## 13. Choosing the Best Classification Split

### The Weighted Impurity Criterion

To compare candidate classification splits we use the **weighted child impurity**:

$$Q = \frac{N_1}{N} Q_1 + \frac{N_2}{N} Q_2$$

where:

- $N$ is the number of observations at the parent,
- $N_1, N_2$ are the numbers routed to each child ($N_1 + N_2 = N$),
- $Q_1, Q_2$ are the children's impurity measures (Gini or entropy),
- the weights $N_1/N, N_2/N$ ensure a large child's impurity matters more.

A good split makes $Q$ **small** (children are purer and/or one child is tiny). Equivalently, since the parent's impurity is fixed, it makes the **gain** $Q_\text{parent} - Q$ large.

### Numerical Demonstration

Suppose the parent node has: **7 Yes, 3 No** (so $p = (0.7, 0.3)$, $G_\text{parent} = 1 - (0.49+0.09) = 0.42$).

Consider a candidate split feature $X \le s$ that routes as follows:

- **Left child**: 6 Yes, 1 No → $N_1=7$, $p=(6/7,1/7)$, $G_1 = 1 - (36/49 + 1/49) = 12/49 \approx 0.2449$.
- **Right child**: 1 Yes, 2 No → $N_2=3$, $p=(1/3,2/3)$, $G_2 = 1 - (1/9+4/9) = 4/9 \approx 0.4444$.

Weighted child impurity:

$$Q = \frac{7}{10}(0.2449) + \frac{3}{10}(0.4444) = 0.1714 + 0.1333 = 0.3048$$

Information gain (Gini reduction):

$$\text{Gain} = 0.42 - 0.3048 = 0.1152$$

The split reduced impurity from $0.42$ to $0.3048$, a gain of about $0.115$. Now compare several candidate splits the same way and pick the one with the smallest $Q$ (largest gain).

For every candidate split the procedure is identical:

1. Split the data into left and right.
2. Compute each child's class proportions.
3. Compute each child's impurity (Gini or entropy).
4. Weight the child impurities by their sizes.
5. Sum to get the split score $Q$.
6. Compare across all features and thresholds.
7. Select the split with the smallest $Q$.

After this process is clear, the code in Section 19 implements exactly these steps.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Classification trees and impurity reduction

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Classification trees

---

## 14. Recursive Binary Splitting and Greedy Optimization

### Recursion in One Paragraph

A **recursive function** is one that calls itself. It has two parts:

- a **base case** (a condition under which it returns an answer immediately, without recursing),
- a **recursive case** (where it calls itself on a smaller version of the problem).

For example, a function that counts down from $n$ prints $n$ and calls itself with $n-1$, stopping when $n = 0$ (the base case).

### Connecting Recursion to Trees

A decision tree is built by a recursive function that, at each node, either returns a leaf (base case) or picks a split and calls itself on the left and right halves:

```text
build_tree(data)
    if stopping_condition(data): return leaf(data)
    (feature, threshold) = best_split(data)
    left, right = split(data, feature, threshold)
    return node(feature, threshold,
                build_tree(left),   # recursive case
                build_tree(right))
```

This is why the method is called **recursive binary splitting**: we take the data, split it into two, and repeatedly split each piece the same way.

### Why It Is Greedy

The full decision-tree optimization problem — find the tree (with any number of splits, in any order) that best predicts — is intractable to solve exactly. Trees therefore use a **greedy** algorithm:

> It chooses the best split available at the current node rather than searching through every possible complete tree to guarantee a globally optimal tree.

At each node the algorithm looks only at that node's data and picks the locally best split. It never looks ahead to see whether a locally second-best split would lead to a better overall tree. This is the same idea as other greedy algorithms: make the best immediate choice and move on. The result is a good tree, not an optimal one. This distinguishes tree construction from closed-form or convex-optimization methods used by linear and logistic regression, which *do* reach a global optimum.

---

## 15. The Full Tree-Building Algorithm

### Regression

```text
Start with all observations
        ↓
Check stopping conditions
        ↓
Search features
        ↓
Search thresholds
        ↓
Calculate split error (squared error of both children)
        ↓
Choose best split (smallest total error)
        ↓
Split data
        ↓
Recursively build children
        ↓
Create leaves (store each region's mean)
```

### Classification

```text
Start with all observations
        ↓
Check stopping conditions
        ↓
Search features
        ↓
Search thresholds
        ↓
Calculate impurity of both children (Gini or entropy)
        ↓
Choose best split (smallest weighted child impurity / largest gain)
        ↓
Split data
        ↓
Recursively build children
        ↓
Create leaves (store majority class and class proportions)
```

### Pseudocode (shared structure)

```text
function BUILD(data, depth):
    if node_count(depth) > max_depth
       or len(data) < min_samples_split
       or data is pure:
        return LEAF(prediction from data)

    best = null
    for each feature j:
        values = sort(unique values of data[:, j])
        for each pair of consecutive values (v_k, v_{k+1}):
            s = (v_k + v_{k+1}) / 2            # candidate midpoint threshold
            score = split_score(data, j, s)    # error or weighted impurity
            if best is null or score < best.score:
                best = (j, s, score)

    (j, s) = feature/threshold of best
    left  = BUILD(row_i where x_ij <= s, depth+1)
    right = BUILD(row_i where x_ij >  s, depth+1)
    return NODE(j, s, left, right)
```

Candidate thresholds are **midpoints between consecutive distinct feature values** (after sorting), exactly as in the manual example of Section 8 and in the implementation of Section 19. The algorithm does not test every raw value as a threshold; it tests one midpoint between each pair of neighbouring distinct values per feature.

This is the algorithmic heart of the chapter. Sections 7-13 explained *which score is used* (squared error for regression, weighted impurity for classification); Sections 15-17 explain *when to stop*.

---

## 16. Stopping Criteria

> Why can't a tree simply split forever?

A tree can always split until each leaf contains identical training values, at which point it has zero training error. But as we saw in the regression example (Section 8), further splits essentially memorize individual points. That is overfitting, so we stop splitting under certain conditions.

### The Stopping Conditions

| Criterion | What it controls |
|-----------|------------------|
| **Maximum depth** (`max_depth`) | The deepest the tree is allowed to grow; depth 0 is a single leaf. |
| **Minimum samples for a split** (`min_samples_split`) | A node must contain at least this many samples before it is allowed to split. |
| **Minimum samples in a leaf** (`min_samples_leaf`) | A split is rejected if it would create a child with fewer than this many samples. |
| **Maximum number of leaves** (`max_leaf_nodes`) | Stop when the tree has this many leaves already. |
| **No useful split** | If no split reduces the score, stop (the best score equals the parent's score). |
| **Pure node** | If all observations share one class (classification), no split can help; make a leaf. |

### Effect on Tree Behavior

| Change | Tree size | Bias | Variance | Overfitting | Underfitting |
|--------|-----------|------|----------|-------------|--------------|
| More restrictive stopping (smaller depth, larger min samples) | Smaller | Higher | Lower | Less | More |
| Less restrictive stopping (deeper, smaller min samples) | Larger | Lower | Higher | More | Less |

- A **very shallow** tree (e.g. depth 1, a single question) cannot capture the structure, so it underfits: high bias.
- A **very deep** tree fits the training data almost exactly, so it overfits: high variance.

The right stopping point balances bias against variance — exactly the bias-variance tradeoff studied earlier (ISL, Section 2.2.2). The hyperparameters listed here *are* the knobs that tune this tradeoff.

---

## 17. Overfitting and the Bias-Variance Tradeoff

### Why Trees Overfit Easily

Decision trees are notoriously prone to overfitting because a tree can be grown deep enough to memorize the training set:

```text
Shallow tree
→ simpler model
→ higher bias
→ lower variance

Deep tree
→ complex model
→ lower training error
→ higher variance
→ potential overfitting
```

With enough splits, every training observation can end up in its own leaf. Training error then drops to (almost) zero, because each training point is essentially its own "region". But on unseen data these tiny regions predict poorly: they encode noise, not signal. This is the familiar story of a model with near-zero training error and high test error.

### Connecting to the Existing Bias-Variance Material

In the earlier regression chapter we decomposed test error as

$$\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Trees illustrate this sharply. A deep tree has low bias (it fits the training structure well) but very high variance (small changes in the training data produce very different trees). A shallow tree has lower variance but higher bias. This is why tree performance on unseen data is governed by controlling complexity, either by stopping early (Section 16) or by pruning (Section 18), and why ensemble methods (Section 29) average many trees to reduce variance.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Tree pruning and complexity control

---

## 18. Pruning

Stopping early is one way to control complexity. **Pruning** is the alternative: grow a full tree first, then remove unnecessary branches.

> A fully grown tree may describe noise in the training data. Pruning removes unnecessary branches to obtain a simpler subtree.

### Key Terms

- **Fully grown tree**: a tree grown until every leaf is pure or otherwise minimal.
- **Subtree**: any tree obtained by removing branches from a larger tree.
- **Complexity**: loosely, the size of the tree, usually measured by the number of leaves $|T|$.
- **Pruning**: removing branches to trade a little training fit for a simpler, better-generalizing tree.
- **Validation**: pruning decisions are made by checking how candidate subtrees perform on held-out data or a complexity penalty, not just on training data.

### Cost-Complexity Pruning

We need a principled way to decide which branches to remove. **Cost-complexity pruning** adds a penalty for tree size to the training error:

$$R_\alpha(T) = R(T) + \alpha |T|$$

where:

- $T$ is a subtree (tree),
- $R(T)$ is the training error of $T$ (RSS for regression; misclassification/impurity measure for classification),
- $|T|$ is the number of leaves (terminal nodes) in $T$ — its complexity,
- $\alpha \ge 0$ is a tuning parameter controlling the penalty for complexity.

The goal is to find the subtree $T$ that minimizes $R_\alpha(T)$. As $\alpha$ increases, larger trees are penalized more heavily, so the optimal $T$ becomes smaller:

```text
Small α
→ complexity is weakly penalized
→ larger tree

Large α
→ complexity is strongly penalized
→ smaller tree
```

- When $\alpha = 0$, $R_\alpha(T) = R(T)$, and a fully grown tree minimizes it (no penalty).
- As $\alpha \to \infty$, the best tree shrinks toward a single leaf (the penalty makes any split not worth its added leaf).

### Pruning vs. Early Stopping

Pruning and early stopping are **related but not identical**:

- **Early stopping** prevents the tree from growing beyond a point during construction (set `max_depth`, `min_samples_split`, etc.). It makes a decision *before* the deep branches exist.
- **Pruning** grows the tree fully first, then removes branches *after* construction, using a validation criterion or a cost-complexity penalty.

Both control complexity and both fight overfitting, but they operate at different times and with different information (pruning can see the fully grown tree and the effect of removing a specific branch). This is captured in scikit-learn by `ccp_alpha` (cost-complexity pruning) standing alongside the early-stopping parameters.

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Tree pruning and cost-complexity control

> Book: ESL  
> Chapter: 10 - Additive Models, Trees, and Related Methods  
> Topic: Cost-complexity pruning

---

## 19. NumPy Implementation From Scratch

Only now, with all the theory in place, do we implement the algorithm. The implementation is deliberately educational: the logic is visible in plain loops, and every step maps to a formula from earlier sections.

### 19.1 Impurity and Error Helpers

```python
import numpy as np

def mean_squared_leaf(y):
    """Leaf prediction minimizes squared error -> the mean."""
    return np.mean(y)

def squared_error(y):
    mu = np.mean(y)
    return np.sum((y - mu) ** 2)

def gini(probs):
    """Gini impurity: 1 - sum(p_k^2)."""
    return 1.0 - np.sum(probs ** 2)

def entropy(probs):
    """Entropy: -sum(p_k log p_k). 0*log(0) = 0."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def class_proportions(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    return counts / counts.sum()
```

`class_proportions` returns the vector $p = (p_1, \dots, p_K)$, which `gini` and `entropy` consume, mirroring Sections 10 and 11.

### 19.2 Best Split Search

The core is finding the feature and threshold that minimize the split score. For regression the score is the total squared error (Section 8); for classification it is the weighted child impurity (Section 13).

```python
def best_regression_split(X, y):
    n_samples, n_features = X.shape
    best_err = np.inf
    best = None
    parent_err = squared_error(y)

    for j in range(n_features):
        values = np.sort(np.unique(X[:, j]))
        for k in range(1, len(values)):
            s = (values[k - 1] + values[k]) / 2.0     # threshold between values
            left  = y[X[:, j] <= s]
            right = y[X[:, j] >  s]
            total = squared_error(left) + squared_error(right)
            if total < best_err:
                best_err, best = total, (j, s)
    return best, best_err
```

Candidate thresholds are placed **midway between consecutive distinct feature values** — exactly the "between values" idea used in the manual regression walkthrough (Section 8). The classification version is analogous, using `gini`/`entropy` and weighting child impurities by sample counts.

### 19.3 Node and the Recursive Builder

```python
def build_tree(X, y, depth, max_depth, min_samples_split):
    n = len(y)

    # stopping conditions
    if depth >= max_depth or n < min_samples_split or len(np.unique(y)) == 1:
        return Leaf(y)

    (j, s) = best_split(X, y)
    if j is None:              # no useful split found
        return Leaf(y)

    left_idx = X[:, j] <= s
    return InternalNode(j, s,
        build_tree(X[left_idx], y[left_idx], depth + 1, max_depth, min_samples_split),
        build_tree(X[~left_idx], y[~left_idx], depth + 1, max_depth, min_samples_split))
```

The base cases (leaf) are the stopping conditions of Section 16; the recursive cases build the left and right subtrees. The full `DecisionTreeRegressor` and `DecisionTreeClassifier` classes in the accompanying scripts wrap this builder with `fit`/`predict`/`predict_proba` methods and add a prediction-traversal routine (Section 20).

### 19.4 API

```python
tree = DecisionTreeRegressor(max_depth=2, min_samples_split=2)
tree.fit(X, y)
predictions = tree.predict(X)
```

```python
tree = DecisionTreeClassifier(criterion='gini', max_depth=2)
tree.fit(X, y)
predictions  = tree.predict(X)
probabilities = tree.predict_proba(X)
```

The important logic is not hidden behind libraries: the split search, impurity computation, recursion, and traversal are all implemented directly with NumPy in the provided scripts. No `sklearn.tree` is used to build the custom tree.

---

## 20. Tree Node Representation

The tree is represented in Python as a structure of **nodes**. A node is one of two kinds:

- an **internal (decision) node**, which stores a split — the feature and threshold — and two children;
- a **leaf (terminal node)**, which stores the prediction.

A convenient representation gives each node these fields:

```text
feature      the index j of the splitting feature   (internal node only)
threshold    the value s of the split                (internal node only)
left         the left child node (feature <= threshold)
right        the right child node (feature > threshold)
prediction   the stored prediction / class / counts  (leaf only)
```

An internal node stores `feature`, `threshold`, `left`, `right`; a leaf stores `prediction`.

**Prediction traversal** sends a new observation down the tree:

```text
if leaf:
    return prediction

if x[feature] <= threshold:
    go left
else:
    go right
```

This corresponds **directly** to traversing the mathematical tree: at each decision node we answer the question "$X_j \le s$?", follow the matching branch, and finally read the leaf's prediction.

```python
def predict_one(node, x):
    if node.is_leaf:
        return node.prediction
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    return predict_one(node.right, x)
```

---

## 21. Tiny Dataset Walkthrough

These walkthroughs let every calculation be reproduced by hand. They are exactly the workflows performed by the provided scripts.

### 21.1 Regression Walkthrough

Use the one-feature data from Section 8: $X = [1,2,3,4,5,6]$, $y = [2,3,4,10,11,12]$.

1. **Start**: all 6 observations at the root.
2. **Candidate splits**: $X \le 1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5$.
3. **Split errors**: $70.00, 39.25, \mathbf{4.00}, 39.25, 70.00$.
4. **Best split**: $X \le 3.5$ (error $4$).
5. **Recursive splitting**: the right child $\{4,5,6\}$ still has $y$-values $[10,11,12]$ that could be split, but for a shallow tree (e.g. `max_depth=1`) we stop here.
6. **Leaves**: left leaf predicts mean$(2,3,4)=3$; right leaf predicts mean$(10,11,12)=11$.
7. **Predictions**: any $X \le 3.5 \to 3$; any $X > 3.5 \to 11$.

Hand check the leaf means: left = $(2+3+4)/3 = 3$; right = $(10+11+12)/3 = 11$. This reproduces the means chosen in Section 8.

### 21.2 Classification Walkthrough

Set up a tiny binary dataset:

| $X$ | class |
|-----|-------|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |
| 5 | 1 |
| 6 | 0 |

1. **Class distributions**: root has 3 zeros and 3 ones → $p=(0.5,0.5)$, $G = 0.5$.
2. **Candidate splits** (thresholds midway between values): $X \le 1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5$.
3. **Evaluate $X \le 2.5$**: left $\{1,2\}$ = 2 zeros, 0 ones ($G_1=0$); right $\{3,4,5,6\}$ = 1 zero, 3 ones ($p=(0.25,0.75)$, $G_2 = 1-(0.0625+0.5625)=0.375$). Weighted: $(2/6)(0) + (4/6)(0.375) = 0.25$. Gain $= 0.5 - 0.25 = 0.25$.
4. **Evaluate $X \le 3.5$**: left $\{1,2,3\}$ = 2 zeros, 1 one ($p=(2/3,1/3)$, $G_1 = 1-(4/9+1/9)=4/9\approx0.444$); right $\{4,5,6\}$ = 1 zero, 2 ones ($G_2=0.444$). Weighted: $0.444$. Gain $= 0.5-0.444=0.056$.
5. **Evaluate remaining splits** similarly; the best is $X \le 2.5$ (lowest weighted impurity $0.25$).
6. **Best split**: $X \le 2.5$.
7. **Recursive splitting**: the right child $\{3,4,5,6\}$ can be split further; a shallow tree stops.
8. **Predictions**: $X \le 2.5 \to$ class 0 (majority); $X > 2.5 \to$ class 1 (majority of that leaf: 3 ones vs 1 zero).

These numbers match the formulas in Sections 10 and 13, and the code in `03_best_split_classification.py` prints the same table.

---

## 22. Visualization

Matplotlib is used to make the predictions concrete.

For **regression**, we plot the training observations and overlay the tree's **piecewise-constant** prediction: a horizontal segment on each region at its leaf mean, jumping at the split thresholds.

For **classification**, we use a simple 2D dataset and color the plane according to the tree's predicted class. The boundary between colors shows the tree's **decision regions**.

An important fact to see in these plots: ordinary decision trees make **axis-aligned splits**. Each split rules on a single feature against a single threshold ($X_j \le s$), so every decision boundary is a vertical or horizontal line in feature space. The resulting regions are axis-aligned rectangles. This is what the earlier two-feature example (Section 6) depicted, and it explains both the interpretability of trees and one of their limitations (Section 24).

![Regression tree piecewise-constant fit](../plots/04_regression_tree.png)
*The fitted regression tree (solid crimson) is a staircase: constant within each region, jumping at the split threshold $X=3.5$. Blue points are the training observations, with horizontal dashed lines marking each leaf's mean.*

![Classification tree decision regions](../plots/05_classification_regions.png)
*Axis-aligned decision regions of a shallow classifier on a simple 2D dataset. Each boundary is a vertical or horizontal line because every split uses a single feature and a single threshold.*

---

## 23. Decision Trees vs. Previously Studied Models

| Property | Linear Regression | Logistic Regression | Decision Tree |
|----------|-------------------|----------------------|---------------|
| Main task | Regression | Classification | Regression / Classification |
| Prediction structure | Linear function | Logistic transformation of a linear function | Piecewise-constant regions |
| Nonlinear relationships | Limited without feature engineering | Limited without feature engineering | Naturally supported |
| Feature scaling | Often useful | Often useful | Usually unnecessary |
| Interactions | Need explicit terms | Need explicit terms | Naturally discovered |
| Interpretability | High | High | High for small trees |
| Overfitting control | Regularization | Regularization | Depth, minimum samples, pruning, etc. |

The differences are more than cosmetic — they come from how each model represents the mapping $f(X)$:

- **Linear regression** assumes $f$ is linear. It needs *explicit* polynomial or interaction terms to capture non-linearity, and coefficients are only comparable after scaling.
- **Logistic regression** assumes the *log-odds* are linear. Like linear regression it is a parametric, global function, and it inherits the same need for feature engineering when the boundary is non-linear.
- **Decision trees** make **no global functional assumption**. They approximate $f$ with locally constant predictions, automatically capturing non-linearity and interactions. Because each split compares a feature to a threshold, the *scale* of a feature does not matter (a threshold is just a number on that feature's own scale), so trees usually need no feature scaling.

Interpretability differs too: a linear coefficient summarizes a global effect, while a small tree reads as a list of if-then rules. Overfitting is controlled differently as well — trees have no "coefficients" to shrink, so control happens through depth, minimum-sample limits, and pruning rather than regularization.

---

## 24. Advantages and Disadvantages

### Advantages

- **Easy to understand and interpret**: a small tree reads like a set of if-then rules; one can explain a prediction by following the path from root to leaf.
- **Easy to visualize**: the whole model can be drawn, unlike a high-dimensional linear function.
- **Little preprocessing required**: missing structure aside, trees do not need dummy-coding tricks for threshold comparisons, do not require scaling, and handle mixed feature types more gracefully.
- **Generally no feature scaling required**: a split compares one feature to a threshold $X_j \le s$; rescaling a feature merely rescales $s$. The split's *quality* is unchanged.
- **Handles nonlinear relationships naturally**: by partitioning, trees follow curved structure without polynomial terms.
- **Handles interactions automatically**: a tree nests questions, so the effect of one feature can depend on another without the user building interaction terms.
- **Works for regression and classification**: the same recursive-partitioning machinery serves both tasks (mean vs. majority class).

### Disadvantages

- **High variance**: small changes in the training data can change the tree structure substantially, because a slightly different top split ripples through all descendants.
- **Sensitive to small changes in training data**: directly follows from high variance; a different first split can produce a very different tree.
- **Prone to overfitting**: a tree can split until it memorizes the training set; complexity must be actively controlled (stopping, pruning).
- **Axis-aligned splits can be restrictive**: every boundary is parallel to a feature axis (Section 22). Relationships that are diagonal, say, may need many splits to approximate, inflating tree size.
- **Often less predictive than ensemble methods**: because of high variance, a single tree is usually outperformed by bagging, random forests, and boosting — which build on single trees (Section 29).

Each disadvantage follows from a mechanism already described: variance (from greedy recursive partitioning), overfitting (from unlimited depth), and axis-alignment (from single-feature, single-threshold splits).

**Reference:**

> Book: ISL  
> Chapter: 9 - Tree-Based Methods  
> Topic: Trees versus linear models; advantages and disadvantages of trees

---

## 25. Hyperparameters

These parameters control the underlying algorithm. The purpose of each is conceptual — this is not a scikit-learn API reference.

### `max_depth`
- **Controls**: how many nested splits the tree is allowed (height).
- **Increases**: deeper, more complex tree, lower training error, more overfitting risk.
- **Decreases**: shallower tree, higher bias, less variance.
- **Complexity**: directly limits the number of splits; the strongest single complexity control.
- **Overfitting**: a small `max_depth` is the simplest way to prevent the tree from memorizing individual points.

### `min_samples_split`
- **Controls**: how many samples a node must contain before it may split.
- **Increases**: fewer nodes split → smaller tree.
- **Decreases**: more nodes split → larger tree.
- **Complexity**: a node that is too small is not allowed to create new branches.
- **Overfitting**: raising it prevents splits with very few samples, which are usually noise.

### `min_samples_leaf`
- **Controls**: the minimum size of any child produced by a split.
- **Increases**: each leaf must contain more samples → smaller, smoother tree.
- **Decreases**: leaves can be tiny → deeper, choppier tree.
- **Complexity**: caps the "resolution" of the predictions; a stricter value lowers variance.
- **Overfitting**: forbids leaves with a handful of samples, which tend to memorize points.

### `max_leaf_nodes`
- **Controls**: the total number of terminal nodes.
- **Increases**: more regions allowed → more complex fit.
- **Decreases**: fewer regions → simpler, higher-bias fit.
- **Complexity**: caps the total size of the tree from the bottom up.
- **Overfitting**: an upper bound on leaf count bounds the model's capacity directly.

### `criterion`
- **Controls**: the score used to judge splits (e.g. `squared_error` for regression; `gini` or `entropy` for classification).
- **Increases/Decreases**: choosing a different impurity measure usually changes the tree only slightly; Gini and entropy behave similarly.
- **Complexity**: the measure affects which splits are *preferred*, not the depth limits.
- **Overfitting**: not a complexity control per se; it selects among candidate splits given the other limits.

### `ccp_alpha`
- **Controls**: the cost-complexity penalty $\alpha$ from Section 18.
- **Increases**: stronger penalty on leaves → smaller pruned subtree.
- **Decreases**: weaker penalty → larger subtree (toward the unpruned tree).
- **Complexity**: directly shrinks the tree after full growth.
- **Overfitting**: growing `ccp_alpha` trades training fit for generalization by removing branches.

In summary, the first five control the tree *during* construction (early stopping), while `ccp_alpha` controls it *after* construction (pruning). All of them tune the bias-variance tradeoff by modifying tree size.

---

## 26. scikit-learn Comparison

After the NumPy implementation, the corresponding library version is a single import away:

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
```

The library provides the **same underlying algorithm** we implemented:

- greedy recursive binary partitioning with the same candidate-threshold idea,
- `squared_error` splits for regression, `gini`/`entropy` for classification,
- the same stopping parameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`) and cost-complexity pruning (`ccp_alpha`),
- `fit`, `predict`, and (for the classifier) `predict_proba`.

So the concepts map one-to-one to our implementation. However, it is **not correct** to imply the custom implementation is equivalent to scikit-learn internally. Important differences:

| Aspect | Educational implementation | scikit-learn |
|--------|----------------------------|--------------|
| Data representation | Python object graph of nodes | Compiled `Tree` structure (C arrays); a Cython-compiled CART |
| Threshold search | Naïve scan of all midpoints | Sorted-feature presort / optimized splits, class-weighting, missing-value support |
| Split criterion | Hand-rolled Gini/entropy/SSE | `criterion` registry with several options |
| Extras | None | Sample weights, class weights, `max_features`, random splitter, cost-complexity path, richer predict APIs |
| Speed | Slow on large data | Highly optimized, designed for production use |

For API-specific behavior, refer to the official scikit-learn Decision Trees documentation rather than assuming our implementation covers every option. Our version exists to teach the *algorithm*; the library exists to use it at scale.

---

## 27. Computational Considerations

Finding the best split is the expensive part of tree construction, because the algorithm may need to examine:

- **multiple features**: the split search loops over all $p$ features at every node,
- **many candidate thresholds**: up to one threshold between each pair of adjacent distinct values of a feature,
- **many observations**: computing the split score involves, in the naïve approach, scanning the samples in each child,
- **many nodes**: this whole search is repeated recursively at every internal node.

So the running time depends on the number of features, the number of distinct thresholds, the sample size, and the number of nodes. An exact closed-form complexity depends on these implementation details, so we avoid an overly precise statement. Two broad regimes are useful:

**Educational naïve implementation**

```text
for each node:
    for each feature:
        for each candidate threshold:
            compute error/impurity over all samples in the node
```

Every candidate split rescans the node's samples, so the cost can grow roughly with the number of nodes times features times thresholds times samples-in-node. Fine on the small teaching datasets in this module; impractical on large data.

**Optimized production implementation**

Libraries use tricks such as:

- **sorting** each feature's values once and updating counts incrementally as the threshold slides (avoiding a full rescan for every threshold);
- **precomputed ordering** and cached impurity statistics per node;
- compiled C/Cython inner loops instead of Python loops.

These reduce the constant factor and per-threshold work dramatically. The *algorithm* is the same greedy recursive partitioning; the *engineering* differs. Do not assume our educational code has production speed — it is written for clarity.

---

## 28. Common Misconceptions

### "Decision Trees need feature scaling."

No. A split compares a single feature to a threshold ($X_j \le s$). Rescaling a feature rescales the threshold with it; the split's *quality* is unchanged. Trees are largely scale-invariant, unlike the gradient-descent methods in earlier chapters, which do need scaled features.

### "A deeper tree is always better."

False. A deeper tree has lower training error but higher variance and overfits (Sections 16-17). Performance on unseen data is best at some intermediate depth. Depth is a bias-variance knob, not a "bigger is better" knob.

### "Zero training error means the model is good."

No. Training error measures fit to the training set, not generalization. A tree that memorizes every training point achieves near-zero training error precisely because it overfits, and performs poorly out of sample. Always evaluate on held-out data.

### "Decision Trees only work for classification."

False. Regression trees predict a continuous response with region means (Sections 6-8), and `DecisionTreeRegressor` in scikit-learn is a regression tree. Both regression and classification use the same partitioning machinery.

### "Gini impurity and the Gini coefficient are the same."

They are different quantities that happen to share a name. Gini impurity ($G = 1 - \sum p_k^2$) measures class mixture in a node (Section 10). The Gini *coefficient* is an unrelated measure of statistical dispersion (e.g., income inequality). Do not conflate them.

### "Decision Trees search every possible tree."

No. They use **greedy recursive binary splitting** (Section 14): at each node they pick the locally best split and never search the space of complete trees. The result is a good, not guaranteed-optimal, tree.

### "Pruning and early stopping are exactly the same."

They are different mechanisms (Section 18). Early stopping prevents growth during construction; pruning grows the tree fully and then removes branches. Both reduce complexity, but at different times and with different information.

### "Decision Trees can represent every possible decision boundary efficiently."

No. Ordinary trees use **axis-aligned** splits (Section 22), so boundaries are vertical/horizontal lines in feature space. Diagonal or otherwise non-axis-aligned boundaries need many splits to approximate, which inflates tree size. Slightly rotated or multi-feature splits can help, but standard trees stay axis-aligned.

---

## 29. Connection to Ensemble Learning

The main topic ends here by looking forward. A single tree has **high variance** (Section 24): small changes in the training data can yield very different trees. Ensemble methods exploit this by combining many trees:

```text
Decision Tree
      ↓
Bagging
      ↓
Random Forest
      ↓
Boosting
```

- **Bagging** trains many trees on resampled versions of the data and averages their predictions, reducing variance.
- **Random Forests** are bagged trees that also randomize the features available at each split, decorrelating the trees further.
- **Boosting** trains trees sequentially, each one focusing on the mistakes of the previous ones, reducing bias.

The shared motivation is that averaging or sequentially correcting many **(weak, high-variance) decision trees** produces a model that generalizes far better than any single tree. This is precisely why decision trees are worth studying even though a single tree is often less predictive than an ensemble.

These topics — bagging, random forests, boosting — will become separate chapters later. Here we only need the connection: **random forests and boosting build on decision trees**, and the reason traces to the tree's variance.

**Reference:**

> Book: ISLP  
> Chapter: 9 - Tree-Based Methods  
> Topic: Bagging, random forests, and boosting

> Book: ESL  
> Chapter: 16 - Random Forests  
> Topic: Bagging and random forests

> Book: ESL  
> Chapter: 11 - Boosting and Additive Trees  
> Topic: Boosting

---

## 30. Key Takeaways

- **Decision tree**: a model that partitions the feature space into regions and predicts with a constant (mean or majority class) per region.
- **Node**: a point in the tree where a question/split is made.
- **Split**: a question "$X_j \le s$" that divides a node's observations into two groups.
- **Threshold**: the value $s$ in the split.
- **Leaf**: a terminal node holding the prediction.
- **Region**: the subset of the feature space corresponding to a leaf.
- **Regression tree**: predicts each region's mean; the leaf prediction is the least-squares constant (Section 7).
- **Classification tree**: predicts each region's majority class and can output class probabilities.
- **Recursive binary splitting**: building the tree by repeatedly splitting data into two, recursively.
- **Squared-error criterion**: the regression split score, $\sum_{R_1}(y_i-\hat c_1)^2 + \sum_{R_2}(y_i-\hat c_2)^2$.
- **Gini impurity**: $1 - \sum p_k^2$; $0$ for a pure node.
- **Entropy**: $-\sum p_k \log p_k$; $0$ for a pure node.
- **Information gain**: $Q_\text{parent} - Q_\text{children}$, the impurity reduction from a split.
- **Weighted impurity**: $\frac{N_1}{N}Q_1 + \frac{N_2}{N}Q_2$, used to compare classification splits.
- **Greedy splitting**: picking the locally best split at each node, not an optimal global tree.
- **Stopping criteria**: depth/sample limits that end growth; they control tree size.
- **Overfitting**: a deep tree's training error is low but test performance poor.
- **Pruning**: removing branches from a fully grown tree.
- **Cost-complexity**: minimizing $R_\alpha(T) = R(T) + \alpha|T|$; larger $\alpha$ gives smaller trees.
- **Bias-variance tradeoff**: tree complexity tunes the balance; this drives both stopping and pruning.
- **Axis-aligned decision boundaries**: every split cuts along a single feature, so regions are rectangles.

---

## 31. References

> **References Summary**
>
> - James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python* (1st ed.). Springer.
> - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
> - scikit-learn developers. *Decision Trees* documentation. https://scikit-learn.org/stable/modules/tree.html
>
> Chapter-level references in this document:
>
> - ISLP **Chapter 9 — Tree-Based Methods**.
> - ESL **Chapter 10 — Additive Models, Trees, and Related Methods**.
> - ESL **Chapter 11 — Boosting and Additive Trees**.
> - ESL **Chapter 16 — Random Forests**.
>
> External references are intentionally chapter/topic level here. Exact page locations vary by edition and printing, so I have avoided unverified page citations.
