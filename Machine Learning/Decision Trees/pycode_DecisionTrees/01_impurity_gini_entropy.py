#----------------------------
#  01: Gini Impurity and Entropy from scratch
#  Node examples used in the documentation (Section 10-11)
#  G = 1 - sum(p_k^2)          (Gini impurity)
#  H = -sum(p_k log p_k)       (entropy)
#----------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------
#  Impurity helpers
#------------------------------
def gini(probs):
    # Gini impurity: 1 - sum(p_k^2)
    return 1.0 - np.sum(probs ** 2)

def entropy(probs):
    # Entropy: -sum(p_k log p_k); ignore classes with p_k = 0 (0 log 0 = 0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def class_proportions(y, n_classes):
    # counts -> proportions p = (p_1, ..., p_K)
    counts = np.bincount(y, minlength=n_classes)
    return counts / counts.sum()

#------------------------------
#  Three example nodes (binary)
#  Node A: 10 Yes / 0 No   -> pure
#  Node B:  5 Yes / 5 No   -> maximally mixed
#  Node C:  8 Yes / 2 No   -> in between
#------------------------------
nodes = {
    "Node A (10 Yes, 0 No)": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    "Node B ( 5 Yes, 5 No)": np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    "Node C ( 8 Yes, 2 No)": np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
}

print("Impurity of the documentation example nodes\n")
print(f"{'Node':<24}{'p(Yes)':>8}{'p(No)':>8}{'Gini':>10}{'Entropy':>10}")
for name, y in nodes.items():
    p = class_proportions(y, 2)
    print(f"{name:<24}{p[1]:>8.2f}{p[0]:>8.2f}{gini(p):>10.4f}{entropy(p):>10.4f}")

print()
print("  Node A is pure        -> Gini = 0,  Entropy = 0")
print("  Node B is most mixed  -> Gini = 0.5 (max for 2 classes)")
print("  Node C is in between  -> Gini = 0.32")

#------------------------------
#  How impurity grows as classes mix (for the plot)
#------------------------------
p_yes = np.linspace(0.0, 1.0, 101)        # probability of class 1
p_no  = 1.0 - p_yes
gini_vals   = [gini(np.array([a, b])) for a, b in zip(p_no, p_yes)]
entropy_vals = [entropy(np.array([a, b])) for a, b in zip(p_no, p_yes)]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_yes, gini_vals,   '-',  color='steelblue',  label='Gini impurity: 1 - sum(p_k^2)')
ax.plot(p_yes, entropy_vals, '--', color='crimson',    label='Entropy: -sum(p_k log p_k)')
ax.axvline(0.5, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Proportion of class 1, $p_1$')
ax.set_ylabel('Impurity')
ax.set_title('Gini impurity and entropy for a two-class node')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / '01_impurity_gini_entropy.png', dpi=150)
plt.close()
print("\n[Plot saved: plots/01_impurity_gini_entropy.png]")
