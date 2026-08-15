"""
Visualizing Table 1 from "Attention Is All You Need" (Section 4):
Complexity per layer, sequential operations, and max path length —
Self-Attention vs Recurrent vs Convolutional layers.

n = sequence length, d = representation dimension, k = kernel size (CNN)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Config (paper-realistic defaults) ----
d = 512          # representation dimension (d_model in the paper)
k = 3            # CNN kernel size
n_values = np.arange(1, 201)  # sequence length from 1 to 200 tokens

# ---- Complexity per layer (Table 1, column 1) ----
complexity_self_attn = n_values**2 * d          # O(n^2 * d)
complexity_rnn       = n_values * d**2          # O(n * d^2)
complexity_cnn        = k * n_values * d**2      # O(k * n * d^2)

# ---- Sequential operations required (Table 1, column 2) ----
seq_ops_self_attn = np.ones_like(n_values)       # O(1)
seq_ops_rnn       = n_values                     # O(n)
seq_ops_cnn       = np.ones_like(n_values)       # O(1)

# ---- Maximum path length (Table 1, column 3) ----
path_self_attn = np.ones_like(n_values)                       # O(1)
path_rnn       = n_values                                     # O(n)
path_cnn       = np.ceil(np.log(n_values) / np.log(k))        # O(log_k(n))
path_cnn[0] = 1  # log(1) = 0, clamp to 1 for n=1

colors = {"Self-Attention": "#EF9F27", "RNN": "#4C72B0", "CNN": "#55A868"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Self-Attention vs RNN vs CNN — Table 1, "Attention Is All You Need"',
             fontsize=13, fontweight="bold")

# --- Plot 1: Complexity per layer ---
ax = axes[0]
ax.plot(n_values, complexity_self_attn, label="Self-Attention  O(n²·d)", color=colors["Self-Attention"], linewidth=2.5)
ax.plot(n_values, complexity_rnn, label="RNN  O(n·d²)", color=colors["RNN"], linewidth=2.5)
ax.plot(n_values, complexity_cnn, label="CNN  O(k·n·d²)", color=colors["CNN"], linewidth=2.5)
ax.set_yscale("log")
ax.set_xlabel("Sequence length (n)")
ax.set_ylabel("Computational complexity per layer (log scale)")
ax.set_title("Complexity per Layer")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.axvline(d, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(d + 3, ax.get_ylim()[0]*3, f"n = d = {d}", fontsize=8, color="gray", rotation=90, va="bottom")

# --- Plot 2: Sequential operations ---
ax = axes[1]
ax.plot(n_values, seq_ops_self_attn, label="Self-Attention  O(1)", color=colors["Self-Attention"], linewidth=2.5)
ax.plot(n_values, seq_ops_rnn, label="RNN  O(n)", color=colors["RNN"], linewidth=2.5)
ax.plot(n_values, seq_ops_cnn, label="CNN  O(1)", color=colors["CNN"], linewidth=2.5, linestyle=(0, (5, 3)))
ax.set_xlabel("Sequence length (n)")
ax.set_ylabel("Minimum sequential operations")
ax.set_title("Sequential Operations Required\n(lower = more parallelizable)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# --- Plot 3: Max path length ---
ax = axes[2]
ax.plot(n_values, path_self_attn, label="Self-Attention  O(1)", color=colors["Self-Attention"], linewidth=2.5)
ax.plot(n_values, path_rnn, label="RNN  O(n)", color=colors["RNN"], linewidth=2.5)
ax.plot(n_values, path_cnn, label=f"CNN  O(log_k(n)), k={k}", color=colors["CNN"], linewidth=2.5)
ax.set_xlabel("Sequence length (n)")
ax.set_ylabel("Max path length between\nany two positions")
ax.set_title("Long-Range Dependency Path Length\n(lower = easier to learn distant relationships)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# annotate the actual values at n=200 so the CNN vs self-attention gap isn't lost visually
n_check = 200
sa_val = 1
cnn_val = path_cnn[n_check - 1]
ax.annotate(f"CNN needs ~{int(cnn_val)} layers\nto connect distant words",
            xy=(n_check, cnn_val), xytext=(n_check - 90, cnn_val + 25),
            fontsize=8, color=colors["CNN"],
            arrowprops=dict(arrowstyle="->", color=colors["CNN"], lw=1))
ax.annotate("Self-Attention: always 1",
            xy=(n_check, sa_val), xytext=(n_check - 90, sa_val + 8),
            fontsize=8, color=colors["Self-Attention"],
            arrowprops=dict(arrowstyle="->", color=colors["Self-Attention"], lw=1))

# --- Plot 3b: zoomed-in, Self-Attention vs CNN only, linear scale ---
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(n_values, path_self_attn, label="Self-Attention  O(1)", color=colors["Self-Attention"], linewidth=2.5)
ax2.plot(n_values, path_cnn, label=f"CNN  O(log_k(n)), k={k}", color=colors["CNN"], linewidth=2.5)
ax2.set_xlabel("Sequence length (n)")
ax2.set_ylabel("Max path length")
ax2.set_title("Zoomed in: Self-Attention vs CNN\n(RNN removed — its O(n) line would dwarf this scale)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 8)
plt.tight_layout()
plt.savefig("attention_vs_cnn_zoomed.png", dpi=200, bbox_inches="tight")

plt.tight_layout()
plt.savefig("attention_vs_cnn_zoomed.png", dpi=200, bbox_inches="tight")
plt.savefig("attention_comparison.png", dpi=200, bbox_inches="tight")
print("Saved.")