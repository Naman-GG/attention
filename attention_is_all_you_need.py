"""
Attention Is All You Need

Implements the core mechanism in Section 3.2 using just NumPy:
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

And Multi-Head Attention (Section 3.2.2):
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # step 1-2 - compare every query against every key, then scale
    scores = (Q @ K.T) / np.sqrt(d_k)          

    # step 3 - mask future positions for decoder self-attention
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # step 4 - softmax turns each row into a probability distribution
    weights = softmax(scores, axis=-1)          

    # step 5 - weighted sum of Value vectors
    output = weights @ V                         

    return output, weights


class MultiHeadAttention:

  #  Runs several attention heads in parallel (section 3.2.2)
  
    def __init__(self, d_model, num_heads, seed=0):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head (64 in the paper: 512 / 8)

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)

        # one shared projection matrix per Q/K/V, split into heads afterward
        self.W_q = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_k = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_v = rng.normal(scale=scale, size=(d_model, d_model))
        self.W_o = rng.normal(scale=scale, size=(d_model, d_model))

    def split_heads(self, x):
        # (seq_len, d_model) -> (num_heads, seq_len, d_k)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def __call__(self, x, mask=None):
        """
        x: (seq_len, d_model) — self-attention, so Q, K, V all come from x
        Returns: (seq_len, d_model), plus per-head attention weights for inspection
        """
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Qh = self.split_heads(Q)   # (num_heads, seq_len, d_k)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)

        head_outputs = []
        head_weights = []
        for h in range(self.num_heads):
            out, w = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask=mask)
            head_outputs.append(out)
            head_weights.append(w)

        seq_len = x.shape[0]
        concat = np.concatenate(head_outputs, axis=-1).reshape(seq_len, self.d_model)

        output = concat @ self.W_o
        return output, np.stack(head_weights)  # weights: (num_heads, seq_len, seq_len)


def causal_mask(seq_len):
#   Lower triangular mask for decoder self attention for position i can only see j <= i.
    return np.tril(np.ones((seq_len, seq_len)))


if __name__ == "__main__":
    # Toy example: 6 tokens, each represented as a d_model=8 vector
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len, d_model, num_heads = len(tokens), 8, 2

    rng = np.random.default_rng(42)
    X = rng.normal(size=(seq_len, d_model))  # stand-in for embeddings

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=1)

    print("=== Encoder self-attention (no mask, sees full sentence) ===")
    out, weights = mha(X)
    print("Output shape:", out.shape)  # (6, 8) — same shape as input, now context-aware
    print("\nHead 0 attention weights (rows=query token, cols=key token):")
    print(np.round(weights[0], 2))

    print("\n=== Decoder self-attention (causal mask, can't see future tokens) ===")
    mask = causal_mask(seq_len)
    out_masked, weights_masked = mha(X, mask=mask)
    print("Head 0 attention weights (upper-right triangle should be ~0):")
    print(np.round(weights_masked[0], 2))