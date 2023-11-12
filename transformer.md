# Transformer

- [Intro to Transformer](https://youtu.be/XfpMkf4rD6E)
- [Transformer in code](https://youtu.be/kCc8FmEb1nY)

## Attention

- Attention is a communication mechanism allowing individual nodes (tokens) within a sequence (sentence) to interact with one another.
```
# x is a token in a sentence.

q = W_q(x)
k = W_k(x)

att = q @ k.T # attention score

v = W_v(x)

out = att @ v
```
- `q`: Current token at hand. The lens through which the token views other tokens.
- `k`: What each token can offer to a query. Represents the trait or attribute other tokens possess.
- `q @ k.T` Compatibility or "affinity" between the query and each of the keys, enabling a "soft-search" as opposed to discrete dictionary lookups.
- `v`: Information each token will pass onto the query, depending on the compatibility manifested by attention score.
- $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
  - Without a normalization by $\sqrt{d_k}$, the var will be higher than 1, and because of softmax, vectors will converge to one-hot vector, defeating the purpose of "soft"-search.
- Tensor dims:
  - `q @ k.T`: `(B, T, C) @ (B, C, T) --> (B, T, T)`
  - `att @ v`: `(B, T, T) @ (B, T, C) --> (B, T, C)`
  - `T`: block size = sequence length = context length = time dim
  - `C`: vector dim of tokens
  - Transpose in `k.T` should only transpose T and C dims, but not the B dim. In actual implementations, it is `k.transpose(-2, -1)`.
- In Self-Attention, q, k, and v comes from the same input x. In Cross-Attention, input for k and v are from the Encoder.
