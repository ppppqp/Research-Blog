---
author: Retep
pubDatetime: 2026-05-28T12:00:00.000-07:00
title: "[Book Club] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
featured: false
draft: false
tags:
  - Book Club
description: "Paper reading notes on FlashAttention, online softmax, IO complexity, and block-sparse attention."
---

Paper: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Dao et al., 2022.

The main point of the paper is simple:

> Attention is not slow only because it has many FLOPs. It is also slow because it moves too much data between GPU memory levels.

FlashAttention keeps the math of exact softmax attention, but changes the order of computation. Instead of materializing the full attention matrix in HBM, it computes attention one tile at a time and keeps temporary values on-chip.

## Standard Attention

For one attention head:

$$
S = QK^T
$$

$$
P = \text{softmax}(S)
$$

$$
O = PV
$$

where:

- $Q \in \mathbb{R}^{N \times d}$: queries
- $K \in \mathbb{R}^{N \times d}$: keys
- $V \in \mathbb{R}^{N \times d}$: values
- $N$: sequence length
- $d$: head dimension

The problem is the middle matrix:

$$
S, P \in \mathbb{R}^{N \times N}
$$

For long sequences, $N \times N$ is huge. Standard attention writes $S$ to HBM, reads it back for softmax, writes $P$ to HBM, then reads $P$ back to multiply by $V$.

That means the GPU spends a lot of time moving intermediate matrices, not just doing math.

## Memory Hierarchy

The paper focuses on three memory levels:

- on-chip SRAM: small, very fast
- GPU HBM: large, slower than SRAM
- CPU DRAM: larger, much slower for GPU kernels

The important gap is SRAM vs HBM. SRAM is fast but too small to hold the full $N \times N$ attention matrix. HBM is large enough, but reading and writing it is expensive.

So the target is:

> minimize HBM reads and writes.

This is what the paper calls **IO-awareness**. The algorithm is designed around memory traffic, not only FLOPs.

## Why Exact Attention Can Be Streamed

At first, softmax looks like it needs the full row of $S$:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

For numerical stability, we usually subtract the row max:

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}
$$

where:

$$
m = \max_j x_j
$$

This seems to require seeing every score before computing the final output. FlashAttention avoids storing every score by using **online softmax**.

For each row, keep two running values:

- $m$: running maximum score
- $\ell$: running softmax denominator

When a new score block arrives, compute:

$$
m_{new} = \max(m_{old}, m_{block})
$$

The old denominator was scaled using $m_{old}$. The new denominator must use $m_{new}$. So we rescale the old value:

$$
\ell_{new}
= e^{m_{old} - m_{new}}\ell_{old}
+ \sum_{x \in block} e^{x - m_{new}}
$$

This is the key correction term. If the new block contains a larger max, the old partial sum is adjusted down.

The output accumulator is corrected in the same way:

$$
O_{new}
= e^{m_{old} - m_{new}}O_{old}
+ \sum_{x_j \in block} e^{x_j - m_{new}}V_j
$$

At the end:

$$
O = \frac{O_{acc}}{\ell}
$$

So we never need to store the full score row or the full probability row.

## FlashAttention Forward Pass

The standard implementation does this:

```text
Q, K, V are in HBM

1. compute S = QK^T
2. write S to HBM
3. read S from HBM
4. compute P = softmax(S)
5. write P to HBM
6. read P from HBM
7. compute O = PV
8. write O to HBM
```

FlashAttention instead uses tiles:

```text
for each K,V tile:
    load K tile into SRAM
    load V tile into SRAM

    for each Q tile:
        load Q tile into SRAM
        load current output state for this Q tile

        compute score tile QK^T
        update online softmax max m
        update online softmax denominator l
        update output accumulator O

        write updated output state back to HBM
```

The score tile is temporary. It lives on-chip only long enough to update the online softmax state.

The paper's Algorithm 1 stores three things per row between tile updates:

- $O$: current output accumulator
- $\ell$: current softmax denominator
- $m$: current max

It does **not** store:

- the full score matrix $S$
- the full probability matrix $P$

That is the main memory saving.

## Simple Example

Suppose one row has scores split into two blocks:

$$
x = [1, 2, 3, 4]
$$

Block 1:

$$
x^{(1)} = [1, 2]
$$

Then:

$$
m_1 = 2
$$

$$
\ell_1 = e^{1 - 2} + e^{2 - 2}
$$

Block 2:

$$
x^{(2)} = [3, 4]
$$

The new max is:

$$
m_2 = 4
$$

The old denominator used max 2, so it must be rescaled:

$$
\ell_2
= e^{2 - 4}\ell_1
+ e^{3 - 4}
+ e^{4 - 4}
$$

This equals the stable softmax denominator for all four scores:

$$
e^{1 - 4} + e^{2 - 4} + e^{3 - 4} + e^{4 - 4}
$$

So online softmax gives the same result as normal softmax, but without keeping every score.

## Complexity Analysis

### FLOPs

FlashAttention does the same main math as standard exact attention:

$$
O(N^2d)
$$

The paper is not claiming a better asymptotic FLOP count for dense attention. It is claiming better memory movement.

The main matrix multiplications are still:

- $QK^T$: $O(N^2d)$
- $PV$: $O(N^2d)$

There is also softmax work over $N^2$ scores. When $d$ is not tiny, the dominant term is still $O(N^2d)$.

### HBM Access

The paper states that standard attention uses:

$$
O(Nd + N^2)
$$

HBM accesses.

The $Nd$ term comes from reading and writing $Q$, $K$, $V$, and $O$. The $N^2$ term comes from materializing the attention matrix.

FlashAttention reduces the HBM access to:

$$
O\left(\frac{N^2d^2}{M}\right)
$$

where $M$ is SRAM size.

The intuition:

- SRAM can only hold a limited tile.
- Larger SRAM means larger tiles.
- Larger tiles mean each loaded $Q$, $K$, and $V$ block can be reused for more work.
- Less reloading means less HBM traffic.

The important comparison is:

```text
standard attention:
    pays for the full N x N matrix in HBM

FlashAttention:
    pays for tiled Q, K, V, O movement
    avoids storing S and P
```

The paper also proves that this IO complexity is optimal for a range of SRAM sizes:

$$
d \le M \le Nd
$$

So within that range, the algorithm is not just better in practice. It is asymptotically optimal in HBM traffic under the paper's model.

## Backward Pass

Training also needs the backward pass. A naive implementation might store the full attention matrix $P$ from the forward pass so backward can reuse it. FlashAttention avoids that too.

During forward, it stores only:

- output $O$
- softmax denominator statistics

During backward, it recomputes the needed score tiles $QK^T$ and softmax probabilities tile by tile. This trades some extra compute for much less memory. That is usually a good trade on modern GPUs because compute is cheaper than HBM traffic. For simplicity we will not include the deduction of backprop math in this blog.

## Block-Sparse Attention

The paper also extends the idea to block-sparse attention. Dense attention allows every query token to attend to every key token. In matrix form, the attention mask is full:

$$
A \in \mathbb{R}^{N \times N}
$$

Block-sparse attention splits this matrix into blocks. Some blocks are active and some blocks are skipped. Let the block size be $B$. Then the attention matrix has:

$$
\frac{N}{B} \times \frac{N}{B}
$$

blocks.

Define a block mask:

$$
\mathcal{M}_{ij} =
\begin{cases}
1 & \text{if query block } i \text{ attends to key block } j \\
0 & \text{otherwise}
\end{cases}
$$

If $\mathcal{M}_{ij} = 0$, that whole score block is ignored. The kernel does not need to load that $K,V$ tile or compute that score tile for the corresponding $Q$ tile.

Example with four blocks:

```text
dense mask:

1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1

causal block mask:

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

local block mask:

1 1 0 0
1 1 1 0
0 1 1 1
0 0 1 1
```

Causal attention is block-sparse if viewed at block granularity, because future blocks are masked out. Local attention is also block-sparse, because each block attends only to nearby blocks.

Here is a simplified pseudocode version:

```python
def block_sparse_flash_attention(Q, K, V, block_mask, B):
    """
    Q, K, V: [N, d]
    block_mask: [N / B, N / B]
        block_mask[i][j] = 1 means query block i attends to key/value block j
    B: block size along sequence length
    """
    N, d = Q.shape
    num_blocks = N // B
    O = zeros_like(Q)

    for q_block_id in range(num_blocks):
        q_start = q_block_id * B
        q_end = q_start + B
        Q_block = Q[q_start:q_end]

        # Online softmax state for this query block.
        running_max = full([B], -inf)
        running_sum = zeros([B])
        output_acc = zeros([B, d])

        for kv_block_id in range(num_blocks):
            if block_mask[q_block_id][kv_block_id] == 0:
                continue

            kv_start = kv_block_id * B
            kv_end = kv_start + B
            K_block = K[kv_start:kv_end]
            V_block = V[kv_start:kv_end]

            scores = Q_block @ K_block.T

            block_max = max(scores, axis=1)
            new_max = maximum(running_max, block_max)

            old_scale = exp(running_max - new_max)
            weights = exp(scores - new_max[:, None])

            running_sum = running_sum * old_scale + sum(weights, axis=1)
            output_acc = output_acc * old_scale[:, None] + weights @ V_block
            running_max = new_max

        O[q_start:q_end] = output_acc / running_sum[:, None]

    return O
```

The important line is:

```python
if block_mask[q_block_id][kv_block_id] == 0:
    continue
```

That `continue` skips a whole attention tile. The rest of the function is still the same FlashAttention idea: compute one score tile, update the online softmax state, and discard the score tile.

If $s$ is the fraction of non-zero blocks, the paper gives the HBM access for block-sparse FlashAttention as:

$$
O\left(Nd + \frac{N^2d^2}{M}s\right)
$$

When $s = 1$, this becomes dense FlashAttention. When $s < 1$, fewer tiles are loaded and computed.

This gives two benefits:

- fewer FLOPs, because skipped blocks do not compute $QK^T$
- fewer HBM accesses, because skipped blocks do not load their $K,V$ tiles

The tradeoff is that block-sparse attention is approximate unless the mask exactly matches the model's desired attention pattern.
