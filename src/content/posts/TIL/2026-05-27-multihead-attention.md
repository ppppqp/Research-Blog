---
author: Retep
pubDatetime: 2026-05-27T12:00:00.000-07:00
title: "[TIL] Multihead Attention"
featured: false
draft: false
tags:
  - TIL
description: ""
---

## Multihead attention in Triton

Today I wrote a small Triton kernel for multihead attention:

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def MHA(Q, K, V, output, N, h, d_head, d_model, scale, HEAD_DIM: tl.constexpr, N_BLOCK: tl.constexpr):
    row_pid = tl.program_id(0)
    h_pid = tl.program_id(1) # head index since we partition by head

    row_indexes = row_pid * N_BLOCK + tl.arange(0, N_BLOCK)
    # NOTE: HEAD_DIM needs to be a constexpr to compile
    h_indexes = h_pid * d_head + tl.arange(0, HEAD_DIM)

    h_masks = tl.arange(0, HEAD_DIM) < d_head


    # consider Q, K, V as N * d_head tensors
    # when doing QK^T, we do not do tiling again. Instead, we just use full d_head as one tile


    running_sum = tl.zeros([N_BLOCK], dtype=tl.float32)
    running_max = tl.full([N_BLOCK], float('-inf'), dtype=tl.float32)

    # load complete d_head as a tile
    q_offsets = row_indexes[:, None] * d_model + h_indexes[None, :]
    row_mask = row_indexes < N
    q_mask = row_mask[:, None] & h_masks[None, :]
    q_tile = tl.load(Q + q_offsets, mask=q_mask, other=0.0)


    out_acc = tl.zeros((N_BLOCK, HEAD_DIM), dtype=tl.float32)
    for n in tl.range(0, N, N_BLOCK):
        acc = tl.zeros((N_BLOCK, N_BLOCK), dtype=tl.float32)
        # online softmax for each N_BLOCK columns in K
        # use row_stride==h for simplicity
        # nth tile of K
        # h indexes are aligned with Q
        k_indexes = n + tl.arange(0, N_BLOCK)
        k_offsets = k_indexes[:, None] * d_model + h_indexes[None, :]
        k_row_mask = k_indexes < N
        k_mask = k_row_mask[:, None] & h_masks[None, :]
        k_tile = tl.load(K + k_offsets, mask=k_mask, other=0.0)

        acc = tl.dot(q_tile, tl.trans(k_tile)) * scale
        acc = tl.where(row_mask[:, None] & k_row_mask[None, :], acc, float('-inf'))
        # compute running max for this tile
        current_max = tl.max(acc, axis=-1)
        new_running_max = tl.maximum(running_max, current_max)
        # compute error correction
        alpha = tl.exp(running_max - new_running_max)

        # compute increment to running sum
        weights = tl.exp(acc - new_running_max[:, None])
        incre = tl.sum(weights, axis=-1)

        # update
        running_sum = tl.fma(running_sum, alpha, incre)
        running_max = new_running_max

        # load v, same offsets as k because we are loading a complete head as a tile
        v_offsets = k_offsets
        v_mask = k_mask
        v_tile = tl.load( V+ v_offsets, mask=v_mask, other=0.0)

        weighted_v = tl.dot(weights, v_tile)
        out_acc = tl.fma(out_acc, alpha[:, None], weighted_v)

    out_acc /= running_sum[:, None]
    out_offsets = q_offsets
    out_mask = q_mask
    tl.store(output + out_offsets, out_acc, mask=out_mask)


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    d_head = d_model // h
    N_BLOCK = 32
    HEAD_DIM = max(16, triton.next_power_of_2(d_head))
    grid = (triton.cdiv(N, N_BLOCK), h)
    # do this kind of compute in driver
    scale= 1 / math.sqrt(d_head)
    MHA[grid](
        Q=Q,
        K=K,
        V=V,
        output=output,
        N=N,
        h=h,
        d_head=d_head,
        d_model=d_model,
        scale=scale,
        HEAD_DIM=HEAD_DIM,
        N_BLOCK=N_BLOCK
    )
```

## One program owns one query block and one head

The launch grid is two-dimensional:

```python
grid = (triton.cdiv(N, N_BLOCK), h)
```

The first program id chooses a block of query rows. The second program id chooses the attention head:

```python
row_pid = tl.program_id(0)
h_pid = tl.program_id(1)
```

So one Triton program computes:

```text
one head
one N_BLOCK x d_head slice of output
```

With `N_BLOCK = 32`, each program owns up to 32 query tokens for a single head:

```text
output, one head

          row_pid=0     row_pid=1     row_pid=2
        +-----------+-----------+-----------+
head 0  | 32 x Dh   | 32 x Dh   | 32 x Dh   |
        +-----------+-----------+-----------+
head 1  | 32 x Dh   | 32 x Dh   | 32 x Dh   |
        +-----------+-----------+-----------+
head 2  | 32 x Dh   | 32 x Dh   | 32 x Dh   |
        +-----------+-----------+-----------+
```

The row offsets are:

```python
row_indexes = row_pid * N_BLOCK + tl.arange(0, N_BLOCK)
```

The head offsets are:

```python
h_indexes = h_pid * d_head + tl.arange(0, HEAD_DIM)
```

The kernel assumes `Q`, `K`, `V`, and `output` are laid out as `[N, d_model]`, with the head dimension packed inside `d_model`. A token row contains all heads next to each other:

```text
token row
+---------+---------+---------+---------+
| head 0  | head 1  | head 2  | head 3  |
+---------+---------+---------+---------+
```

For a fixed `h_pid`, `h_indexes` selects one of those slices.

## Why HEAD_DIM is separate from d_head

Triton wants block shapes to be compile-time constants. The real head size is runtime data:

```python
d_head = d_model // h
```

but the tile shape is:

```python
HEAD_DIM = max(16, triton.next_power_of_2(d_head))
```

So the program may allocate a slightly larger block than the real head size. For example, if `d_head = 48`, then `HEAD_DIM = 64`. The extra lanes are masked out:

```python
h_masks = tl.arange(0, HEAD_DIM) < d_head
q_mask = row_mask[:, None] & h_masks[None, :]
```

This is a common Triton pattern:

1. Use a static power-of-two block shape so the kernel can compile.
2. Use masks so the extra elements do not participate in the real computation.

## Loading Q

The query tile is loaded once before the loop over keys and values:

```python
q_offsets = row_indexes[:, None] * d_model + h_indexes[None, :]
q_tile = tl.load(Q + q_offsets, mask=q_mask, other=0.0)
```

The shape is:

```text
q_tile: [N_BLOCK, HEAD_DIM]
```

This is the block of query vectors for the current output rows and current head. It stays live while the program streams through all key/value blocks.

## Streaming through K and V

The loop walks over the sequence dimension in chunks:

```python
for n in tl.range(0, N, N_BLOCK):
    k_indexes = n + tl.arange(0, N_BLOCK)
```

For each chunk, the program loads:

```text
k_tile: [N_BLOCK, HEAD_DIM]
v_tile: [N_BLOCK, HEAD_DIM]
```

The score tile is:

```python
acc = tl.dot(q_tile, tl.trans(k_tile)) * scale
```

which has shape:

```text
acc: [N_BLOCK, N_BLOCK]
```

Each row is one query token from the current query block. Each column is one key token from the current key block.

Conceptually, the program is doing:

```text
for this head:
    for 32 query tokens:
        compare against key tokens 0..31
        compare against key tokens 32..63
        compare against key tokens 64..95
        ...
```

The mask turns invalid row or key positions into `-inf`:

```python
acc = tl.where(row_mask[:, None] & k_row_mask[None, :], acc, float('-inf'))
```

That matters for the tail blocks when `N` is not divisible by `N_BLOCK`.

## What is different from simple softmax attention

The interesting part here is not the online softmax itself. That is already the same idea as the previous attention/softmax kernel: keep per-row max and per-row denominator while streaming across key blocks.

The difference is the shape of the tile. In a simple softmax attention kernel, it is natural to think about the attention matrix:

```text
scores: [N, N]
```

and tile only the sequence dimension. In multihead attention, the input is logically:

```text
Q, K, V: [N, h, d_head]
```

This kernel assigns one program to one head and one query block. Inside that program, it uses the entire attention head as the feature tile:

```text
q_tile:  [N_BLOCK, HEAD_DIM]
k_tile:  [N_BLOCK, HEAD_DIM]
v_tile:  [N_BLOCK, HEAD_DIM]
```

So `d_head` is not split further. The reduction dimension for `QK^T` is the whole head:

```text
[N_BLOCK, HEAD_DIM] x [HEAD_DIM, N_BLOCK] -> [N_BLOCK, N_BLOCK]
```

Then the second multiply consumes the value tile:

```text
[N_BLOCK, N_BLOCK] x [N_BLOCK, HEAD_DIM] -> [N_BLOCK, HEAD_DIM]
```

This keeps the implementation close to the math:

```text
for one head:
    Q block compares with every K block
    softmax weights immediately multiply the matching V block
    final output is one [N_BLOCK, d_head] tile
```

That is a good simple pattern when `d_head` is small enough, such as 32, 64, or 128. If `d_head` becomes large, this becomes a heavier tile because `q_tile`, `v_tile`, and `out_acc` all carry the full head dimension.

## The direct pointer arithmetic version

In my first version, the head slice is selected manually:

```python
h_indexes = h_pid * d_head + tl.arange(0, HEAD_DIM)
h_masks = tl.arange(0, HEAD_DIM) < d_head
```

Then every load builds offsets by broadcasting row indexes and head indexes:

```python
q_offsets = row_indexes[:, None] * d_model + h_indexes[None, :]
q_mask = row_mask[:, None] & h_masks[None, :]
q_tile = tl.load(Q + q_offsets, mask=q_mask, other=0.0)
```

The K and V loads repeat the same idea:

```python
k_offsets = k_indexes[:, None] * d_model + h_indexes[None, :]
k_mask = k_row_mask[:, None] & h_masks[None, :]
k_tile = tl.load(K + k_offsets, mask=k_mask, other=0.0)
```

This is explicit and easy to debug. The downside is that boundary handling is mixed into every pointer expression. For every tile, the code has to remember two separate masks:

- sequence mask: `row_indexes < N` or `k_indexes < N`
- head mask: `tl.arange(0, HEAD_DIM) < d_head`

That is fine for a small kernel, but it gets noisy as soon as the layout changes.

## A cleaner block pointer version

The second version changes the data layout before launching:

```python
Q = Q.reshape(N, h, d_key).permute(1, 0, 2).contiguous()
K = K.reshape(N, h, d_key).permute(1, 2, 0).contiguous()
V = V.reshape(N, h, d_key).permute(1, 0, 2).contiguous()
output = output.reshape(N, h, d_key).permute(1, 0, 2)
```

Now the tensors have head as the outer dimension:

```text
Q:      [h, N, d_key]
K:      [h, d_key, N]
V:      [h, N, d_key]
output: [h, N, d_key]
```

That is a better layout for the two dot products. `Q` and `V` are naturally loaded as `[BLOCK_SIZE_N, HEAD_DIM]`, while `K` is naturally loaded as `[HEAD_DIM, BLOCK_SIZE_N]`. The code no longer needs to load K as `[N_BLOCK, HEAD_DIM]` and transpose it inside the dot:

```python
k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
qkt = tl.dot(q, k) * scale
```

The bigger change is that tile shape and memory layout are described once with `tl.make_block_ptr`:

```python
Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + start_h * stride_qh,
    shape=(N, d_key),
    strides=(stride_qn, stride_qk),
    offsets=(start_n * BLOCK_SIZE_N, 0),
    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
    order=(1, 0),
)
```

This says:

```text
inside this head:
    logical matrix shape is [N, d_key]
    current tile starts at [start_n * BLOCK_SIZE_N, 0]
    tile shape is [BLOCK_SIZE_N, HEAD_DIM]
```

Then the load becomes:

```python
q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
```

The boundary behavior is attached to the tile itself. If the tile runs past `N` or past `d_key`, Triton pads out-of-bounds elements with zero. This removes the manual load mask:

```python
q_mask = row_mask[:, None] & h_masks[None, :]
```

and replaces it with the block pointer's declared shape:

```python
shape=(N, d_key)
boundary_check=(0, 1)
padding_option="zero"
```

That is a cleaner pattern because the valid region is expressed once as the logical tensor shape. The load site does not need to rebuild the same mask every time.

## Advancing tiles instead of rebuilding offsets

The first version rebuilds K and V offsets on every loop iteration:

```python
k_indexes = n + tl.arange(0, N_BLOCK)
k_offsets = k_indexes[:, None] * d_model + h_indexes[None, :]
k_tile = tl.load(K + k_offsets, mask=k_mask, other=0.0)
```

The block pointer version creates the K and V tile pointers once:

```python
K_block_ptr = tl.make_block_ptr(
    base=K_ptr + start_h * stride_kh,
    shape=(d_key, N),
    strides=(stride_kk, stride_kn),
    offsets=(0, 0),
    block_shape=(HEAD_DIM, BLOCK_SIZE_N),
    order=(1, 0),
)

V_block_ptr = tl.make_block_ptr(
    base=V_ptr + start_h * stride_vh,
    shape=(N, d_key),
    strides=(stride_vn, stride_vk),
    offsets=(0, 0),
    block_shape=(BLOCK_SIZE_N, HEAD_DIM),
    order=(1, 0),
)
```

Then the inner loop advances the logical tile positions:

```python
K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_N))
V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))
```

This matches the math more directly:

```text
K moves across columns: [d_key, N]
V moves down rows:      [N, d_key]
```

The loop no longer has to manually reconstruct pointer offsets from raw indexes. It says "move this tile to the next K/V block," and the block pointer keeps the shape, stride, block shape, and boundary rules attached.


Block pointers improve load boundary handling:
```python
tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
```

But attention still needs a logical mask for the score tile:

```python
qkt = tl.where((offset_m[:, None] < N) & (offset_n[None, :] < N), qkt, 0.0)
```

The load boundary check protects memory accesses. The score mask protects the math. These are different responsibilities:

- `boundary_check`: do not read outside the logical Q/K/V tile.
- `tl.where`: do not let padded query/key positions contribute to the softmax numerator.

One subtle point: in this version, `m_i` is computed before the score mask is applied:

```python
m_i = tl.maximum(m, tl.max(qkt, axis=1))
qkt = tl.exp(qkt - m_i[:, None])
qkt = tl.where((offset_m[:, None] < N) & (offset_n[None, :] < N), qkt, 0.0)
```

Because out-of-bounds K loads are padded with zero, invalid key columns can contribute a score of zero to the row max. The final normalized value can still come out correctly because numerator and denominator are scaled together, but it is cleaner to mask invalid score positions before the max:

```python
qkt = tl.where((offset_m[:, None] < N) & (offset_n[None, :] < N), qkt, -float("inf"))
m_i = tl.maximum(m, tl.max(qkt, axis=1))
qkt = tl.exp(qkt - m_i[:, None])
qkt = tl.where((offset_m[:, None] < N) & (offset_n[None, :] < N), qkt, 0.0)
```

Complete code:
```py
import torch
import triton
import triton.language as tl
import math

@triton.jit
def mha_inner_fwd(
    q, K_block_ptr, V_block_ptr, m, l,
    offset_m, acc, N, scale, BLOCK_SIZE_N
):
    for n_idx in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_n = n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k = tl.load(K_block_ptr, boundary_check = (0, 1), padding_option = 'zero')
        qkt = tl.dot(q, k) * scale
        m_i = tl.maximum(m, tl.max(qkt, axis = 1))
        qkt = tl.exp(qkt - m_i[:, None])
        qkt = tl.where((offset_m[:, None] < N) & (offset_n[None, :] < N), qkt, 0.0)
        l_i = tl.sum(qkt, axis = 1)
        alpha = tl.exp(m - m_i)
        l = alpha * l + l_i
        v = tl.load(V_block_ptr, boundary_check = (0, 1), padding_option = 'zero')
        acc = alpha[:, None] * acc
        acc = tl.dot(qkt, v, acc)
        m = m_i 
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_N, 0))
    
    return l, acc

@triton.jit
def mha_fwd(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    N, d_model, h, d_key, scale,
    stride_qh, stride_qn, stride_qk,
    stride_kh, stride_kk, stride_kn,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_on, stride_ok,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    start_n = tl.program_id(axis = 0)
    start_h = tl.program_id(axis = 1)
    offset_m = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + start_h * stride_qh,
        shape = (N, d_key),
        strides = (stride_qn, stride_qk),
        offsets = (start_n * BLOCK_SIZE_N, 0),
        block_shape = (BLOCK_SIZE_N, HEAD_DIM),
        order = (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + start_h * stride_kh,
        shape = (d_key, N),
        strides = (stride_kk, stride_kn),
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_N),
        order = (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + start_h * stride_vh,
        shape = (N, d_key),
        strides = (stride_vn, stride_vk),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_N, HEAD_DIM),
        order = (1, 0),
    )

    m = tl.full((BLOCK_SIZE_N,), -float('inf'), dtype = tl.float32)
    l = tl.zeros((BLOCK_SIZE_N,), dtype = tl.float32)
    acc = tl.zeros((BLOCK_SIZE_N, HEAD_DIM), dtype = tl.float32)

    q = tl.load(Q_block_ptr, boundary_check = (0, 1), padding_option = 'zero')

    l, acc = mha_inner_fwd(
        q, K_block_ptr, V_block_ptr, m, l,
        offset_m, acc, N, scale, BLOCK_SIZE_N
    )

    acc = acc / l[:, None]
    offset_d = tl.arange(0, HEAD_DIM)
    output_ptrs = output_ptr + start_h * stride_oh + offset_m[:, None] * stride_on + offset_d[None, :] * stride_ok
    tl.store(output_ptrs, acc, mask = (offset_m[:, None] < N) & (offset_d[None, :] < d_key))

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    h: int,
):
    assert d_model % h == 0, "d_model must be divisible by num_head"
    d_key = d_model // h
    BLOCK_SIZE_N = 32
    HEAD_DIM = max(16, triton.next_power_of_2(d_key))

    Q = Q.reshape(N, h, d_key).permute(1, 0, 2).contiguous()
    K = K.reshape(N, h, d_key).permute(1, 2, 0).contiguous()
    V = V.reshape(N, h, d_key).permute(1, 0, 2).contiguous()
    output = output.reshape(N, h, d_key).permute(1, 0, 2)
    grid = (triton.cdiv(N, BLOCK_SIZE_N), h)
    scale = 1 / math.sqrt(d_key)
    mha_fwd[grid](
        Q, K, V, output, N, d_model, h, d_key, scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_SIZE_N, HEAD_DIM, num_warps = 2, num_stages = 1
    )
    output = output.transpose(0, 1).reshape(N, -1)
```


## What gets reused

The query tile is reused across every key/value block:

```text
q_tile [32, d_head]
    dot K block 0
    dot K block 1
    dot K block 2
    ...
```

Each key tile is used to produce a `32 x 32` score tile. Each value tile is used immediately after the corresponding score weights are computed:

```python
weighted_v = tl.dot(weights, v_tile)
```

So the kernel's memory behavior is:

- `Q`: loaded once per program.
- `K`: streamed through all sequence blocks.
- `V`: streamed through all sequence blocks.
- `output`: stored once at the end.

The important part is that the intermediate attention matrix is not stored. For one head, a direct attention implementation has an `N x N` score matrix. Here, one program only keeps a `N_BLOCK x N_BLOCK` score tile and a `N_BLOCK x d_head` output accumulator.

The block-pointer version keeps the same reuse pattern, but describes it more cleanly:

```text
Q tile: loaded once
K tile: block pointer advances across sequence
V tile: block pointer advances across sequence
O tile: one manual masked store at the end
```

The final store is still manual:

```python
output_ptrs = (
    output_ptr
    + start_h * stride_oh
    + offset_m[:, None] * stride_on
    + offset_d[None, :] * stride_ok
)

tl.store(output_ptrs, acc, mask=(offset_m[:, None] < N) & (offset_d[None, :] < d_key))
```

That is reasonable. Stores often need an explicit mask, and this one clearly says the output is valid only for real sequence rows and real head-dimension lanes.

## The shape tradeoff

This version tiles along the sequence dimension but not along the head dimension. It loads the whole `d_head` for a head:

```python
q_tile:   [N_BLOCK, HEAD_DIM]
k_tile:   [N_BLOCK, HEAD_DIM]
v_tile:   [N_BLOCK, HEAD_DIM]
out_acc:  [N_BLOCK, HEAD_DIM]
```

That keeps the code simple. The dot products are:

```text
[N_BLOCK, HEAD_DIM] x [HEAD_DIM, N_BLOCK] -> [N_BLOCK, N_BLOCK]
[N_BLOCK, N_BLOCK] x [N_BLOCK, HEAD_DIM] -> [N_BLOCK, HEAD_DIM]
```

For common small head dimensions like 32, 64, or 128, this is a reasonable mental model. For very large `d_head`, the accumulator and tile sizes become more expensive, and we may want to split the head dimension too.

`N_BLOCK = 32` is also a tuning choice. Larger blocks improve reuse and make larger score tiles, but they increase register pressure because the program carries:

- `acc`: `N_BLOCK x N_BLOCK`
- `weights`: `N_BLOCK x N_BLOCK`
- `out_acc`: `N_BLOCK x HEAD_DIM`

