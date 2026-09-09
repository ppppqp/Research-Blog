---
author: Retep
pubDatetime: 2026-09-08T12:00:00.000-07:00
title: "[TIL] Block exchange"
featured: false
draft: false
tags:
  - TIL
description: ""
---
Learning source: [rocPRIM](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.2.1/block_ops/ops_classes/exchange.html#classrocprim_1_1block__exchange)
A thread/value layout describes how a logical sequence is distributed across those threads. For example, a logical sequence `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` can be distributed across 4 threads

```text
thread 0: [0, 1, 2]
thread 1: [3, 4, 5]
thread 2: [6, 7, 8]
thread 3: [9, 10, 11]
```

It can be expressed as `(4, 3):(3, 1)`, which means the outer layout is 4 threads with 3 as stride, and the inner layout is 3 registers with 1 as stride. The global index can be calculated as `thread_id * 4 + local_id`

In a striped layout, consecutive elements are distributed across consecutive threads:

```text
thread 0: [0, 4, 8]
thread 1: [1, 5, 9]
thread 2: [2, 6, 10]
thread 3: [3, 7, 11]
```
Which can be expressed as `(4, 1):(3, 4)`. Notice that the stride for register dimenson becomes #total_threads. The global index (or logical index) is `thread_id + local_id * block_size`. `blocked_to_striped` redistributes values between these ownership patterns without changing the logical sequence, which basically means transforming from the first layout to the second.

## Warp-Striped Layout

A warp-striped layout applies striping independently inside each warp or wave. For two four-thread warps with two items per thread:

```text
warp 0:
thread 0: [0, 4]
thread 1: [1, 5]
thread 2: [2, 6]
thread 3: [3, 7]

warp 1:
thread 4: [8,  12]
thread 5: [9,  13]
thread 6: [10, 14]
thread 7: [11, 15]
```

`blocked_to_warp_striped` never moves values between warps. This often permits efficient wave-local instructions instead of shared memory and block-wide barriers.

## Scatter and Gather
While fixed transformations such as blocked-to-striped have predetermined permutations, scatter and gather accept explicit ranks. For `scatter_to_blocked` each input item provides its destination rank (the linearized index). The rank determines its final blocked location:
```text
destination_thread = rank // items_per_thread
destination_item   = rank % items_per_thread
```
For `gather_from_striped`, each output item provides the rank of the striped input it wants.

For a striped source:
```text
source_thread = rank % block_size
source_item   = rank // block_size
```
Some examples:
```
initial:
thread 0: values [A, B], ranks [7, 0]
thread 1: values [C, D], ranks [6, 1]
thread 2: values [E, F], ranks [5, 2]
thread 3: values [G, H], ranks [4, 3]

match rank:
rank:   0  1  2  3  4  5  6  7
value:  B  D  F  H  G  E  C  A

After scatter:
thread 0: [B, D]   # ranks 0, 1
thread 1: [F, H]   # ranks 2, 3
thread 2: [G, E]   # ranks 4, 5
thread 3: [C, A]   # ranks 6, 7
```

Gather is the other way around:
```
initial:
thread 0: [A, E], ranks [7, 6]
thread 1: [B, F], ranks [5, 4]
thread 2: [C, G], ranks [3, 2]
thread 3: [D, H], ranks [1, 0]

After gather with the rank:

thread 0: [H, G]
thread 1: [F, E]
thread 2: [D, C]
thread 3: [B, A]

```
## Redistribution

Different GPU operations prefer different value distributions. One layout may enable coalesced memory access, while another may match an MMA instruction or a cooperative algorithm.Redistribution bridges these stages by moving register-held values between threads. Depending on the source and destination layouts, an implementation can use:

- Local register shuffles when ownership does not change
- Lane operations for wave-local movement
- Shared memory or LDS when values cross waves

Treating these exchanges as layout transformations makes them reusable and creates opportunities for the compiler to select the cheapest implementation.
