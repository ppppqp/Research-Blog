---
author: Retep
pubDatetime: 2026-07-02T12:00:00.000-07:00
title: "[TIL] GPU Shared Memory Swizzle"
featured: false
draft: false
tags:
  - TIL
description: ""
---

I used to vaguely understand swizzling as "some layout trick that makes GPU
memory access faster". That description is not wrong, but it is too broad to be
useful. In the context of CUDA kernels and matrix multiplication, swizzling
mostly matters for **shared memory**, and the reason is very concrete: shared
memory is divided into banks.

The main problem is not continuous access. Continuous access is usually already
good. The problem is strided or column-like access, where many threads in the
same warp accidentally hit the same shared-memory bank. Swizzling changes the
physical layout inside a small repeated block so that the same logical access
pattern spreads across banks.

# Shared Memory Banks

A useful mental model for NVIDIA shared memory is:

$$
\text{number of banks} = 32
$$

$$
\text{bank width} = 4\ \text{bytes}
$$

Given a byte address, the bank index is approximately:

$$
\text{bank} = \left\lfloor \frac{\text{byte address}}{4} \right\rfloor \bmod 32
$$

For an aligned `float32` array, each element is 4 bytes. If we index the array by
`i`, then:

$$
\text{byte address} = \text{base} + 4i
$$

Ignoring the aligned base, the bank becomes:

$$
\text{bank} = i \bmod 32
$$

This is why contiguous `float32` access is good. If 32 lanes in a warp read
`smem[0]` through `smem[31]`, they touch banks 0 through 31:

```cpp
int lane = threadIdx.x % 32;
float x = smem[lane];
```

Logically:

```text
lane 0  -> smem[0]  -> bank 0
lane 1  -> smem[1]  -> bank 1
...
lane 31 -> smem[31] -> bank 31
```

There is no bank conflict here.

The bad case is a strided access. For example:

```cpp
int lane = threadIdx.x % 32;
float x = smem[lane * 32];
```

Now every lane maps to the same bank:

```text
lane 0  -> smem[0]    -> bank 0
lane 1  -> smem[32]   -> bank 0
lane 2  -> smem[64]   -> bank 0
...
lane 31 -> smem[992]  -> bank 0
```

This is a 32-way bank conflict. The access can degrade toward serialized
service, because the warp is asking one bank to provide many different
addresses. This is the core issue that shared-memory swizzling tries to solve.

# What Is a Lane?

A **lane** is a thread's position inside a warp. On NVIDIA GPUs:

$$
1\ \text{warp} = 32\ \text{threads}
$$

So we usually talk about lanes 0 through 31:

```text
lane 0  = first thread in the warp
lane 1  = second thread in the warp
...
lane 31 = last thread in the warp
```

When a warp executes a shared-memory load, all active lanes issue the same
instruction together, but each lane may compute a different address. Bank
conflicts are therefore a property of the **lane-to-address mapping**.

This point is important. A single thread loading from shared memory does not
create an inter-lane bank conflict by itself. The conflict happens when multiple
lanes in the same warp instruction access different addresses that map to the
same bank.

This is also why "the load is only 16B" does not automatically make swizzling
irrelevant. A 16B vector load by one lane touches four 4B bank slots. If all
lanes load contiguous vectors, the pattern can still be fine. But if all lanes
load strided vectors, each transaction can still have conflicts.

# Continuous Access vs Column Access

For global memory, we usually care about coalescing: neighboring lanes should
access neighboring global addresses so the hardware can combine the request into
efficient memory transactions. A typical good pattern is:

```text
lane 0 -> A[i + 0]
lane 1 -> A[i + 1]
lane 2 -> A[i + 2]
...
```

Shared memory has a different bottleneck. It is already on chip, but it is
banked. The question is not only whether addresses are continuous. The question
is whether the lanes spread across banks.

This distinction shows up naturally in matmul. A tiled matmul often loads a
matrix tile from global memory into shared memory. The global-memory load should
be coalesced. Later, the tile may be consumed in a different orientation, such
as column-wise, transposed, or in a tensor-core fragment layout. That second
access is where shared-memory bank conflicts often appear.

In other words, loading data from global memory into shared memory and loading
data back out of shared memory are two different layout problems. The first one
usually wants neighboring lanes to read neighboring global addresses. The second
one wants the lanes in a warp to hit different shared-memory banks. Swizzling is
mostly about this second problem.

# What Swizzling Does

Swizzling changes the physical address used for a logical shared-memory
coordinate. Instead of:

$$
\text{physical offset} = \text{row} \cdot \text{stride} + \text{col}
$$

we use a small permutation:

$$
\text{physical offset} = \operatorname{swizzle}(\text{row}, \text{col})
$$

The exact formula depends on the GPU instruction, the library, the data type,
and the chosen layout. But the usual shape of the algorithm is to split the
tile into fixed-size byte groups, such as 64B or 128B, and then permute the
lower address bits inside each group. Elements stay inside their local group;
the same local permutation is repeated for the next group.

If \(G\) is the swizzle group size, then a byte offset can be decomposed into a
group base and a local offset:

$$
\text{group base} = \left\lfloor \frac{\text{offset}}{G} \right\rfloor G
$$

$$
\text{local offset} = \text{offset} \bmod G
$$

The swizzle changes the local offset but keeps the group base:

$$
\text{physical offset}
  = \text{group base} + \operatorname{permute}(\text{local offset})
$$

Many swizzle layouts can be understood as XOR-ing some higher coordinate bits
into lower address or bank bits. In a 2D explanation, it often looks like:

$$
\text{swizzled col} = \text{col} \oplus f(\text{row})
$$

Then:

$$
\text{physical offset} =
  \text{row} \cdot \text{stride} + \text{swizzled col}
$$

This is not meant to preserve row-major order. It is meant to break the pattern
where column access maps every lane to the same bank. The logical matrix still
has the same shape. Only the physical placement in shared memory changes.

# 64B vs 128B Swizzle

When we say "64B swizzle" or "128B swizzle", we are usually talking about the
byte granularity of the local permutation:

$$
\text{64B swizzle}: \text{permute offsets inside each 64-byte group}
$$

$$
\text{128B swizzle}: \text{permute offsets inside each 128-byte group}
$$

The 128B number is natural because a full warp reading one `float32` per lane
touches:

$$
32\ \text{lanes} \times 4\ \text{bytes} = 128\ \text{bytes}
$$

So for many warp-level shared-memory patterns, 128B is the natural full-bank
span. If the access pattern and tile shape naturally operate on 128B chunks,
128B swizzling is often the right choice.

But 128B is not always better. If the logical row or repeated local block is
only 64B wide, a 128B swizzle may cross a boundary that we would rather keep
separate. In that case, 64B swizzling can be a better match. I think of the
choice as matching the swizzle group to the natural width of the access: use the
largest group that fits the thing being loaded or stored without crossing a
logical boundary.

# Data Type Changes the Shape

The byte size of the data type determines how many elements fit inside a 64B or
128B swizzle group.

For `float32`:

$$
1\ \text{element} = 4\ \text{bytes}
$$

$$
128\text{B} = 32\ \text{float32 elements}
$$

$$
64\text{B} = 16\ \text{float32 elements}
$$

For `float16`:

$$
1\ \text{element} = 2\ \text{bytes}
$$

$$
128\text{B} = 64\ \text{float16 elements}
$$

$$
64\text{B} = 32\ \text{float16 elements}
$$

This means the same byte-level swizzle corresponds to different logical matrix
shapes for different dtypes. A 128B atom could be `4 x 8` for `float32`, because
that is 32 elements:

$$
4 \times 8 \times 4\text{B} = 128\text{B}
$$

But for `float16`, 128B could be `8 x 8`, because that is 64 elements:

$$
8 \times 8 \times 2\text{B} = 128\text{B}
$$

This is why swizzle mode should be chosen together with dtype, tile shape, and
the instruction that consumes the tile.

# Atom

An **atom** is the smallest repeated layout block. The large tile is built by
repeating atoms, and the swizzle is applied inside each atom.

Conceptually, the atom is the unit that carries the fixed internal layout. A
larger tile is just many copies of that atom placed next to each other.

For a 128B swizzle, it is common for the atom to contain 128B of data, but the
2D shape is still a layout choice. For `float32`, 128B is 32 elements, so all of
these shapes have the same byte size:

```text
1 x 32 float32 = 128B
2 x 16 float32 = 128B
4 x 8  float32 = 128B
8 x 4  float32 = 128B
```

They are not equivalent. The correct atom shape depends on the lane mapping and
the instruction that will read or write the data.

For example, a `float32` tile of shape `4 x 8` contains:

$$
4 \times 8 \times 4\text{B} = 128\text{B}
$$

So this tile can be one 128B atom.

A `float32` tile of shape `8 x 16` contains:

$$
8 \times 16 \times 4\text{B} = 512\text{B}
$$

If we choose `4 x 8` as the atom shape, then the tile has four atoms:

$$
\frac{512\text{B}}{128\text{B}} = 4
$$

The swizzle happens independently inside each of those four atoms. It does not
usually mix values from the first atom into the second atom. A good mental model
is:

$$
\text{physical address}
  = \text{atom base}(\text{atom id})
  + \text{swizzled local offset}(\text{local coordinate})
$$

So swizzling is local. Atom tiling is how we repeat the local swizzle pattern
over a larger tile.

# How Lane Mapping and Swizzling Fit Together

The reason atom shape matters is that the hardware instruction has an expected
lane-to-element pattern. For one access pattern, `4 x 8` may spread lanes nicely
across banks. For another access pattern, `2 x 16` or `1 x 32` may be better.

Suppose the logical access is column-like. In a row-major layout, lane `l` might
access:

$$
i_l = 32l
$$

Then:

$$
\text{bank}_l = i_l \bmod 32 = 0
$$

for every lane. This is the bank conflict.

With a swizzled layout, the same logical coordinate is mapped to a different
physical offset:

$$
i_l' = \operatorname{swizzle}(i_l)
$$

The goal is not that \(i_l'\) is continuous. The goal is:

$$
\text{bank}_l =
\left\lfloor \frac{\text{byte address}(i_l')}{4} \right\rfloor \bmod 32
$$

should vary across lanes. Ideally, the active lanes distribute over many banks
instead of all colliding on one.

This is the key connection between lanes and swizzling. The lane mapping tells
us which logical elements are loaded together by one warp instruction.
Swizzling changes where those logical elements live physically. The bank
mapping then determines whether the instruction is conflict-free or serialized
by bank conflicts.

# Takeaway

Swizzling is not primarily about making memory addresses continuous. Continuous
row-wise access is already the easy case. Swizzling is about making a repeated
shared-memory access pattern bank-friendly, especially when the logical access
is strided, column-wise, transposed, or tensor-core-fragment shaped.

The useful way to reason about it is to start from the warp instruction. Which
logical values do the lanes access together? Which banks would those values hit
without a swizzle? How many elements fit into a 64B or 128B region for the dtype
we are using? What atom shape matches the instruction's access pattern? Once
those questions are clear, swizzling stops being a mysterious layout trick. It
becomes a small address permutation inside each atom, chosen so that the warp's
lanes do not fight over the same shared-memory banks.
