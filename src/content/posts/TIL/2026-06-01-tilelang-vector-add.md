---
author: Retep
pubDatetime: 2026-06-01T12:00:00.000-07:00
title: "[TIL] Tilelang Element-wise Op & Vectorization"
featured: false
draft: false
tags:
  - TIL
  - TileLang
description: "Notes on three TileLang vector-add kernels, how vectorization appears in IR/CUDA/PTX, and why one-element-per-thread leaves bandwidth on the table."
---


This is a small reading note on three TileLang vector-add kernels following the official [walkthrough](https://tilelang.com/deeplearning_operators/elementwise.html):

- `v1`: one element per CUDA thread
- `v2`: eight contiguous elements per CUDA thread
- `v3`: eight contiguous elements per thread, staged through explicit local fragments


```text
for i in 0..N:
    c[i] = a[i] + b[i]
```

For each computation, assuming `dtype=float32`, every element costs 3 memory IO:

```text
load a[i]  = 4 bytes
load b[i]  = 4 bytes
store c[i] = 4 bytes
```

So vector add moves 12 bytes for one floating-point add. It is memory-bandwidth bound.

## Three Patterns

The three kernels differ in how much work each thread does.

### V1: Scalar Per Thread
Block covers 256 elements, and each thread computes one element.

```py
@tilelang.jit
def vector_add_v1(n: int, block_size: int = 256, dtype: str = "float32"):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as block_idx:
            for thread_idx in T.Parallel(block_size):
                i = block_idx * block_size + thread_idx
                c[i] = a[i] + b[i]

    return kernel
```

The TileLang PrimFunc reflects that directly. For example, for 2^20 elements, it need to issue 4096=2^20/256 blocks.

```python
bx = T.launch_thread("blockIdx.x", 4096)
tx = T.launch_thread("threadIdx.x", 256)

for thread_idx in T.parallel(256):
    i: T.int32 = bx * 256 + thread_idx
    c[i] = a[i] + b[i]
```

After lowering, TileLang binds the parallel loop to CUDA threads:

```python
bx = T.launch_thread("blockIdx.x", 4096)
tx = T.launch_thread("threadIdx.x", 256)

c_1[bx * 256 + tx] = a_1[bx * 256 + tx] + b_1[bx * 256 + tx]
```

The emitted CUDA is scalar:

```cpp
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
  c[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (a[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + b[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))]);
}
```

The PTX is scalar too:

```ptx
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<11>;
	.loc	1 20 0

	ld.param.b64 	%rd1, [kernel_kernel_param_0];
	ld.param.b64 	%rd2, [kernel_kernel_param_1];
	ld.param.b64 	%rd3, [kernel_kernel_param_2];
	.loc	1 21 3
	cvta.to.global.u64 	%rd4, %rd3;
	cvta.to.global.u64 	%rd5, %rd2;
	cvta.to.global.u64 	%rd6, %rd1;
	mov.u32 	%r1, %ctaid.x;
	shl.b32 	%r2, %r1, 8;
	mov.u32 	%r3, %tid.x;
	or.b32 	%r4, %r2, %r3;
	mul.wide.u32 	%rd7, %r4, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.nc.b32 	%r5, [%rd8];
	add.s64 	%rd9, %rd5, %rd7;
	ld.global.nc.b32 	%r6, [%rd9];
	add.f32 	%r7, %r5, %r6;
	add.s64 	%rd10, %rd4, %rd7;
	st.global.b32 	[%rd10], %r7;
	.loc	1 22 1
	ret;
```

This is still coalesced across the warp: neighboring threads read neighboring floats. But each thread only contributes one scalar load from `a`, one scalar load from `b`, and one scalar store to `c`.

For `N = 1,048,576` and `block_size = 256`, this launches:

```text
1,048,576 / 256 = 4096 blocks
```

### V2: Coarsened And Vectorized



The second version gives each thread eight adjacent elements:

```py
@tilelang.jit
def vector_add_v2(
    n: int,
    block_size: int = 256,
    num_per_thread: int = 8,
    dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size * num_per_thread), threads=block_size) as block_idx:
            for thread_idx, j in T.Parallel(block_size, num_per_thread):
                i = (block_idx * block_size + thread_idx) * num_per_thread
                c[i + j] = a[i + j] + b[i + j]

    return kernel
```


Each block covers 256 * 8 = 2048 elements, and each thread t computes 8 elements locally.
```
base = (block_id * 256 + thread_id) * 8
c[base + 0] = a[base + 0] + b[base + 0]
...
c[base + 7] = a[base + 7] + b[base + 7]
```

The PrimFunc has a two-dimensional parallel loop:
```python
bx = T.launch_thread("blockIdx.x", 512)
tx = T.launch_thread("threadIdx.x", 256)

for thread_idx in T.parallel(256):
    for j in T.parallel(8):
        i: T.int32 = (bx * 256 + thread_idx) * 8
        c[i + j] = a[i + j] + b[i + j]
```

The important idea is that `j` is a contiguous inner dimension. TileLang can recognize each thread owns a contiguous 8-element slice.
The lowered device IR makes that visible.

```python
c_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8] =
    a_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8] +
    b_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8]
```

That slice form is the key to vectorization. The emitted CUDA uses 256-bit vector loads and stores:

```cpp
ulonglong4 v_ = tl::load_global_256(
    &(*(ulonglong4*)(a + blockIdx.x * 2048 + threadIdx.x * 8)));

ulonglong4 v__1 = tl::load_global_256(
    &(*(ulonglong4*)(b + blockIdx.x * 2048 + threadIdx.x * 8)));

*(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
*(float2*)(&(__1.y)) = tl::add2(*(float2*)(&(v_.y)), *(float2*)(&(v__1.y)));
*(float2*)(&(__1.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
*(float2*)(&(__1.w)) = tl::add2(*(float2*)(&(v_.w)), *(float2*)(&(v__1.w)));

tl::store_global_256(
    &(*(ulonglong4*)(c + blockIdx.x * 2048 + threadIdx.x * 8)), __1);
```

Eight `float32` values are 32 bytes, or 256 bits:

```text
8 floats * 4 bytes = 32 bytes
```

That is why the CUDA uses `ulonglong4`: four 64-bit lanes make one 256-bit transaction. Same in PTX:

```ptx
{
	.reg .b32 	%r<7>;
	.reg .b64 	%rd<21>;
	.loc	1 20 0

	ld.param.b64 	%rd17, [kernel_kernel_param_0];
	ld.param.b64 	%rd18, [kernel_kernel_param_1];
	ld.param.b64 	%rd19, [kernel_kernel_param_2];
	.loc	1 22 5
	mov.u32 	%r2, %ctaid.x;
	shl.b32 	%r3, %r2, 11;
	mov.u32 	%r4, %tid.x;
	shl.b32 	%r5, %r4, 3;
	or.b32 	%r6, %r3, %r5;
	mul.wide.u32 	%rd20, %r6, 4;
	add.s64 	%rd5, %rd17, %rd20;
	.loc	1 22 5
	.loc	2 77 3, function_name $L__info_string0, inlined_at 1 22 5
	.loc	2 21 3, function_name $L__info_string1, inlined_at 2 77 3
	mov.b32 	%r1, 1;
	mov.b64 	%rd6, 0;
	// begin inline asm
	{
  .reg .pred p;
	.loc	2 23 3, function_name $L__info_string1, inlined_at 2 77 3
  setp.ne.b32 p, %r1, 0;
	.loc	2 24 3, function_name $L__info_string1, inlined_at 2 77 3
  mov.b64 %rd1, %rd6;
	.loc	2 25 3, function_name $L__info_string1, inlined_at 2 77 3
  mov.b64 %rd2, %rd6;
	.loc	2 26 3, function_name $L__info_string1, inlined_at 2 77 3
  mov.b64 %rd3, %rd6;
	.loc	2 27 3, function_name $L__info_string1, inlined_at 2 77 3
  mov.b64 %rd4, %rd6;
  @p ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd4}, [%rd5];
}
```

This version launches
```text
1,048,576 / 2048 = 512 blocks
```

So compared with `v1`, `v2` has:
- 8x fewer blocks
- 8 elements of useful work per thread
- vectorized global loads and stores
- less scalar indexing overhead per element


### V3: Explicit Register Staging

The third version keeps the same `256 * 8` work shape as `v2`, but writes the staging explicitly. Notice that we copy over the data for each block first, so that we don't need to care about `block_idx` in the loop.

```py
@tilelang.jit
def vector_add_v3(
    n: int,
    block_size: int = 256,
    num_per_thread: int = 8,
    dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size * num_per_thread), threads=block_size) as block_idx:
            A_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            B_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            C_register = T.alloc_fragment((block_size * num_per_thread), dtype)
            s_start = block_idx * block_size * num_per_thread
            s_end = (block_idx + 1) * block_size * num_per_thread

            # load 8 values from a into A_register
            T.copy(a[s_start:s_end], A_register)

            # load 8 values from b into B_register
            T.copy(b[s_start:s_end], B_register)

            # for i in 0..8:
            #   C_register[i] = A_register[i] + B_register[i]
            for thread_idx, j in T.Parallel(block_size, num_per_thread):
                i = (thread_idx * num_per_thread) + j
                C_register[i] = A_register[i] + B_register[i]

            # store C_register to c
            T.copy(C_register, c[s_start:s_end])

    return kernel
```


The PrimFunc uses fragments:

```python
A_register = T.alloc_fragment((2048,), dtype)
B_register = T.alloc_fragment((2048,), dtype)
C_register = T.alloc_fragment((2048,), dtype)

T.copy(a[s_start:s_end], A_register)
T.copy(b[s_start:s_end], B_register)

for thread_idx, j in T.Parallel(256, 8):
    i = thread_idx * 8 + j
    C_register[i] = A_register[i] + B_register[i]

T.copy(C_register, c[s_start:s_end])
```

The lowered IR assigns each thread local buffers of length 8:

```python
A_register = T.alloc_buffer((8,), scope="local")
B_register = T.alloc_buffer((8,), scope="local")
C_register = T.alloc_buffer((8,), scope="local")

A_register_1[0:8] = a_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8]
B_register_1[0:8] = b_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8]

for i in T.unroll(8):
    C_register_1[i] = A_register_1[i] + B_register_1[i]

c_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8] = C_register_1[0:8]
```

The CUDA also uses vectorized loads and stores:

```cpp
float A_register[8];
float B_register[8];
float C_register[8];

*(ulonglong4*)(A_register + 0) = tl::load_global_256(...);
*(ulonglong4*)(B_register + 0) = tl::load_global_256(...);

#pragma unroll
for (int i = 0; i < 8; ++i) {
  C_register[i] = A_register[i] + B_register[i];
}

tl::store_global_256(..., *(ulonglong4*)(C_register + 0));
```

This style is useful when a tile is reused, transformed, or participates in a more complex computation. For pure vector add, each value is used once. The explicit local arrays may not help, and they can increase register pressure. Based on my understanding, V3 is purly to demonstrate different flavor of TileLang API, not necessarily optimizing the kernel.

## Deep Dive Into V2

Now zoom into `v2`, because it is the most instructive one. The high-level flow is TileLang source code -> TileLang PrimFunc -> IR -> CUDA -> PTX -> SASS -> execution on my 5060Ti.

The TileLang source is:

```python
@tilelang.jit
def vector_add_v2(
    n: int,
    block_size: int = 256,
    num_per_thread: int = 8,
    dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size * num_per_thread), threads=block_size) as block_idx:
            for thread_idx, j in T.Parallel(block_size, num_per_thread):
                i = (block_idx * block_size + thread_idx) * num_per_thread
                c[i + j] = a[i + j] + b[i + j]

    return kernel
```

### `T.Tensor((n,), dtype)` for buffer allocation

These declare the logical buffers:
```python
a: T.Tensor((n,), dtype)
b: T.Tensor((n,), dtype)
c: T.Tensor((n,), dtype)
```

TileLang later lowers them into one-dimensional global buffers:
```python
a = T.match_buffer(a_handle, (1048576,), strides=(1,))
b = T.match_buffer(b_handle, (1048576,), strides=(1,))
c = T.match_buffer(c_handle, (1048576,), strides=(1,))
```

### `T.Kernel(...)` for kernel meta

The launch shape is:

```python
T.ceildiv(n, block_size * num_per_thread)
```
Since we need 512 blocks if each block process 256 * 8 elements, the total block number is materialized during compilation. The PrimFunc has:

```python
bx = T.launch_thread("blockIdx.x", 512)
tx = T.launch_thread("threadIdx.x", 256)
```

The lowered IR records the same launch configuration in the function attributes:

```python
T.func_attr({
    "thread_extent": {
        "blockIdx.x": 512,
        "threadIdx.x": 256,
        "threadIdx.y": 1,
        "threadIdx.z": 1,
    },
    ...
})
```

At the CUDA level:

```cpp
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(...)
```

`__launch_bounds__(256, 1)` tells the CUDA compiler this kernel is launched with at most 256 threads per block and at least one block per SM.

### `T.Parallel(block_size, num_per_thread)`

This line:

```python
for thread_idx, j in T.Parallel(block_size, num_per_thread):
```

creates a two-dimensional parallel iteration space (nested loop)

```text
thread_idx in 0..255
j          in 0..7
```

The important part is the layout. `thread_idx` maps to CUDA's `threadIdx.x`; `j` becomes each thread's contiguous per-thread vector lane.That is why the lowered IR does not remain a nested scalar loop. It becomes a vector slice:

```python
c_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8] =
    a_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8] +
    b_1[bx * 2048 + tx * 8 : bx * 2048 + tx * 8 + 8]
```

Conceptually:
```text
for this CUDA thread:
    load 8 contiguous floats from a
    load 8 contiguous floats from b
    add them elementwise
    store 8 contiguous floats to c
```


### CUDA Vector Loads
Now let's look at some cuda specific stuff for memory movement. The generated CUDA begins by loading 8 floats from `a`. The cast reinterprets the address as a 256-bit vector pointer and tell the helper to perform a 256-bit global load.
```cpp
ulonglong4 v_ = tl::load_global_256(
    &(*(ulonglong4*)(a + blockIdx.x * 2048 + threadIdx.x * 8)));
```


`ulonglong4` is four unsigned 64-bit (8 byte) values for vectorization, so total is 32 bytes. `float32` is 4 bytes, so it matches `num_per_thread = 8` (That's why these hyperparameter matters!!)


### CUDA Add
The CUDA then adds two floats at a time:

```cpp
*(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
*(float2*)(&(__1.y)) = tl::add2(*(float2*)(&(v_.y)), *(float2*)(&(v__1.y)));
*(float2*)(&(__1.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
*(float2*)(&(__1.w)) = tl::add2(*(float2*)(&(v_.w)), *(float2*)(&(v__1.w)));
```

Each `ulonglong4` lane is 64 bits. A `float2` is also 64 bits (2 * float32). So the four calls cover eight floats:

```text
__1.x -> elements 0,1
__1.y -> elements 2,3
__1.z -> elements 4,5
__1.w -> elements 6,7
```

The result is packed back into `__1`.

### CUDA Store

Finally:

```cpp
tl::store_global_256(
    &(*(ulonglong4*)(c + blockIdx.x * 2048 + threadIdx.x * 8)), __1);
```

This writes the eight `float32` results back with one 256-bit store.

### PTX: Parameters And Registers
Now let's look at the generated PTX code.
```ptx
.visible .entry kernel_kernel(
    .param .u64 .ptr .align 1 kernel_kernel_param_0,
    .param .u64 .ptr .align 1 kernel_kernel_param_1,
    .param .u64 .ptr .align 1 kernel_kernel_param_2
)
.maxntid 256
.minnctapersm 1
{
    .reg .b32 %r<7>;
    .reg .b64 %rd<21>;
```

The three params are the addresses of `a`, `b`, and `c`. The register declarations reserve temporary registers:

- `%r` registers are 32-bit
- `%rd` registers are 64-bit

Pointers and vector lanes need 64-bit registers. Thread/block IDs and offsets can use 32-bit registers until widened for pointer arithmetic.

### PTX: Load Kernel Parameters

```ptx
ld.param.b64 %rd17, [kernel_kernel_param_0];
ld.param.b64 %rd18, [kernel_kernel_param_1];
ld.param.b64 %rd19, [kernel_kernel_param_2];
```

These load the three pointer arguments into registers:
```text
%rd17 = a
%rd18 = b
%rd19 = c
```

### PTX: Compute The Base Offset

```ptx
mov.u32 %r2, %ctaid.x;
shl.b32 %r3, %r2, 11;
mov.u32 %r4, %tid.x;
shl.b32 %r5, %r4, 3;
or.b32  %r6, %r3, %r5;
mul.wide.u32 %rd20, %r6, 4;
```

This is the lowered version of:

```text
element_offset = blockIdx.x * 2048 + threadIdx.x * 8
byte_offset = element_offset * 4
```

Why shifts?

```text
2048 = 2^11, so blockIdx.x * 2048 becomes shl by 11
8    = 2^3,  so threadIdx.x * 8 becomes shl by 3
4    = sizeof(float32), so byte offset multiplies by 4
```

The compiler uses `or.b32` because the block portion and thread portion occupy non-overlapping low bits:

```text
blockIdx.x << 11
threadIdx.x << 3
```

Combining them with OR is equivalent to addition in this layout.

### PTX: Address For `a`

```ptx
add.s64 %rd5, %rd17, %rd20;
```

This computes:

```text
%rd5 = a + byte_offset
```

### PTX: Vector Load

The generated inline assembly contains:

```ptx
@p ld.global.v4.u64 {%rd1, %rd2, %rd3, %rd4}, [%rd5];
```

This is the PTX version of:

```cpp
tl::load_global_256(...)
```

It loads four 64-bit values:

```text
4 * 64 bits = 256 bits = 32 bytes = 8 float32 values
```

The same pattern appears again for `b`.

### PTX: Add And Store

Later in the PTX, the four 64-bit chunks are added as packed pairs of
`float32` values:

```ptx
add.rn.f32x2 %rd13, %rd1, %rd7;
add.rn.f32x2 %rd14, %rd2, %rd8;
add.rn.f32x2 %rd15, %rd3, %rd9;
add.rn.f32x2 %rd16, %rd4, %rd10;
```

Each `f32x2` instruction handles two `float32` values packed inside one
64-bit register. Four instructions cover eight elements.

Then PTX computes the output address:

```ptx
add.s64 %rd12, %rd19, %rd20;
```

and stores the four 64-bit chunks:

```ptx
@p st.global.v4.u64 [%rd12], {%rd13, %rd14, %rd15, %rd16};
```

So v2 is vectorized at both the memory level and the packed-arithmetic level
in this PTX.
