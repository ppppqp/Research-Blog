---
author: Retep
pubDatetime: 2026-09-08T12:00:00.000-07:00
title: "[TIL] Optimizing paged attention"
featured: false
draft: false
tags:
  - TIL
description: ""
---

This blog will cover the strategy to profile and optimize over an paged attention kernel, using Nsight Compute.

## Paged Attention
Attention is a pretty basic concept: getting a sequence of token, we want to somehow derive the latent state of the sequence. Assuming the sequence is a list of 1D vectors, we want to train three matrix, $Q$, $K$ and $V$. Every token vector $e$ first multiply $Q$ and $K$ to get its projection $Q(e)$ and $K(e)$ (I like to use the function representation here to understand this as a projection). Then, given the query token $e_q$, we calculate the liklihood of it being the next token in the sequence by the following method: for all $e$ in the sequence, calculate $\dfrac{Q(e_q)\cdot K(e)}{scale}$, then calculate the softmax across all $e$ to regularize, and finally inner product with each $e$'s $V(e)$ respectively. Intuitively, it's simulating the process of "given a query, look up how much it matches the key, and based on the match level, scale the key's corresponding value's weight".

Paged attention is basically a cache reuse strategy. When deriving a sequence, we already calculated the $K(e)$ and $V(e)$ of the existing sequence, so we can reuse them instead of redoing the matrix product.

## Data Layout
For this lab, the data layout is hierarchical. First of all, the core part is storing the KV cache. The highest level is batch, and for each batch we see a list of logical pages, which is essentially pointers that points to physical page in memory. In this lab, we simplify that to just be list index to a contiguous global memory. Each physical page is guaranteed to be contiguous in memory, and stores `page_size` tokens. Each token consists of `kv_heads` kv head vectors in total, each has a length of `head_dim`. The input(query) and output(latent state of the query) are essentially the same. For this lab, each batch we have just one input query, and it contains `query_heads` query head vectors, each has a length of `head_dim`.

![abstraction](@/assets/images/paged-attention-1.png)
![abstraction](@/assets/images/paged-attention-2.png)
![abstraction](@/assets/images/paged-attention-3.png)


## Baseline
The baseline is made intentionally simple.

```cpp
#include "paged_attention/api.cuh"

#include <cmath>

namespace pa {
namespace {

// Each output element redundantly
// recomputes every Q.K dot product. It is simple, deterministic, and correct,
// making it a useful profiling baseline rather than a performance target.
__global__ void paged_attention_baseline_kernel(Problem p, Inputs in, float *__restrict__ out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;
  int const d_out = threadIdx.x;
  if (d_out >= p.head_dim)
    return;

  int const kv_head = q_head / p.queries_per_kv();
  int const context = in.context_lengths[batch];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  for (int token = 0; token < context; ++token) {
    // For each token t, get its cached K(t) and V(t)
    // Assuming for this session, token is continously stored in pages
    // e.g. the ith token is stored in i / page_size th page

    int const logical_page = token / p.page_size;
    int const token_in_page = token % p.page_size;
    // block tables maps token index to the physical page
    // each batch(or "stream") can contain max_pages_per_sequence pages
    /*
    stream 1:   p0,  p1,   p2,   #,  #,  #,  #,  #
    stream 2:   p8,  t9,   t10,  #,  #,  #,  #,  #
    stream 3:   t16, t17,  t18,  #,  #,  #,  #,  #

    # for placeholder (blank page), assuming max_pages_per_sequence=8

    */
    int const physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];

    // this assumes pages are stored continously. therefore the start of the page is
    // physical_page * page_size entries away from the base.
    // Each entry has kv_heads * head_dim fp16
    // Finally add the current kv_head we are computing against, which means another kv_head *
    // head_dim fp16 offset
    std::size_t const kv_base =
        ((static_cast<std::size_t>(physical_page) * p.page_size + token_in_page) * p.kv_heads +
         kv_head) *
        p.head_dim;

    // get the query of the current query token
    std::size_t const q_base =
        (static_cast<std::size_t>(batch) * p.query_heads + q_head) * p.head_dim;

    float score = 0.0f;
    for (int d = 0; d < p.head_dim; ++d) {
      // manually perform dot product of query_head * kv_head
      score = fmaf(in.query[q_base + d], in.key_cache[kv_base + d], score);
    }
    score *= in.scale;

    float const new_max = fmaxf(running_max, score);
    float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
    float const new_factor = __expf(score - new_max);
    running_sum = running_sum * old_factor + new_factor;
    accumulator = accumulator * old_factor + new_factor * in.value_cache[kv_base + d_out];
    running_max = new_max;
  }
  // softmax across all tokens, store to the output

  std::size_t const out_index =
      (static_cast<std::size_t>(batch) * p.query_heads + q_head) * p.head_dim + d_out;
  out[out_index] = accumulator / running_sum;
}

} // namespace

void launch_baseline(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  dim3 const grid(p.query_heads, p.batch);
  int const threads = p.head_dim <= 64 ? 64 : 128;
  paged_attention_baseline_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}

} // namespace pa
```

Each block owns a batch and a query head, and each thread owns one output dimension. It loops through all tokens, computes all the dot product for all dimensions to get the sum and max, and return only the output dimension it owns. The online softmax here is almost doing the more work than just vanilla softmax since thread loops through all dimensions anyways. The kernel profile looks like this:
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz          13.79
    SM Frequency                    Ghz           2.60
    Elapsed Cycles                cycle    312,174,573
    Memory Throughput                 %          89.78
    DRAM Throughput                   %           2.40
    Duration                         ms         119.98
    L1/TEX Cache Throughput           %          91.72
    L2 Cache Throughput               %           2.23
    SM Active Cycles              cycle 305,463,204.67
    Compute (SM) Throughput           %          89.78
    ----------------------- ----------- --------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle     39,683,480
    Total DRAM Elapsed Cycles        cycle  6,617,162,752
    Average L1 Active Cycles         cycle 305,463,204.67
    Total L1 Elapsed Cycles          cycle 11,234,537,952
    Average L2 Active Cycles         cycle 235,037,434.56
    Total L2 Elapsed Cycles          cycle  4,453,487,280
    Average SM Active Cycles         cycle 305,463,204.67
    Total SM Elapsed Cycles          cycle 11,234,537,952
    Average SMSP Active Cycles       cycle 305,463,153.95
    Total SMSP Elapsed Cycles        cycle 44,938,151,808
    -------------------------- ----------- --------------
```
The baseline is simultaneously busy on the SM and L1 while barely exercising DRAM. On the first sight you might think it's a good thing: a lot of cache traffic and good occupancy. Well it's not always good because the algorithm can easily be busy doing redundant work, and as we said before, it's doing `head_dim` times of redundant work. For `head_dim=128`, that's 128x more work and the high occupancy cannot compensate for doing roughly two orders of magnitude too much score work.

## V1 (Cooporative workload)

So the next optimization is naturally cutting off redundant work. It keeps one CTA per `(batch, query_head)`. Threads cooperate on a single QK dot product, reduce warp partials through shared memory, and then update distinct output dimensions using online softmax. This removes the baseline's replicated dot products but introduces CTA-wide communication and barriers for every token.

```cpp
#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {

namespace {

__device__ float warp_reduce_sum(float value) {
  // each warp has 32 lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    // 0xfffffff means all lanes participate
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_attention_fused_kernel(Problem p, Inputs in, float *out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;

  int const d = threadIdx.x;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  const int num_warps = blockDim.x / 32;

  // each thread computes one dimension, and reduce to become dot product
  if (d >= p.head_dim) {
    return;
  }
  int const kv_head = q_head / p.queries_per_kv();

  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + q_head) * p.head_dim;
  float q = in.query[q_base + d];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator = 0.0f;

  const int num_pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    for (int token = 0; token < valid_tokens; ++token) {
      // load K, V
      const std::size_t kv_base =
          ((static_cast<std::size_t>(physical_page)) * p.page_size + token) * p.kv_heads *
              p.head_dim +
          kv_head * p.head_dim;
      // only load d dimension
      float k_d = in.key_cache[kv_base + d];
      float v_d = in.value_cache[kv_base + d];
      float partial_sum = q * k_d;
      float warp_sum = warp_reduce_sum(partial_sum);
      // assume head_dim = 128, we need 4 warps
      __shared__ float warp_sums[4];
      // check if I am the leader warp
      if (lane == 0) {
        warp_sums[warp] = warp_sum;
      }
      __syncthreads();
      // warp 0 reduces all warps
      float block_sum = 0.0f;
      __shared__ float shared_score;
      if (warp == 0) {
        block_sum = lane < num_warps ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
          shared_score = block_sum * in.scale;
        }
      }
      __syncthreads();
      float const new_max = fmaxf(running_max, shared_score);
      float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      float const increment = __expf(shared_score - new_max);
      // new factor is the incremental
      // prev_sum * old_max / new_max + increment / new_max
      running_sum = running_sum * old_factor + increment;
      // NOTE THAT accumulator is different for each thread
      // because v_d is different
      accumulator = accumulator * old_factor + increment * v_d;
      running_max = new_max;
    }
  }

  std::size_t const out_index = (batch * p.query_heads + q_head) * p.head_dim + d;
  out[out_index] = accumulator / running_sum;
}
} // namespace
void launch_fused_v1(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 1): replace this forwarding call with a cooperative CTA kernel.
  // Compute each Q.K once per warp/CTA, use warp reductions, then broadcast the
  // score while lanes update disjoint output dimensions with online softmax.
  dim3 const grid(p.query_heads, p.batch);
  int const threads = p.head_dim <= 64 ? 64 : 128;
  paged_attention_fused_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}
} // namespace pa
```

For reduction, we first run warp-level butterfly reduction with `__shfl_down_sync`. This does not need explicit barrier because all threads in the same warp are issued together. Then, each warp stores the reduced partial result into shared memory. Warp 0 will load those results from shared memory, and perform another warp reduction to get the final result, which is stored to shared memory. Then after `syncthreads`, all threads load the final result from shared memory to compute its own dimension.

Profiling under the same parameters yields this result:
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz         13.79
    SM Frequency                    Ghz          2.60
    Elapsed Cycles                cycle    19,510,057
    Memory Throughput                 %         60.59
    DRAM Throughput                   %         37.92
    Duration                         ms          7.50
    L1/TEX Cache Throughput           %         61.40
    Waves Per SM                                 4.74
    L2 Cache Throughput               %         28.18
    SM Active Cycles              cycle 19,237,620.33
    Compute (SM) Throughput           %         60.59
    ----------------------- ----------- -------------


    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    39,235,040
    Total DRAM Elapsed Cycles        cycle   413,871,104
    Average L1 Active Cycles         cycle 19,237,620.33
    Total L1 Elapsed Cycles          cycle   701,828,280
    Average L2 Active Cycles         cycle 16,855,800.56
    Total L2 Elapsed Cycles          cycle   277,861,376
    Average SM Active Cycles         cycle 19,237,620.33
    Total SM Elapsed Cycles          cycle   701,828,280
    Average SMSP Active Cycles       cycle 19,237,596.24
    Total SMSP Elapsed Cycles        cycle 2,807,313,120
    -------------------------- ----------- -------------
```

The performance improved by 16x (both duration and elapsed cycles). Compute and L1 percentages fall because redundant work was removed. DRAM throughput rises from 2.40% to 37.92%, even though DRAM active cycles remain near 39 million. The similar DRAM activity represents a similar unique KV working set completed in much less time.

The profile's busiest compute component is the LSU path at 60.59%, while issue active is 40.05%. The kernel is no longer dominated by scalar FMA redundancy; load/store work and synchronization are now proportionally important.

## V2 (Vectorization)

Well, V1 actually looks pretty good already. To further improve it, we need to utilize hardware features, and one of the most important is vectorization. Notice that each head is mostly `128` dimension, which can be split into 32 `float4` vectors, and can exactly fit into one warp. This gives us the benefit of:
1. Vectorized load and store is faster, because hardware usually only need to issue one instruction for continguous memory address read.
2. We no longer need block-level synchronization if everything can be done within a warp.

```cpp
#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {
namespace {
__device__ float warp_reduce_sum(float value) {
  // each warp has 32 lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    // 0xfffffff means all lanes participate
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_attention_vectorized_kernel(Problem p, Inputs in, float *out) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;

  constexpr int vec_width = 4;
  int const chunks = p.head_dim / vec_width;
  int const lane = threadIdx.x & 31;

  // if head_dim = 64, only 16 lanes will be used
  const bool is_valid_lane = lane < chunks;

  int const kv_head = q_head / p.queries_per_kv();

  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + q_head) * p.head_dim;
  // reintepret Q as float4 aligned
  auto const *query4 = reinterpret_cast<float4 const *>(in.query + q_base);
  // load Q aligned
  float4 q = is_valid_lane ? query4[lane] : make_float4(0, 0, 0, 0);
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float4 accumulator = make_float4(0, 0, 0, 0);

  const int num_pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    for (int token = 0; token < valid_tokens; ++token) {
      // load K, V
      const std::size_t kv_base =
          ((static_cast<std::size_t>(physical_page)) * p.page_size + token) * p.kv_heads *
              p.head_dim +
          kv_head * p.head_dim;

      float4 const *key4 = reinterpret_cast<float4 const *>(in.key_cache + kv_base);
      float4 const *value4 = reinterpret_cast<float4 const *>(in.value_cache + kv_base);

      // load d aligned
      float4 k_d = is_valid_lane ? key4[lane] : make_float4(0, 0, 0, 0);
      float4 v_d = is_valid_lane ? value4[lane] : make_float4(0, 0, 0, 0);
      float partial_sum = q.x * k_d.x + q.y * k_d.y + q.z * k_d.z + q.w * k_d.w;
      float shared_score = warp_reduce_sum(partial_sum);
      // need to broad cast to all lanes
      shared_score = __shfl_sync(0xffffffff, shared_score, 0);
      shared_score *= in.scale;
      // assume head_dim = 128, we only need 1 warp
      float const new_max = fmaxf(running_max, shared_score);
      float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      float const increment = __expf(shared_score - new_max);
      // new factor is the incremental
      // prev_sum * old_max / new_max + increment / new_max
      running_sum = running_sum * old_factor + increment;
      accumulator.x = accumulator.x * old_factor + increment * v_d.x;
      accumulator.y = accumulator.y * old_factor + increment * v_d.y;
      accumulator.z = accumulator.z * old_factor + increment * v_d.z;
      accumulator.w = accumulator.w * old_factor + increment * v_d.w;
      running_max = new_max;
    }
  }
  float inverse_sum = 1.0f / running_sum;
  accumulator.x *= inverse_sum;
  accumulator.y *= inverse_sum;
  accumulator.z *= inverse_sum;
  accumulator.w *= inverse_sum;

  std::size_t const out_base = (batch * p.query_heads + q_head) * p.head_dim;
  float4 *out4 = reinterpret_cast<float4 *>(out + out_base);
  if (is_valid_lane) {
    out4[lane] = accumulator;
  }
}

} // namespace

void launch_vectorized_v2(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 2): start from fused_v1. Add aligned 16-byte Q/K/V accesses,
  // compare cache layouts, and preserve a scalar tail path if needed.
  dim3 const grid(p.query_heads, p.batch);

  // only need one warp
  int const threads = 32;
  paged_attention_vectorized_kernel<<<grid, threads, 0, stream>>>(p, in, out);
}
} // namespace pa
```

V2 have one thread owns 4 output dimensions, stored in a `float4`. Notice that it means we also need `float4` registers to load the `K`, `V` values and `accumulator` for performing the online softmax. Instead of using a scalar `d` to index dimension, we now count in `chunck`s, and we reintepret the pointers into `float4` for vectorized load and store. For reduction, the `sum` and `max` do not need to be vectorized since they are shared across all dimensions. Notice that we just need on warp level reduction to reduce to lane 0, and use `__shfl_sync` to broadcast the reduction result to all lanes.


```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz         13.79
    SM Frequency                    Ghz          2.60
    Elapsed Cycles                cycle    12,869,678
    Memory Throughput                 %         57.42
    DRAM Throughput                   %         57.42
    Duration                         ms          4.95
    L1/TEX Cache Throughput           %         34.10
    Waves Per SM                                 2.37
    L2 Cache Throughput               %         42.14
    SM Active Cycles              cycle 12,578,048.19
    Compute (SM) Throughput           %         17.02
    ----------------------- ----------- -------------


    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    32
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  2,048
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              64
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              36
    Stack Size                                                 1,024
    Threads                                   thread          65,536
    # TPCs                                                        18
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                2.37
    Theoretical Occupancy                          %              50
    Achieved Occupancy                             %           37.58
    -------------------------------- --------------- ---------------
```
The kernel becomes faster even though occupancy drops from 89.36% to 37.58%. The one-warp CTA is subject to the hardware's resident-block limit, so it can reach only 24 of the GPU's 48 possible active warps per SM. Each lane also owns more live state, increasing register use from 31 to 64. Nevertheless, warp-only reduction and lower synchronization/instruction cost outweigh reduced latency hiding. The kernel shifts strongly toward memory: DRAM is now the most utilized major unit at 57.42%, while compute is only 17.02%.

## V3 (Shared memory data reuse)

The next opportunity is data reuse, since memory becomes the bottleneck again. Four query heads share each KV head, but v2 independently loads the same logical K/V data for each query head. Make that reuse explicit by staging a page once per KV head and letting four query warps consume it. 

```cpp
#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {

namespace {
__device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}
__global__ void paged_attention_gqa_reuse_kernel(Problem p, Inputs in, float *out) {
  const int kv_head = blockIdx.x;
  const int batch = blockIdx.y;
  constexpr int vec_width = 4;
  constexpr int warp_size = 32;
  const int warp = threadIdx.x / warp_size;
  const int lane = threadIdx.x % warp_size;
  // since we are vectorizing, we will use chunks instead of head_dim
  const int chunks = p.head_dim / vec_width;
  bool is_valid_lane = lane < chunks;
  // each CTA contains four warps (handles 4 kv_heads)
  // each CTA handles 1 kv_head and (queries_per_kv) query_heads
  // each warp handles 1 query_head

  // all shared_k and shared_v for one page.
  extern __shared__ float4 shared[];
  float4 *shared_k = shared;
  float4 *shared_v = shared + p.page_size * chunks;
  // every page hosts (page_size * chunks) chunks
  const int query_head = kv_head * p.queries_per_kv() + warp;
  // the query base for the warp (query_head already contains warp info)
  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + query_head) * p.head_dim;
  // each warp is responsible for one query head
  float4 const *global_q = reinterpret_cast<float4 const *>(in.query + q_base);
  // the actual q this thread is responsible for
  const float4 q = is_valid_lane ? global_q[lane] : make_float4(0, 0, 0, 0);

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float4 accumulator = make_float4(0, 0, 0, 0);

  const int pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  for (int logical_page = 0; logical_page < pages; ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    const std::size_t kv_base =
        ((static_cast<std::size_t>(physical_page)) * p.page_size) * p.kv_heads * p.head_dim +
        kv_head * p.head_dim;
    float4 const *global_k4 = reinterpret_cast<float4 const *>(in.key_cache + kv_base);
    float4 const *global_v4 = reinterpret_cast<float4 const *>(in.value_cache + kv_base);

    for (int index = threadIdx.x; index < valid_tokens * chunks; index += blockDim.x) {
      // the step is the block size, so that all threads in the block participate in loading evenly
      int token = index / chunks;
      int chunk = index % chunks;
      shared_k[index] = global_k4[token * p.kv_heads * chunks + chunk];
      shared_v[index] = global_v4[token * p.kv_heads * chunks + chunk];
    }
    __syncthreads();

    for (int token = 0; token < valid_tokens; ++token) {
      // each threads computes a chunk
      float4 k = is_valid_lane ? shared_k[token * chunks + lane] : make_float4(0, 0, 0, 0);
      float4 v = is_valid_lane ? shared_v[token * chunks + lane] : make_float4(0, 0, 0, 0);
      float partial = q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
      float score = warp_reduce_sum(partial);
      score = __shfl_sync(0xffffffff, score, 0);
      score *= in.scale;

      const float new_max = fmaxf(running_max, score);
      const float alpha = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      const float increment = __expf(score - new_max);
      running_sum = running_sum * alpha + increment;
      accumulator.x = accumulator.x * alpha + increment * v.x;
      accumulator.y = accumulator.y * alpha + increment * v.y;
      accumulator.z = accumulator.z * alpha + increment * v.z;
      accumulator.w = accumulator.w * alpha + increment * v.w;
      running_max = new_max;
    }
    __syncthreads();
  }
  accumulator.x /= running_sum;
  accumulator.y /= running_sum;
  accumulator.z /= running_sum;
  accumulator.w /= running_sum;

  const std::size_t out_base = (batch * p.query_heads + query_head) * p.head_dim;
  float4 *out4 = reinterpret_cast<float4 *>(out + out_base);
  if (is_valid_lane) {
    out4[lane] = accumulator;
  }
}

} // namespace
void launch_gqa_reuse_v3(Problem const &p, Inputs const &in, float *out, cudaStream_t stream) {
  // TODO(stage 3): assign a CTA to (batch, kv_head), load a K/V tile once,
  // and process multiple query heads. Measure the reuse/register tradeoff.
  const dim3 grid(p.kv_heads, p.batch);
  const int threads = p.queries_per_kv() * 32; // four warps

  // for each block, allocate one page for k, one page for v
  const int shared_bytes = p.page_size * p.head_dim * 2 * sizeof(float);
  paged_attention_gqa_reuse_kernel<<<grid, threads, shared_bytes, stream>>>(p, in, out);
}
} // namespace pa
```

In this version, instead of each thread loading it's own KV from pages, all threads in the block cooperatively load one page before the computation. In order to make sure that all threads in one block can reuse that loaded KV, we change the ownership from one block owning `(batch, query_head)` to `(batch, kv_head)`, which consists of four query heads. Each warp will now own a query head instead: 
```cpp
const int query_head = kv_head * p.queries_per_kv() + warp;
```

Notice that we distribute the load evenly across all threads in the block, and make sure all loads are unique:
```cpp
    for (int index = threadIdx.x; index < valid_tokens * chunks; index += blockDim.x) {
      // the step is the block size, so that all threads in the block participate in loading evenly
      int token = index / chunks;
      int chunk = index % chunks;
      shared_k[index] = global_k4[token * p.kv_heads * chunks + chunk];
      shared_v[index] = global_v4[token * p.kv_heads * chunks + chunk];
    }
    __syncthreads();
```
After the entire page is loaded, we then walk through each token in this page, and load its corresponding KV from the shared memory. The rest is pretty much the same as V2.


```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        13.79
    SM Frequency                    Ghz         2.60
    Elapsed Cycles                cycle    8,712,909
    Memory Throughput                 %        84.86
    DRAM Throughput                   %        84.86
    Duration                         ms         3.35
    L1/TEX Cache Throughput           %        30.32
    Waves Per SM                                2.84
    L2 Cache Throughput               %        31.54
    SM Active Cycles              cycle 8,193,819.39
    Compute (SM) Throughput           %        28.46
    ----------------------- ----------- ------------


    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   128
    Cluster Scheduling Policy                           PolicySpread
    Cluster Size                                                   0
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                    512
    Preferred Cluster Size                                         0
    Registers Per Thread             register/thread              39
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block           16.38
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              36
    Stack Size                                                 1,024
    Threads                                   thread          65,536
    # TPCs                                                        18
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                2.84
    Theoretical Occupancy                          %           41.67
    Achieved Occupancy                             %           36.79
    -------------------------------- --------------- ---------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    39,229,512
    Total DRAM Elapsed Cycles        cycle   184,905,728
    Average L1 Active Cycles         cycle  8,193,819.39
    Total L1 Elapsed Cycles          cycle   313,420,716
    Average L2 Active Cycles         cycle  7,569,240.88
    Total L2 Elapsed Cycles          cycle   124,075,968
    Average SM Active Cycles         cycle  8,193,819.39
    Total SM Elapsed Cycles          cycle   313,420,716
    Average SMSP Active Cycles       cycle  8,193,791.33
    Total SMSP Elapsed Cycles        cycle 1,253,682,864
    -------------------------- ----------- -------------
```


This is the fastest complete kernel in the reports. Explicit reuse reduces the grid from 2,048 query-head CTAs to 512 KV-head CTAs and avoids duplicate per-query-head global/L1 load instructions. Register use also falls from 64 to 39 because K/V staging changes live values and address-generation requirements.

Shared memory limits residency to roughly five 128-thread CTAs, or 20 warps per SM, which explains the 41.67% theoretical occupancy. Despite that cost, the kernel wins because reuse is more valuable than maximum residency. DRAM throughput reaches 84.86% while average DRAM active cycles remain almost unchanged. The optimization has exposed the irreducible memory working set: the kernel is now credibly DRAM-bound on this workload.

Further improvement must reduce bytes or hide memory latency more effectively. Candidate experiments are:
- FP16/BF16 K/V storage with FP32 accumulation;
- INT8 or FP8 KV-cache storage with explicit scale handling;
- asynchronous or double-buffered page staging where supported;
- prefetching the next physical-page table entry;

## V4 (Context-axis parallelism)
For V4, I exprimented with parallelizing the context axis. For a sequence with large context, computing the complete context for a query head within one block can be slow. The idea is that we split the context into smaller chunks and complete one chunk per block, and run a second merge kernel to merge the result. This is more of a scalability concern instead of pure performance.

```cpp
#include "paged_attention/api.cuh"
#include <cmath>
namespace pa {
namespace {
__device__ float warp_reduce_sum(float value) {
  // each warp has 32 lanes
  for (int offset = 16; offset > 0; offset /= 2) {
    // 0xfffffff means all lanes participate
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void paged_attention_split_kernel(Problem p, Inputs in, float *global_partial_max,
                                             float *global_partial_sum,
                                             float *global_partial_output, int num_splits) {
  int const q_head = blockIdx.x;
  int const batch = blockIdx.y;
  int const split = blockIdx.z;

  constexpr int vec_width = 4;
  int const chunks = p.head_dim / vec_width;
  int const lane = threadIdx.x & 31;

  // if head_dim = 64, only 16 lanes will be used
  const bool is_valid_lane = lane < chunks;

  int const kv_head = q_head / p.queries_per_kv();

  const std::size_t q_base =
      ((static_cast<std::size_t>(batch)) * p.query_heads + q_head) * p.head_dim;
  // reintepret Q as float4 aligned
  auto const *query4 = reinterpret_cast<float4 const *>(in.query + q_base);
  // load Q aligned
  float4 q = is_valid_lane ? query4[lane] : make_float4(0, 0, 0, 0);
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float4 accumulator = make_float4(0, 0, 0, 0);

  const int num_pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  const int num_pages_per_split = (num_pages + num_splits - 1) / num_splits;

  for (int logical_page = num_pages_per_split * split;
       logical_page < min(num_pages_per_split * (split + 1), num_pages); ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    for (int token = 0; token < valid_tokens; ++token) {
      // load K, V
      const std::size_t kv_base =
          ((static_cast<std::size_t>(physical_page)) * p.page_size + token) * p.kv_heads *
              p.head_dim +
          kv_head * p.head_dim;

      float4 const *key4 = reinterpret_cast<float4 const *>(in.key_cache + kv_base);
      float4 const *value4 = reinterpret_cast<float4 const *>(in.value_cache + kv_base);

      // load d aligned
      float4 k_d = is_valid_lane ? key4[lane] : make_float4(0, 0, 0, 0);
      float4 v_d = is_valid_lane ? value4[lane] : make_float4(0, 0, 0, 0);
      float partial_sum = q.x * k_d.x + q.y * k_d.y + q.z * k_d.z + q.w * k_d.w;
      float shared_score = warp_reduce_sum(partial_sum);
      // need to broad cast to all lanes
      shared_score = __shfl_sync(0xffffffff, shared_score, 0);
      shared_score *= in.scale;
      // assume head_dim = 128, we only need 1 warp
      float const new_max = fmaxf(running_max, shared_score);
      float const old_factor = running_max == -INFINITY ? 0.0f : __expf(running_max - new_max);
      float const increment = __expf(shared_score - new_max);
      // new factor is the incremental
      // prev_sum * old_max / new_max + increment / new_max
      running_sum = running_sum * old_factor + increment;
      accumulator.x = accumulator.x * old_factor + increment * v_d.x;
      accumulator.y = accumulator.y * old_factor + increment * v_d.y;
      accumulator.z = accumulator.z * old_factor + increment * v_d.z;
      accumulator.w = accumulator.w * old_factor + increment * v_d.w;
      running_max = new_max;
    }
  }
  std::size_t const out_base =
      (batch * p.query_heads * num_splits + p.query_heads * split + q_head) * p.head_dim;
  float *split_partial_max =
      global_partial_max + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float *split_partial_sum =
      global_partial_sum + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float4 *split_partial_output = reinterpret_cast<float4 *>(global_partial_output + out_base);
  if (lane == 0) {
    // partial max and partial sum are per-cta
    split_partial_max[0] = running_max;
    split_partial_sum[0] = running_sum;
  }
  if (is_valid_lane) {
    split_partial_output[lane] = accumulator;
  }
}
__global__ void paged_attention_merge_kernel(Problem p, Inputs in, float *global_partial_max,
                                             float *global_partial_sum,
                                             float *global_partial_output, int num_splits,
                                             float *out) {
  const int q_head = blockIdx.x;
  const int batch = blockIdx.y;
  constexpr int vec_width = 4;
  const int chunks = p.head_dim / vec_width;
  const int lane = threadIdx.x & 31;
  const bool is_valid_lane = lane < chunks;

  float combined_max = -INFINITY;
  float combined_sum = 0.0f;
  float4 combined_output = make_float4(0, 0, 0, 0);

  for (int split = 0; split < num_splits; ++split) {
    float split_max = 0.0f;
    float split_sum = 0.0f;
    int state_index = batch * p.query_heads * num_splits + p.query_heads * split + q_head;
    if (lane == 0) {
      split_max = global_partial_max[state_index];
      split_sum = global_partial_sum[state_index];
    }
    split_max = __shfl_sync(0xffffffff, split_max, 0);
    split_sum = __shfl_sync(0xffffffff, split_sum, 0);
    float4 *global_partial_output4 = reinterpret_cast<float4 *>(global_partial_output);
    float4 split_output = is_valid_lane ? global_partial_output4[state_index * chunks + lane]
                                        : make_float4(0, 0, 0, 0);

    if (split_sum != 0.0f) {
      // NOTE: if split_sum is 0.0f, it means the split has no element
      // merging it will lead to -INF / -INF error
      const float new_max = fmaxf(combined_max, split_max);
      const float alpha = combined_max == -INFINITY ? 0.0f : __expf(combined_max - new_max);
      // have to use alpha-beta here, because we have no idea of the token level score now
      // so we can not do increment = __expf(score - new_max);
      // have to directly adjust the already calculated split_output
      const float beta = __expf(split_max - new_max);
      combined_sum = alpha * combined_sum + beta * split_sum;
      combined_output.x = combined_output.x * alpha + beta * split_output.x;
      combined_output.y = combined_output.y * alpha + beta * split_output.y;
      combined_output.z = combined_output.z * alpha + beta * split_output.z;
      combined_output.w = combined_output.w * alpha + beta * split_output.w;
      combined_max = new_max;
    }
  }
  combined_output.x /= combined_sum;
  combined_output.y /= combined_sum;
  combined_output.z /= combined_sum;
  combined_output.w /= combined_sum;
  if (is_valid_lane) {
    float4 *out4 = reinterpret_cast<float4 *>(out + (batch * p.query_heads + q_head) * p.head_dim);
    out4[lane] = combined_output;
  }
}

void launch_split_kv_partial(Problem const &p, Inputs const &in, SplitKvWorkspace const &workspace,
                             cudaStream_t stream) {
  const dim3 grid(p.query_heads, p.batch, workspace.num_splits);
  const int threads = 32;

  paged_attention_split_kernel<<<grid, threads, 0, stream>>>(
      p, in, workspace.partial_max, workspace.partial_sum, workspace.partial_output,
      workspace.num_splits);
}
void launch_split_kv_merge(Problem const &p, Inputs const &in, SplitKvWorkspace const &workspace,
                           cudaStream_t stream, float *out) {
  const dim3 grid(p.query_heads, p.batch);
  const int threads = 32;
  paged_attention_merge_kernel<<<grid, threads, 0, stream>>>(
      p, in, workspace.partial_max, workspace.partial_sum, workspace.partial_output,
      workspace.num_splits, out);
}
} // namespace

void launch_split_kv_v4(Problem const &p, Inputs const &in, float *out,
                        SplitKvWorkspace const &workspace, cudaStream_t stream) {
  // TODO(stage 4): split long contexts across CTAs. Store each split's max,
  // sum, and partial output, then combine them with log-sum-exp correction.
  // Add workspace ownership to the public API when implementing this stage.

  launch_split_kv_partial(p, in, workspace, stream);
  // no need to sync between them if they are launched into the same CUDA stream
  launch_split_kv_merge(p, in, workspace, stream, out);
}
} // namespace pa

```

In V4, we basically have two kernels: one computes the split partial and stores the result to global memory. The other read from global memory, perform merge and write the output. The split kernel is just a small tweak of V2: instead of reading all pages, we only read the pages of the split that the block owns:
```cpp
  const int num_pages = (in.context_lengths[batch] + p.page_size - 1) / p.page_size;
  const int num_pages_per_split = (num_pages + num_splits - 1) / num_splits;

  for (int logical_page = num_pages_per_split * split;
       logical_page < min(num_pages_per_split * (split + 1), num_pages); ++logical_page) {
    const int physical_page = in.block_tables[batch * p.max_pages_per_sequence + logical_page];
    int valid_tokens = min(p.page_size, in.context_lengths[batch] - logical_page * p.page_size);
    for (int token = 0; token < valid_tokens; ++token) {
```
Another thing to note is that we need to explicitly allocate global memory to store the partial results (it's in the driver code which is not included here). We bascially need to store three things:
- partial max: the local max of the current split (for the owned query head)
- partial sum: the local sum of the current split (for the owned query head)
- partial output: the local output (partial QK dot product times V for the owned dimensions)

Partial max and sum are a scalar because it is reduced across the query head. Partial output is a `float4` because all dimensions have their own output. Note that output is not normalized by partial sum, because we will need to correct it with global sum anyways.

```cpp
  std::size_t const out_base =
      (batch * p.query_heads * num_splits + p.query_heads * split + q_head) * p.head_dim;
  float *split_partial_max =
      global_partial_max + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float *split_partial_sum =
      global_partial_sum + batch * p.query_heads * num_splits + p.query_heads * split + q_head;
  float4 *split_partial_output = reinterpret_cast<float4 *>(global_partial_output + out_base);
  if (lane == 0) {
    // partial max and partial sum are per-cta
    split_partial_max[0] = running_max;
    split_partial_sum[0] = running_sum;
  }
  if (is_valid_lane) {
    split_partial_output[lane] = accumulator;
  }
```

For the merge kernel, each block walks through all the split for that query head, and basically perform online softmax on split level.

For split kernel:
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz         13.79
    SM Frequency                    Ghz          2.59
    Elapsed Cycles                cycle    10,178,623
    Memory Throughput                 %         73.41
    DRAM Throughput                   %         73.41
    Duration                         ms          3.92
    L1/TEX Cache Throughput           %         42.05
    Waves Per SM                                18.96
    L2 Cache Throughput               %         52.91
    SM Active Cycles              cycle 10,178,884.22
    Compute (SM) Throughput           %         21.58
    ----------------------- ----------- -------------
```

For merge kernel:
```
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        13.79
    SM Frequency                    Ghz         2.57
    Elapsed Cycles                cycle       57,995
    Memory Throughput                 %        85.68
    DRAM Throughput                   %        85.68
    Duration                         us        22.56
    L1/TEX Cache Throughput           %        15.04
    Waves Per SM                                2.37
    L2 Cache Throughput               %        32.28
    SM Active Cycles              cycle    54,381.42
    Compute (SM) Throughput           %        11.52
    ----------------------- ----------- ------------
```


The merge is strongly memory-bound, as expected: it reads eight (suppose 8 split in total) partial output vectors plus scalar softmax state and performs little arithmetic. Its 85.68% DRAM throughput looks high, but its absolute duration is only 22.56 us and its average DRAM active cycles are only 0.27 million. It is not the primary limiter for this shape. Splitting improves the complete profiled kernel pair relative to unsplit v2. It does this by exposing many more independent CTAs and shortening each CTA's serial token loop. Achieved occupancy approaches its theoretical ceiling, and workload distribution is more uniform than the one-warp unsplit kernel.

The merge cost is small here, but the combined component time is still slower than GQA reuse's 3.35 ms. That is expected on a batch-64 workload: there was already abundant grid-level parallelism, while split-KV does not reuse KV data across the four GQA query heads. The relative merge cost will also grow with split count because workspace size and merge reads are proportional to the number of splits.