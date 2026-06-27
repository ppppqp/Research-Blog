---
author: Retep
pubDatetime: 2026-06-26T12:50:00.000-07:00
title: "How to Write Layout Correctly: A Deep Dive into TileLang Layout Inference"
featured: false
draft: false
tags:
  - TileLang
description: ""
---

Disclaimer: I am not a core TileLang developer. This article is based on reading the source code, so it may not be entirely correct. Treat it as a starting point for discussion and reference. If you find any mistakes, corrections are very welcome and deeply appreciated.

# Introduction

Layout inference is one of TileLang's most elegant parts. Compared with CUDA's SIMT programming model, TileLang lets users program at a higher block level and relies on the compiler for scheduling, achieving near-SOTA performance with very little code. One key contributor is layout inference: the algorithm that maps highly abstract block-level code to thread-level instructions.

While writing TileLang operators, I often ran into a strange phenomenon. For example, consider the following operator:

```py
# kernel
for i in T.Parallel(BLOCK):
  A[i] = B[i, 0]
```

Sometimes compilation fails with:

```
  File "/workspace/tilelang/3rdparty/tvm/src/arith/iter_affine_map.cc", line 2568, in void tvm::arith::InverseAffineIterMapTransformer::Visit_(const tvm::arith::IterSumExpr&)
    TVM_FFI_ICHECK(analyzer_->CanProveEqual(abs(source->scale), 1));

tvm.error.InternalError: Check failed: (analyzer_->CanProveEqual(abs(source->scale), 1)) is false: 
```

However, if we use `T.Fragment` to add a layout annotation to `B`, or change the value of `BLOCK`, it compiles successfully. Out of curiosity, I started reading TileLang's layout inference source code.

# What Is Layout Inference?

First, let's introduce the concept of layout inference. Readers already familiar with it can skip this section. In Triton, every kernel we write obtains its own `program_id`, then uses it to infer the index range covered by the current thread/program and compute physical data addresses. For example:

```py
@triton.jit
def matrix_transpose_kernel(
    input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    src_offset = input + row_pid * stride_ir + col_pid * stride_ic
    tar_offset = output + col_pid * stride_or + row_pid * stride_oc
    val = tl.load(src_offset)
    tl.store(tar_offset, val)
```

In TileLang, we do not deal with threads directly. Instead, we declare what we want to do using tensor buffers:

```py
# Conceptually similar to the following
with T.Kernel(1, threads=32): 
  for i, j in T.Parallel(BLOCK_I, BLOCK_J):
    output[i, j] = input[j, i]
```

(So elegant.)

During JIT compilation, TileLang assigns the work of each `T.Parallel` loop to threads in an optimal way, achieving SIMT execution. This "assignment" algorithm is layout inference. In simple terms, it is a mapping from loop variables to threads. By answering "given loop variables `i` and `j`, which thread should execute the operation?", we obtain the mapping `forward_thread(i, j) = thread`. Then we invert it to obtain `forward_thread^{-1}(thread) = i, j`, meaning "for the current thread, what are the loop variable values?". This automatically derives the layout algebra that, in Triton, we have to write by hand.

Sounds not too hard, right? In reality, it is.

1. From the **correctness** perspective, TileLang lets us declare thread-local buffers, namely fragment buffers. These fragments can be used across multiple interleaved `T.Parallel` regions, but we must guarantee that they remain thread-local, meaning they must be bound to fixed threads.
2. From the **performance** perspective, expanding `T.Parallel` is not simply a matter of assigning iterations by loop count. We also need to consider things like vectorization, so memory accesses can be coalesced and latency reduced.

For fragment buffers, we need to consider one additional variable: replication. An element may need to appear on multiple threads, so element indices and threads are not necessarily bijective. Also, when programming, we always use block-level fragment buffer indices. But after compiling to the thread level, only part of the fragment buffer is actually copied to a given thread. Therefore, we also need a mapping from block-level indices to thread-level indices. In summary, for a fragment buffer we need to answer two questions:

1. Given a fragment buffer index `(i, j)` and replica `rep`, which thread should this element be assigned to? That is, `forward_thread(i, j) => thread`.
2. Given a fragment buffer index `(i, j)` and replica `rep`, which index in the thread-local buffer should be used? That is, `forward_index(i, j, rep) => local_index`.

Below is a terminology table used throughout this article. It mostly follows the TileLang source code naming. Translating these terms is really hard.

| Term | Meaning |
| --- | --- |
| `forward_thread` | Maps logical indices plus replica to the physical thread that owns the element. |
| `forward_index` | Maps logical indices plus replica to the local index inside that thread. |
| `replicate` / `rep` | Represents multiple physical copies of the same logical element, usually so multiple threads can read it. |
| `loop_layout` | The `Fragment` layout bound to a `T.Parallel` loop. |
| `buffer_layout` | The `Fragment` layout bound to a fragment buffer. |
| `ThreadExtent()` | Number of logical threads used by the layout expression. |
| `thread_bounds` | Actual thread range of the CUDA block, such as `threadIdx.x in [0, 64)`. |

In short, because we code at a higher level, the compiler has more room for global optimization, but the complexity also increases. We need to understand the internal principles well in order to fully exploit the compiler.

# A Broad Reading of the Layout Inference Framework

Let's first understand the layout inference flow at a high level.

In TileLang, layout inference is implemented by the `tilelang.transform.LayoutInference` pass, which is bound through FFI to `tvm::tl::LayoutInference`. The purpose of this pass is to perform thread binding for operations in the IR:

```cpp
tvm::transform::Pass LayoutInference() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    //...
    f = LayoutInferencer::Substitute(std::move(f), skip_thread_partition);
    //...
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

class LayoutInferencer : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f, bool skip_thread_partition = false) {
    // ...
    BufferUseDefCollector collector(skip_thread_partition);
    collector.Collect(f);
    // Run layout inference. This is the entry point of layout inference.
    auto result = collector.Run();
    LayoutInferencer substituter(result, &analyzer);
    // Substitute according to the layout inference result.
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }
```

From the two requirements of "correctness" and "performance", we can infer that layout inference is a constraint optimization process: we need to improve the performance of the compiled result as much as possible while satisfying thread-local constraints. `collector.Run` solves this problem through BFS. The core steps are listed below. We will briefly go over them first, then discuss them in detail later.

1. Enqueue all operations that need inference. These operations include explicit `ParallelOp`s and implicit ones such as `CopyOp`s. Due to space limits, this article mainly discusses `T.Parallel`.
2. Step 0: if there are "floating" fragment buffers, replicate them to all threads.
3. Step 1: perform strict layout inference for each operation and update the layout map.
4. Step 2: perform common layout inference for operations in the queue, using BFS until convergence, and update the layout map.
5. Step 3: perform free layout inference for operations in the queue, using BFS until convergence, and update the layout map.

```cpp
LayoutInferenceResult Run() {
    // ...
    Map<Buffer, Layout> layout_map = annotated_layout_map_;
    Map<Buffer, Layout> strict_layout_map;
    int num_infer = infer_list_.size();
    // Prepare BFS queue for iterative inference
    std::deque<int> q;
    std::vector<bool> in_queue(num_infer, true);
    for (int i = 0; i < num_infer; i++) {
      // Check that each thread_var_vec_ entry is defined
      if (!thread_var_vec_[i].defined() && skip_thread_partition_) {
        thread_var_vec_[i] = thread_var_;
      }
      q.push_back(i);
    }

    // step 0: set fully replicated layout for floating fragment buffers
    // Floating buffers are accessed outside TileOps (e.g., in if conditions),
    // so they must be replicated across all threads.
    for (const auto &[buffer, thread_bounds] : floating_fragment_buffers_) {
      if (layout_map.count(buffer))
        continue;
      auto frag =
          Fragment::FullyReplicated(buffer->shape, thread_bounds->extent)
              ->BindThreadRange(thread_bounds);
      layout_map.Set(buffer, frag);
    }

    // step 1: infer strict layout
    for (int i = 0; i < num_infer; i++) {
      RunInferStep(i, InferLevel::kStrict, false, layout_map, strict_layout_map,
                   q, in_queue);
    }
    // step 2: infer common layout with BFS
    FinishInferQueue(InferLevel::kCommon, layout_map, strict_layout_map, q,
                     in_queue);
    // step 3: relax constraints to free and re-run
    InferInFreeMode(layout_map, strict_layout_map);
    // ...
  }
```

**Floating fragment buffer**: this refers to cases like the following. `A` is used under an `if` statement and will not be analyzed for layout, so `A` is directly replicated to all threads to guarantee correctness.

```py
A = T.alloc_fragment(...)
for i, j in T.Parallel:
  if A[i, j] > 0:
    B[i, j ] = 0
  else:
    B[i, j] = 1
```

The remaining three layout inference passes all eventually call the same method, `RunInferStep`, to converge.

- Run `RunInferStep` on the first operation currently in the queue.
- Call this operation's `InferLayout` method, which returns layout updates for each buffer. For example, `Parallel` and `Copy` each implement their own method, which we will discuss later.
- If the buffer layout does not exist yet, update it.
- If the buffer layout already exists, check whether it is compatible and try to merge it.
- Perform alias propagation. For example, if `buffer_A` is updated this time, and `op_X` and `op_Y` use `buffer_A`, enqueue those two operations for the next iteration.

```cpp
 void RunInferStep(int cur_infer_id, InferLevel level, bool update_queue,
                    LayoutMap &layout_map, const LayoutMap &strict_layout_map,
                    std::deque<int> &q, std::vector<bool> &in_queue) {
    // ...
    auto num_infer = infer_list_.size();
    auto &next = infer_list_[cur_infer_id];

    // ...
    // Run InferLayout
    auto updates = next->InferLayout(LayoutInferArgs{target_,
                                                     thread_bounds,
                                                     layout_map,
                                                     cur_analyzer,
                                                     buffer_oob,
                                                     {},
                                                     bind_var_to_expr_,
                                                     false},
                                     level);

    // Process the returned updates
    for (const auto &[buffer, layout] : updates) {
      // ...
      // Helper: propagate inferred layout to alias buffers (same data Var)
      auto propagate_alias = [&](const Buffer &src_buffer,
                                 const Layout &src_layout) {
        // ...
        const auto &siblings = buffer_data_to_buffers_[src_buffer->data];
        for (const auto &sib : siblings) {
          // some checks
        }
      };

      if (layout_map.count(buffer)) {
        // If new layout contains the old one, update map
        // ...
        // If already in map, check if they are structurally equal
        if (!layout->IsEqual(layout_map[buffer].get())) {
          //...
        }
        // Ensure aliases are consistent too
        propagate_alias(buffer, layout);
      } else {
        // Otherwise, update map
        // ...
        layout_map.Set(buffer, layout);
        // Propagate to alias buffers (may enqueue their users)
        propagate_alias(buffer, layout);
        }

        // Push back into BFS queue
        for (int idx : use_list_[buffer]) {
          EnqueueWithPriority(idx, q, in_queue, cur_infer_id, layout_map);
        }
      }
    }
```

## Strict Layout Inference

In the example below, `B` only has a constant index and is independent of the loop variables, so it can be treated as a free variable. From the loop body's perspective, `B` is an external constant, so we can simply copy one replica to every thread.

```py
for i, j in T.Parallel(...):
    A[i, j] = B[0]
```

Note that the following is illegal and will cause data racing.

```py
for i, j in T.Parallel(...):
    B[0] = A[i, j]
```

The following is the strict layout portion of `ParallelOpNode::InferLayout`. The key line is `const PrimExpr &forward_thread = rep;`. It means `forward_thread(i, j, rep) = rep`, making the bound thread depend only on replication and not on the index. In other words, the value is fully replicated.

```cpp
LayoutMap ParallelOpNode::InferLayout(const LayoutInferArgs &T,
                                      InferLevel level) const {
  // ...

  if (level == InferLevel::kStrict) {
    LayoutMap results;
    // Deduce buffers that should be complicated replicated.
    // For example:
    // for i in T.Parallel(m):
    //   fragment[0] = x[i]
    // then fragment[0] must be replicated on all threads.
    // ...

      // Only set layout if all indices are zero
      if (all_indices_zero) {
        Array<IterVar> forward_vars;
        for (const auto &s : buffer->shape) {
          forward_vars.push_back(
              IterVar(Range(0, s), Var(), IterVarType::kDataPar));
        }
        Var rep;
        auto rep_iter =
            IterVar({0, T.thread_bounds->extent}, rep, IterVarType::kDataPar);

        const PrimExpr &forward_thread = rep;
        auto frag = Fragment(forward_vars, /*forward_index=*/{}, forward_thread,
                             rep_iter)
                        ->BindThreadRange(T.thread_bounds);
        results.Set(buffer, frag);
      }
    }
    return results;
  }
  // ...
}
```

## Common/Free Layout Inference

Next, common layout inference performs layout propagation. For example, in the following code, if a previous BFS iteration has produced the layout of `A[i,j]`, then because it is used in the second loop, the second loop's layout should be derived from `A[i,j]`. As the source-code comment below also says, propagation is needed when non-constant indices exist.

```py
for i, j in T.Parallel(...):
  A[i, j] = B[i, j]

# need propagation
for i, j in T.Parallel(...):
  C[i, j] = A[i, j]

# don't need propagation, because A[0,0] is fully replicated
for i, j in T.Parallel(...):
  D[i,j] = A[0, 0]
```

```cpp
  // ...
  // Collect fragment buffers with const index and all fragment_buffers
  std::vector<Buffer> const_index_fragment_buffer, fragment_buffers;
  for (const auto &buffer : access_order_) {
    const auto &access = GetAccessInfo(buffer);
    if (!IsFragmentBuffer(buffer))
      continue;
    fragment_buffers.push_back(buffer);

    bool is_const_index = true;
    for (const auto &index : access.indices) {
      if (!index.as<IntImmNode>()) {
        is_const_index = false;
        break;
      }
    }
    if (is_const_index) {
      const_index_fragment_buffer.push_back(buffer);
    }
  }

  // Determine if common layout propagation should be applied.
  // If there are fragment buffers with non-constant indices, we need to
  // propagate the common layout pattern to ensure consistency across all
  // fragments. Example cases:
  //   - Need propagation: frag_a[0] = T.min(frag_a[0], frag_b[i])
  //     (const index frag_a interacts with non-const index frag_b)
  //   - No propagation needed: shared_a[i] = frag_a[0]
  //     (const index frag_a with non-fragment buffer)

  bool allow_layout_propgate =
      const_index_fragment_buffer.empty() ||
      (fragment_buffers.size() > const_index_fragment_buffer.size());
```

Because a loop may have multiple source buffers that can be used for inference, we need to choose one. The priority is listed below.

- Highest priority: `T.Fragment`/`T.Layout` annotations
- Next: the written buffer
- Next: the most-read buffer. The more reads there are, the more accurate inference is.
- Next again: a fully replicated write buffer. I do not fully understand this one either. Why would there be a replicated write buffer?
- Last: free inference

```cpp
  // ...

  // Step 1: try to infer loop's partition from a source fragment
  Buffer source_buffer, read_source_buffer;
  bool source_buffer_is_write = false;
  Buffer replicated_write_buffer; // Backup: fully replicated write buffer
  for (const auto &buffer : access_order_) {
    const auto &access = GetAccessInfo(buffer);
    if (T.layout_map.count(buffer)) {
      // skip reducers with rep=ALL
      if (auto info = reducer_info_map_.Get(buffer->data);
          info && info.value()->rep == ReducerRepType::ALL)
        continue;

      bool is_fully_replicated =
          IsBufferCompletelyReplicated(buffer, T.layout_map);

      if (access.is_write) {
        source_buffer = buffer;
        source_buffer_is_write = true;
      } else {
        // Keep the buffer with largest number of indices
        // (which means the inference based on that buffer is more accurate)
        // as read_source_buffer to get more accurate layout
        // if the buffer is completed replicated, we don't need to infer the
        // layout from this buffer.
        if ((!read_source_buffer.defined() ||
             access.indices.size() >
                 GetAccessInfo(read_source_buffer).indices.size())) {
          read_source_buffer = buffer;
        }
        // If the buffer is not replicated and shape is equal to the
        // source_buffer, use it as source_buffer because the layout inference
        // is more accurate
        auto frag = T.layout_map[buffer].as<Fragment>();
        if (frag.has_value() && is_one(frag.value()->ReplicateExtent()) &&
            !source_buffer.defined()) {
          source_buffer = buffer;
          source_buffer_is_write = false;
        }
      }
    }
  }
```

After selecting a source buffer, the core inference logic is the `ComputeLoopLayoutFromBuffer` function. If no source buffer can be found, free inference is performed through `ComputePlanCandidate`. These two functions are relatively important, so we will expand on them separately later. In free inference mode, the code computes one layout through each of these two functions, then chooses the better one with `ChooseBestCandidate`.

```cpp
  // ...
  // Try to infer loop layout from buffers in order of preference only if we
  // don't already have a layout (e.g., from annotations):
  // 1. Annotated loop layout
  // 2. Non-replicated write buffer (most reliable)
  // 3. Non-replicated read buffer
  // 4. Fully replicated write buffer (backup, may cause issues)
  // 5. Free inference mode (no source buffer)
  if (!loop_layout_.defined() && annotated_layout_unbound_.defined()) {
    // use annotation
    // ...
  } else if (!loop_layout_.defined() && source_buffer.defined() &&
             (allow_layout_propgate || source_buffer_is_write)) {
    loop_layout_ = ComputeLoopLayoutFromBuffer(source_buffer, T);
  } else if (!loop_layout_.defined() && level == InferLevel::kFree) {
    // For free layout inference
    // In free inference, try two mechanisms and prefer the one that
    // minimizes replication while remaining compatible:
    // 1) compute_loop_layout_from_buffer (always correct but may
    // over-replicate) 2) PlanLoopPartition (often smaller replication)
    Fragment candidate_from_buffer;
    Fragment candidate_from_plan;
    bool selected_plan_candidate = false;

    if (read_source_buffer.defined() && allow_layout_propgate) {
      candidate_from_buffer =
          ComputeLoopLayoutFromBuffer(read_source_buffer, T);
    }

    // try to infer loop layout with two mechanisms and choose the best one
    {
      candidate_from_plan = ComputePlanCandidate(T);
    }

    // Choose the best candidate:
    if (candidate_from_buffer.defined() && candidate_from_plan.defined()) {
      loop_layout_ =
          ChooseBestCandidate(candidate_from_buffer, candidate_from_plan, T);
    } else if (candidate_from_plan.defined()) {
      loop_layout_ = candidate_from_plan;
    } else if (candidate_from_buffer.defined()) {
      loop_layout_ = candidate_from_buffer;
    }
    loop_layout_requires_padding_guard_ =
        selected_plan_candidate && indice_map_.empty();
  } else if (!loop_layout_.defined()) {
    return {};
  }
```

The next step is some finishing work. First, `DetectInjective` checks whether the inferred loop layout is a valid injective mapping, ensuring that two different logical elements do not map to the same physical address and cause a collision.

```cpp
  // ...
  // Non-fragment SIMT loops may deliberately over-cover a ragged iteration
  // space; PartitionLoop emits guards for the padded points. Fragment/reducer
  // loops stay strict because padding would change per-thread ownership.
  auto injective_res =
      loop_layout_->DetectInjective(loop_layout_requires_padding_guard_);
  if (!injective_res->errors.empty()) {
    // ...
    throw LoopLayoutInjectiveException(oss.str());
  }
```

Then predicates are added to ensure that threads not mapped by the layout become no-ops.

```cpp
  // ...
  PrimExpr loop_thread_extent = loop_layout_->ThreadExtent();

  auto block_size = T.thread_bounds->extent;
  if (loop_layout_.defined()) {
    if (loop_layout_->ThreadRange().defined()) {
      auto thread_range = loop_layout_->ThreadRange();
      block_size = thread_range->extent;
      AddPredicate(GE(InputPlaceholder(0), thread_range->min));
      AddPredicate(
          LT(InputPlaceholder(0), thread_range->min + thread_range->extent));
    }
  }

  if (!analyzer_.CanProveEqual(loop_thread_extent, block_size)) {
    AddPredicate(
        LT(InputPlaceholder(0), loop_thread_extent + T.thread_bounds->min));
  }
```

We also need to ensure that the inferred layout is compatible with all already-inferred buffer layouts. If it is incompatible, the inference process is wrong and compilation should fail directly.

```cpp
  // Step 2: Check that the loop's partition can correctly align with all source
  // fragment, and infer layout only when it's not yet layout-ed.
  ValidateCandidateAgainstFragments(loop_layout_, T, /*throw_on_error=*/true,
                                    /*check_forward_index=*/false,
                                    source_buffer);
```

Next, check whether there is replication while writing shared or global memory, which would cause data racing. If replication only writes to its own fragment buffer, it is legal.

```cpp
  // Step 3: Build replication guards
  BuildReplicationGuardsIfNeeded(
      T, store_shared_global_buffers_, store_fragment_buffers_,
      has_cross_thread_access_, const_index_fragment_buffer);
```

And we are done. Return the layouts to BFS and move on to the next iteration.

```cpp
  // Step 4: Collect buffer fragments
  LayoutMap results;
  for (const auto &buffer : access_order_) {
    if (!T.layout_map.count(buffer)) {
      auto dst_layout =
          CompleteBufferFragment(buffer)->BindThreadRange(T.thread_bounds);
      results.Set(buffer, dst_layout);
    }
  }
  loop_layout_inferred_ = true;
  return results;
```

# A Close Reading of the Inference Algorithm

In this section, we will read several key functions in detail and investigate the following questions:

- What is the structure of `Layout`, and how is it modeled?
- How do we infer a loop layout from a known buffer layout?
- How is free inference performed?
- How do we check compatibility between buffer layouts?

## Layout Model

In TileLang, a layout is an expression `forward_index`. Its inputs are loop variables `forward_var`, and its return value is the corresponding mapped index.

For a plain `Layout`, `forward_index` maps logical input coordinates to coordinates in the transformed output shape. It is not necessarily a byte offset or a final flattened memory address. The actual memory offset is computed later from the output coordinate and buffer strides. For example:

```text
Layout((M, N) -> (M / 16, N, 16),
       (i, j) -> (i / 16, j, i % 16))
```

This represents the mapping `A[i, j] -> B[i / 16, j, i % 16]`. If the output buffer is compact row-major, the final address may later become the familiar Triton-style expression `(i / 16) * stride0 + j * stride1 + (i % 16) * stride2`.

The source code for `Layout` is:

```cpp
class Layout : public ObjectRef {
public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index);
  TVM_DLL Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

```

`Fragment` is a special layout. As mentioned earlier, it needs two mappings: `forward_index` and `forward_thread`, mapping to the thread-local index and the corresponding thread, respectively. In this case, the mapping inputs are the loop variables `forward_var` and the replication parameter `thread_replicate`.

```cpp
class Fragment : public Layout {
public:
  TVM_DLL Fragment(Array<IterVar> forward_var, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, IterVar thread_replicate);

  TVM_DLL Fragment(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
                   PrimExpr forward_thread, PrimExpr replicate_size,
                   Optional<Var> replicate_var);
}
```

## `ComputeLoopLayoutFromBuffer`

`ComputeLoopLayoutFromBuffer` takes a `Buffer` and returns a `Fragment` layout. It roughly consists of the following steps:

- `IsCommonAccessIndice` fast path
- Expression substitution
- Expression validity check, namely whether inner variables exist
- Layout construction
- Thread-range binding

First, `IsCommonAccessIndice` is a fast path. If this buffer's access-index structure is the same as the loop variables, then the buffer's layout can be used directly. Access index means the index with which this buffer is used inside the loop.

```py
for h, i, j in T.Parallel(2, 16, 32):
  x = Q[h, i, j]
```

Here, the loop variables are `(h, i, j)`, and `Q`'s access index is also `(h, i, j)`. They match exactly, so the layout can be used directly.

```cpp
Fragment
ParallelOpNode::ComputeLoopLayoutFromBuffer(const Buffer &buffer,
                                            const LayoutInferArgs &T) const {
  Fragment src_layout = T.layout_map[buffer].as<Fragment>().value();
  Fragment result;

  if (IsCommonAccessIndice(buffer)) {
    result = src_layout;
  } else {
    // ...
```

If the fast path cannot be used, we obtain the source buffer's index information in the current loop and feed it into the source layout's `forward_thread`, producing a thread-mapping expression called `loop_var_to_thread`. For example, suppose our source layout is:

- `forward_thread(i, j, rep) = i * 16 + j`
- `forward_index(i, j, rep) = 0`, meaning each thread stores only one element

and the current loop is:

```
for group, row_in_group in T.Parallel(2, 2):
    row = group * 2 + row_in_group
    x = Q[row, 0]
```

Then substituting `i = group * 2 + row_in_group` and `j = 0` into the original expression gives the layout of this loop:

- `forward_thread(group, row_in_group, rep) = (group * 2 + row_in_group) * 16 + 0`
- `forward_index(group, row_in_group, rep) = 0`

After simplifying this expression, we need to check its validity. `PostOrderVisit` traverses the variables in the expression and ensures that all variables come from loop variables rather than inner variables `inner_vars_`. The reason is that during lowering, we need to expand the loop and compute which thread receives each piece of work. If the expression depends on inner variables, this computation is no longer possible. For example, in the following case, the first loop annotates `Q` with `forward_thread(i, j) = i * M + j`, but when applying it to the second loop, `j` is no longer a loop variable, which makes the layout illegal and throws a `LayoutConflictException`.

```py
Q = T.alloc_fragment((M, N))
for i, j in T.Parallel(M, N):
    Q[i, j] = A[i, j]

for i in T.Parallel(M):
  for j in T.Serial(N):
    O[i, j] = Q[i, j]
```

The source code is shown below:

```cpp
// ...
else {
    Var rep("_rep");
    auto rep_iter =
        IterVar({0, src_layout->ReplicateExtent()}, rep, IterVarType::kDataPar);
    PrimExpr loop_var_to_thread =
        src_layout->ForwardThread(GetAccessInfo(buffer).indices, rep);
    loop_var_to_thread = analyzer_.Simplify(loop_var_to_thread);
    PostOrderVisit(loop_var_to_thread, [&](const ObjectRef &objref) {
      if (auto opt_var = objref.as<Var>();
          opt_var && inner_vars_.count(*opt_var)) {
        std::ostringstream oss;
        oss << "loop_var_to_thread = " << loop_var_to_thread
            << "contains inner var" << *opt_var;
        throw LayoutConflictException(oss.str());
      }
    });
  // ...
```

Finally, we construct the new layout using the loop variables `loop_vars_`, the substituted expression `loop_var_to_thread`, and the original layout's replication parameter `rep_iter`. Then we bind the thread range, which defines which threads this layout will use.

```cpp
//..
    try {
      result = Fragment(loop_vars_, {}, loop_var_to_thread, rep_iter)
                   ->BindThreadRange(T.thread_bounds);
    } catch (const Error &err) {
      // some error
    }
  }
  return result;
}
```

## `ComputePlanCandidate`

`ComputePlanCandidate` freely infers a loop layout when there is no known fragment layout that can be propagated. If `ComputeLoopLayoutFromBuffer` means "derive the loop layout from an existing buffer", then `ComputePlanCandidate` means "derive a good loop layout from the loop shape itself".

The function mainly does three things:

- Estimate the vectorization width.
- Compute the total number of loop iterations.
- Call `PlanLoopPartition` to construct the layout.

### Estimating Vectorization Width

We can roughly understand vectorization as binding multiple loop variables into a group and assigning work to different threads by group. Of course, this only makes sense when memory addresses are contiguous enough to support coalesced access.

First, the code rewrites the loop if some buffers have already been remapped, because vectorization must be computed on the final access pattern rather than the original source pattern. For example:

```py
for i in T.Parallel(4):
  frag[i] = shared[i]
```

But `shared[i]` may actually be non-contiguous under its layout. For example, if `forward_index(i) = i * 16`, then the remapped indices for `i = [1, 2, 3, 4]` are `[0, 16, 32, 48]`, which are not contiguous in physical space and cannot be vectorized. Due to space limits, we skip the details of `GetVectorizeSize`.

The source code for this part:

```cpp
Fragment ParallelOpNode::ComputePlanCandidate(const LayoutInferArgs &T) const {
  // Vectorize Size must be aware of the buffer_remap
  // As the pass will do post processing to the layout
  auto maybe_remapped_root_ =
      IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
  int vector_size =
      GetVectorizeSize(maybe_remapped_root_, T.analyzer, T.layout_map);
```

### Computing Total Loop Iterations

The flattened loop size is computed from the extents of the parallel loops. For example, the loop size of `for i, j in T.Parallel(4, 16):` is 64.

```cpp
  PrimExpr loop_total_size = 1;
  for (Stmt l = root_; l.as<For>().has_value(); l = l.as<For>().value()->body)
    loop_total_size = loop_total_size * l.as<For>().value()->extent;
```

<!-- If `threads=64` and `vector_size=1`, the natural plan is:

```text
flat(i, j) = i * 16 + j
thread     = flat(i, j)
local      = 0
```

So every logical element gets one thread. -->

### Calling `PlanLoopPartition` to Construct the Layout

The actual layout construction happens in `PlanLoopPartition`. The source code is:

```cpp
Fragment Partition(const For &op, int num_thread, int vectorize_size) {
    this->VisitStmt(op);
    DataType dtype = DataType::Int(32);
    if (!loop_vars_.empty()) {
      dtype = loop_vars_.back()->var.dtype();
    }
    PrimExpr flattened = make_const(dtype, 0);
    PrimExpr vector_extent = make_const(dtype, vectorize_size);
    PrimExpr thread_extent_const = make_const(dtype, num_thread);
    for (size_t i = 0; i < loop_vars_.size(); i++) {
      PrimExpr extent = loop_vars_[i]->dom->extent;
      flattened = flattened * extent + loop_vars_[i]->var;
    }
    PrimExpr access_idx = FloorDiv(flattened, vector_extent);
    PrimExpr thd = FloorMod(access_idx, thread_extent_const);
    PrimExpr idx = FloorDiv(access_idx, thread_extent_const) * vector_extent +
                   FloorMod(flattened, vector_extent);

    auto fragment = Fragment(loop_vars_, {idx}, {thd}, {});
    if (has_fragment_) {
      // for fragment buffer, we don't need to replicate the loop layout
      auto thread_extent = *as_const_int(fragment->ThreadExtent());
      auto num_thread_fragment = num_thread / thread_extent;
      fragment = fragment->Replicate(num_thread_fragment);
    }
    return fragment;
  }
```

We can see that this is manually constructing an expression. `flattened` is the flattened loop variable, such as `i * 16 + j`. Since we group by `vector_extent`, the grouped index, `access_index`, is `flattened / vector_extent`. This index is evenly assigned across `num_thread` threads, giving `forward_thread = access_index % num_thread`. `forward_index` has two parts: `(access_index / num_thread) * vector_extent` gives the corresponding vector group, and `flattened % vector_size` gives the relative index inside the group.

For `T.Parallel(4, 16)`, `threads=64`, and `vector_size=4`:

```text
flattened(i, j) = i * 16 + j
access_index(i, j) = (i * 16 + j) / 4
thread         = ((i * 16 + j) / 4) % 64 
               = (i * 16 + j) / 4 (because it is always less than 64)
local_idx      = (i * 16 + j) / 64 +  (i * 16 + j) % 4 
               = (i * 16 + j) % 4 (because the first term is always 0)
```

### Special Checks for Fragment Buffers

Fragment loops are stricter than ordinary copy/shared-memory loops. For fragment accesses, TileLang tries to avoid padded logical points because padding creates fake fragment ownership. For non-fragment SIMT loops, padding can be skipped with guards during lowering.

```cpp
  bool has_fragment_access = !indice_map_.empty();
  if (has_fragment_access) {
    while (
        !analyzer_.CanProve(floormod(loop_total_size, T.thread_bounds->extent *
                                                          vector_size) == 0) &&
        vector_size > 1) {
      vector_size /= 2;
    }
  } else if (!root_->annotations.count(attr::kCoalescedWidth)) {
    vector_size = SelectMinPaddingVectorSize(
        vector_size, loop_total_size, T.thread_bounds->extent, &analyzer_);
  }
```

This `floormod` divisibility check ensures `loop_total_size % (threads * vector_size) == 0`. We assign multiple groups of size `vector_size` to threads, and we want every group assigned to every thread to be complete, without padded elements. This is because fragment buffers represent real thread-local storage. If the planner covers more logical elements than the loop actually has, those padded elements would still look like fragment elements and could create invalid ownership. For non-fragment loops, the compiler can usually emit a predicate and skip the padded elements. For fragments, padding can make correctness guarantees more complicated. For example:

```py
with T.Kernel(threads=64)
for i in T.Parallel(256):
    fragment[i] = ...
```

Assume the initial `vector_size=8`. The planner first checks `256 % (64 * 8) = 256 % 512 = 256`. In other words, one complete SIMT/vector tile would cover 512 logical elements, but the loop only has 256. That would require 256 padded logical fragment elements, so TileLang halves the vector size and gets `256 % (64 * 4) = 256 % 256 = 0`, meaning every group is complete and no padding is needed. Note that here we assume all `num_threads` are used. Sometimes, reducing `num_threads` might achieve better vectorization, but the compiler does not implement that. My guess is that if we assume all parameters are powers of two, non-divisible cases are relatively rare and the extra complexity is not very worthwhile.

## `ValidateCandidateAgainstFragments`

`ValidateCandidateAgainstFragments` checks whether the selected loop layout is compatible with every already-known fragment layout used by this loop. A loop can touch multiple fragments. Even if the candidate layout looks valid by itself, it may disagree with a buffer's existing owner-thread mapping.

The function iterates over all fragment buffers accessed by the loop:

```cpp
bool ParallelOpNode::ValidateCandidateAgainstFragments(
    const Fragment &candidate, const LayoutInferArgs &T, bool throw_on_error,
    bool check_forward_index, const Buffer &source_buffer) const {
  auto vars =
      loop_vars_.Map([](const IterVar &iv) { return PrimExpr(iv->var); });
  for (const auto &buffer : access_order_) {
    const auto &access = GetAccessInfo(buffer);
    if (!T.layout_map.count(buffer))
      continue;
    // ...
    auto fragment = T.layout_map[buffer].as<Fragment>().value();
```

For reads, the requirement is "containment": the loop layout must be able to read elements from the source fragment layout. This is intuitive. If the loop iteration maps to a thread that does not own the element it needs to read, then the read is invalid.

```cpp
if (access.is_read &&
    !ProveFragmentContains(candidate, fragment, vars, access.indices,
                           analyzer_, check_forward_index)) {
  success = false;
}
```

Let's continue with the implementation of `ProveFragmentContains`:

```cpp
bool ProveFragmentContains(Fragment small_frag, Fragment large_frag,
                           Array<PrimExpr> small_frag_indices,
                           Array<PrimExpr> large_frag_indices,
                           Analyzer &analyzer, bool check_forward_index) {
  bool large_physical_is_fully_replicated = large_frag->IsCompletedReplicated();
  if (large_physical_is_fully_replicated) {
    return true;
  }
  // ...
```

`small_frag` represents the target fragment's layout, while `large_frag` represents the existing fragment's layout. The function first handles a special case: if the large fragment is fully replicated, every thread has a valid copy, so containment is trivially true.

When `check_forward_index` is true, it also checks whether the two layouts use the same physical local index.

```cpp
if (check_forward_index) {
  auto small_physical = small_frag->Forward(small_frag_indices);
  auto large_physical = large_frag->Forward(large_frag_indices);
  if (small_physical.size() != large_physical.size()) {
    return false;
  }
  for (size_t i = 0; i < small_physical.size(); i++) {
    auto diff = analyzer.Simplify(small_physical[i] - large_physical[i]);
    if (!analyzer.CanProve(diff == 0)) {
      return false;
    }
  }
}
```

Next, the function introduces a symbolic replicate variable for the small fragment:

```cpp
Var rep_small("__checking_frag_contains_rep");
analyzer.Bind(rep_small,
              Range(IntImm(small_frag->ReplicateExtent()->dtype, 0),
                    small_frag->ReplicateExtent()),
              true);
auto thread = small_frag->ForwardThread(small_frag_indices, rep_small);
```

This says: for any valid replica of the candidate layout, compute the thread that executes this access.

Next, the function asks the large fragment: "if I use the large fragment's physical local index together with the small fragment's thread, what logical large-fragment element would this physical coordinate correspond to?"

```cpp
auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
large_frag_physical_and_thread.push_back(thread);
auto inv_large_frag = large_frag->Inverse();
auto inv_large_frag_logical_and_rep =
    inv_large_frag->Forward(large_frag_physical_and_thread);
```

`large_frag->Forward(large_frag_indices)` gives the local index. Combined with the inferred thread, it forms the physical coordinate `(large local index, candidate thread)`. The inverse of the large fragment maps this physical coordinate back to `(large logical indices, large replica)`. The last element is the recovered replica.

```cpp
auto inv_large_frag_rep =
    inv_large_frag_logical_and_rep[inv_large_frag_logical_and_rep.size() - 1];
```

Finally, the function recomputes the large fragment's owning thread using this recovered replica:

```cpp
auto check_thread =
    large_frag->ForwardThread(large_frag_indices, inv_large_frag_rep);

auto diff = analyzer.Simplify(thread - check_thread);
return analyzer.CanProve(diff == 0);
```

If the recomputed large-fragment thread equals the original candidate thread, then the candidate thread is a valid owner for that large-fragment access. Otherwise, the candidate loop is trying to read or write an element from the wrong thread.

Here is a successful read example:

```text
candidate:
  thread(i, j) = i * 16 + j

fragment:
  thread(i, j) = i * 16 + j
  local(i, j)  = 0

access:
  frag[i, j]
```

`ProveFragmentContains` computes:

```text
small thread = i * 16 + j
large local  = 0
inverse large(0, i * 16 + j) -> (i, j, rep=0)
check_thread = i * 16 + j
diff         = 0
```

So the access is valid. Now consider a failing example:

```text
candidate:
  thread(i, j) = i

fragment:
  thread(i, j) = i * 16 + j
  local(i, j)  = 0

access:
  frag[i, j]
```

Now:

```text
small thread = i
large local  = 0
inverse large(0, i) -> some element owned by thread i
check_thread = i * 16 + j
diff         = i - (i * 16 + j)
```

The analyzer cannot prove this is zero. Therefore the candidate layout cannot safely read `frag[i, j]`.

For writes, the check is stricter:

```cpp
if (access.is_write &&
    (!ProveFragmentContains(fragment, candidate, access.indices, vars,
                            analyzer_, check_forward_index) ||
     !ProveFragmentContains(candidate, fragment, vars, access.indices,
                            analyzer_, check_forward_index))) {
  success = false;
}
```

Here, the function uses containment checks in both directions, effectively requiring exact owner-thread compatibility, meaning the ownership of all accessed elements must be equal. A read only needs one-way containment, because the loop only needs to prove that the values it reads are available on its executing threads. A write needs equality, because the write updates the buffer layout's physical state.

For example, in `for i, j in T.Parallel(2, 2):`, suppose the buffer layout is:

```text
forward_thread(i, j) = i * 2 + j
forward_index(i, j) = 0
```

and the candidate layout is:

```text
forward_thread(i, j) = i
forward_index(i, j) = j
```

The candidate layout may cover the same logical shape. For example, when thread 0 and thread 1 want to read, they can always read something. But because each element is stored in a different physical thread/local slot, read/write inconsistency can occur, so it should be rejected.

## Common Failure Cases

To summarize, we have mentioned three different layout inference failure modes. The first failure mode is a non-injective layout: for two different elements `x` and `y`, we have `forward_thread(x) == forward_thread(y)` and `forward_index(x) == forward_index(y)`.

The second failure mode is incompatible ownership. The candidate itself may be injective, but it does not access the fragment from the thread that owns the value. `ValidateCandidateAgainstFragments` and `ProveFragmentContains` are designed to catch this case.

The third failure mode is sparse-but-injective layout:

```text
(group, row_in_group) -> thread = group * 32 + row_in_group * 16
```

This maps four logical elements to threads `{0, 16, 32, 48}`. There is no collision, so it is injective. However, lowering needs to invert the thread expression. The inverse requires extra divisibility guards such as `thread % 16 == 0`. Current lowering goes through TVM's affine inverse machinery, and some sparse single-component expressions cannot be represented there. This is the failure mode behind the `abs(source->scale) == 1` assertion mentioned at the beginning of this article.

# An Example

Let's use an example to connect all the pieces:

```py
@tilelang.jit
def plan_then_buffer_layout():
    @T.prim_func
    def main(
        A: T.Tensor((4, 16), T.float32),
        B: T.Tensor((4,), T.float32),
    ):
        with T.Kernel(1, threads=64):
            fragment = T.alloc_fragment((4, 16), T.float32)

            for row, col in T.Parallel(4, 16):
                fragment[row, col] = A[row, col]

            for group, row_in_group in T.Parallel(2, 2):
                row = group * 2 + row_in_group
                B[row] = fragment[row, 0]

    return main
```

The full layout inference timeline is:

```text
Initial:
  layout_map = {}

Strict:
  loop 1 -> no update, because fragment[row, col] is not a constant-index access
  loop 2 -> cannot derive a useful layout from fragment yet

Common:
  loop 1 -> no source fragment layout to propagate
  loop 2 -> still waits for fragment's layout

Free:
  loop 1 -> ComputePlanCandidate establishes fragment's layout
  use_list_[fragment] re-enqueues loop 2

Revisit:
  loop 2 -> ComputeLoopLayoutFromBuffer propagates fragment's layout through fragment[row, 0]
```

The first loop writes the whole fragment:

```py
for row, col in T.Parallel(4, 16):
    fragment[row, col] = A[row, col]
```

At the beginning of layout inference, `fragment` has no known layout. In strict mode, this loop cannot infer a fully replicated layout because the fragment access is not a constant index. In common mode, there is still no known source layout to propagate. In free mode, `ComputePlanCandidate` is used.

This loop has a total iteration count of `64`, exactly equal to the number of `threads`, so through the algebraic derivation above, the first loop obtains the fragment layout:

- `forward_thread(row_col) = row * 16 + col`
- `forward_index(row, col) = 0`

This update is returned to the BFS driver. The driver stores it in `layout_map`, then uses `use_list_` to re-enqueue operations that use `fragment`. That is how the second loop gets another chance to infer its layout with the newly available information.

When inferring the second loop, `fragment` already has a known layout, so `ComputeLoopLayoutFromBuffer` can run. It substitutes the access indices into the source fragment's thread expression. Substituting `row = group * 2 + row_in_group` and `col = 0` gives:

- `forward_thread(group, row_in_group) = group * 32 + row_in_group * 16`

The four logical iterations map to thread 0, thread 16, thread 32, and thread 48. The problem is that **this layout is sparse**. From a mathematical perspective, the mapping is still injective: no two logical iterations map to the same physical location. However, it is hard for the current inverse-lowering path to represent. Lowering needs to invert from the physical thread coordinate back to logical loop variables, but the derivation requires divisibility constraints:

```text
group = thread / 32
row_in_group = (thread % 32) / 16
```

The current path goes through TVM's affine inverse machinery, and this sparse single-component map can produce an `IterSplitExpr` whose source scale is not `1` or `-1`. That is where the failure comes from:

```text
TVM_FFI_ICHECK(analyzer_->CanProveEqual(abs(source->scale), 1));
```

This explains the strange behavior at the beginning of the article. A kernel may fail not because the layout is obviously non-injective, but because the inferred layout is sparse in thread space and the inverse step does not have the right guard representation.

There are several practical ways to avoid this:

- Add an explicit `T.Fragment` annotation so the source fragment layout produces a dense thread mapping for the later projected access.
- Change the loop structure so the projected-away dimension becomes part of the local index rather than part of the thread index.
- Change `BLOCK` or `threads` so the propagated layout happens to be dense enough for the current inverse machinery. This one is a bit mystical and luck-based.

Here is a concrete annotated version. We can define a layout that puts `col` on the thread axis and `row` on the local axis:

```py
def source_layout(row, col):
    return col, row

T.annotate_layout({
    fragment: T.Fragment((4, 16), forward_fn=source_layout)
})
```

Then `forward_thread(row, col) = col`. Reading `fragment[row, 0]` gives:

```text
thread = 0
local  = row
```

This is a replicated or single-thread style access rather than the sparse `{0, 16, 32, 48}` pattern. It may not be the best layout for performance, but it runs.

# Closing

TileLang layout inference is powerful because it lets us write block-level tensor code while the compiler derives thread-level ownership. The price is that fragment layouts become a global constraint: once a fragment element is assigned to a thread/local slot, every later use must respect that assignment.

The internal principles of layout inference inspire my debugging process: focus on:

1. Which loop first establishes the fragment layout?
2. Which later loop propagates that layout through a different access expression?
3. Does the propagated thread expression become non-injective, sparse, or dependent on an inner serial variable?
