---
author: Retep
pubDatetime: 2026-06-20T12:50:00.000-07:00
title: "如何正确地写Layout：TileLang Layout Inference深度解析"
featured: false
draft: true
tags:
  - TileLang
description: ""
---

Disclaimer：笔者不是TileLang核心开发者，啃源码写的文章，不一定完全正确。权作抛砖引玉，供大家参考。如有纰漏，欢迎指正，不胜感激。

# 引子

Layout inference是TileLang的精华。相比Triton和CUDA的SIMT模式，TileLang在更高的block层面进行编程，由编译器进行调度，实现了以极少的代码量达到接近SOTA的性能。这其中的一位功臣就是layout inference，即将高度抽象的block层代码映射到thread层指令的算法。

笔者在写TileLang算子的过程中，经常遇到一个很奇怪的现象。比如如下的算子：

```py
# kernel
for i in T.Parallel(BLOCK):
  A[i] = B[i, 0]
```

有时会编译报错：

```
  File "/workspace/tilelang/3rdparty/tvm/src/arith/iter_affine_map.cc", line 2568, in void tvm::arith::InverseAffineIterMapTransformer::Visit_(const tvm::arith::IterSumExpr&)
    TVM_FFI_ICHECK(analyzer_->CanProveEqual(abs(source->scale), 1));

tvm.error.InternalError: Check failed: (analyzer_->CanProveEqual(abs(source->scale), 1)) is false: 
```

然而，如果用`T.Fragment`对`B`进行layout annotation，或者更改`BLOCK`的大小，就可以通过编译。出于好奇，笔者开始阅读TileLang layout inference的源码。

# 什么是Layout Inference?

首先为大家介绍一下layout inference的概念，熟悉的同学可以跳过。在Triton中，我们编写的kernel都会取自己的`program_id`，然后根据它来推算当前thread/program覆盖的索引范围，计算数据的物理地址。比如：

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

在TileLang中，我们并不和thread直接打交道，而是使用Tensor buffer声明要做的事即可：

```py
# 类似以下形式
with T.Kernel(1, threads=32): 
  for i, j in T.Parallel(BLOCK_I, BLOCK_J):
    output[i, j] = input[j, i]
```

（太优雅了。）

在实际JIT时，TileLang会将`T.Parallel`的每个循环所做的事以最优的方式分配到threads上，达到SIMT执行。这个“分配”的算法就是layout inference。简单来说，就是一个从循环变量到thread的映射。通过回答“给定循环变量`i`和`j`，操作应该放在哪个thread”，得到映射 `forward_thread(i, j) = thread`，然后对其进行求逆得到 `forward_thread^{-1}(thread) = i, j`，即“对于当前thread，循环变量的值是什么”。这就达到了自动推理出layout algebra的目的，而在Triton中，我们必须苦逼地手搓。

看上去没那么难是不是？其实并不。

1. 从**正确性**的角度，在TileLang中，我们可以声明thread的本地buffer，即fragment buffer。这些fragment可以在多个`T.Parallel`中穿插使用，但我们需要保证它们始终是thread-local的，即需要绑定到固定的thread。
2. 从**性能**的角度，`T.Parallel`的展开并不是简单直接按循环次数分配即可，还需要结合考虑如向量化来合并访问内存，减少延迟。

对于fragment buffer，我们需要考虑一个额外变量：replication，即一个元素可能需要在多个thread上出现，导致元素索引和thread并不是双射。此外，我们在编程时总是使用block-level的fragment buffer索引，但编译到thread level时，只有一部分fragment buffer会真正被拷贝到这个thread level，所以还需要一个block-level索引到thread-level索引的映射。综上，对于fragment buffer需要回答两个问题：

1. 对于fragment buffer的索引`(i, j)`和replica `rep`，该元素应该分配到哪个thread。即 `forward_thread(i, j) => thread`。
2. 对于fragment buffer的索引`(i, j)`和replica `rep`，在使用时，应该取thread buffer中的哪个索引。即 `forward_index(i, j, rep) => local_index`。

下面是本文会用到的一张术语表，基本与TileLang源码保持一致（想翻译真是一件难事）。

| 术语 | 含义 |
| --- | --- |
| `forward_thread` | 将逻辑索引加replica映射到拥有该元素的物理thread。 |
| `forward_index` | 将逻辑索引加replica映射到该thread内部的local index。 |
| `replicate` / `rep` | 表示同一个逻辑元素的多个物理副本，通常为了被多个thread来读取。 |
| `loop_layout` | 绑定到`T.Parallel`循环上的`Fragment` layout。 |
| `buffer_layout` | 绑定到fragment buffer上的`Fragment` layout。 |
| `ThreadExtent()` | layout表达式使用的逻辑thread数量。 |
| `thread_bounds` | CUDA block中的实际thread范围，比如`threadIdx.x in [0, 64)`。 |

总之，由于我们在更高层级上编码，编译器能在全局上的优化能力就更大，复杂性也会同步增加。我们需要充分了解内部原理，才能极致地发挥编译器的能力。

# 泛读Layout Inference框架

我们先在宏观上了解一下layout inference的流程。

Layout inference在TileLang中通过`tilelang.transform.LayoutInference`这个pass实现，通过FFI绑定到`tvm::tl::LayoutInference`。可以看到这个pass的目的就是将IR中的操作进行thread binding：

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

通过“正确性”和“性能”的两个要求，可以推出inference是一个constraint optimization过程：我们需要在满足thread-local约束下，尽可能提升编译结果的性能。`collector.Run`通过BFS来解决这个问题，核心是以下几步。我们先简略带过，后面细说。

1. 将所有需要infer的operation入队。这里的operation有显式的`ParallelOp`，也有隐式的如`CopyOp`。本文限于篇幅主要讨论`T.Parallel`。
2. Step 0：如果有“floating”的fragment buffer，则将它replicate到所有thread。
3. Step 1：对每一个operation进行strict layout推导，更新layout map。
4. Step 2：对队列中的operation进行common layout推导，BFS直到收敛，更新layout map。
5. Step 3：对队列中的operation进行free layout推导，BFS直到收敛，更新layout map。

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

**floating fragment buffer**：类似如下情况。`A`在`if`语句下使用，不会被进行layout分析，所以会将`A`直接replicate到所有thread上保证正确性。

```py
A = T.alloc_fragment(...)
for i, j in T.Parallel:
  if A[i, j] > 0:
    B[i, j ] = 0
  else:
    B[i, j] = 1
```

剩下的三个layout inference pass，最终都是调用同一个方法`RunInferStep`进行收敛。

- 对当前队列中的首个operation元素执行`RunInferStep`。
- 调用这个operation的`InferLayout`方法，返回每个buffer的layout update。如`Parallel`和`Copy`就实现了各自的方法，我们之后会讲到。
- 如果该buffer的layout还不存在，则更新。
- 如果该buffer的layout已经存在，则检查是否兼容，并尝试合并。
- 进行alias propagation。比如这次更新了`buffer_A`，然后`op_X`和`op_Y`用了`buffer_A`，则将这两个operation加入队列进行下一轮迭代。

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

在下面的例子中，`B`只有常数索引，和循环变量无关，所以可以算作自由变量。从循环体的视角看，`B`就是一个外部常量，所以直接给每个thread都复制一份即可。

```py
for i, j in T.Parallel(...):
    A[i, j] = B[0]
```

注意下面这样是非法的，会导致data racing。

```py
for i, j in T.Parallel(...):
    B[0] = A[i, j]
```

以下是`ParallelOpNode::InferLayout`的strict layout部分源码。其核心就是`const PrimExpr &forward_thread = rep;`。这句的含义是`forward_thread(i, j, rep) = rep`，即使得绑定的thread只和replication有关，和index无关。换句话说，这个值被fully replicated。

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

下一步，common layout inference会进行layout propagation。比如下面的代码中，如果前一次BFS迭代已经得到了`A[i,j]`的layout，那么由于它被用在了第二个循环中，所以第二个循环的layout要根据`A[i,j]`进行推导。下面源码注释也讲到，propagation需要发生的条件是存在非常数索引。

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

由于一个循环可能存在多个可以用于推导的source buffer，我们需要选择其中一个。下面列出了考虑的优先级。

- 最优先：`T.Fragment`/`T.Layout`的标注
- 其次：写入的buffer
- 再次：读取最多的buffer。读取越多，推导越准确。
- 再再次：fully replicated write buffer。（这个我也不是很理解，为什么会有replicated write buffer？
- 最后：自由推导

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

选好source buffer以后，推导的核心是`ComputeLoopLayoutFromBuffer`函数。如果找不到source buffer，就通过`ComputePlanCandidate`进行自由推导。这两个函数相对重要，我们后面单独展开。可以看到在自由推导模式下，代码会通过这两个函数分别计算一个layout，然后用`ChooseBestCandidate`取一个更优解。

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

下一步是一些收尾工作。首先通过`DetectInjective`判断推导出的loop layout是否是合法的单射，以此保证两个不同的逻辑元素不会映射到同一个物理地址上造成collision。

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

然后加一些predicate来保证layout没有映射到的thread是no-op。

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

我们还需要保证推导出的layout和所有已经推导出的buffer layout是兼容的。如果不兼容，说明推导过程错误，直接报错。

```cpp
  // Step 2: Check that the loop's partition can correctly align with all source
  // fragment, and infer layout only when it's not yet layout-ed.
  ValidateCandidateAgainstFragments(loop_layout_, T, /*throw_on_error=*/true,
                                    /*check_forward_index=*/false,
                                    source_buffer);
```

接着检查是否存在replication并且写shared/global memory的情况，因为这会导致data racing。如果replication只写自己的fragment buffer，则是合法的。

```cpp
  // Step 3: Build replication guards
  BuildReplicationGuardsIfNeeded(
      T, store_shared_global_buffers_, store_fragment_buffers_,
      has_cross_thread_access_, const_index_fragment_buffer);
```

大功告成。将layout作为返回值发回BFS，进行下一步迭代。

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

# 精读推导算法

这个部分我们会详细阅读几个关键函数，探究几个问题：

- `Layout`是什么结构，是如何建模的？
- 如何通过已知buffer layout推导到loop layout？
- 如何进行自由推导？
- 如何检查buffer layout之间是否兼容？

## Layout Model

在TileLang中，一个layout是一个表达式`forward_index`，入参是循环变量`forward_var`，返回值是对应的映射后索引。

对于普通`Layout`来说，`forward_index`将逻辑输入坐标映射到变换后输出shape中的坐标。它不一定是byte offset，也不一定是最终flatten后的内存地址。实际内存offset会在之后根据输出坐标和buffer strides计算出来。比如：

```text
Layout((M, N) -> (M / 16, N, 16),
       (i, j) -> (i / 16, j, i % 16))
```

表示`A[i, j] -> B[i / 16, j, i % 16]`这个映射。如果输出buffer是compact row-major，那么最终地址之后可能变成（我们在Triton中喜闻乐见的）`(i / 16) * stride0 + j * stride1 + (i % 16) * stride2`


`Layout`源码如下
```cpp
class Layout : public ObjectRef {
public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index);
  TVM_DLL Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Layout, ObjectRef, LayoutNode);
};

```

`Fragment`是一个特殊的layout。如我们之前所说，它需要两个映射：`forward_index`和`forward_thread`，分别映射到thread-local索引和对应的thread。此时映射的入参是循环变量`forward_var`和replication参数`thread_replicate`。

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

`ComputeLoopLayoutFromBuffer`接受一个`Buffer`，返回一个`Fragment` layout。大致分为以下几步：

- `IsCommonAccessIndice`快速通道
- 表达式替换
- 表达式合法性检查，即是否存在内部变量
- 构造layout
- 绑定thread范围

首先，`IsCommonAccessIndice`是一个快速通道，即如果这个buffer的access index结构和loop的循环变量一致，那么就可以直接使用这个buffer的layout。Access index指这个buffer在这个循环中被使用的索引。

```py
for h, i, j in T.Parallel(2, 16, 32):
  x = Q[h, i, j]
```

此时循环变量是`(h, i, j)`，`Q`的access index也是`(h, i, j)`，完全一致，所以可以直接使用。

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

如果不能走快速通道，我们会拿到当前循环中source buffer的索引信息，传入其`forward_thread`得到thread映射表达式，即`loop_var_to_thread`。比如我们的source layout是：

- `forward_thread(i, j, rep) = i * 16 + j`
- `forward_index(i, j, rep) = 0`，即每个thread都只保存一个元素

而当前loop是：

```
for group, row_in_group in T.Parallel(2, 2):
    row = group * 2 + row_in_group
    x = Q[row, 0]
```

那么我们将`i = group * 2 + row_in_group`和`j = 0`替换入原来的表达式，就可以得到这个loop的layout：

- `forward_thread(group, row_in_group, rep) = (group * 2 + row_in_group) * 16 + 0`
- `forward_index(group, row_in_group, rep) = 0`

将这个表达式简化后，需要进行合法性检查。通过`PostOrderVisit`遍历这个表达式的变量，保证其所有变量都来源于循环变量而不是内部变量`inner_vars_`。原因是在lower的过程中，我们需要展开循环，计算出每份工作分配到的thread。依赖内部变量后就没法计算这个表达式了。比如在下面的例子中，第一个循环将`Q`标注为`forward_thread(i, j) = i * M + j`，但是应用到第二个循环中，发现`j`不再是循环变量，导致layout非法，抛出`LayoutConflictException`。

```py
Q = T.alloc_fragment((M, N))
for i, j in T.Parallel(M, N):
    Q[i, j] = A[i, j]

for i in T.Parallel(M):
  for j in T.Serial(N):
    O[i, j] = Q[i, j]
```

这段源码如下：

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

最后，我们通过循环变量`loop_vars_`、替换后的表达式`loop_var_to_thread`、原layout的replication参数`rep_iter`来构造新的layout。然后绑定thread范围，即定义这个layout会用到哪些thread。

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

当没有已知的fragment layout可以传播时，`ComputePlanCandidate`会自由推导一个loop layout。如果说`ComputeLoopLayoutFromBuffer`表示“从已有buffer推导loop layout”，那么`ComputePlanCandidate`表示“从loop shape本身推导一个好的loop layout”。

这个函数主要做三件事：

- 估计向量化宽度。
- 计算loop迭代总数。
- 调用`PlanLoopPartition`构造layout。


### 估计向量化宽度

我们可以粗浅地将向量化理解为将多个循环变量绑定成一个组，以组的单位分配给不同thread。当然，这只有在存在诸如内存地址连续，可以进行合并访问时才有意义。
首先会在某些buffer已经被重新映射的情况下重写loop，因为向量化必须基于最终access pattern计算，而不是基于原始source pattern。举个例子：

```py
for i in T.Parallel(4):
  frag[i] = shared[i]
```
但其实`shared[i]`的layout是不连续。比如其`forward_index(i) = i * 16`，那么`i = [1, 2, 3, 4]`重新映射后的索引就是`[0, 16, 32, 48]`，在物理空间上不连续，没法向量化。限于篇幅，我们跳过`GetVectorizeSize`的细节。

这部分的源码：
```cpp
Fragment ParallelOpNode::ComputePlanCandidate(const LayoutInferArgs &T) const {
  // Vectorize Size must be aware of the buffer_remap
  // As the pass will do post processing to the layout
  auto maybe_remapped_root_ =
      IfBufferRemapLoopGenerator::run(root_, T.buffer_remap, T.layout_map);
  int vector_size =
      GetVectorizeSize(maybe_remapped_root_, T.analyzer, T.layout_map);
```

### 计算loop迭代总数

通过计算parallel loops的extent来计算拍平后的loop size。例如`for i, j in T.Parallel(4, 16):`的loop size就是64。
```cpp
  PrimExpr loop_total_size = 1;
  for (Stmt l = root_; l.as<For>().has_value(); l = l.as<For>().value()->body)
    loop_total_size = loop_total_size * l.as<For>().value()->extent;
```



<!-- 如果`threads=64`且`vector_size=1`，自然的plan就是：

```text
flat(i, j) = i * 16 + j
thread     = flat(i, j)
local      = 0
```

所以每个逻辑元素都得到一个thread。 -->

### 调用`PlanLoopPartition`构造layout
实际layout构造发生在`PlanLoopPartition`中。源码如下：

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

可以看到这里我们在手搓一个表达式。`flattened`就是一个拍平的循环变量，比如`i * 16 + j`。由于我们按照`vector_extent`进行分组，分组后的索引(`access_index`)就是`flattened / vector_extent`。按照这个索引平均分配到`num_thread`个thread上，得到`forward_thread=access_index % num_thread`。`forward_index` 由两部分组成：`(access_index/num_thread) * vector_extent` 得到对应的向量组，`flattened % vector_size`得到组内的相对索引。

对于`T.Parallel(4, 16)`、`threads=64`、`vector_size=4`：

```text
flattened(i, j) = i * 16 + j
access_index(i, j) = (i * 16 + j) / 4
thread         = ((i * 16 + j) / 4) % 64 
               = (i * 16 + j) / 4 (因为总小于64)
local_idx      = (i * 16 + j) / 64 +  (i * 16 + j) % 4 
               = (i * 16 + j) % 4 （因为前一项恒等于0）
```

### Fragment buffer的特殊检查
Fragment loop比普通copy/shared-memory loop更严格。对于fragment access，TileLang会尽量避免padded logical points，因为padding会制造假的fragment ownership。对于非fragment的SIMT loop，padding可以在lowering时通过guard跳过。

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

这个`floormod`的整除检查为了保证 `loop_total_size % (threads * vector_size) == 0`。我们将多个大小是`vector_size`的组分配给threads，我们希望每个thread拿到的组都是完整的，没有padded元素。这是因为fragment buffer代表真实的thread-local storage。如果planner覆盖的逻辑元素比loop实际拥有的更多，这些padded元素仍然会看起来像fragment的元素，可能造成非法ownership。对于非fragment loop，编译器通常可以写一个predicate跳过这些padded element。对于fragment，padding可能会使得正确性的保证变得更加复杂。比如以下例子：

```py
with T.Kernel(threads=64)
for i in T.Parallel(256):
    fragment[i] = ...
```
假设最开始得到的`vector_size=8`。planner首先检查 `256 % (64 * 8) = 256 % 512 = 256`。也就是说一个完整的SIMT/vector tile会覆盖512个逻辑元素，但loop只有256个。这会需要256个padded logical fragment elements，因此TileLang将vector size减半，得到`256 % (64 * 4) = 256 % 256 = 0`,说明每个组都是完整的，不用pad。注意此时我们假设的是所有`num_threads`都被用上的情况。有些时候，我们可能可以减少`num_threads`来达到更好的向量化，但编译器没有这么实现。我猜测可能是因为如果我们假设所有的参数都是二的幂，非整除的情况较少，没那么大的意义。

## `ValidateCandidateAgainstFragments`

`ValidateCandidateAgainstFragments`会检查选出的loop layout是否和这个loop使用到的所有已知fragment layout兼容。一个loop可能触碰多个fragments。即使候选layout本身看起来合法，它也可能和某个buffer已有的owner-thread mapping不一致。

函数会遍历loop访问的所有fragment buffers：

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

对于read，要求是“包含”，即loop layout需要能从source fragment layout中读取元素。这很好理解，如果loop iteration映射到的thread并不拥有需要读取的元素，那么这个读取就是非法的。

```cpp
if (access.is_read &&
    !ProveFragmentContains(candidate, fragment, vars, access.indices,
                           analyzer_, check_forward_index)) {
  success = false;
}
```

我们继续看`ProveFragmentContains`的实现：

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

`small_frag`表示目标fragment的layout，`large_frag`表示已经存在的fragment的layout。函数首先有一个特殊情况：如果large fragment是fully replicated，每个thread都有一份合法拷贝，那么containment显然为真。

当`check_forward_index`为true时，还会检查两个layout是否使用相同的物理local index。

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


接着，函数为small fragment引入一个符号replicate变量：
```cpp
Var rep_small("__checking_frag_contains_rep");
analyzer.Bind(rep_small,
              Range(IntImm(small_frag->ReplicateExtent()->dtype, 0),
                    small_frag->ReplicateExtent()),
              true);
auto thread = small_frag->ForwardThread(small_frag_indices, rep_small);
```

这表示：对于candidate layout的任意合法replica，计算执行这次access的thread。

下一步，函数会问large fragment：“如果我使用large fragment的物理local index，再加上small fragment的thread，那么这个物理坐标对应large fragment中的哪个逻辑元素？”

```cpp
auto large_frag_physical_and_thread = large_frag->Forward(large_frag_indices);
large_frag_physical_and_thread.push_back(thread);
auto inv_large_frag = large_frag->Inverse();
auto inv_large_frag_logical_and_rep =
    inv_large_frag->Forward(large_frag_physical_and_thread);
```

`large_frag->Forward(large_frag_indices)`给出local index，结合推理出的thread，形成物理坐标`(large local index, candidate thread)`。Large fragment的inverse会把这个物理坐标映射回`(large logical indices, large replica)`。最后一个元素就是恢复出的replica。

```cpp
auto inv_large_frag_rep =
    inv_large_frag_logical_and_rep[inv_large_frag_logical_and_rep.size() - 1];
```

最后，函数使用这个恢复出的replica重新计算large fragment的owning thread：

```cpp
auto check_thread =
    large_frag->ForwardThread(large_frag_indices, inv_large_frag_rep);

auto diff = analyzer.Simplify(thread - check_thread);
return analyzer.CanProve(diff == 0);
```

如果重新算出来的large-fragment thread等于原来的candidate thread，那么candidate thread就是这个large-fragment access的合法owner。否则，candidate loop就试图从错误的thread读写元素。

这里是一个成功的读取的例子：

```text
candidate:
  thread(i, j) = i * 16 + j

fragment:
  thread(i, j) = i * 16 + j
  local(i, j)  = 0

access:
  frag[i, j]
```

`ProveFragmentContains`计算：

```text
small thread = i * 16 + j
large local  = 0
inverse large(0, i * 16 + j) -> (i, j, rep=0)
check_thread = i * 16 + j
diff         = 0
```

所以这个access合法。我们再来看一个失败的例子：

```text
candidate:
  thread(i, j) = i

fragment:
  thread(i, j) = i * 16 + j
  local(i, j)  = 0

access:
  frag[i, j]
```

此时：

```text
small thread = i
large local  = 0
inverse large(0, i) -> some element owned by thread i
check_thread = i * 16 + j
diff         = i - (i * 16 + j)
```

analyzer无法证明它为0。因此candidate layout不能安全地读取`frag[i, j]`。

对于write，检查更严格：

```cpp
if (access.is_write &&
    (!ProveFragmentContains(fragment, candidate, access.indices, vars,
                            analyzer_, check_forward_index) ||
     !ProveFragmentContains(candidate, fragment, vars, access.indices,
                            analyzer_, check_forward_index))) {
  success = false;
}
```

这里使用两个方向的containment check，实际上要求owner-thread完全兼容，即所有被访问元素的ownership相等。Read只需要单向containment，因为loop只需要证明它所读取的值在执行它的thread上可用。Write需要相等，因为write会更新buffer layout的物理状态。

例如，在`for i, j in T.Parallel(2, 2):`中，假设buffer layout是
```text
forward_thread(i, j) = i * 2 + j
forward_index(i, j) = 0
```
candidate layout 是
```text
forward_thread(i, j) = i
forward_index(i, j) = j
```



Candidate layout可能覆盖相同的逻辑shape（比如thread0 和thread1想要读取的时候，总是可以读取），但由于把每个元素存到了不同的物理thread/local slot，所以会存在读写不一致性，应该被拒绝。




## 常见失败案例

总结一下以上我们提到的三种不同layout inference失败模式。第一种失败模式是非单射layout，即对于两个不同的元素`x`, `y`，有`forward_thread(x) == forward_thread(y)`且`forward_index(x) == forward_index(y)`。


第二种模式是ownership不兼容。candidate本身可能是单射的，但它并没有从拥有该值的thread访问fragment。`ValidateCandidateAgainstFragments`和`ProveFragmentContains`就是用来捕获这种情况。

第三种失败模式是稀疏但单射的layout：

```text
(group, row_in_group) -> thread = group * 32 + row_in_group * 16
```
这会将四个逻辑元素映射到thread `{0, 16, 32, 48}`。没有collision，所以它是单射的。然而lowering需要对thread表达式求逆。这个inverse需要额外的整除guard，比如`thread % 16 == 0`。当前lowering路径会经过TVM的affine inverse machinery，而某些稀疏的single-component表达式无法在那里表示。这就是本文开篇提到的`abs(source->scale) == 1`断言背后的失败模式。

# 举个例子

我们用一个例子把所有东西串起来：

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

完整的layout inference时序图如下：
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

第一个loop写整个fragment：

```py
for row, col in T.Parallel(4, 16):
    fragment[row, col] = A[row, col]
```

在layout inference开始时，`fragment`没有已知layout。在strict mode下，因为fragment access不是常数索引，所以这个loop无法推导出fully replicated layout。在common mode下，仍然没有已知source layout可以传播。在free mode下，会使用`ComputePlanCandidate`。

这个loop有的总迭代次数是`64`，刚好是`threads`的总数，所以进过上述的代数推导，得到了第一个loop的fragment layout：
- `forward_thread(row_col) = row * 16 + col`
- `forward_index(row, col) = 0`


这个update会返回给BFS driver。driver将它存入`layout_map`，然后通过`use_list_`重新入队使用`fragment`的operation。这就是第二个loop如何在已有信息下获得第二次推导机会。

推导第二个loop时，`fragment`已经有已知layout，所以`ComputeLoopLayoutFromBuffer`可以运行。它会将access indices替换进source fragment的thread表达式。带入`row = group * 2 + row_in_group`和，`col = 0`得到：
- `forward_thread(group, row_in_group) = group * 32 + row_in_group * 16`

四个逻辑iteration分别映射到thread 0, thread 16, thread 32, thread 48。问题是，**这个layout是稀疏的**。从数学角度看，这个映射仍然是单射：没有两个逻辑iteration映射到同一个物理位置。然而，当前inverse-lowering路径很难表示它。Lowering需要从物理thread坐标反推出逻辑loop变量，但推导过程需要整除约束：
```text
group = thread / 32
row_in_group = (thread % 32) / 16
```
当前路径会走TVM的affine inverse machinery，而这个稀疏single-component map可能会产生一个source scale不是`1`或`-1`的`IterSplitExpr`。这就是失败来源：

```text
TVM_FFI_ICHECK(analyzer_->CanProveEqual(abs(source->scale), 1));
```

这解释了本文开头那个诡异的行为。kernel失败可能并不是因为layout明显非单射，而是因为推导出的layout在thread space中是稀疏的，并且inverse步骤没有合适的guard表示。

有几个实用方法可以避免这个问题：

- 加一个显式`T.Fragment` annotation，让source fragment layout为后续投影access产生dense thread mapping。
- 改变loop结构，让被投影掉的dimension成为local index的一部分，而不是thread index的一部分。
- 改变`BLOCK`或`threads`，让传播出的layout对当前inverse machinery来说恰好dense。（这个就有点玄学和碰运气了）

下面是一个具体的annotation版本。我们可以定义一个layout，把`col`放到thread轴，把`row`放到local轴：

```py
def source_layout(row, col):
    return col, row

T.annotate_layout({
    fragment: T.Fragment((4, 16), forward_fn=source_layout)
})
```
于是`forward_thread(row, col) = col`。那么读取`fragment[row, 0]`会得到：

```text
thread = 0
local  = row
```

这就是一种replicated或single-thread风格的access，而不是稀疏的`{0, 16, 32, 48}`模式。不一定是性能最好的layout，但它能跑。

# 收尾

TileLang layout inference很强大，因为它让我们可以写block-level tensor code，而由编译器推导thread-level ownership。代价是fragment layout会变成一个全局约束：一旦某个fragment element被分配到某个thread/local slot，之后的每次使用都必须遵守这个分配。


layout inference内部原理给我关于调试的启发是，可以关注
1. 哪个loop最先建立了fragment layout？
2. 哪个后续loop通过不同的access expression传播了这个layout？
3. 传播出的thread expression是否变成了非单射、稀疏，或者依赖内部serial变量？