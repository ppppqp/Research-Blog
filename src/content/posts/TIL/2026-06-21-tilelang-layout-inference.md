---
author: Retep
pubDatetime: 2026-06-20T12:50:00.000-07:00
title: "如何正确地写Layout：TileLang Layout Inference深度解析"
featured: false
draft: false
tags:
  - TileLang
description: ""
---

Disclaimer：笔者不是TileLang核心开发者，啃源码写的文章，不一定完全正确。权作抛砖引玉，供大家参考。如有纰漏，欢迎指正，不胜感激。

# 引子

Layout inference是TileLang的精华。相比Triton和CUDA的SIMT模式，TileLang在更高的block的层面进行编程，由编译器进行调度，实现了以极少的代码量达到接近sota的性能。这其中的一位功臣就是Layout inference，即将高度抽象的block层代码映射到thread层指令的算法。

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
然而，如果用`T.Fragment`对B进行layout annotation，或者更改`BLOCK`的大小，就可以通过编译。处于好奇，笔者开始阅读TileLang layout inference的源码。


# 什么是Layout Inference?

首先为大家介绍一下Layout Inference的概念，熟悉的同学可以跳过。在Triton中，我们编写的kernel都会取自己的`program_id`，然后根据它来推算当前thread覆盖的索引范围，计算数据的物理地址。比如：
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


在TileLang中，我们并不和thread打交道，而是使用Tensor buffer声明要做的事即可：

```py
# 类似以下形式
with T.Kernel(1, threads=32): 
  for i, j in T.Parallel(BLOCK_I, BLOCK_J):
    output[i, j] = input[j, i]
```
（太优雅了）

在实际JIT时，TileLang会将`T.Parallel`的每个循环所做的事以最优的方式分配到threads上，达到SIMT。这个“分配”的算法就是Layout Inference。简单来说，就是一个从循环变量到thread的映射。通过回答“给定循环变量i和j，操作应该放在哪个thread”，得到映射 $f(i, j) = thread$，然后对其进行求逆得到$f^{-1}(thread) = i, j$, 即“对于当前thread，循环变量的值是什么”，达到自动推理出layout algebra的目的（而在Triton中，我们必须苦逼地手搓）。

看上去没那么难是不是？其实并不。
1. 从正确性的角度，在TileLang中，我们可以声明thread的本地buffer，即fragment buffer。这些Fragment可以在多个`T.Parallel`中穿插使用，但我们需要保证它们始终是thread local的，即需要绑定到固定的thread。
2. 从性能的角度，`T.Parallel`的展开并不是简单直接按循环次数即可，还需要结合考虑如向量化来合并访问内存，减少延迟。

对于fragment buffer，我们需要考虑一个额外变量：replication，即一个元素可能需要在多个thread上出现，导致元素索引和thread并不是双射。此外，我们在编程时总是用block level的fragment buffer索引，但编译到thread level时，只有一部分fragment buffer会真正被拷贝到这个thread level，所以还需要一个block level索引到thread level索引的映射。综上，对于fragment buffer需要回答两个问题：
1. 对于fragment buffer的索引`(i, j)`和replica `rep`，该元素应该分配到哪个thread。即$f(i, j) \rightarrow thread$
2. 对于fragment buffer的索引`(i, j)`和replica `rep`，在使用时，应该取thread buffer中的哪个索引，即$f(i, j, rep) \rightarrow local_index$

总之，由于我们在更高层级上编码，编译器能在全局上的优化能力就更大，复杂性也会同步增加。我们需要充分了解内部原理，才能极致地发挥编译器的能力。


## High Level Flow
我们先在宏观上了解一下Layout Inference的流程。
Layout Inference在TileLang中通过`tilelang.transform.LayoutInference`这个pass实现，通过ffi绑定到`tvm::tl::LayoutInference`。可以看到这个pass的目的就是将IR中的操作进行thread binding:
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
    // 执行layout inference，这个是layout inference的入口
    auto result = collector.Run();
    LayoutInferencer substituter(result, &analyzer);
    // 根据layout inference的结果进行替换
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }
```
通过“正确性“和”性能“的两个要求，可以推出Inference是一个constraint optimization过程：我们需要在满足thread local的约束下，尽可能提升编译结果的性能。`collector.Run`通过BFS来解决这个问题，核心是以下几步（先简略带过，后面细说）：

1. 将所有需要infer的operation入队。这里的operation有显式的`ParallelOp`，也有隐式的如`CopyOp`。本文限于篇幅主要讨论`T.Parallel`。
2. step 0: 如果有“floating”的fragment buffer，则将它replicate到所有thread。
3. step 1: 对每一个operation进行strict layout推导，更新layout map。
4. step 2：对队列中的operation进行common layout推导，BFS直到收敛，更新layout map。
5. step 3：对队列中的operation进行free layout推导，BFS直到收敛，更新layout map。

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

### 什么是“floating”的fragment buffer
类似如下情况：A在`if`语句下使用，是不会被进行layout分析的，所以会将A直接replicate到所有的thread保证正确性。
```py
A = T.alloc_fragment(...)
for i, j in T.Parallel:
  if A[i, j] > 0:
    B[i, j ] = 0
  else:
    B[i, j] = 1
```

剩下的三个layout inference pass，最终都是调用同一个方法(`RunInferStep`)进行收敛。
- 对当前队列中的首个operation元素执行`RunInferStep`
- 调用这个operation的`InferLayout`方法，返回每个buffer layout update。如`Parallel`和`Copy`就实现了各自的方法，我们之后会讲到。
- 如果该buffer的layout还不存在，则更新。
- 如果该buffer的layout已经存在，则检查是否兼容，并尝试合并。
- 进行alias propagation，比如这次更新了buffer_A, 然后op_X和op_Y用了buffer_A，则将这两个operation加入队列进行下一轮的迭代。

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


### Strict layout inference
在下面的例子中，`B`只有常数索引，和循环变量无关，所以可以算作自由变量(free variable)。在循环体的视角下，B就是一个外部的常量，所以直接给每个thread都复制一份即可。
```py
for i, j in T.Parallel(...):
    A[i, j] = B[0]
```
注意这样是非法的，会导致data racing。
```py
for i, j in T.Parallel(...):
    B[0] = A[i, j]
```

以下是`ParallelOpNode
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

```cpp
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
  LOG(INFO) << "[LayoutTrace] SOURCE"
            << " source="
            << (source_buffer.defined() ? source_buffer->name
                                        : String("<none>"))
            << " source_is_write=" << source_buffer_is_write << " read_source="
            << (read_source_buffer.defined() ? read_source_buffer->name
                                             : String("<none>"))
            << " allow_propagate=" << allow_layout_propgate;
  // moved to ComputeLoopLayoutFromBuffer

  // Try to infer loop layout from buffers in order of preference only if we
  // don't already have a layout (e.g., from annotations):
  // 1. Annotated loop layout
  // 2. Non-replicated write buffer (most reliable)
  // 3. Non-replicated read buffer
  // 4. Fully replicated write buffer (backup, may cause issues)
  // 5. Free inference mode (no source buffer)
  if (!loop_layout_.defined() && annotated_layout_unbound_.defined()) {
    loop_layout_ =
        annotated_layout_unbound_.value()->BindThreadRange(T.thread_bounds);
    LOG(INFO) << "[LayoutTrace] LOOP SELECT"
              << " source=annotation"
              << "\nlayout=" << loop_layout_->DebugOutput();
    loop_layout_requires_padding_guard_ = annotated_requires_padding_guard_;
    if (annotated_predicate_.defined()) {
      predicate_ = annotated_predicate_.value();
    }
  } else if (!loop_layout_.defined() && source_buffer.defined() &&
             (allow_layout_propgate || source_buffer_is_write)) {
    loop_layout_ = ComputeLoopLayoutFromBuffer(source_buffer, T);
    LOG(INFO) << "[LayoutTrace] LOOP SELECT"
              << " source=known-buffer buffer=" << source_buffer->name
              << "\nlayout=" << loop_layout_->DebugOutput();
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

  // Non-fragment SIMT loops may deliberately over-cover a ragged iteration
  // space; PartitionLoop emits guards for the padded points. Fragment/reducer
  // loops stay strict because padding would change per-thread ownership.
  auto injective_res =
      loop_layout_->DetectInjective(loop_layout_requires_padding_guard_);
  if (!injective_res->errors.empty()) {
    // ...
    throw LoopLayoutInjectiveException(oss.str());
  }

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

  // case for threads=48, total_threads=64, replication=1
  if (!analyzer_.CanProveEqual(loop_thread_extent, block_size)) {
    AddPredicate(
        LT(InputPlaceholder(0), loop_thread_extent + T.thread_bounds->min));
  }

  // Step 2: Check that the loop's partition can correctly align with all source
  // fragment, and infer layout only when it's not yet layout-ed.
  ValidateCandidateAgainstFragments(loop_layout_, T, /*throw_on_error=*/true,
                                    /*check_forward_index=*/false,
                                    source_buffer);

  // Step 3: Build replication guards
  BuildReplicationGuardsIfNeeded(
      T, store_shared_global_buffers_, store_fragment_buffers_,
      has_cross_thread_access_, const_index_fragment_buffer);

  //...
  return results;
}
```