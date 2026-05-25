---
author: Retep
pubDatetime: 2026-05-24T20:14:13.000-07:00
title: "[Book Club] TileLang: Bridge Programmability And Performance In Modern Neural Kernels"
featured: false
draft: false
tags:
  - Book Club
description: ""
---
# TileLang

A blog post that connects the [TileLang paper](https://arxiv.org/abs/2504.17577) with [code](https://github.com/tile-ai/tilelang). Note that all the code pointers in this blog post is based on commit 69bc43e2cfbc38976b0c5e1c4b2299b66e8970ee.


## 1. Code Walkthrough

Let's first take a look at the overall compilation lifecycle of TileLang.

### Capture

TileLang has two frontend capture paths. We can either use `@T.prim_func` for static compilation or `@tilelang.jit` for JIT.

In tilelang code, we write: 
```py
with T.Kernel(blocks, threads=threads) as bx:
   acc_o = T.alloc_fragment(...)
```
Here we are calling the TileLang API, that lives in `tilelang/language`. `T.Kernel(...)` creates a kernel launch frame with block and thread bindings
through `_ffi_api.KernelLaunch` in `tilelang/language/kernel.py:267`. This calls the `KernelLaunch` in `src/ir.cc` that initializes a kernel launch frame.

```py
# tilelang/language/kernel.py
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
    is_cpu: bool = False,
    prelude: str | None = None,
):
```


Then when a kernel is compiled, TileLang walks through the AST of the code and compile to TIR. The script-style `@T.prim_func` path
uses the TVM parser (`tilelang/language/parser/entry.py:37`). The eager JIT path uses `Builder`, which owns an `IRBuilder`, enters a `tirx.prim_func` frame, and returns a `PrimFunc` (`tilelang/language/eager/builder.py:168`, `tilelang/language/eager/builder.py:193`, `tilelang/language/eager/builder.py:236`).
`@tilelang.jit` wraps this in `JITImpl`; at runtime it obtains TIR and creates a
`JITKernel` (`tilelang/jit/__init__.py:390`, `tilelang/jit/kernel.py:59`).

```py
# tilelang/language/eager/builder.py
class Builder(BaseBuilder):
    def __init__(self):
        self.frames: list[AnyFrame] = []
        self.ir_builder = IRBuilder()
        self.name_inside_frame: dict[str, AnyFrame] = {}
        self.macro_arg_annot = {}
        self.out_idx = []
        self.out_tensor_cnt = 0

    @contextmanager
    def prim_func(self, name):
        thread_local_storage.builder = self
        clear_let_values()
        try:
            with self.ir_builder, self.with_frame(tirx.prim_func()):
                tirx.func_name(name)
                yield
            if self.eager_jit != "phase1" and len(self.out_idx) != self.out_tensor_cnt:
                raise RuntimeError("Not all tensor allocated from `T.empty` are returned")
        finally:
            clear_let_values()
            del thread_local_storage.builder

    def get(self) -> PrimFunc:
        return self.ir_builder.get()
```


There are three kinds of things to compile:
- Control flow: Rewritten to corresponding TIR branch instructions via `DSLMutator` in `tilelang/language/eager/ast.py`. However, note that if branching can already been resolved in compile time (e.g. `if block_m == 128`), only the branch that matters will be kept.
- Python primitives: They are mostly treated as meta values unless participated in a TIR expression, which means most of them are not lowered.
- TileLang API: They (e.g. `T.alloc_fragment`) are translated to specific user defined TIR by the rules defined (e.g. `tilelang/language/allocate.py`) The C++ side registers each tile-op through `TIR_REGISTER_TL_TILE_OP` in `src/op/operator.h:172`, and reconstructs a typed `TileOperator` from the intrinsic call through `ParseOperator` in `src/op/operator.cc:34`.

During the compilation, the TileLang autotuner will apply layout recommendation, which is explicitly called out by TileLang as a core contribution. The entry is in `tilelang/carver/template/base.py`, and we can also explicitly use in user code like:
```py
roller_hints = carve_template.recommend_hints(topk=topk)
...
config["block_M"] = block_m
config["block_N"] = block_n
config["block_K"] = hint.rstep[0]
config["num_stages"] = hint.pipeline_stage
config["thread_num"] = block_rows * block_cols * 32
```
It generate the optimal layout by considering input/output shapes, dependency analysis and hardware spec. The produced config will be part of the pre-lowered code and materialized during compilation.
```py
with T.Kernel(..., threads=thread_num):
   # block_M gets the recommendation on compilation
   A_shared = T.alloc_shared((block_M, block_K), dtype)
```
### Optimization

After we have the IR, we do optimization and emit machine code. Compilation enters `JITKernel._compile_and_create_adapter`, opens a TVM
`PassContext`, and calls `tilelang.lower` (`tilelang/jit/kernel.py:204`,
`tilelang/jit/kernel.py:246`). `lower_to_host_device_ir` binds the target, runs
semantic checks, then runs `LowerAndLegalize` and `OptimizeForTarget`
(`tilelang/engine/lower.py:274`, `tilelang/engine/lower.py:300`). These two passes are the core contribution of TileLang.

```py
# tilelang/engine/lower.py
# Before lowering, do semantic check
PreLowerSemanticCheck(mod)

# Phase 1: Lower and legalize the IR
mod = LowerAndLegalize(mod, target)

# Phase 2: Optimize the IR for the target
mod = OptimizeForTarget(mod, target)

host_mod = tirx.transform.Filter(_is_host_call)(mod)
device_mod = tirx.transform.Filter(_is_device_call)(mod)
```


`LowerAndLegalize` performs the frontend-to-low-level transition: add wrappers,
legalize negative indices, verify parallel loops, simplify, infer reducer
layouts, optionally warp-specialize, plan/inject software pipeline, run
`LayoutInference`, lower tile ops, legalize vectorization and memory access, and
simplify (`tilelang/engine/phase.py:143`). `LowerTileOp` turns each typed
tile-op into lower-level TIR by calling the operator's `Lower` method
(`src/transform/lower_tile_op.cc:1132`). Custom operators like GEMM are lowered by
`tl.gemm.lower`, with a specific backend implementation that emits tensor-core macro
TIR (`src/op/gemm.cc:185`, `tilelang/tileop/gemm/__init__.py:18`,
`tilelang/cuda/op/gemm/gemm_mma.py:71`).

```py
# tilelang/engine/phase.py
# Run pipeline planning and software-pipeline rewriting before layout
# inference so inferred layouts see the final pipelined structure directly.
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.Simplify()(mod)

# Infer memory layouts for fragments and shared memory
mod = tilelang.transform.LayoutInference()(mod)
LayoutVisual(mod)

# Lower high-level tile operations to low-level operations
mod = tilelang.transform.LowerTileOp()(mod)
```

`OptimizeForTarget` then performs target-facing cleanup: TMEM/shared barrier
lowering, allocation planning, flattening, vectorization, storage rewrite,
unrolling, allreduce lowering, LDG/STG and Hopper intrinsic lowering,
host/device splitting, shared-memory merge, sync/fence injection, packed API,
kernel launch lowering, and persistent block transforms
(`tilelang/engine/phase.py:231`).

The other optimization that TileLang paper explicitly called out is layout inference. Layout inference is registered as a `LowerAndLegalize` pass as `LayoutInference` in `tilelang/transform`, which calls the C++ API in `src/transform/layout_inference.cc`. The compiler's layout-inference FTG is built implicitly by
`BufferUseDefCollector` in `src/transform/layout_inference.cc:108`.
During `Collect`, it records all buffers by storage `Var`, scans the function
body, and appends every tile op or parallel loop to `infer_list_`
(`src/transform/layout_inference.cc:480`, `src/transform/layout_inference.cc:544`,
`src/transform/layout_inference.cc:662`). Fragment buffers form graph edges:
`addToUseList` maps each fragment buffer to the tile-op indices that touch it
(`src/transform/layout_inference.cc:642`). Aliased buffers sharing the same data
`Var` are tracked in `buffer_data_to_buffers_` and propagated together
(`src/transform/layout_inference.cc:169`, `src/transform/layout_inference.cc:398`).

### Code Gen

Finally after legalization and optimization is done, `lower` codegens device source through `device_codegen_without_compile`
or compiled runtime modules through `device_codegen` (`tilelang/engine/lower.py:232` to compile to machine code in various backend. CUDA uses
`target.build.tilelang_cuda` or the no-compile variant
(`tilelang/engine/lower.py:237`). Host codegen uses LLVM or TileLang C host
codegen (`tilelang/engine/lower.py:198`). For `tvm_ffi`, the device code is
compiled by the registered CUDA callback using NVCC
(`tilelang/engine/lower.py:103`). For the `nvrtc` backend, the adapter wraps
the generated source, compiles it into a loadable CUDA library, obtains kernel
handles, and exposes a PyTorch-facing callable
(`tilelang/jit/adapter/nvrtc/adapter.py:27`).

Next let's deep dive a bit to those two core contribution of TileLang:

## 2. Tile Recommendation

In short, tile recommendation models each operation to optimize the memory and computation placement (or in other words, optimize based on roofline model).

$$Time = \max_{i,j}(\dfrac{MemoryTraffic_i}{Bandwidth_i}, \dfrac{Computation_j}{Performance_j}) + t_{intrinsic} $$
> where i indexes memory hierarchy levels (e.g., HBM, L2, L1), j indexes compute unit types (e.g. tensor cores, CUDA cores, SFUs), and $t_{intrinsic}$ accounts for inherent overheads such as kernel launch latency and loop prologue/epilogue costs.


The implemented recommendation system is `tilelang/carver`. Public templates
such as `MatmulTemplate` build an equivalent TE/TIR tensor representation, which will then be used for analysis on operation shape, reduction axes and tensorization opportunities/memory traffic.

### Recommendation Data Flow

The recommendation path is intentionally separate from the TileLang program
that will eventually be compiled. A template first builds an equivalent TE/TIR
view of the operation. Carver analyzes that view, emits `Hint` objects, and the
caller materializes one hint back into the TileLang kernel's meta-parameters.

```text
Problem shape and dtypes
M, N, K, transposes, heads, seq length
        |
        v
Template
MatmulTemplate / FlashAttentionTemplate
        |
        v
Equivalent PrimFunc or output graph
TE placeholders, compute, reduce axes
        |
        v
Tensorization detected?
        |
        +-- yes --> TensorCorePolicy
        |           MMA/WGMMA/TCGEN-aware constraints
        |
        +-- no  --> DefaultPolicy
                    CUDA-core heuristic schedule
        |
        v
Candidate Hint list
        |
        v
User/autotuner config
block_M, block_N, block_K, thread_num, num_stages, rasterization
        |
        v
TileLang kernel source
T.Kernel / alloc_shared / alloc_fragment / T.Pipelined
        |
        v
LowerAndLegalize + OptimizeForTarget
```

For a plain GEMM, the data carried by a winning `Hint` maps almost one-to-one to
kernel meta-parameters:



Carver Hint              |    TileLang kernel parameter/use
--|--
hint.block = [BM, BN]   |     block_M, block_N, output tile per CTA
hint.rstep = [BK]       |    block_K, reduction slice loaded per pipeline step
hint.warp = [WM, WN]     |    tensor-core warp tile shape, used to derive threads
hint.thread = [...]       |   CUDA-core thread partition when not using TC
hint.pipeline_stage        |  T.Pipelined(..., num_stages=...)
hint.use_async            |   async-copy pipeline preference
hint.rasterization_plan    |  optional blockIdx remapping for L2 locality
hint.cached_tensors        |  tensors expected to reside in shared memory
hint.pass_context          |  lowering knobs such as static shared merge

The important boundary is that Carver does not rewrite the user's kernel by
itself. It recommends static parameters; the benchmark/autotune code then
instantiates the TileLang program with those parameters before lowering.

### Hardware Models
The model is heuristic, not learned. Hardware characteristics come from
`TileDevice` implementations. CUDA records shared memory capacity, SM count,
warp size, register capacity, transaction sizes, approximate bandwidth, and
available tensor instruction shapes (`tilelang/carver/arch/cuda.py:124`).
`DefaultPolicy.emit_config` builds candidate tiles, estimates traffic,
shared-memory usage, register usage, block residency, and waves, then ranks
candidates with `(traffic + 1) * num_wave`
(`tilelang/carver/roller/policy/default.py:72`,
`tilelang/carver/roller/policy/default.py:96`,
`tilelang/carver/roller/policy/default.py:537`).
Reduction steps are selected by coalescing and transaction-size heuristics
(`tilelang/carver/roller/policy/default.py:241`). Block sizes are scored against
warp size and SM partitioning (`tilelang/carver/roller/policy/default.py:202`,
`tilelang/carver/roller/policy/default.py:599`).

```py
# tilelang/carver/roller/policy/default.py
def emit_config(self, topk: int) -> list[Hint]:
    base_tile = self.get_base_tile()
    if base_tile is None:
        return []

    rstep_map = {node: self._assign_reduce_step(node) for node in self.ordered_nodes}
    smem_tile_condidates = self.dfs_smem_tile(base_tile, rstep_map)
    results = []
    for td in smem_tile_condidates:
        if not self.check_tile_shape_isvalid(td):
            continue

        self._expand_reduce_axis(td)
        for codegen_dicts in self.assign_block_size(td):
            if isinstance(codegen_dicts, dict) and len(codegen_dicts) == 1:
                results.append(list(codegen_dicts.values())[0])
```

```py
# tilelang/carver/roller/policy/default.py
def prio(td: TileDict):
    return (td.traffic + 1) * td.num_wave
```

The candidate search can be read as a constrained best-first expansion over
output tile sizes:

```text
Base tile
minimum tile that reduces redundant compute
        |
Assign initial rstep
coalescing/transaction heuristic
        |
Expand output tile factors
DFS/PriorityQueue over factors and {2,4,8,16,32}
        |
Compute TileDict
        |
Propagate tile to producer nodes
node.propagate_inputs / node.propagate_outputs
        |
Estimate global traffic
coalesced HBM reads and writes
        |
Estimate shared memory
node footprint + BestFit lifetime allocator
        |
Fits smem cap?
        |
        +-- no  --> reject
        |
        +-- yes --> Estimate registers
                    2 * max(tile elems * dtype bits / 32)
                         |
                    Fits reg cap?
                         |
                         +-- no  --> reject
                         |
                         +-- yes --> Estimate residency
                                     block_per_SM = min(smem, regs, sm partition)
                                          |
                                     Estimate waves
                                     ceil(grid_size / (block_per_SM * SMs))
                                          |
                                     Priority
                                     (traffic + 1) * num_wave
                                          |
                                     Expand reduce axis if smem budget allows
                                          |
                                     Assign block size / thread or warp tile
                                          |                 
                                     Emit Hint
```

The score makes two tradeoffs visible:

- Smaller tiles often reduce per-CTA shared/register pressure and improve
  occupancy, but they increase `grid_size` and may produce more waves.
- Larger tiles can reduce global-memory traffic through reuse, but they are
  rejected or penalized when shared memory/register pressure lowers residency.

```py
# tilelang/carver/roller/policy/default.py
td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
if td.smem_cost > self.arch.smem_cap:
    td.valid = False
    return td
output_shape = self.output_nodes[0].get_space_dim()
td.grid_size = int(np.prod([(y + x - 1) // x for x, y in zip(output_tile, output_shape)]))
reg_usage = int(2 * max([np.prod(td.get_tile(node)) * node.get_dtype().bits / 32 for node in self.ordered_nodes]))
if reg_usage > self.arch.reg_cap:
    td.valid = False
    return td
td.block_per_SM = min(
    self.arch.max_smem_usage // max(td.smem_cost, 1),
    self.arch.reg_cap // max(reg_usage, 1),
    self.arch.sm_partition,
)
td.num_wave = int(np.ceil(td.grid_size / int(td.block_per_SM * self.arch.compute_max_core)))
```

### Tensor core
`TensorCorePolicy` adds tensor-core-specific rules: pipeline stage and async-copy
defaults by architecture, K-step multiples of `wmma_k`, tensor-intrinsic shape
validity, warp tile assignment, and shared-memory limits
(`tilelang/carver/roller/policy/tensorcore.py:30`,
`tilelang/carver/roller/policy/tensorcore.py:86`,
`tilelang/carver/roller/policy/tensorcore.py:206`,
`tilelang/carver/roller/policy/tensorcore.py:262`).

```py
# tilelang/carver/roller/policy/tensorcore.py
if self.arch.compute_capability in {"sm_80", "sm_90", "sm_90a"}:
    self.pipeline_stage = 2
else:
    self.pipeline_stage = 1
use_async_copy = self.prim_func_node.get_tag("use_async_copy")
if use_async_copy:
    self.use_async_copy = use_async_copy
else:
    if self.arch.compute_capability in {"sm_80", "sm_90", "sm_90a"}:
        self.use_async_copy = True
    else:
        self.use_async_copy = False
```

For tensor-core GEMM, the recommender additionally has to keep the macro tile
compatible with intrinsic fragments:

```text
Output tile [block_M, block_N]
+-----------------------------------------------+
| CTA computes one C tile                       |
|                                               |
|  +--------------+--------------+-----------+  |
|  | warp tile    | warp tile    | ...       |  |
|  | [warp_M,N]   | [warp_M,N]   |           |  |
|  +--------------+--------------+-----------+  |
|  | warp tile    | warp tile    | ...       |  |
|  +--------------+--------------+-----------+  |
|                                               |
| Each warp tile is an integer multiple of      |
| the selected MMA/WGMMA/TCGEN intrinsic shape. |
+-----------------------------------------------+

Reduction slice [block_K]
+--------------------+      +--------------------+
| A_shared BM x BK   |  x   | B_shared BN x BK   |
| staged per pipe    |      | staged per pipe    |
+--------------------+      +--------------------+

Pipeline depth multiplies the live shared-memory footprint.
```

For `FlashAttentionTemplate`, Carver is not looking at the final online-softmax
kernel body directly. The template represents the core compute as a small graph
of two tensorized matmuls and connects their nodes:

```text
Q: [B*H, S_q, D]  ----+
                      |
               +-------------+
K: [B*H,       | MMA0: QK^T  |
   S_kv, D] -->| space:      |
               | [B*H,S_q,   |
               |  S_kv]      |
               | reduce: D   |
               +-------------+
                      |
                      | edge: score tile shape
               +-------------+
V: [B*H,       | MMA1:       |
   S_kv, D] -->| Score * V   |
               | space:      |
               | [B*H,S_q,D] |
               | reduce:     |
               | S_kv        |
               +-------------+
                      |
               O: [B*H, S_q, D]
```

That graph lets `DefaultPolicy`/`TensorCorePolicy` compute tile propagation,
temporary storage lifetime, and traffic across both matmul nodes instead of
ranking each matmul independently.

From my understanding, in order for tile recommendation (auto tuning) to benefit my kernel development, I need to implement a custom template for the kernel as well, or reuse the existing public template.


Also it's worth noting that separately from Carver, the compiler has per-op hardware specific tuning. For example, CUDA GEMM instruction selection chooses TCGEN05 on Blackwell when legal, WGMMA on Hopper when legal, otherwise MMA (`src/backend/cuda/op/gemm.cc:265`). Warp partitioning is heuristic over `GemmWarpPolicy`, warp count, and matrix dimensions (`src/backend/cuda/op/gemm.cc:117`, `src/backend/cuda/op/gemm.cc:187`,`src/backend/cuda/op/gemm.cc:289`).

```cpp
// src/backend/cuda/op/gemm.cc
if (AllowTcgen5Mma(op, target)) {
  return kCudaTCGEN05;
}
if (AllowWgmma(op, block_size, target)) {
  return kCudaWGMMA;
}
return kCudaMMA;
```

## 3. Tile Inference

Layout inference runs in `LowerAndLegalize` before tile-op lowering
(`tilelang/engine/phase.py:203`). The pass entry is `LayoutInference` in
`src/transform/layout_inference.cc:1288`.

The pass starts with user annotations from `T.annotate_layout`/`layout_map`
(`src/transform/layout_inference.cc:759`), then runs four main steps:

1. Give floating fragment buffers fully replicated layouts when they are used
   outside tile ops (`src/transform/layout_inference.cc:372`,
   `src/transform/layout_inference.cc:895`).
2. Run strict inference once over all operators (`src/transform/layout_inference.cc:383`).
3. Run common inference with a BFS queue over buffer-use edges
   (`src/transform/layout_inference.cc:393`).
4. Run free inference by connected component search
   (`src/transform/layout_inference.cc:396`, `src/transform/layout_inference.cc:1030`).

```cpp
// src/transform/layout_inference.cc
// step 0: set fully replicated layout for floating fragment buffers
for (const auto &[buffer, thread_bounds] : floating_fragment_buffers_) {
  if (layout_map.count(buffer))
    continue;
  auto frag =
      Fragment::FullyReplicated(buffer->shape, thread_bounds->extent);
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
```

The result is attached to each block as `attr::kLayoutMap`, and parallel loops
receive `attr::kParallelLoopLayout` plus an optional predicate
(`src/transform/layout_inference.cc:1222`, `src/transform/layout_inference.cc:1250`).
`LowerTileOp` later consumes those annotations to remap buffer shapes and indices
(`src/transform/lower_tile_op.cc:40`, `src/transform/lower_tile_op.cc:1009`,
`src/transform/lower_tile_op.cc:1178`, `src/transform/lower_tile_op.cc:1281`).

### Inference Graph Shape

`LayoutInference` constructs the graph implicitly while visiting the TIR body.
The nodes are inferable operations, and the edges are fragment buffers used by
multiple operations. Aliased buffers that share the same `data` `Var` are tied
together so a layout inferred for one view can be propagated to sibling views.

```text
BufferUseDefCollector

Tile op / Parallel loop #0
        |
        | fragment buffer A_local
        v
Tile op / Parallel loop #1
        |
        | fragment buffer C_local
        v
Tile op / Parallel loop #2

Derived indexes:

fragment buffer A_local ----> use_list_[A_local] = [0, 1]
fragment buffer C_local ----> use_list_[C_local] = [1, 2]
alias buffer view ----------> buffer_data_to_buffers_[var]

alias buffer view shares the same data Var as C_local, so layout updates can
propagate between the two views when their shapes are compatible.
```

The `infer_list_` ordering is the pass's worklist universe. When a buffer gets a
new layout, every operation in `use_list_[buffer]` becomes eligible to run
`InferLayout` again, because its local constraints may now be solvable.

```text
Start
  |
  v
Seed layouts
  - explicit annotations
  - floating fully replicated fragments
  |
  v
Strict inference
  - run each op once at kStrict
  - record strict_layout_map
  |
  v
Common inference
  - BFS from newly inferred buffers
  - propagate through use_list_
  |
  v
Free inference
  - build connected components
  - try each op as root
  - choose the plan with minimum fragment registers
  |
  v
Annotate IR
  - attach kLayoutMap to blocks
  - attach kParallelLoopLayout to loops
  |
  v
LowerTileOp
```

At a high level, the three inference levels answer different questions:

```text
Level     Main question                         Typical source of constraints
-------   ------------------------------------  --------------------------------
Strict    What layout must this op use?          GEMM/MMA shared/register layout
Common    What layout follows from neighbors?    copy/parallel/reduce propagation
Free      What layout is legal and cheapest?     synthesized loop partitions,
                                                minimum fragment register count
```

There are three sub-category of layout inference, according to TileLang's paper:

### Strict Layout Inference

Strict inference is operator-specific and handles hardware-sensitive constraints.
For example, GEMM's strict inference delegates to Python via `tl.gemm.infer_layout`
(`src/op/gemm.cc:229`, `tilelang/tileop/gemm/__init__.py:12`). CUDA MMA returns
swizzled shared-memory layouts for shared operands and MMA store/load fragment
layouts for register operands (`tilelang/cuda/op/gemm/gemm_mma.py:42`). WGMMA
and TCGEN05 choose full/half/quarter bank swizzles or linear layouts from the continuous dimension size (`tilelang/cuda/op/gemm/gemm_wgmma.py:28`,
`tilelang/cuda/op/gemm/gemm_tcgen05.py:46`). Accumulator layouts come from the tensor-core emitter's `make_mma_store_layout`.

For shared layouts, CUDA MMA may preserve an existing shared layout to avoid conflicts, while WGMMA/TCGEN05 always set strict shared layouts (`src/op/gemm.cc:237`, `src/backend/cuda/op/gemm.cc:305`). This is the concrete
implementation of "operator-specific constraints" for tensor-core primitives.

The strict phase is where hardware instructions constrain otherwise abstract
fragment buffers:

```text
Inputs to strict GEMM inference:

target + block_size
MMA / WGMMA / TCGEN05
        |
        v
tl.gemm.infer_layout
        |
        +--> shared operand layout for A
        |    swizzle or existing layout
        |          |
        |          v
        |      A_shared ----+
        |                   |
        +--> shared operand layout for B
        |    swizzle or existing layout
        |          |
        |          v
        |      B_shared ----+--> tl.gemm --> C_local
        |                                      ^
        +--> accumulator fragment layout       |
             mma store/load layout -----------+
```

```cpp
// src/op/gemm.cc
LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = Function::GetGlobal("tl.gemm.infer_layout")) {
    auto inferred_layouts = Downcast<LayoutMap>(
        (*f)(GetRef<Gemm>(this), T.target, T.thread_bounds));
    auto block_size = *as_const_int(T.thread_bounds->extent);
    String gemm_inst = getGemmInstructionKey(block_size, T.target);
    bool reuse_existing_shared_layout =
        ResolveGemmImpl(T.target).reuse_existing_shared_layout(gemm_inst);
```

### Common Layout Inference
The paper define the layout inference process as a "constrained propagation algorithm", and common inference is the BFS propagation phase. When an inferred layout is added for a buffer, all tile ops that use that buffer are enqueued (`src/transform/layout_inference.cc:296`). `ParallelOpNode::InferLayout` tries to derive the loop layout from an already-layouted source fragment, preferring non-replicated writes/reads and honoring explicit loop annotations (`src/op/parallel.cc:357`, `src/op/parallel.cc:397`). It then completes layouts for other touched buffers from that loop layout (`src/op/parallel.cc:490`).

```text
InferLayout(op)
        |
        | returns {buffer: layout}
layout_map
        |
        | propagate aliases sharing the same data Var
use_list_
        |
        | find ops that touch the newly layouted buffer
BFS queue
        |
        | enqueue dependent op ids
InferLayout(dependent op)
        |
        | rerun at kCommon with a richer layout_map
...
```

```cpp
// src/transform/layout_inference.cc
// Push back into BFS queue
for (int idx : use_list_[buffer]) {
  ICHECK_GE(idx, 0)
      << "Index in use_list_ for buffer " << buffer << " is negative.";
  ICHECK_LT(idx, num_infer)
      << "Index in use_list_ for buffer " << buffer
      << " out of range: " << idx << " >= " << num_infer << ".";

  EnqueueWithPriority(idx, q, in_queue, cur_infer_id, layout_map);
}
```

Reduction is a good example of structurally aligned propagation. `ReduceOp` does not infer in strict mode, but in common/free mode it derives the destination fragment layout from the source fragment by removing the reduced dimension and folding it into replication (`src/op/reduce.cc:169`, `src/op/reduce.cc:233`).

```text
Source fragment layout over [M, N, K]
        reduce over K
              |
              v
Destination fragment layout over [M, N]

The removed K dimension is not discarded entirely: its work is folded into
replication so the destination layout still describes which threads own the
post-reduction values.
```

### Free Layout Inference

Free inference handles remaining unconstrained components. `InferInFreeMode`
builds connected components with union-find over operators that share fragment
buffers and over alias buffers sharing the same storage `Var` (`src/transform/layout_inference.cc:1039`). For each component it tries every member as the inference root, runs free inference, catches layout/normalization
failures, and scores successful layouts by total fragment output shape size, used as register count (`src/transform/layout_inference.cc:1090`,
`src/transform/layout_inference.cc:1113`,
`src/transform/layout_inference.cc:1143`). The minimum-register plan wins (`src/transform/layout_inference.cc:1163`). So it's more like a brute-force enumeration process.

```cpp
// src/transform/layout_inference.cc
// Compute the total register number for this layout
int64_t reg_num = 0;
for (const auto &[buffer, layout] : tmp_layout_map) {
  if (auto frag = layout.as<Fragment>()) {
    int64_t frag_reg_num = 1;
    for (auto i : frag.value()->OutputShape()) {
      auto pci = as_const_int(i);
      ICHECK(pci != nullptr)
          << "Can not use non-constant range to "
             "iterate over a fragment/local "
             "buffer. Non-constant shape expr is: "
          << i
          << ". This is possibly because you use symbolic shape when "
             "accessing a fragment/local buffer.";
      frag_reg_num *= *pci;
    }
    reg_num += frag_reg_num;
  }
}
// Update the best plan if this one uses fewer registers
if (reg_num < min_reg_num ||
    (reg_num == min_reg_num &&
     attempt_infer_root < min_reg_num_infer_root)) {
```

Inside `ParallelOpNode`, free mode can synthesize a layout via `ComputePlanCandidate`/loop partitioning when no source layout exists, and it chooses between buffer-derived and plan-derived candidates
(`src/op/parallel.cc:413`).

The free phase is deliberately last because it is the most permissive. It
handles a connected component only after strict and common propagation fail to
settle every fragment layout:

```text
Unresolved connected component
ops connected by fragment buffers
        |
Pick attempt root
        |
Run root InferLayout at kFree
        |
BFS propagate kFree updates
        |
All component buffers resolved?
        |
        +-- no  --> Try another root
        |             |
        |        Pick attempt root
        |
        +-- yes --> Normalize layouts and count registers
                    |
               Lowest register count?
                    |
                    +-- no  --> Try another root
                    |
                    +-- yes --> Keep as best plan
                                  |
                           Commit best layout_map updates
```

This explains why free inference can recover from underconstrained user code
without silently choosing the first legal mapping it finds: each successful root
attempt is scored by the total output shape of fragment layouts, which is used
as a proxy for register pressure.
