---
author: Retep
pubDatetime: 2026-08-16T12:00:00.000-07:00
title: "[TIL] Triton Kernel to LLVM IR Walkthrough"
featured: false
draft: false
tags:
  - TIL
description: ""
---

This post follows a minimal vector-add kernel through Triton's compiler:

```text
Python DSL → TTIR → TTGIR → LLVM IR → PTX
```

The main lesson is that Triton tensors are logical tensors, not hardware
vectors. TTGIR assigns their elements to GPU threads; LLVM IR then describes the
scalar work performed by each thread.

## The kernel

```python
@triton.jit
def add_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, values, mask=mask)
```

Compile it with `n = BLOCK = 128`, four warps, and an SM80 target. Both
compile-time arguments disappear from the function signature and become
constants.

## TTIR: logical tensor computation

The essential TTIR is:

```mlir
%n = arith.constant dense<128> : tensor<128xi32>
%block = arith.constant 128 : i32
%pid = tt.get_program_id x : i32
%base = arith.muli %pid, %block : i32
%range = tt.make_range {start = 0, end = 128} : tensor<128xi32>
%base_vector = tt.splat %base : i32 -> tensor<128xi32>
%offsets = arith.addi %base_vector, %range : tensor<128xi32>
%mask = arith.cmpi slt, %offsets, %n : tensor<128xi32>

%x_vector = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
%x_ptrs = tt.addptr %x_vector, %offsets
%x_values = tt.load %x_ptrs, %mask

%y_vector = tt.splat %y : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
%y_ptrs = tt.addptr %y_vector, %offsets
%y_values = tt.load %y_ptrs, %mask

%values = arith.addf %x_values, %y_values : tensor<128xf32>
%out_vector = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
%out_ptrs = tt.addptr %out_vector, %offsets
tt.store %out_ptrs, %values, %mask
```

Some details are easy to misread:

- `dense<128> : tensor<128xi32>` means 128 `i32` elements, each with value
  128. The attribute supplies the value; the result type supplies shape and
  element type.
- `tt.splat` explicitly broadcasts a scalar into a logical tensor.
- `tt.addptr` performs element-scaled pointer arithmetic.
- The type printed by `arith.cmpi` describes its operands; its result contains
  `i1` elements. Similarly, the printed type on `tt.load` describes its pointer
  operand, while its result is inferred from the pointee type.

At this stage, `tensor<128xf32>` says only that there are 128 logical values. It
does not say which threads compute them.

Complete IR:
```
#loc = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":8:1)
#loc11 = loc("x"(#loc))
#loc12 = loc("y"(#loc))
#loc13 = loc("out"(#loc))
module {
  tt.func public @add_kernel(%x: !tt.ptr<f32> loc("x"(#loc)), %y: !tt.ptr<f32> loc("y"(#loc)), %out: !tt.ptr<f32> loc("out"(#loc))) attributes {noinline = false} {
    %mask = arith.constant dense<128> : tensor<128xi32> loc(#loc14)
    %c128_i32 = arith.constant 128 : i32 loc(#loc2)
    %offsets = tt.get_program_id x : i32 loc(#loc15)
    %offsets_0 = arith.muli %offsets, %c128_i32 : i32 loc(#loc15)
    %offsets_1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32> loc(#loc16)
    %offsets_2 = tt.splat %offsets_0 : i32 -> tensor<128xi32> loc(#loc15)
    %offsets_3 = arith.addi %offsets_2, %offsets_1 : tensor<128xi32> loc(#loc15)
    %mask_4 = arith.cmpi slt, %offsets_3, %mask : tensor<128xi32> loc(#loc14)
    %values = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>> loc(#loc17)
    %values_5 = tt.addptr %values, %offsets_3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc(#loc17)
    %values_6 = tt.load %values_5, %mask_4 : tensor<128x!tt.ptr<f32>> loc(#loc18)
    %values_7 = tt.splat %y : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>> loc(#loc19)
    %values_8 = tt.addptr %values_7, %offsets_3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc(#loc19)
    %values_9 = tt.load %values_8, %mask_4 : tensor<128x!tt.ptr<f32>> loc(#loc20)
    %values_10 = arith.addf %values_6, %values_9 : tensor<128xf32> loc(#loc18)
    %0 = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>> loc(#loc9)
    %1 = tt.addptr %0, %offsets_3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc(#loc9)
    tt.store %1, %values_10, %mask_4 : tensor<128x!tt.ptr<f32>> loc(#loc10)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":10:12)
#loc2 = loc(unknown)
#loc3 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":9:15)
#loc4 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":9:42)
#loc5 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:22)
#loc6 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:14)
#loc7 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:56)
#loc8 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:48)
#loc9 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":12:14)
#loc10 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":12:5)
#loc14 = loc("mask"(#loc1))
#loc15 = loc("offsets"(#loc3))
#loc16 = loc("offsets"(#loc4))
#loc17 = loc("values"(#loc5))
#loc18 = loc("values"(#loc6))
#loc19 = loc("values"(#loc7))
#loc20 = loc("values"(#loc8))
```


## TTGIR: assign values to threads

For this kernel, TTGIR looks almost identical. Its important addition is a
layout encoding:

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [4],
  order = [0]
}>
```

The unencoded TTIR type:

```mlir
tensor<128xf32>
```

becomes:

```mlir
tensor<128xf32, #blocked>
```

`tensor` is MLIR's builtin ranked-tensor type. Ranked tensors deliberately have
an optional encoding field, and `#blocked` is a TritonGPU dialect attribute
stored in that field. It is compile-time type information, not an SSA operand or
runtime value.

The layout covers the tensor exactly:

```text
1 element/thread × 32 threads/warp × 4 warps/CTA = 128 elements
```

Thus the ownership is effectively:

```text
tensor element = warp_id × 32 + lane_id = threadIdx.x
```

TTGIR also records the execution configuration:

```mlir
module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  ttg.target = "cuda:80"
}
```

No `ttg.convert_layout` or shared-memory operations are needed because all
values use the same layout and no thread needs another thread's value. More
complex kernels—especially reductions and matrix multiplication—make TTGIR look
substantially different from TTIR.


Complete IR:

```
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":8:1)
#loc11 = loc("x"(#loc))
#loc12 = loc("y"(#loc))
#loc13 = loc("out"(#loc))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @add_kernel(%x: !tt.ptr<f32> loc("x"(#loc)), %y: !tt.ptr<f32> loc("y"(#loc)), %out: !tt.ptr<f32> loc("out"(#loc))) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %cst = arith.constant dense<128> : tensor<128xi32, #blocked> loc(#loc1)
    %offsets = tt.get_program_id x : i32 loc(#loc14)
    %offsets_0 = arith.muli %offsets, %c128_i32 : i32 loc(#loc14)
    %offsets_1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked> loc(#loc15)
    %offsets_2 = tt.splat %offsets_0 : i32 -> tensor<128xi32, #blocked> loc(#loc14)
    %offsets_3 = arith.addi %offsets_2, %offsets_1 : tensor<128xi32, #blocked> loc(#loc14)
    %mask = arith.cmpi slt, %offsets_3, %cst : tensor<128xi32, #blocked> loc(#loc16)
    %values = tt.splat %x : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked> loc(#loc17)
    %values_4 = tt.addptr %values, %offsets_3 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked> loc(#loc17)
    %values_5 = tt.load %values_4, %mask : tensor<128x!tt.ptr<f32>, #blocked> loc(#loc18)
    %values_6 = tt.splat %y : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked> loc(#loc19)
    %values_7 = tt.addptr %values_6, %offsets_3 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked> loc(#loc19)
    %values_8 = tt.load %values_7, %mask : tensor<128x!tt.ptr<f32>, #blocked> loc(#loc20)
    %values_9 = arith.addf %values_5, %values_8 : tensor<128xf32, #blocked> loc(#loc18)
    %0 = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked> loc(#loc9)
    %1 = tt.addptr %0, %offsets_3 : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked> loc(#loc9)
    tt.store %1, %values_9, %mask : tensor<128x!tt.ptr<f32>, #blocked> loc(#loc10)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":9:15)
#loc3 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":9:42)
#loc4 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":10:12)
#loc5 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:22)
#loc6 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:14)
#loc7 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:56)
#loc8 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":11:48)
#loc9 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":12:14)
#loc10 = loc("/home/qiping-pan/Documents/workspace/triton/triton_ir_walkthrough.py":12:5)
#loc14 = loc("offsets"(#loc2))
#loc15 = loc("offsets"(#loc3))
#loc16 = loc("mask"(#loc4))
#loc17 = loc("values"(#loc5))
#loc18 = loc("values"(#loc6))
#loc19 = loc("values"(#loc7))
#loc20 = loc("values"(#loc8))
```

## LLVM IR: one scalar program per thread

After TTGIR lowering, tensor types and layouts disappear. The kernel requires
128 threads:

```llvm
attributes #0 = { nounwind "nvvm.reqntid"="128" }
```

Its first three parameters are `x`, `y`, and `out` in global address space 1.
Triton also appends global-scratch and profile-scratch pointers, unused here.

The logical offsets become scalar index arithmetic:

```llvm
%pid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%base = shl i32 %pid, 7
%tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%local = and i32 %tid, 127
%offset = or disjoint i32 %base, %local
%mask = icmp slt i32 %offset, 128
```

This is equivalent to:

```c
int offset = blockIdx.x * 128 + threadIdx.x;
bool mask = offset < 128;
```

Multiplication by 128 becomes a shift by seven. The final addition becomes
`or disjoint` because the shifted program ID and seven-bit local index cannot
have overlapping set bits.

Each pointer tensor becomes one address per thread:

```llvm
%index64 = sext i32 %offset to i64
%x_ptr = getelementptr [4 x i8], ptr addrspace(1) %x, i64 %index64
```

Indexing `[4 x i8]` advances four bytes per element, the size of `f32`.

Triton emits predicated global memory operations as inline PTX. A load is
essentially:

```ptx
mov.u32 result, 0;
@predicate ld.global.b32 result, [address];
```

The zero initialization defines the masked-out value. The loaded raw `i32` bits
are bitcast to `float`, the addition becomes one scalar instruction per thread,
and the result is bitcast back for the store:

```llvm
%x_float = bitcast i32 %x_bits to float
%y_float = bitcast i32 %y_bits to float
%sum = fadd float %x_float, %y_float
%sum_bits = bitcast float %sum to i32
```

The masked store is:

```ptx
@predicate st.global.b32 [address], value;
```

The complete per-thread behavior is therefore:

```c
int offset = blockIdx.x * 128 + threadIdx.x;
bool mask = offset < 128;

float xv = mask ? x[offset] : 0.0f;
float yv = mask ? y[offset] : 0.0f;
float result = xv + yv;

if (mask)
    out[offset] = result;
```

Complete IR:
```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

; Function Attrs: nounwind
define ptx_kernel void @add_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree readnone captures(none) %3, ptr addrspace(1) nofree readnone captures(none) %4) local_unnamed_addr #0 !dbg !4 {
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !11
  %7 = shl i32 %6, 7, !dbg !11
  %8 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !12
  %9 = and i32 %8, 127, !dbg !12
  %10 = or disjoint i32 %7, %9, !dbg !11
  %11 = icmp slt i32 %10, 128, !dbg !13
  %12 = sext i32 %10 to i64, !dbg !14
  %13 = getelementptr [4 x i8], ptr addrspace(1) %0, i64 %12, !dbg !14
  %14 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b"(ptr addrspace(1) %13, i1 %11) #2, !dbg !15
  %15 = bitcast i32 %14 to float, !dbg !15
  %16 = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %12, !dbg !16
  %17 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b"(ptr addrspace(1) %16, i1 %11) #2, !dbg !17
  %18 = bitcast i32 %17 to float, !dbg !17
  %19 = fadd float %15, %18, !dbg !15
  %20 = getelementptr [4 x i8], ptr addrspace(1) %2, i64 %12, !dbg !18
  %21 = bitcast float %19 to i32, !dbg !19
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %21, ptr addrspace(1) %20, i1 %11) #2, !dbg !19
  ret void, !dbg !20
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { nounwind "nvvm.reqntid"="128" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "triton_ir_walkthrough.py", directory: "/home/qiping-pan/Documents/workspace/triton")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
!4 = distinct !DISubprogram(name: "add_kernel", linkageName: "add_kernel", scope: !1, file: !1, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{null, !7, !7, !7, !9, !9}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !8, size: 64, dwarfAddressSpace: 1)
!8 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !10, size: 64, dwarfAddressSpace: 1)
!10 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!11 = !DILocation(line: 9, column: 15, scope: !4)
!12 = !DILocation(line: 9, column: 42, scope: !4)
!13 = !DILocation(line: 10, column: 12, scope: !4)
!14 = !DILocation(line: 11, column: 22, scope: !4)
!15 = !DILocation(line: 11, column: 14, scope: !4)
!16 = !DILocation(line: 11, column: 56, scope: !4)
!17 = !DILocation(line: 11, column: 48, scope: !4)
!18 = !DILocation(line: 12, column: 14, scope: !4)
!19 = !DILocation(line: 12, column: 5, scope: !4)
!20 = !DILocation(line: 8, column: 1, scope: !4)

```

## The vertical mapping

| Triton source | TTIR/TTGIR | LLVM/PTX |
|---|---|---|
| `tl.program_id(0)` | `tt.get_program_id x` | `ctaid.x` |
| `tl.arange(0, 128)` | `tt.make_range`, `#blocked` | `tid.x` |
| `pid * 128 + range` | `arith.muli`, `arith.addi` | `shl`, `or disjoint` |
| `offsets < n` | `arith.cmpi slt` | `icmp slt` / PTX predicate |
| `x + offsets` | `tt.splat`, `tt.addptr` | scalar `getelementptr` |
| `tl.load` | `tt.load` | predicated `ld.global.b32` |
| tensor addition | `arith.addf` | scalar `fadd` per thread |
| `tl.store` | `tt.store` | predicated `st.global.b32` |

The critical boundary is TTIR to TTGIR: logical tensors gain layout encodings.
Once ownership is known, lowering can replace every tensor operation in this
kernel with scalar work distributed across 128 GPU threads.
