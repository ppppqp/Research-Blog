---
author: Retep
pubDatetime: 2026-08-17T12:00:00.000-07:00
title: "[TIL] Triton Linear Layout"
featured: false
draft: false
tags:
  - TIL
  - Triton
description: ""
---

# Triton LinearLayout

Similar to TileLang, Triton tensors are logical tensors and programed in blocked level. This means that you can simply write `tl.dot` in a triton program, but under the hood multiple threads coorporatively realize that operator, each owning a portion of tensor elements. Before lowering them to LLVM IR, the compiler must answer a physical question: which register, lane, warp, and CTA owns each tensor element?

Triton's `LinearLayout` provides a common representation for that answer. It is
a function:

```text
(register, lane, warp, block) → (tensor dimension 0, tensor dimension 1, ...)
```

If `L(register=2, lane=7, warp=1) = (12, 7)`, that hardware location contains
logical tensor element `[12, 7]`.


## Linear Layout
At first sight I thought linear layout is similar to affine mapping, but it's not the case. An affine map uses ordinary integer arithmetic:

```text
(i, j) → (4*i + j + 3, j floordiv 2)
```

A `LinearLayout` treats coordinates as bit vectors and represents a linear map over GF(2), the two-element field. In simple words, it's a vector space where the possible values are just `{0, 1}`. In this space, addition is XOR and multiplication by a coefficient is AND. The linearity is represented by

```text
L(x XOR y) = L(x) XOR L(y)
```

This is backed by the fact that for addition, we have 
- `0 + 0 = 0`, and `0 xor 0 = 0`
- `0 + 1 = 1`, and `0 xor 1 = 1`
- `1 + 1 = 10`, and `1 xor 1 = 0` (stripping the major 1 because it's a 1 bit space)
 
This model naturally expresses bit permutations, broadcasts, warp/lane
distribution, MMA fragments, and XOR shared-memory swizzles. It generally requires power-of-two dimension sizes and has no constant offset: `L(0) = 0`. Base pointers and offsets are modeled separately.

## Basis-vector representation

The core data member is:

```cpp
// LinearLayout.h
llvm::MapVector<
  StringAttr,
    std::vector<std::vector<int32_t>>
> bases;
```

It's a map of "dimension name" to a list of base vectors. Each innermost vector contains one coordinate for every output dimension. For input dimensions `lane` and `warp`, and output dimensions `row` and `col`, consider

```text
lane bit 0: L(lane=1, warp=0) = (1,1)
lane bit 1: L(lane=2, warp=0) = (2,2)
warp bit 0: L(lane=0, warp=1) = (0,1)
warp bit 1: L(lane=0, warp=2) = (0,2)
```

The corresponding structure is conceptually:

```cpp
bases["lane"] = {{1, 1}, {2, 2}};
bases["warp"] = {{0, 1}, {0, 2}};
```

The linear layout it describes is

```text
L(lane, warp) = (lane, lane XOR warp)
```
You can observe that lane is already the same as the row coordinate, and for column coordinate,
- `1 xor 0 = 1`
- `2 xor 0 = 2`
- `0 xor 1 = 1`
- `0 xor 2 = 2`

Only four values are stored, but they define the complete 4×4 input space. `MapVector` provides lookup by dimension name while retaining deterministic minor-to-major dimension order for printing, reshaping, and matrix conversion.

## `LinearLayout::apply`

```cpp
// LinearLayout.cpp
SmallVector<std::pair<StringAttr, int32_t>>
LinearLayout::apply(ArrayRef<std::pair<StringAttr, int32_t>> ins) const {
  assertDimsEqualIgnoringOrder(llvm::make_first_range(ins), getInDimNames());

  SmallVector<std::pair<StringAttr, int32_t>> ret;
  for (StringAttr outDim : getOutDimNames()) {
    int32_t outVal = 0;
    for (auto &[inDim, val] : ins) {
      for (int i = 0; i < getInDimSizeLog2(inDim); i++) {
        if (val & (1 << i))
          outVal ^= getBasis(inDim, i, outDim);
      }
    }
    ret.push_back({outDim, outVal});
  }
  return ret;
}
```
The logic is basically
```
for each output coordinate:
  for each input coordinate:
    for each bit in this input coordinate:
      if this bit is set:
        output coordinate xor with that base vector (as contribution)
```
Think about the input as a set of coefficient for base vectors. E.g. if input is `(3, 2)`, it means the output coordinate is conceptually `3 * base_0 + 2 * base_1`. However, because it's GF(2), we need to replace `* ` with `AND` and `+` with `XOR`. This is how we derive `val & (1 << i)` and `outVal ^= getBasis(inDim, i, outDim);`. Notice the `AND` is distributed, so it's not a conventional `AND`: it represents the which base vector participates. So it's like

```
(0, 1) and (base_00 xor base_01)
xor
(1, 1) and (base_10 xor base_11)
```
is actually
```
(0 * base_00) xor (1 * base_01)
xor
(1 * base_01) xor (1 * base_11)
```


Let's walk through an example. Consider the input we are interested in is
```text
lane = 3 = 0b11
warp = 2 = 0b10
```

`val & (1 << i)` tests whether input bit `i` is set. A set bit means the corresponding GF(2) coefficient is one, so its basis vector participates in the result. For `lane=3`, bits 0 and 1 select:
```text
(1,1) XOR (2,2) = (3,3)
```

For `warp=2`, only bit 1 selects:

```text
(0,2)
```

Combining all selected bases gives:

```text
(3,3) XOR (0,2) = (3,1)
```

Thus:

```text
L(lane=3, warp=2) = (row=3, col=1)
```

The implementation computes each output coordinate separately:

```text
row = 1 XOR 2 XOR 0 = 3
col = 1 XOR 2 XOR 2 = 1
```

This is matrix-vector multiplication over GF(2), performed directly from the matrix columns rather than by materializing the full matrix.

## Blocked-layout example

Consider the GEMM layout:

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [1, 32],
  warpsPerCTA = [4, 1],
  order = [1, 0]
}>
```

For a 32×32 tensor, its `LinearLayout` is approximately:

```text
register bases: (4,0), (8,0), (16,0)
lane bases:     (0,1), (0,2), (0,4), (0,8), (0,16)
warp bases:     (1,0), (2,0)
```

The coverage is:

```text
8 registers/thread × 32 lanes × 4 warps = 1024 = 32×32
```

For `(register=5, lane=3, warp=2)`:

```text
register 5 = 0b101 → (4,0) XOR (16,0) = (20,0)
lane     3 = 0b011 → (0,1) XOR (0,2)   = (0,3)
warp     2 = 0b010 → (2,0)

result = (20,0) XOR (0,3) XOR (2,0) = (22,3)
```

That register holds logical tensor element `[22,3]`.

## Surjectivity, injectivity, and broadcasts

The constructor calls `checkInvariants`, which verifies basis sizes, power-of-two
output dimensions, valid basis coordinates, and—when requested—surjectivity.

A layout is surjective when every logical tensor element is represented by at
least one hardware location. It is injective when no two hardware locations map
to the same tensor element. Triton computes these properties from the rank of
the GF(2) matrix, using row reduction from `f2reduce`:

```cpp
bool isSurjective() const {
  return rank == getTotalOutDimSizeLog2();
}

bool isInjective() const {
  return rank == getTotalInDimSizeLog2();
}
```

Layouts need not be injective. Zero bases represent broadcasting. If every
`warp` basis is zero, changing the warp ID does not change the logical tensor
coordinate, so the same values are duplicated across warps.

## Combining layouts

`LinearLayout` supports several algebraic operations:

- `compose(outer)` constructs `outer ∘ this` by applying `outer` to every basis
  vector of `this`.
- `operator*` forms a direct-sum-like product across dimensions; despite the
  notation, it is not function composition.
- `transposeIns`, `transposeOuts`, `reshapeIns`, and `reshapeOuts` reorganize
  named dimensions and their bits.
- `sublayout` restricts the input and output dimensions.
- `divideLeft` and `divideRight` factor compatible layouts.

Composition is simple because a linear function is fully defined by its bases:

```cpp
for (each basis of this)
  newBasis = outer.apply(basis);
```

## `invertAndCompose`: derive data movement

`invertAndCompose` is central to lowering memory and layout conversions.
Suppose:

```text
R: (register, lane, warp) → tensor coordinate
S: shared-memory offset   → tensor coordinate
```

To find the shared-memory address for a register value, Triton needs:

```text
(register, lane, warp)
  ──R──> tensor coordinate
  ──S⁻¹─> shared offset
```

This is computed by:

```cpp
R.invertAndCompose(S)
```

Layouts can contain broadcasts and therefore may not have ordinary inverses.
The implementation solves a GF(2) linear system and chooses a suitable
pseudoinverse. See `LinearLayout::invertAndCompose` and its focused tests in
`LinearLayoutTest.cpp`.


## From layout to LLVM IR

TTGIR encodings are first normalized with `toLinearLayout`. Lowering then calls
`applyLinearLayout` to emit the integer bit operations corresponding to `LinearLayout::apply`.

The effects depend on the encoding:

```text
#blocked → thread/lane index arithmetic and global-memory addresses
#shared  → swizzled shared-memory offsets
#dot_op  → ldmatrix addresses and register fragments
#mma     → per-lane MMA inputs and accumulator registers
```

For `ttg.convert_layout`, source and destination encodings become two
`LinearLayout`s. Their relationship determines the implementation:

```text
same-thread ownership → reorder registers
different lanes       → warp shuffles
different warps       → shared-memory exchange and barriers
```

The layouts themselves do not survive as LLVM metadata. They are compiled into
per-thread struct types, index arithmetic, register permutations, shared-memory
addresses, shuffles, `ldmatrix`, and MMA instructions.
