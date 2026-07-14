---
author: Retep
pubDatetime: 2026-07-14T12:00:00.000-07:00
title: "[TIL] A small note on layout computation"
featured: false
draft: false
tags:
  - TIL
description: ""
---

Mostly based off this [guide](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tirx_layout_api/index.html#chap-tirx-layout-api). As mentioned in the previous blog for tilelang layout inference, what layout entails is the way to derive from a logical tensor coordinate to the actual physical coordinate.
In a CUDA like model, the "scopes" for data placement are:

Scope|notation
|---|---|
|grid|/|
|block cluster| `(cbx, cby, cbz)`
|block | `(bx, by, bz)`
|warp group | `wgid` `tid_in_wg`, `wid_in_wg`
|warp| `warpid`
|thread/lane| `laneid`, `TLane`, `TCol` 

This means a physical coordinate is a coordinate in all these scope, for example `(block_1, warp_group_2, warp_3, lane_4)`. Some special coordinates are:
- `tid_in_wg`: the thread ID within a warp group. `tid_in_wg = threadIdx.x % 128`. A warp group consists of 4 warps, and a warp consists of 32 threads.
- `TLane`, `TCol`: stands for "thread lane" and "thread column". Representing which lane and column that participate in fragment/load/store layout. This is mostly used for `TMEM`.

There are also some special "logical coordinates", like replication, which is bascially another stand alone axis that represents how data is replicated across.

The notation for a layout is usually `(shape, stride)`. For example, `(8, 16), (16, 1)` means that there are 8 rows and 16 columns, and it's a row-major tensor. You can also do named axis: `(8, 16), (16@laneid, 1@warpid)`, which specifies the placement for each physical coordinate.

Then let's talk about how to convert a logical coordinate into physical one.
for a logical coordinate like
```
x = (x0, x1, ..., xr-1)
```
Assume we have the logical shape of
```
(S0, S1, ..., Sr-1)
```

We first convert them into flat index, which means compress it into a 1-D index:
```
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

Assume that the physical shape (or shard extent) is

```
(e0, e1, ..., en-1)
```

So we derive the physical coordinate `c0, c1, ..., cn-1` by the following:

```
c0 = flat / (e0 * e1 * ... * en-1) % e0
c1 = flat / (e1 * e2 * ... * en-1) % e1
c2 = flat / (e3 * e4 * ... * en-1) % e2
...
cn-1 = flat / 1 % en-1
```

For example, say we have logical shape `(8, 16)` and physical shard extent `S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]`. The flat index is 
```
flat = 16 * i + j
```

The physical coordinate is (the redundant mod omitted)
```
c0 = i
c1 = j / 8
c2 = j / 2 % 4
c3 = j % 2
```

Then with the coordinate and stride, we can compute the shard contribution:
```
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

This gives us the final placement:
```
laneid = 4 * i + j / 2 % 4
warpid = j / 8 
m      = j % 2
```
