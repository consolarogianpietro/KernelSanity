# Explicit multiple-input cases

`<kernelsanity/multi_compare.hpp>` adds `ks::compare_multi(reference, optimized)`
for a small set of explicit FP32 cases. The original one-input `ks::compare` API
and its v2 artifacts remain unchanged. No multi-input random generator is
exposed yet.

## Operand model

Each `ks::operand` owns physical `storage`, logical `dimensions`, element
`strides`, and a starting `offset`. For logical coordinates `(i, j)`, a rank-2
operand reads `storage[offset + i*strides[0] + j*strides[1]]`. Strides and offset
are measured in FP32 elements. Storage includes any padding; kernels receive
all of it as immutable input. Positive strides may overlap, so aliased logical
positions are allowed. Negative and zero strides are not supported.

A case is a `ks::multi_input` (vector of operands). Each case needs 1..16
operands, each operand needs positive dimensions, and the total physical
storage must contain at most 1,048,576 FP32 values. Each logical view must fit
inside its own storage. The library checks those bounds and integer overflow,
but leaves relationships between operands, such as GEMM's shared K dimension,
to the kernels. Outputs remain flat `ks::tensor` vectors with no output shape
metadata. This is enough for reductions and GEMM comparisons, but not a full
tensor-layout abstraction.

`examples/multi_compare.cpp` runs a strided reduction and GEMM. Its A operand
is row-major 2x3 with strides `{3, 1}`. B is column-major 3x2 with strides
`{1, 3}`. The logical B values are `[[7, 8], [9, 10], [11, 12]]`; its physical
storage is `{7, 9, 11, 8, 10, 12}`. The reference output is
`{58, 64, 139, 154}` in row-major order.

## Replay and shrinking

A failure saved by `ks::multi_result::save()` uses `KernelSanity-v3` and a
`multi-case-INDEX.txt` filename. The file records case index, absolute and
relative tolerances, operand count, each operand's shape/strides/offset/storage,
and both output vectors. FP32 storage and outputs use eight-digit hexadecimal
bit patterns. `ks::load_multi_case()` validates the entire file and rejects
v2, truncated, invalid-view, and trailing data. Replay runs the supplied
kernels on the saved physical storage and descriptors.

`ks::saved_multi_case::shrink_values(reference, optimized, budget)` can reduce
physical storage values to zero or half toward zero. It preserves operand
count, shapes, strides, offsets, and storage lengths. The budget counts the
initial replay, candidate kernel exceptions are rejected, and no global
minimality is guaranteed. Shrinking logical shapes or changing layout is left
for a later milestone because changing a view can alter which physical values
are reachable. Save shrunk results to a different directory to keep originals.

The v3 format has no seed because this API accepts explicit cases only. When
random generation is added, its metadata and format version must be reviewed
before extending the API.
