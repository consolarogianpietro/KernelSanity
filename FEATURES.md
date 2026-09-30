# Accomplished features

This is the project's local inventory of implemented behavior. Update it when a
feature lands, then use it with [ROADMAP.md](ROADMAP.md) to reconsider priorities.
Last reviewed: 2026-09-30.

| Area | Implemented behavior | Limits |
| --- | --- | --- |
| Core API | Header-only C++17 `ks::compare(reference, optimized)` for one contiguous FP32 input and FP32 output per kernel. | The original API remains single-input; multiple inputs use a separate explicit-case adapter. |
| Case generation | Validated explicit shapes and seeded rank/dimension generation; deterministic uniform input values in [-1, 1). | Maximum 1,048,576 input values; modulo-selected random dimensions are slightly biased. |
| Boundary cases | Opt-in `.boundary_values()` cycles exact FP32 signed zeros, smallest subnormals, smallest normals, largest finite values, infinities, and quiet NaNs. | Fixed corpus and order; all elements use boundary values in this mode. |
| Comparison | Absolute/relative tolerance, output-length checks, NaN failure, and matching infinities. | Kernel exceptions propagate during normal comparison. |
| Saved cases | Version 2 text artifacts record seed, case index, tolerances, shape, input, and both outputs as exact FP32 bits; loader validates and replays stored input. | Version 1 loading is unsupported; saving overwrites a matching filename. |
| Shrinking | A saved failure can be greedily reduced by shape and input value under an explicit evaluation budget, then saved and replayed. | Rank is fixed, shape reduction keeps the flattened prefix, and global minimality is not guaranteed. |
| Multiple-input cases | Explicit 1..16 operand FP32 cases with physical storage, positive strides, offsets, v3 save/replay, and budgeted value shrinking. | No multi-input random generation, output shape metadata, or layout/shape shrinking. |
| Validation | CTest, runnable comparison/replay/boundary/multi-input examples, and Linux GCC/Clang ASan/UBSan CI. | Hardware-specific GPU checks are not present. |

The project is randomized differential testing, not coverage-guided fuzzing.
FP16, BF16, random multi-input generation, and GPU execution remain planned work.
