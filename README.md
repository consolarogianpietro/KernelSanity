# KernelSanity

Property-based testing and fuzzing for high-performance numerical kernels.

An early C++17 prototype for randomized differential testing of CPU FP32 kernels.
Supply a trusted reference and a candidate; KernelSanity generates inputs and
compares their outputs. No third-party dependencies.

```cpp
#include <kernelsanity/compare.hpp>

// Callables: ks::tensor(const ks::tensor&, const ks::shape&).
auto report = ks::compare(reference, optimized)
    .shapes({{1, 4096}, {17, 4096}, {128, 4096}})
    .random_shapes(10000) // default rank 2, dimensions 1..64
    .seed(42)
    .tolerance(1e-3, 1e-3) // absolute, relative
    .run();
report.save("failures");
```

`result` carries the seed and tolerances used by `run()`. Each failure is saved
as `case-SEED-INDEX.txt`; saving again to the same path overwrites it. To replay
an input with the same comparison settings, load the file and supply the kernels:

```cpp
auto saved = ks::load_case("failures/case-42-0.txt");
auto replayed = saved.replay(reference, optimized);
if (!replayed.ok()) {
    // The current kernels still disagree on the saved input.
}
```

`replay()` runs both supplied kernels on the saved input. Its result contains
fresh outputs; `saved.original` retains the recorded outputs for inspection.
The saved seed identifies the original run, but replay uses the stored input
rather than regenerating it. Kernel exceptions propagate to the caller.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/ks_example
./build/ks_replay_example
./build/ks_replay_example failures/case-42-0.txt
```

The replay example writes one known failure on its first invocation, then loads
and reproduces it. The optional path replays an existing artifact with the
example's kernels. Consumers can use `add_subdirectory` and link
`ks::kernelsanity`.

## Current contract

- Both kernels receive the same contiguous FP32 input, sampled from [-1, 1).
- Finite outputs pass when `abs(reference - actual) <= atol + rtol * abs(reference)`.
- NaNs always fail. Equal signed infinities pass; other infinity comparisons fail.
- Output length mismatches fail. Kernel exceptions propagate to the caller.
- Shapes have positive dimensions and at most 1,048,576 elements per case.
- Random dimensions use a deterministic modulo mapping (slightly biased).
- Artifact v2 stores seed, case index, absolute and relative tolerance, shape,
  input, and both outputs. FP32 values are stored as eight-digit hexadecimal bit
  patterns to preserve every bit. Input length must match the shape; each stored
  vector is limited to 1,048,576 values. Invalid, truncated, trailing, or
  unsupported-version data throws `std::runtime_error`. The earlier v1 format
  lacks tolerance metadata and is explicitly unsupported by `load_case()`.
- All failures are retained in memory. Saving writes one file per failure.
- This is randomized differential testing, not coverage-guided fuzzing. A
  passing result is evidence for the tested inputs, not proof of correctness.

## Linux / WSL development

WSL 2 with Ubuntu is a suitable development environment. Prefer a checkout in
the Linux filesystem (`~/src/KernelSanity`) for Linux build performance.

```sh
sudo apt update
sudo apt install -y build-essential cmake ninja-build clang git
git clone https://github.com/consolarogianpietro/KernelSanity.git ~/src/KernelSanity
cd ~/src/KernelSanity
```

Run the build/test commands above. CI exercises GCC and Clang with address and
undefined-behavior sanitizers on Linux. GPU adapters will need separate hardware
and driver validation.

## Next milestones

1. Shape and value shrinking with clearly defined minimality and budgets.
2. Boundary inputs: signed zero, subnormals, extreme magnitudes and non-finite values.
3. Multiple-input adapters for GEMM/reductions, strides and layouts.
4. Explicit FP16/BF16 storage and accumulation semantics, then GPU adapters.

No FP16 support or smallest-failure claim is exposed in this prototype.
