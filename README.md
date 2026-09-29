# KernelSanity

Property-based testing and fuzzing for high-performance numerical kernels.

An early C++17 prototype for differential testing of CPU FP32 kernels. Supply a
trusted reference and a candidate; KernelSanity generates inputs and compares
their outputs. No third-party dependencies.

```cpp
#include <kernelsanity/compare.hpp>

// Callables: ks::tensor(const ks::tensor&, const ks::shape&).
auto report = ks::compare(reference, optimized)
    .shapes({{1, 4096}, {17, 4096}, {128, 4096}})
    .random_shapes(10000) // default rank 2, dimensions 1..64
    .seed(42)
    .tolerance(1e-3, 1e-3) // absolute, relative
    .run();
report.save("failures", 42);
```

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/ks_example
```

Consumers can use `add_subdirectory` and link `ks::kernelsanity`.

## Current contract

- Both kernels receive the same contiguous FP32 input, sampled from [-1, 1).
- Finite outputs pass when `abs(reference - actual) <= atol + rtol * abs(reference)`.
- NaNs always fail. Equal signed infinities pass; other infinity comparisons fail.
- Output length mismatches fail. Kernel exceptions propagate to the caller.
- Shapes have positive dimensions and at most 1,048,576 elements per case.
- Random dimensions use a deterministic modulo mapping (slightly biased).
- A seed and identical configuration reproduce input generation. Saved text cases
  also include the exact input and both outputs, with round-trip float precision.
  Use the same seed in `save` as in the comparison. Files with the same seed/index
  are overwritten. All failures are retained in memory.
- This is randomized differential testing, not yet a coverage-guided fuzzer.
  A passing result is evidence for the tested inputs, not proof of correctness.

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

1. Artifact loader and standalone replay; keep seed metadata in the result.
2. Shape and value shrinking with clearly defined minimality and budgets.
3. Boundary inputs: signed zero, subnormals, extreme magnitudes and non-finite values.
4. Multiple-input adapters for GEMM/reductions, strides and layouts.
5. Explicit FP16/BF16 storage and accumulation semantics, then GPU adapters.

No FP16 support or smallest-failure claim is exposed in this first version.
