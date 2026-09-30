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

For FP32 boundary inputs, opt in with `.boundary_values()` before `.run()`.
This replaces uniform input values with a repeating corpus of signed zeros,
smallest subnormals and normals, largest finite values, infinities, and quiet
NaNs. Case index shifts the corpus, so twelve single-element cases cover it all:

```cpp
auto boundary_report = ks::compare(reference, optimized)
    .shapes(std::vector<ks::shape>(12, ks::shape{1}))
    .boundary_values().seed(42).run();
```

Boundary mode still advances the seeded generator for each input element, so
random shape sequences stay the same when switching modes. Quiet NaNs always
fail the comparison, even if both kernels return the same NaN.

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

A reproducing saved case can be shrunk with an explicit evaluation budget:

```cpp
auto shrunk = saved.shrink(reference, optimized, 1000);
if (!shrunk.report.ok()) shrunk.report.save("shrunk-failures");
```

The budget counts kernel-pair evaluations, including the initial replay. The
search keeps rank fixed, tries smaller dimensions in axis order, and retains the
flattened input prefix when shape size falls. It then tries replacing each value
with zero or halving it toward zero. Only candidates that still produce a
numerical or output-length mismatch are accepted. Candidate kernel exceptions
are rejected; an exception on the initial replay propagates. The search is
deterministic and greedy, with no global-minimality guarantee. A passing
`shrunk.report` means the saved failure no longer reproduces; its `evaluations`
and `budget_exhausted` fields show how much search ran. Save shrunk artifacts to
a different directory if you want to keep the original file.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/ks_example
./build/ks_replay_example
./build/ks_replay_example failures/case-42-0.txt
./build/ks_boundary_example
```

The replay example writes one known failure on its first invocation, then loads
and reproduces it. The optional path replays an existing artifact with the
example's kernels. Consumers can use `add_subdirectory` and link
`ks::kernelsanity`.

## Current contract

- Both kernels receive the same contiguous FP32 input. Uniform values in
  [-1, 1) are the default; the fixed boundary corpus is opt-in.
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

## Development plan

See [FEATURES.md](FEATURES.md) for the implemented feature inventory and
[ROADMAP.md](ROADMAP.md) for sequenced milestones and validation gates.
No FP16 support or smallest-failure claim is exposed in this prototype.
