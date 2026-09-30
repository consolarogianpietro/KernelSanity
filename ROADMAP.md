# Development plan

KernelSanity stays a small C++17 library for randomized differential testing of
numerical kernels. [FEATURES.md](FEATURES.md) records what is already implemented. Each milestone needs a runnable example, focused tests, and
updated documentation before it is considered complete. Existing seeded input
and saved-case behavior should remain reproducible.

1. **Bounded shrinking of saved failures — complete.** Add a deterministic,
   caller-budgeted search that tries smaller dimensions and simpler input values.
   Preserve seed, tolerances, and case index, and save the resulting failure in
   the existing artifact format. Document exactly how candidates are formed,
   how exceptions are handled, and that greedy shrinking does not prove global
   minimality.
2. **Boundary input generation — complete.** Add an explicit opt-in mode for signed zeros,
   subnormals, extreme finite values, infinities, and NaNs. Keep the current
   seeded uniform generator as the default. Test exact bit patterns and repeatability.
3. **Multiple-input adapter design.** Define the smallest useful representation
   for at least a reduction and GEMM case, including shapes, layouts, and strides.
   Implement it only after the replay and shrinking contract can preserve those
   inputs without ambiguity.
4. **Additional dtypes.** Define FP16 and BF16 storage, conversion, accumulation,
   and tolerance semantics before exposing either dtype. Add CPU reference tests
   before considering GPU adapters.
5. **GPU and coverage-guided fuzzing.** Treat these as separate integrations,
   with hardware-specific validation and clear claims about what each explores.

For each change: run CTest in a normal Linux build and under GCC and Clang
ASan/UBSan, then review the README and example output. A passing run is evidence
for tested inputs, not a proof that a kernel is correct.
