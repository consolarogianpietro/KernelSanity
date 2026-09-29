#include <kernelsanity/compare.hpp>
#include <iostream>
int main() {
    auto reference = [](const ks::tensor& x, const ks::shape&) {
        auto y = x;
        for (auto& v : y) v *= 2;
        return y;
    };
    auto candidate = reference;
    auto report = ks::compare(reference, candidate)
        .shapes({{1, 4096}, {17, 4096}, {128, 4096}})
        .random_shapes(1000).seed(42).tolerance(1e-6, 1e-5).run();
    report.save("failures", 42);
    std::cout << report.passed << " tests passed\n"
              << report.failures.size() << " numerical failures\n";
    return report.ok() ? 0 : 1;
}
