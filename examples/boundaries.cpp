#include <kernelsanity/compare.hpp>
#include <iostream>
int main() {
    auto identity = [](const ks::tensor& input, const ks::shape&) { return input; };
    auto report = ks::compare(identity, identity)
        .shapes(std::vector<ks::shape>(12, ks::shape{1}))
        .boundary_values().seed(42).run();
    report.save("boundary-failures");
    std::cout << report.passed << " boundary cases passed, "
              << report.failures.size() << " failed (quiet NaNs always fail)\n";
    return report.passed == 10 && report.failures.size() == 2 ? 0 : 1;
}
