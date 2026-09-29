#pragma once
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ks {
using shape = std::vector<std::size_t>;
using tensor = std::vector<float>;
// Each kernel receives the same immutable input and returns its output.
using kernel = std::function<tensor(const tensor&, const shape&)>;
struct failure {
    shape dimensions;
    tensor input, expected, actual;
    std::size_t case_index;
};
struct result {
    std::size_t passed = 0;
    std::vector<failure> failures;
    bool ok() const { return failures.empty(); }
    // Text format v1: seed/index, shape, then input/reference/actual vectors.
    void save(const std::filesystem::path& directory, std::uint32_t seed) const {
        if (failures.empty()) return;
        std::filesystem::create_directories(directory);
        for (const auto& f : failures) {
            auto path = directory / ("case-" + std::to_string(seed) + "-" + std::to_string(f.case_index) + ".txt");
            std::ofstream out(path);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << "KernelSanity-v1\n" << seed << ' ' << f.case_index << '\n';
            out << f.dimensions.size();
            for (auto d : f.dimensions) out << ' ' << d;
            out << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
            for (const auto* values : {&f.input, &f.expected, &f.actual}) {
                out << values->size();
                for (auto v : *values) out << ' ' << v;
                out << '\n';
            }
        }
    }
};
class comparison {
    kernel reference_, optimized_;
    std::vector<shape> shapes_;
    std::size_t random_count_ = 0, rank_ = 2, max_dim_ = 64;
    std::uint32_t seed_ = 0;
    double atol_ = 1e-5, rtol_ = 1e-5;
    static std::size_t elements(const shape& s) {
        if (s.empty()) throw std::invalid_argument("shape must have at least one dimension");
        std::size_t n = 1;
        for (auto d : s) {
            if (!d || d > 1048576 / n) throw std::invalid_argument("shape must contain 1..1048576 elements");
            n *= d;
        }
        return n;
    }
    bool equal(float a, float b) const {
        if (std::isnan(a) || std::isnan(b)) return false;
        if (a == b) return true; // Includes matching signed infinities and zeros.
        if (!std::isfinite(a) || !std::isfinite(b)) return false;
        return std::abs(double(a) - double(b)) <= atol_ + rtol_ * std::abs(double(a));
    }
public:
    comparison(kernel reference, kernel optimized)
        : reference_(std::move(reference)), optimized_(std::move(optimized)) {
        if (!reference_ || !optimized_) throw std::invalid_argument("kernels must be callable");
    }
    comparison& shapes(std::vector<shape> values) {
        for (const auto& s : values) elements(s);
        shapes_ = std::move(values); return *this;
    }
    comparison& random_shapes(std::size_t count, std::size_t rank = 2, std::size_t max_dimension = 64) {
        if (!rank || rank > 20 || !max_dimension) throw std::invalid_argument("invalid random shape bounds");
        elements(shape(rank, max_dimension));
        random_count_ = count; rank_ = rank; max_dim_ = max_dimension; return *this;
    }
    comparison& seed(std::uint32_t value) { seed_ = value; return *this; }
    comparison& tolerance(double absolute, double relative = 0) {
        if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0 || relative < 0)
            throw std::invalid_argument("tolerances must be finite and nonnegative");
        atol_ = absolute; rtol_ = relative; return *this;
    }
    result run() const {
        if (shapes_.empty() && !random_count_) throw std::invalid_argument("no test cases configured");
        std::mt19937 rng(seed_);
        result report;
        auto check = [&](const shape& s, std::size_t index) {
            tensor input(elements(s));
            // Defined mapping avoids implementation-dependent uniform distributions.
            for (auto& v : input) v = float(rng() >> 8) / 8388608.0f - 1.0f;
            auto expected = reference_(input, s);
            auto actual = optimized_(input, s);
            bool matches = expected.size() == actual.size();
            for (std::size_t i = 0; matches && i < expected.size(); ++i) matches = equal(expected[i], actual[i]);
            if (matches) ++report.passed;
            else report.failures.push_back({s, std::move(input), std::move(expected), std::move(actual), index});
        };
        std::size_t index = 0;
        for (const auto& s : shapes_) check(s, index++);
        for (std::size_t i = 0; i < random_count_; ++i) {
            shape s(rank_);
            for (auto& d : s) d = 1 + rng() % max_dim_;
            check(s, index++);
        }
        return report;
    }
};
inline comparison compare(kernel reference, kernel optimized) {
    return comparison(std::move(reference), std::move(optimized));
}
} // namespace ks
