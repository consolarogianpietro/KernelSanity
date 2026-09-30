#pragma once
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ks {
using shape = std::vector<std::size_t>;
using tensor = std::vector<float>;
static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
              "KernelSanity requires IEEE 754 binary32 float");
// Each kernel receives the same immutable input and returns its output.
using kernel = std::function<tensor(const tensor&, const shape&)>;
struct failure {
    shape dimensions;
    tensor input, expected, actual;
    std::size_t case_index;
};
namespace detail {
constexpr std::size_t max_elements = 1048576;
inline std::size_t elements(const shape& s) {
    if (s.empty()) throw std::invalid_argument("shape must have at least one dimension");
    std::size_t n = 1;
    for (auto d : s) {
        if (!d || d > max_elements / n) throw std::invalid_argument("shape must contain 1..1048576 elements");
        n *= d;
    }
    return n;
}
inline bool equal(float a, float b, double atol, double rtol) {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (a == b) return true; // Includes matching signed infinities and zeros.
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    return std::abs(double(a) - double(b)) <= atol + rtol * std::abs(double(a));
}
inline bool matches(const tensor& expected, const tensor& actual, double atol, double rtol) {
    if (expected.size() != actual.size()) return false;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (!equal(expected[i], actual[i], atol, rtol)) return false;
    return true;
}
inline void validate_tolerance(double absolute, double relative) {
    if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0 || relative < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
}
inline std::string token(std::istream& in) {
    std::string value;
    if (!(in >> value)) throw std::runtime_error("truncated KernelSanity artifact");
    return value;
}
template <typename T> T number(std::istream& in) {
    const auto value = token(in);
    T parsed{};
    const auto end = value.data() + value.size();
    auto result = std::from_chars(value.data(), end, parsed);
    if (result.ec != std::errc{} || result.ptr != end)
        throw std::runtime_error("invalid integer in KernelSanity artifact");
    return parsed;
}
inline double tolerance(std::istream& in) {
    std::istringstream text(token(in));
    text.imbue(std::locale::classic());
    double value;
    if (!(text >> value) || (text >> std::ws, !text.eof()) || !std::isfinite(value) || value < 0)
        throw std::runtime_error("invalid tolerance in KernelSanity artifact");
    return value;
}
inline std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}
inline float float_from_bits(std::uint32_t value) {
    float result;
    std::memcpy(&result, &value, sizeof result);
    return result;
}
inline void write_tensor(std::ostream& out, const tensor& values) {
    if (values.size() > max_elements) throw std::invalid_argument("artifact vector exceeds 1048576 elements");
    out << values.size();
    for (float value : values)
        out << ' ' << std::hex << std::setw(8) << std::setfill('0') << bits(value) << std::dec;
    out << '\n';
}
inline tensor read_tensor(std::istream& in) {
    const auto count = number<std::size_t>(in);
    if (count > max_elements) throw std::runtime_error("artifact vector exceeds 1048576 elements");
    tensor values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto value = token(in);
        if (value.size() != 8) throw std::runtime_error("invalid FP32 bits in KernelSanity artifact");
        std::uint32_t parsed{};
        auto result = std::from_chars(value.data(), value.data() + value.size(), parsed, 16);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
            throw std::runtime_error("invalid FP32 bits in KernelSanity artifact");
        values.push_back(float_from_bits(parsed));
    }
    return values;
}
} // namespace detail
struct result {
    std::uint32_t seed = 0;
    double atol = 1e-5, rtol = 1e-5;
    std::size_t passed = 0;
    std::vector<failure> failures;
    bool ok() const { return failures.empty(); }
    // Version 2 stores exact FP32 bit patterns, including signed zero and NaN payloads.
    void save(const std::filesystem::path& directory) const {
        if (failures.empty()) return;
        detail::validate_tolerance(atol, rtol);
        std::filesystem::create_directories(directory);
        for (const auto& f : failures) {
            if (f.input.size() != detail::elements(f.dimensions))
                throw std::invalid_argument("failure input does not match shape");
            if (f.expected.size() > detail::max_elements || f.actual.size() > detail::max_elements)
                throw std::invalid_argument("artifact vector exceeds 1048576 elements");
            auto path = directory / ("case-" + std::to_string(seed) + "-" + std::to_string(f.case_index) + ".txt");
            std::ofstream out(path);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.imbue(std::locale::classic());
            out << "KernelSanity-v2\n" << seed << ' ' << f.case_index << '\n';
            out << std::setprecision(std::numeric_limits<double>::max_digits10)
                << atol << ' ' << rtol << '\n';
            out << f.dimensions.size();
            for (auto d : f.dimensions) out << ' ' << d;
            out << '\n';
            detail::write_tensor(out, f.input);
            detail::write_tensor(out, f.expected);
            detail::write_tensor(out, f.actual);
            out.close();
        }
    }
};
struct shrink_result {
    result report;
    std::size_t evaluations = 0;
    bool budget_exhausted = false;
};
struct saved_case {
    std::uint32_t seed;
    double atol, rtol;
    failure original;
    result replay(kernel reference, kernel optimized) const {
        if (!reference || !optimized) throw std::invalid_argument("kernels must be callable");
        result report;
        report.seed = seed; report.atol = atol; report.rtol = rtol;
        auto expected = reference(original.input, original.dimensions);
        auto actual = optimized(original.input, original.dimensions);
        if (detail::matches(expected, actual, atol, rtol)) ++report.passed;
        else report.failures.push_back({original.dimensions, original.input,
                                        std::move(expected), std::move(actual), original.case_index});
        return report;
    }
    // Greedy rank-preserving shape reduction followed by input-value reduction.
    // The budget includes the initial replay; no global minimality is claimed.
    shrink_result shrink(kernel reference, kernel optimized, std::size_t budget) const {
        if (!budget) throw std::invalid_argument("shrink budget must be positive");
        shrink_result output;
        output.report = replay(reference, optimized);
        output.evaluations = 1;
        if (output.report.ok()) return output; // The saved failure no longer reproduces.

        auto& current = output.report.failures.front();
        auto try_candidate = [&](shape dimensions, tensor input) {
            if (output.evaluations == budget) return false;
            ++output.evaluations;
            try {
                auto expected = reference(input, dimensions);
                auto actual = optimized(input, dimensions);
                if (!detail::matches(expected, actual, atol, rtol)) {
                    current = {std::move(dimensions), std::move(input),
                               std::move(expected), std::move(actual), original.case_index};
                    return true;
                }
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception&) {
                // A candidate rejected by either kernel is not a numerical mismatch.
            }
            return false;
        };

        for (std::size_t axis = 0; axis < current.dimensions.size() && output.evaluations < budget; ++axis) {
            bool reduced = true;
            while (reduced && output.evaluations < budget) {
                reduced = false;
                const auto d = current.dimensions[axis];
                const std::size_t choices[] = {1, d / 2, d - 1};
                for (std::size_t choice = 0; choice < 3; ++choice) {
                    const auto smaller = choices[choice];
                    if (!smaller || smaller >= d) continue;
                    bool duplicate = false;
                    for (std::size_t earlier = 0; earlier < choice; ++earlier)
                        duplicate |= choices[earlier] == smaller;
                    if (duplicate) continue;
                    auto dimensions = current.dimensions;
                    dimensions[axis] = smaller;
                    auto input = current.input;
                    input.resize(detail::elements(dimensions)); // Keep the flattened prefix.
                    if (try_candidate(std::move(dimensions), std::move(input))) {
                        reduced = true;
                        break;
                    }
                    if (output.evaluations == budget) break;
                }
            }
        }
        for (std::size_t i = 0; i < current.input.size() && output.evaluations < budget; ++i) {
            bool reduced = true;
            while (reduced && output.evaluations < budget) {
                reduced = false;
                const float value = current.input[i];
                for (float smaller : {0.0f, value / 2.0f}) {
                    if (detail::bits(smaller) == detail::bits(value)) continue;
                    auto input = current.input;
                    input[i] = smaller;
                    if (try_candidate(current.dimensions, std::move(input))) {
                        reduced = true;
                        break;
                    }
                    if (output.evaluations == budget) break;
                }
            }
        }
        output.budget_exhausted = output.evaluations == budget;
        return output;
    }
};
inline saved_case load_case(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open KernelSanity artifact: " + path.string());
    in.imbue(std::locale::classic());
    if (detail::token(in) != "KernelSanity-v2")
        throw std::runtime_error("unsupported KernelSanity artifact version");
    saved_case saved;
    saved.seed = detail::number<std::uint32_t>(in);
    saved.original.case_index = detail::number<std::size_t>(in);
    saved.atol = detail::tolerance(in);
    saved.rtol = detail::tolerance(in);
    const auto rank = detail::number<std::size_t>(in);
    if (!rank || rank > detail::max_elements) throw std::runtime_error("invalid artifact rank");
    saved.original.dimensions.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i)
        saved.original.dimensions.push_back(detail::number<std::size_t>(in));
    std::size_t count;
    try { count = detail::elements(saved.original.dimensions); }
    catch (const std::invalid_argument&) { throw std::runtime_error("invalid artifact shape"); }
    saved.original.input = detail::read_tensor(in);
    if (saved.original.input.size() != count) throw std::runtime_error("artifact input length does not match shape");
    saved.original.expected = detail::read_tensor(in);
    saved.original.actual = detail::read_tensor(in);
    std::string extra;
    if (in >> extra) throw std::runtime_error("trailing data in KernelSanity artifact");
    if (in.bad()) throw std::runtime_error("error reading KernelSanity artifact");
    return saved;
}
class comparison {
    kernel reference_, optimized_;
    std::vector<shape> shapes_;
    std::size_t random_count_ = 0, rank_ = 2, max_dim_ = 64;
    std::uint32_t seed_ = 0;
    double atol_ = 1e-5, rtol_ = 1e-5;
public:
    comparison(kernel reference, kernel optimized)
        : reference_(std::move(reference)), optimized_(std::move(optimized)) {
        if (!reference_ || !optimized_) throw std::invalid_argument("kernels must be callable");
    }
    comparison& shapes(std::vector<shape> values) {
        for (const auto& s : values) detail::elements(s);
        shapes_ = std::move(values); return *this;
    }
    comparison& random_shapes(std::size_t count, std::size_t rank = 2, std::size_t max_dimension = 64) {
        if (!rank || rank > 20 || !max_dimension) throw std::invalid_argument("invalid random shape bounds");
        detail::elements(shape(rank, max_dimension));
        random_count_ = count; rank_ = rank; max_dim_ = max_dimension; return *this;
    }
    comparison& seed(std::uint32_t value) { seed_ = value; return *this; }
    comparison& tolerance(double absolute, double relative = 0) {
        detail::validate_tolerance(absolute, relative);
        atol_ = absolute; rtol_ = relative; return *this;
    }
    result run() const {
        if (shapes_.empty() && !random_count_) throw std::invalid_argument("no test cases configured");
        std::mt19937 rng(seed_);
        result report;
        report.seed = seed_; report.atol = atol_; report.rtol = rtol_;
        auto check = [&](const shape& s, std::size_t index) {
            tensor input(detail::elements(s));
            // Defined mapping avoids implementation-dependent uniform distributions.
            for (auto& v : input) v = float(rng() >> 8) / 8388608.0f - 1.0f;
            auto expected = reference_(input, s);
            auto actual = optimized_(input, s);
            if (detail::matches(expected, actual, atol_, rtol_)) ++report.passed;
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
