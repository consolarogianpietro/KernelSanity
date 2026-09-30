#pragma once
#include <kernelsanity/compare.hpp>

namespace ks {
// Strides and offset are measured in float elements of physical storage.
struct operand {
    tensor storage;
    shape dimensions;
    shape strides;
    std::size_t offset = 0;
};
using multi_input = std::vector<operand>;
using multi_kernel = std::function<tensor(const multi_input&)>;
struct multi_failure {
    multi_input inputs;
    tensor expected, actual;
    std::size_t case_index;
};
namespace detail {
inline void validate_inputs(const multi_input& inputs) {
    if (inputs.empty() || inputs.size() > 16)
        throw std::invalid_argument("multi-input case needs 1..16 operands");
    std::size_t total_storage = 0;
    for (const auto& input : inputs) {
        elements(input.dimensions);
        if (input.strides.size() != input.dimensions.size())
            throw std::invalid_argument("operand strides must match rank");
        if (input.storage.empty() || input.storage.size() > max_elements - total_storage)
            throw std::invalid_argument("multi-input storage exceeds 1048576 elements");
        total_storage += input.storage.size();
        if (input.offset >= input.storage.size())
            throw std::invalid_argument("operand offset exceeds storage");
        std::size_t last = input.offset;
        for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
            const auto stride = input.strides[axis];
            if (!stride || (input.dimensions[axis] - 1) > (input.storage.size() - 1 - last) / stride)
                throw std::invalid_argument("operand view exceeds storage");
            last += (input.dimensions[axis] - 1) * stride;
        }
    }
}
inline void write_inputs(std::ostream& out, const multi_input& inputs) {
    out << inputs.size() << '\n';
    for (const auto& input : inputs) {
        out << input.dimensions.size();
        for (auto d : input.dimensions) out << ' ' << d;
        out << '\n';
        out << input.strides[0];
        for (std::size_t axis = 1; axis < input.strides.size(); ++axis)
            out << ' ' << input.strides[axis];
        out << '\n' << input.offset << '\n';
        write_tensor(out, input.storage);
    }
}
inline multi_input read_inputs(std::istream& in) {
    const auto count = number<std::size_t>(in);
    if (!count || count > 16) throw std::runtime_error("invalid operand count in artifact");
    multi_input inputs;
    inputs.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        operand input;
        const auto rank = number<std::size_t>(in);
        if (!rank || rank > max_elements) throw std::runtime_error("invalid operand rank in artifact");
        input.dimensions.reserve(rank);
        input.strides.reserve(rank);
        for (std::size_t axis = 0; axis < rank; ++axis)
            input.dimensions.push_back(number<std::size_t>(in));
        for (std::size_t axis = 0; axis < rank; ++axis)
            input.strides.push_back(number<std::size_t>(in));
        input.offset = number<std::size_t>(in);
        input.storage = read_tensor(in);
        inputs.push_back(std::move(input));
    }
    try { validate_inputs(inputs); }
    catch (const std::invalid_argument&) { throw std::runtime_error("invalid operand in artifact"); }
    return inputs;
}
} // namespace detail
struct multi_result {
    double atol = 1e-5, rtol = 1e-5;
    std::size_t passed = 0;
    std::vector<multi_failure> failures;
    bool ok() const { return failures.empty(); }
    void save(const std::filesystem::path& directory) const {
        if (failures.empty()) return;
        detail::validate_tolerance(atol, rtol);
        std::filesystem::create_directories(directory);
        for (const auto& f : failures) {
            detail::validate_inputs(f.inputs);
            if (f.expected.size() > detail::max_elements || f.actual.size() > detail::max_elements)
                throw std::invalid_argument("artifact vector exceeds 1048576 elements");
            auto path = directory / ("multi-case-" + std::to_string(f.case_index) + ".txt");
            std::ofstream out(path);
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out.imbue(std::locale::classic());
            out << "KernelSanity-v3\n" << f.case_index << '\n';
            out << std::setprecision(std::numeric_limits<double>::max_digits10)
                << atol << ' ' << rtol << '\n';
            detail::write_inputs(out, f.inputs);
            detail::write_tensor(out, f.expected);
            detail::write_tensor(out, f.actual);
            out.close();
        }
    }
};
struct multi_shrink_result {
    multi_result report;
    std::size_t evaluations = 0;
    bool budget_exhausted = false;
};
struct saved_multi_case {
    double atol, rtol;
    multi_failure original;
    multi_result replay(multi_kernel reference, multi_kernel optimized) const {
        if (!reference || !optimized) throw std::invalid_argument("kernels must be callable");
        multi_result report;
        report.atol = atol; report.rtol = rtol;
        auto expected = reference(original.inputs);
        auto actual = optimized(original.inputs);
        if (detail::matches(expected, actual, atol, rtol)) ++report.passed;
        else report.failures.push_back({original.inputs, std::move(expected),
                                        std::move(actual), original.case_index});
        return report;
    }
    // Preserve all descriptors; simplify physical storage values only.
    multi_shrink_result shrink_values(multi_kernel reference, multi_kernel optimized,
                                      std::size_t budget) const {
        if (!budget) throw std::invalid_argument("shrink budget must be positive");
        multi_shrink_result output;
        output.report = replay(reference, optimized);
        output.evaluations = 1;
        if (output.report.ok()) return output;
        auto& current = output.report.failures.front();
        for (std::size_t operand_index = 0;
             operand_index < current.inputs.size() && output.evaluations < budget; ++operand_index) {
            for (std::size_t i = 0;
                 i < current.inputs[operand_index].storage.size() && output.evaluations < budget; ++i) {
                bool reduced = true;
                while (reduced && output.evaluations < budget) {
                    reduced = false;
                    const float value = current.inputs[operand_index].storage[i];
                    for (float smaller : {0.0f, value / 2.0f}) {
                        if (detail::bits(smaller) == detail::bits(value)) continue;
                        auto inputs = current.inputs;
                        inputs[operand_index].storage[i] = smaller;
                        ++output.evaluations;
                        try {
                            auto expected = reference(inputs);
                            auto actual = optimized(inputs);
                            if (!detail::matches(expected, actual, atol, rtol)) {
                                current = {std::move(inputs), std::move(expected),
                                           std::move(actual), original.case_index};
                                reduced = true;
                                break;
                            }
                        } catch (const std::bad_alloc&) {
                            throw;
                        } catch (const std::exception&) {
                            // A rejected candidate is not a numerical mismatch.
                        }
                        if (output.evaluations == budget) break;
                    }
                }
            }
        }
        output.budget_exhausted = output.evaluations == budget;
        return output;
    }
};
inline saved_multi_case load_multi_case(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open KernelSanity artifact: " + path.string());
    in.imbue(std::locale::classic());
    if (detail::token(in) != "KernelSanity-v3")
        throw std::runtime_error("unsupported KernelSanity artifact version");
    saved_multi_case saved;
    saved.original.case_index = detail::number<std::size_t>(in);
    saved.atol = detail::tolerance(in);
    saved.rtol = detail::tolerance(in);
    saved.original.inputs = detail::read_inputs(in);
    saved.original.expected = detail::read_tensor(in);
    saved.original.actual = detail::read_tensor(in);
    std::string extra;
    if (in >> extra) throw std::runtime_error("trailing data in KernelSanity artifact");
    if (in.bad()) throw std::runtime_error("error reading KernelSanity artifact");
    return saved;
}
class multi_comparison {
    multi_kernel reference_, optimized_;
    std::vector<multi_input> cases_;
    double atol_ = 1e-5, rtol_ = 1e-5;
public:
    multi_comparison(multi_kernel reference, multi_kernel optimized)
        : reference_(std::move(reference)), optimized_(std::move(optimized)) {
        if (!reference_ || !optimized_) throw std::invalid_argument("kernels must be callable");
    }
    multi_comparison& cases(std::vector<multi_input> values) {
        for (const auto& inputs : values) detail::validate_inputs(inputs);
        cases_ = std::move(values);
        return *this;
    }
    multi_comparison& tolerance(double absolute, double relative = 0) {
        detail::validate_tolerance(absolute, relative);
        atol_ = absolute; rtol_ = relative;
        return *this;
    }
    multi_result run() const {
        if (cases_.empty()) throw std::invalid_argument("no test cases configured");
        multi_result report;
        report.atol = atol_; report.rtol = rtol_;
        for (std::size_t index = 0; index < cases_.size(); ++index) {
            const auto& inputs = cases_[index];
            auto expected = reference_(inputs);
            auto actual = optimized_(inputs);
            if (detail::matches(expected, actual, atol_, rtol_)) ++report.passed;
            else report.failures.push_back({inputs, std::move(expected), std::move(actual), index});
        }
        return report;
    }
};
inline multi_comparison compare_multi(multi_kernel reference, multi_kernel optimized) {
    return multi_comparison(std::move(reference), std::move(optimized));
}
} // namespace ks
