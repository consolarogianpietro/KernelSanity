#include <kernelsanity/multi_compare.hpp>
#include <iostream>

int main() {
    // One strided vector: logical values are 1, 2, 3, 4.
    ks::operand vector{{99, 1, 99, 2, 99, 3, 99, 4}, {4}, {2}, 1};
    auto reduction = [](const ks::multi_input& inputs) {
        const auto& x = inputs.at(0);
        float sum = 0;
        for (std::size_t i = 0; i < x.dimensions[0]; ++i)
            sum += x.storage[x.offset + i * x.strides[0]];
        return ks::tensor{sum};
    };
    auto reduction_report = ks::compare_multi(reduction, reduction).cases({{vector}}).run();

    // A is row-major 2x3; B is column-major 3x2.
    ks::operand a{{1, 2, 3, 4, 5, 6}, {2, 3}, {3, 1}, 0};
    ks::operand b{{7, 9, 11, 8, 10, 12}, {3, 2}, {1, 3}, 0};
    auto gemm = [](const ks::multi_input& inputs) {
        const auto& left = inputs.at(0);
        const auto& right = inputs.at(1);
        ks::tensor output(left.dimensions[0] * right.dimensions[1]);
        for (std::size_t i = 0; i < left.dimensions[0]; ++i)
            for (std::size_t j = 0; j < right.dimensions[1]; ++j)
                for (std::size_t k = 0; k < left.dimensions[1]; ++k)
                    output[i * right.dimensions[1] + j] +=
                        left.storage[left.offset + i * left.strides[0] + k * left.strides[1]] *
                        right.storage[right.offset + k * right.strides[0] + j * right.strides[1]];
        return output;
    };
    auto buggy = [gemm](const ks::multi_input& inputs) {
        auto output = gemm(inputs);
        output[0] += 1;
        return output;
    };
    auto gemm_report = ks::compare_multi(gemm, buggy).cases({{a, b}}).run();
    gemm_report.save("multi-failures");
    auto saved = ks::load_multi_case("multi-failures/multi-case-0.txt");
    auto replayed = saved.replay(gemm, buggy);
    auto shrunk = saved.shrink_values(gemm, buggy, 100);
    shrunk.report.save("multi-shrunk-failures");
    std::cout << reduction_report.passed << " reduction passed; "
              << replayed.failures.size() << " GEMM failure reproduced; "
              << shrunk.evaluations << " shrink evaluations\n";
    return reduction_report.ok() && !replayed.ok() && !shrunk.report.ok() ? 0 : 1;
}
