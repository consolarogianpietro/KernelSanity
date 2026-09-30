#include <kernelsanity/multi_compare.hpp>
#include <iostream>

void require(bool value) { if (!value) throw std::runtime_error("multi-input test failed"); }
template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw);
}
int main() {
    ks::operand x{{99, 1, 99, 2, 99, 3, 99, 4}, {4}, {2}, 1};
    auto sum = [](const ks::multi_input& inputs) {
        const auto& v = inputs[0];
        float total = 0;
        for (std::size_t i = 0; i < v.dimensions[0]; ++i)
            total += v.storage[v.offset + i * v.strides[0]];
        return ks::tensor{total};
    };
    require(ks::compare_multi(sum, sum).cases({{x}}).run().passed == 1);
    auto off_by_one = [sum](const ks::multi_input& inputs) {
        auto output = sum(inputs); output[0] += 1; return output;
    };
    auto report = ks::compare_multi(sum, off_by_one).cases({{x}}).tolerance(.25).run();
    require(report.failures.size() == 1 && report.failures[0].expected == ks::tensor{10});
    require(report.failures[0].actual == ks::tensor{11});
    require(report.atol == .25 && report.failures[0].case_index == 0);

    ks::operand a{{1, 2, 3, 4, 5, 6}, {2, 3}, {3, 1}, 0};
    ks::operand b{{7, 9, 11, 8, 10, 12}, {3, 2}, {1, 3}, 0};
    auto gemm = [](const ks::multi_input& inputs) {
        const auto& left = inputs[0]; const auto& right = inputs[1];
        ks::tensor output(left.dimensions[0] * right.dimensions[1]);
        for (std::size_t i = 0; i < left.dimensions[0]; ++i)
            for (std::size_t j = 0; j < right.dimensions[1]; ++j)
                for (std::size_t k = 0; k < left.dimensions[1]; ++k)
                    output[i * right.dimensions[1] + j] +=
                        left.storage[left.offset + i * left.strides[0] + k * left.strides[1]] *
                        right.storage[right.offset + k * right.strides[0] + j * right.strides[1]];
        return output;
    };
    auto expected = gemm({a, b});
    require(expected == (ks::tensor{58, 64, 139, 154}));
    auto bad_gemm = [gemm](const ks::multi_input& inputs) {
        auto output = gemm(inputs); output[0] += 1; return output;
    };
    auto gemm_result = ks::compare_multi(gemm, bad_gemm).cases({{a, b}}).run();
    require(gemm_result.failures.size() == 1);
    auto directory = std::filesystem::temp_directory_path() /
        ("ks-multi-test-" + std::to_string(std::random_device{}()));
    report.save(directory);
    auto saved = ks::load_multi_case(directory / "multi-case-0.txt");
    require(saved.atol == .25 && saved.original.inputs.size() == 1);
    require(saved.original.inputs[0].offset == 1 && saved.original.inputs[0].strides == ks::shape{2});
    require(saved.original.inputs[0].storage == x.storage);
    require(saved.replay(sum, off_by_one).failures.size() == 1);
    require(saved.replay(sum, sum).ok());
    auto shrunk = saved.shrink_values(sum, off_by_one, 100);
    require(!shrunk.report.ok() && !shrunk.budget_exhausted);
    require(shrunk.report.failures[0].inputs[0].dimensions == x.dimensions);
    require(shrunk.report.failures[0].inputs[0].strides == x.strides);
    require(shrunk.report.failures[0].inputs[0].offset == x.offset);
    require(shrunk.report.failures[0].inputs[0].storage == ks::tensor(8, 0));
    shrunk.report.save(directory / "shrunk");
    require(!ks::load_multi_case(directory / "shrunk/multi-case-0.txt")
        .replay(sum, off_by_one).ok());
    auto limited = saved.shrink_values(sum, off_by_one, 1);
    require(limited.evaluations == 1 && limited.budget_exhausted);
    require(limited.report.failures[0].inputs[0].storage == x.storage);
    require(saved.shrink_values(sum, sum, 4).report.ok());

    gemm_result.save(directory / "gemm");
    auto saved_gemm = ks::load_multi_case(directory / "gemm/multi-case-0.txt");
    require(saved_gemm.original.inputs.size() == 2);
    require(saved_gemm.original.inputs[1].strides == ks::shape({1, 3}));
    require(saved_gemm.replay(gemm, bad_gemm).failures.size() == 1);

    // Physical padding survives the artifact round trip bit for bit.
    auto nan = ks::detail::float_from_bits(0x7fc01234);
    x.storage[0] = nan;
    auto padded = ks::compare_multi(sum, off_by_one).cases({{x}}).run();
    padded.save(directory / "padding");
    auto loaded = ks::load_multi_case(directory / "padding/multi-case-0.txt");
    require(ks::detail::bits(loaded.original.inputs[0].storage[0]) == 0x7fc01234u);

    rejects([&] { ks::compare_multi(sum, sum).cases({{}}); });
    rejects([&] { ks::compare_multi(sum, sum).cases({{{{1}, {2}, {1}, 0}}}); });
    rejects([&] { ks::compare_multi(sum, sum).cases({{{{1}, {1}, {0}, 0}}}); });
    rejects([&] { ks::compare_multi(sum, sum).cases({{{{1}, {1}, {1}, 1}}}); });
    rejects([&] { ks::compare_multi(sum, sum).cases({{{{1}, {2}, {2}, 0}}}); });
    rejects([&] { ks::compare_multi(sum, sum).run(); });
    rejects([&] { saved.shrink_values(sum, off_by_one, 0); });
    auto write_bad = [&](const std::string& name, const std::string& contents) {
        auto path = directory / name;
        std::ofstream out(path); out << contents; out.close();
        rejects([&] { ks::load_multi_case(path); });
    };
    write_bad("old.txt", "KernelSanity-v2\n");
    write_bad("truncated.txt", "KernelSanity-v3\n0\n0 0\n1\n1 2\n1\n0\n2 3f800000\n");
    write_bad("bad-view.txt", "KernelSanity-v3\n0\n0 0\n1\n1 2\n2\n0\n2 3f800000 40000000\n1 00000000\n1 00000000\n");
    write_bad("trailing.txt", "KernelSanity-v3\n0\n0 0\n1\n1 1\n1\n0\n1 00000000\n1 00000000\n1 00000000\nextra\n");
    std::filesystem::remove_all(directory);
    std::cout << "Multi-input checks passed\n";
}
