#include <kernelsanity/compare.hpp>
#include <iostream>

void require(bool value) { if (!value) throw std::runtime_error("test failed"); }
template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::runtime_error&) { threw = true; }
    require(threw);
}
int main() {
    auto identity = [](const ks::tensor& x, const ks::shape&) { return x; };
    auto broken = [](const ks::tensor& x, const ks::shape&) { auto y = x; y[0] += 1; return y; };
    auto good = ks::compare(identity, identity).random_shapes(100).run();
    require(good.ok() && good.passed == 100);
    auto comparison = ks::compare(identity, broken).shapes({{1}, {3, 7}}).random_shapes(10).seed(123).tolerance(.001, .002);
    auto first = comparison.run(), second = comparison.run();
    require(first.seed == 123 && first.atol == .001 && first.rtol == .002);
    require(first.failures.size() == 12 && first.passed == 0);
    for (std::size_t i = 0; i < first.failures.size(); ++i) {
        require(first.failures[i].input == second.failures[i].input);
        require(first.failures[i].dimensions == second.failures[i].dimensions);
    }
    auto scalar = [](float v) { return [v](const ks::tensor&, const ks::shape&) { return ks::tensor{v}; }; };
    auto accepts = [&](float a, float b, double at, double rt) {
        return ks::compare(scalar(a), scalar(b)).shapes({{1}}).tolerance(at, rt).run().ok();
    };
    require(accepts(100, 101, 0, .01));
    require(!accepts(100, 102, 0, .01));
    require(accepts(0, .01f, .02, 0));
    const auto inf = std::numeric_limits<float>::infinity();
    require(accepts(inf, inf, 0, 0));
    require(!accepts(inf, -inf, 0, 0));
    require(!accepts(0, inf, 0, 0));
    require(!accepts(NAN, NAN, 0, 0));
    require(!ks::compare(identity, scalar(0)).shapes({{2}}).run().ok());
    for (const auto& s : std::vector<ks::shape>{{}, {0}, {1048577}, {1024, 1025}}) {
        bool threw = false;
        try { ks::compare(identity, identity).shapes({s}); } catch (const std::invalid_argument&) { threw = true; }
        require(threw);
    }

    auto directory = std::filesystem::temp_directory_path() / ("ks-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(directory);
    first.save(directory);
    auto path = directory / "case-123-0.txt";
    auto saved = ks::load_case(path);
    require(saved.seed == 123 && saved.atol == .001 && saved.rtol == .002);
    require(saved.original.case_index == 0 && saved.original.dimensions == ks::shape{1});
    require(saved.original.input == first.failures[0].input);
    require(saved.original.expected == first.failures[0].expected);
    require(saved.original.actual == first.failures[0].actual);
    auto reproduced = saved.replay(identity, broken);
    require(reproduced.seed == 123 && reproduced.failures.size() == 1);
    require(reproduced.failures[0].input == saved.original.input);
    require(reproduced.failures[0].expected == saved.original.expected);
    require(reproduced.failures[0].actual == saved.original.actual);
    require(saved.replay(identity, identity).ok());
    auto near = [](const ks::tensor& x, const ks::shape&) { auto y = x; y[0] += .0005f; return y; };
    require(saved.replay(identity, near).ok()); // Stored absolute tolerance is applied.
    auto wrong_length = [](const ks::tensor&, const ks::shape&) { return ks::tensor{}; };
    require(saved.replay(identity, wrong_length).failures.size() == 1);
    auto throwing = [](const ks::tensor&, const ks::shape&) -> ks::tensor {
        throw std::runtime_error("kernel error");
    };
    rejects([&] { saved.replay(identity, throwing); });

    auto shrunk = saved.shrink(identity, broken, 32);
    require(!shrunk.report.ok() && !shrunk.budget_exhausted);
    require(shrunk.evaluations <= 32);
    require(shrunk.report.seed == 123 && shrunk.report.atol == .001);
    require(shrunk.report.failures[0].case_index == 0);
    require(shrunk.report.failures[0].dimensions == ks::shape{1});
    require(shrunk.report.failures[0].input == ks::tensor{0.0f});
    auto repeated_shrink = saved.shrink(identity, broken, 32);
    require(repeated_shrink.evaluations == shrunk.evaluations);
    require(repeated_shrink.report.failures[0].input == shrunk.report.failures[0].input);
    shrunk.report.save(directory / "shrunk");
    require(!ks::load_case(directory / "shrunk/case-123-0.txt").replay(identity, broken).ok());
    auto limited = saved.shrink(identity, broken, 1);
    require(limited.budget_exhausted && limited.evaluations == 1);
    require(limited.report.failures[0].input == saved.original.input);
    require(saved.shrink(identity, identity, 8).report.ok());
    bool invalid_budget = false;
    try { saved.shrink(identity, broken, 0); }
    catch (const std::invalid_argument&) { invalid_budget = true; }
    require(invalid_budget);

    // A value-dependent mismatch can survive halving while zero is rejected.
    ks::saved_case value_case{5, 0, 0, {{1}, {.8f}, {0}, {1}, 2}};
    auto value_bug = [](const ks::tensor& x, const ks::shape&) {
        return ks::tensor{x[0] > .25f ? 1.0f : 0.0f};
    };
    auto value_shrunk = value_case.shrink(scalar(0), value_bug, 16);
    require(!value_shrunk.report.ok() && !value_shrunk.budget_exhausted);
    require(value_shrunk.report.failures[0].input[0] == .4f);

    // Shape candidates rejected by a kernel do not become shrunk failures.
    ks::saved_case shape_case{5, 0, 0, {{4}, {1, 2, 3, 4}, {0}, {1}, 3}};
    auto shape_bug = [](const ks::tensor& x, const ks::shape& s) -> ks::tensor {
        if (s[0] < 2) throw std::runtime_error("unsupported shape");
        return {x[0] == 1 && x[1] == 2 ? 1.0f : 0.0f};
    };
    auto shape_shrunk = shape_case.shrink(scalar(0), shape_bug, 32);
    require(shape_shrunk.report.failures[0].dimensions == ks::shape{2});
    require(shape_shrunk.report.failures[0].input == (ks::tensor{1, 2}));

    // Exact FP32 storage preserves sign of zero, infinities and NaN payload bits.
    ks::result edge;
    edge.seed = 9; edge.atol = 0; edge.rtol = 0;
    edge.failures.push_back({{4}, {0.0f, -0.0f, inf, ks::detail::float_from_bits(0x7fc01234)},
                             {0.0f, -inf}, {-0.0f, inf}, 7});
    edge.save(directory);
    auto edge_case = ks::load_case(directory / "case-9-7.txt");
    for (std::size_t i = 0; i < 4; ++i)
        require(ks::detail::bits(edge_case.original.input[i]) == ks::detail::bits(edge.failures[0].input[i]));
    require(ks::detail::bits(edge_case.original.expected[1]) == ks::detail::bits(-inf));
    require(ks::detail::bits(edge_case.original.actual[0]) == ks::detail::bits(-0.0f));
    require(!edge_case.replay(scalar(NAN), scalar(NAN)).ok());
    require(edge_case.replay(scalar(inf), scalar(inf)).ok());

    auto write_bad = [&](const std::string& name, const std::string& contents) {
        auto file = directory / name;
        std::ofstream out(file); out << contents; out.close();
        rejects([&] { ks::load_case(file); });
    };
    write_bad("old.txt", "KernelSanity-v1\n");
    write_bad("truncated.txt", "KernelSanity-v2\n1 2\n0 0\n1 1\n1");
    write_bad("shape.txt", "KernelSanity-v2\n1 2\n0 0\n1 0\n1 00000000\n0\n0\n");
    write_bad("length.txt", "KernelSanity-v2\n1 2\n0 0\n1 2\n1 00000000\n0\n0\n");
    write_bad("bits.txt", "KernelSanity-v2\n1 2\n0 0\n1 1\n1 g0000000\n0\n0\n");
    write_bad("trailing.txt", "KernelSanity-v2\n1 2\n0 0\n1 1\n1 00000000\n0\n0\nextra\n");
    write_bad("tol.txt", "KernelSanity-v2\n1 2\n-1 0\n1 1\n1 00000000\n0\n0\n");
    rejects([&] { ks::load_case(directory / "missing.txt"); });
    std::filesystem::remove_all(directory);
    std::cout << "All checks passed\n";
}
