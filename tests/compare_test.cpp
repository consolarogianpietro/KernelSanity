#include <kernelsanity/compare.hpp>
#include <iostream>
#include <sstream>
void require(bool value) { if (!value) throw std::runtime_error("test failed"); }
int main() {
    auto identity = [](const ks::tensor& x, const ks::shape&) { return x; };
    auto broken = [](const ks::tensor& x, const ks::shape&) { auto y = x; y[0] += 1; return y; };
    auto good = ks::compare(identity, identity).random_shapes(100).run();
    require(good.ok() && good.passed == 100);
    auto comparison = ks::compare(identity, broken).shapes({{1}, {3, 7}}).random_shapes(10).seed(123);
    auto first = comparison.run(), second = comparison.run();
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
    first.save(directory, 123);
    std::ifstream file(directory / "case-123-0.txt");
    std::string header; std::getline(file, header);
    require(header == "KernelSanity-v1");
    unsigned seed; std::size_t index, rank, dim, count; float value;
    file >> seed >> index >> rank >> dim >> count >> value;
    require(seed == 123 && index == 0 && rank == 1 && dim == 1 && count == 1);
    require(value == first.failures[0].input[0]);
    file.close();
    std::filesystem::remove_all(directory);
    std::cout << "All checks passed\n";
}
