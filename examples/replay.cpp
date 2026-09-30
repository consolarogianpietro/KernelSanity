#include <kernelsanity/compare.hpp>
#include <iostream>
int main(int argc, char** argv) {
    auto reference = [](const ks::tensor& input, const ks::shape&) { return input; };
    auto buggy = [](const ks::tensor& input, const ks::shape&) {
        auto output = input;
        output[0] += 1.0f;
        return output;
    };
    std::filesystem::path path;
    if (argc > 1) {
        path = argv[1];
    } else {
        auto report = ks::compare(reference, buggy).shapes({{4}}).seed(42).tolerance(1e-5).run();
        report.save("failures");
        path = "failures/case-42-0.txt";
    }
    auto saved = ks::load_case(path);
    auto replayed = saved.replay(reference, buggy);
    std::cout << path << ": " << (replayed.ok() ? "passes" : "failure reproduced") << '\n';
    if (!replayed.ok()) {
        auto shrunk = saved.shrink(reference, buggy, 100);
        shrunk.report.save("shrunk-failures");
        const auto& failure = shrunk.report.failures.front();
        std::cout << "shrunk to " << failure.input.size() << " input value(s) after "
                  << shrunk.evaluations << " evaluations\n";
    }
    return replayed.ok() ? 1 : 0;
}
