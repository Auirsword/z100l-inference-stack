#include "../src/ds4/ds4_deterministic_math.hpp"

#include <cstdio>

namespace gg = strata::ds4::gguf;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    CHECK(gg::native_ordered::parse(nullptr) == 0);
    CHECK(gg::native_ordered::parse("0") == 0);
    CHECK(gg::native_ordered::parse("1") == 1);

    const char* invalid[] = {"", "2", "-1", "01", "true", " 1", "1 ", "0x1"};
    for (const char* value : invalid) CHECK(gg::native_ordered::parse(value) == -1);

    const char* modes[] = {nullptr, "0", "1"};
    for (int det = 0; det <= 1; ++det)
        for (const char* ordered : modes)
            for (const char* canonical : modes) {
                const bool expect_ordered = det != 0 || gg::native_ordered::parse(ordered) == 1;
                const bool expect_canonical = det != 0 || gg::native_canonical::parse(canonical) == 1;
                CHECK(gg::deterministic::ordered_merge(det != 0, ordered) == expect_ordered);
                CHECK(gg::deterministic::canonical_math(det != 0, canonical) == expect_canonical);
            }

    std::puts("DS4_NATIVE_ORDERED_COMBINE: default off, strict values and independent dispatch passed");
}
