#include "../src/ds4/ds4_deterministic_math.hpp"

#include <cstdio>

namespace gg = strata::ds4::gguf;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    CHECK(gg::native_canonical::parse(nullptr) == 0);
    CHECK(gg::native_canonical::parse("0") == 0);
    CHECK(gg::native_canonical::parse("1") == 1);

    const char* invalid[] = {"", "2", "-1", "01", "true", " 1", "1 ", "0x1"};
    for (const char* value : invalid) CHECK(gg::native_canonical::parse(value) == -1);

    CHECK(!gg::deterministic::canonical_math(false, nullptr));
    CHECK(!gg::deterministic::canonical_math(false, "0"));
    CHECK(gg::deterministic::canonical_math(false, "1"));
    CHECK(gg::deterministic::canonical_math(true, nullptr));
    CHECK(gg::deterministic::canonical_math(true, "0"));
    CHECK(gg::deterministic::canonical_math(true, "1"));
    CHECK(!gg::deterministic::canonical_math(false, "invalid"));

    std::puts("DS4_NATIVE_CANONICAL: default off, strict values and canonical dispatch passed");
}
