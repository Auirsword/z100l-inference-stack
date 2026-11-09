#include "../src/ds4/ds4_fused_ar_control.hpp"
#include "../src/ds4/fused_ar.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace control = strata::ds4::fused_ar_control;
namespace fused = strata::ds4::p2p::fused;
#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
    CHECK(control::parse_mode(nullptr) == 0);
    CHECK(control::parse_mode("0") == 0);
    CHECK(control::parse_mode("1") == 1);
    CHECK(control::parse_mode("2") == 2);
    const char* invalid[] = {"", "3", "-1", "01", "true", " 1", "1 ", "0x1"};
    for (const char* value : invalid) CHECK(control::parse_mode(value) == -1);
    CHECK(control::parse_init_mask(nullptr) == 7);
    CHECK(control::parse_init_mask("0") == 0 && control::parse_init_mask("7") == 7);
    for (const char* value : {"", "8", "-1", "10", "07"}) CHECK(control::parse_init_mask(value) == -1);
    CHECK(control::parse_group_stage(nullptr) == 4);
    for (int stage = 0; stage <= 4; ++stage) {
        const char value[] = {static_cast<char>('0' + stage), 0};
        CHECK(control::parse_group_stage(value) == stage);
    }
    for (const char* value : {"", "5", "-1", "10", "04", " 4", "4 "}) CHECK(control::parse_group_stage(value) == -1);
    CHECK(control::parse_test_mode(nullptr) == 0 && control::parse_test_mode("") == 0);
    CHECK(control::parse_test_mode("drop-member-1") == 1);
    CHECK(control::parse_test_mode("drop-member-0") == -1);

    const control::WindowSource sources[] = {control::WindowSource::kPrompt, control::WindowSource::kStep,
                                             control::WindowSource::kVerify, control::WindowSource::kMulti};
    for (int mode = 0; mode <= 2; ++mode)
        for (control::WindowSource source : sources)
            for (int T : {1, 2, 16}) {
                const bool expected = mode != 0 && source == control::WindowSource::kStep && T == 1;
                CHECK(control::window_enabled(mode, source, T) == expected);
            }
    CHECK(!control::window_enabled(-1, control::WindowSource::kStep, 1));
    CHECK(!control::window_enabled(3, control::WindowSource::kStep, 1));
    CHECK(control::cuda_pdl_allowed(0, nullptr));
    CHECK(!control::cuda_pdl_allowed(1, nullptr));
    CHECK(!control::cuda_pdl_allowed(2, "1"));
    CHECK(!control::cuda_pdl_allowed(1, "00"));
    CHECK(control::cuda_pdl_allowed(1, "0") && control::cuda_pdl_allowed(2, "0"));

    for (int mode = 0; mode <= 2; ++mode)
        for (std::size_t members : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5}})
            CHECK(control::group_enabled(mode, members) == (mode != 0 && (members == 2 || members == 4)));

    const std::vector<int> physical{10, 11, 12, 13};
    std::vector<int> mapped;
    CHECK(control::physical_devices({2, 0, 3, 1}, physical, mapped));
    CHECK(mapped == std::vector<int>({12, 10, 13, 11}));
    CHECK(!control::physical_devices({-1, 0}, physical, mapped) && mapped.empty());
    CHECK(!control::physical_devices({4, 0}, physical, mapped) && mapped.empty());

    fused::Config config;
    config.n = 16384;
    config.slots = 2;
    std::string error;
    CHECK(fused::validate_config(config, 4, error) && error.empty());
    config.n = 16383;
    CHECK(!fused::validate_config(config, 4, error));
    alignas(16) float row[8] = {};
    CHECK(reinterpret_cast<std::uintptr_t>(row) % 16 == 0);
    CHECK(reinterpret_cast<std::uintptr_t>(row + 1) % 16 != 0);

    unsigned fused_seq[4] = {};
    unsigned native_seq[4] = {3, 5, 7, 9};
    for (int member = 0; member < 4; ++member) {
        ++native_seq[member];
        CHECK(fused_seq[member] == 0);
    }
    for (int layer = 0; layer < 40; ++layer) {
        for (int call = 0; call < 2; ++call) {
            for (int member = 0; member < 4; ++member) {
                const unsigned expected = static_cast<unsigned>(2 * layer + call + 1);
                CHECK(control::next_sequence(fused_seq[member]) == expected);
                int slot = -1;
                CHECK(fused::slot_for_sequence(expected, 2, slot));
                CHECK(slot == static_cast<int>(expected % 2));
                unsigned previous = 0;
                CHECK(fused::reuse_sequence(expected, 2, previous));
                if (expected > 2) CHECK(previous == expected - 2);
                CHECK(native_seq[member] == static_cast<unsigned>(4 + 2 * member));
            }
        }
    }
    for (int member = 0; member < 4; ++member) {
        CHECK(fused_seq[member] == 80);
        const unsigned before = fused_seq[member];
        ++native_seq[member];
        CHECK(fused_seq[member] == before);
    }

    control::ErrorState broken;
    CHECK(broken.error().empty());
    for (int member = 0; member < 4; ++member)
        broken.fail("member " + std::to_string(member) + " sticky error");
    const std::string persistent = broken.error();
    for (int member = 0; member < 4; ++member)
        CHECK(persistent.find("member " + std::to_string(member) + " sticky error") != std::string::npos);
    CHECK(broken.error() == persistent);

    std::puts("DS4_FUSED_AR host policy: strict modes/stages, decode-only routing, mapping, sequences, slots and sticky errors passed");
}
