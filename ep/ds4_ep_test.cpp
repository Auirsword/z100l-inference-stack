#include "../src/ds4/ds4_ep.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

using namespace strata::ds4::ep;

int main() {
    Config config;
    std::string config_error;
    assert(validate_ep4(config, config_error) && config_error.empty());
    config.gpus = {0, 1, 2};
    assert(!validate_ep4(config, config_error) && config_error.find("--gpus 0,1,2,3") != std::string::npos);
    config = Config{};
    config.sync = false;
    assert(!validate_ep4(config, config_error));
    config = Config{};
    config.layers = 39;
    assert(!validate_ep4(config, config_error));

    Layout layout(384, 4);
    assert(layout.owner(0) == 0 && layout.owner(95) == 0);
    assert(layout.owner(96) == 1 && layout.owner(191) == 1);
    assert(layout.owner(192) == 2 && layout.owner(383) == 3);
    bool rejected = false;
    try { (void) layout.owner(384); } catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);

    const int experts[] = {0, 96, 192, 383, 96, 5};
    const float weights[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    Plan plan = plan_window(1, 6, experts, weights, 8, layout,
                            [](int expert) { return expert == 0; },
                            [](int, int expert) { return expert != 5; });
    assert(plan.picks.size() == 6 && plan.groups.size() == 4 && plan.cpu_picks.size() == 1);
    assert(plan.picks[0].tier == Tier::main_vram);
    assert(plan.picks[1].tier == Tier::owner_vram && plan.picks[4].tier == Tier::owner_vram);
    assert(plan.picks[5].tier == Tier::cpu_cold);
    size_t remote = 0;
    for (const Group& group : plan.groups) remote += group.picks.size();
    assert(remote == 5);
    const Group& repeated = plan.groups[1];
    assert(repeated.owner == 1 && repeated.expert == 96 && repeated.picks.size() == 2);
    assert(repeated.picks[0] == 1 && repeated.picks[1] == 4);
    for (size_t i = 0; i < plan.picks.size(); ++i) {
        assert(plan.picks[i].weight == weights[i]);
        assert(output_row(plan.picks[i], 8) == (int) i);
    }
    const int two_tokens[16] = {0, 1, 2, 3, 4, 5, -1, -1, 96, 97, 192, 193, 383, 6, -1, -1};
    float two_weights[16] = {};
    for (int i = 0; i < 16; ++i) two_weights[i] = 0.25f * i;
    Plan two = plan_window(2, 6, two_tokens, two_weights, 8, layout,
                           [](int expert) { return expert < 96; }, [](int, int) { return true; });
    const int expected_experts[12] = {0, 1, 2, 3, 4, 5, 96, 97, 192, 193, 383, 6};
    for (int i = 0; i < 12; ++i) {
        const int row = i / 6 * 8 + i % 6;
        assert(two.picks[(size_t) i].expert == expected_experts[i]);
        assert(two.picks[(size_t) i].weight == two_weights[row]);
        assert(output_row(two.picks[(size_t) i], 8) == row);
    }
    assert(two.picks[11].tier == Tier::main_vram);
    for (const Group& group : two.groups)
        for (const size_t index : group.picks) {
            assert(index < two.picks.size());
            assert(two.picks[index].token == (int) (index / 6));
            assert(two.picks[index].pick == (int) (index % 6));
            assert(two.picks[index].expert == expected_experts[index]);
            assert(output_row(two.picks[index], 8) == (int) (index / 6 * 8 + index % 6));
        }
    Plan two_cold = plan_window(2, 6, two_tokens, two_weights, 8, layout,
                                [](int) { return false; }, [](int, int) { return false; });
    assert(two_cold.cpu_picks.size() == two_cold.picks.size());
    for (const size_t index : two_cold.cpu_picks) {
        assert(index < two_cold.picks.size());
        assert(two_cold.picks[index].token == (int) (index / 6));
        assert(two_cold.picks[index].pick == (int) (index % 6));
        assert(two_cold.picks[index].expert == expected_experts[index]);
    }

    const float compact[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const int destinations[] = {5, 1, 13};
    float full[16 * 2];
    scatter_unweighted_rows(compact, destinations, 3, 2, full, 16, 16);
    assert(full[0] == 0.0f && full[1] == 0.0f);   // compact row 0 is global row 5, not combine row 0
    assert(full[1 * 2] == 3.0f && full[1 * 2 + 1] == 4.0f);
    assert(full[5 * 2] == 1.0f && full[5 * 2 + 1] == 2.0f);
    assert(full[13 * 2] == 5.0f && full[13 * 2 + 1] == 6.0f);
    for (int row = 0; row < 16; ++row) {
        if (row == 1 || row == 5 || row == 13) continue;
        assert(full[row * 2] == 0.0f && full[row * 2 + 1] == 0.0f);
    }
    const int duplicate[] = {5, 5};
    rejected = false;
    try { scatter_unweighted_rows(compact, duplicate, 2, 2, full, 16, 16); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
    const int outside[] = {13};
    rejected = false;
    try { scatter_unweighted_rows(compact, outside, 1, 2, full, 16, 12); }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);

    Completion completion(plan, 4);
    assert(!completion.can_combine());
    for (int owner = 0; owner < 4; ++owner) completion.complete_owner(owner);
    assert(!completion.can_combine());
    completion.complete_cpu();
    assert(completion.can_combine());

    Completion reordered(plan, 4);
    reordered.complete_cpu();
    for (int owner : {3, 1, 2, 0}) reordered.complete_owner(owner);
    assert(reordered.can_combine());
    reordered.fail("late owner STOP");
    assert(!reordered.can_combine() && reordered.error() == "late owner STOP");

    const int reverse_destinations[] = {13, 5, 1};
    const float reverse_compact[] = {5.0f, 6.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    float reverse_full[16 * 2];
    scatter_unweighted_rows(reverse_compact, reverse_destinations, 3, 2, reverse_full, 16, 16);
    assert(std::memcmp(reverse_full, full, sizeof full) == 0);

    Completion failed(plan, 4);
    failed.complete_owner(0);             // a valid partial answer is still not a combinable result
    failed.complete_cpu();
    failed.fail("injected owner timeout");
    failed.complete_owner(1);
    assert(!failed.can_combine() && failed.error() == "injected owner timeout");

    Plan cold = plan_window(1, 1, experts + 5, weights + 5, 8, layout,
                            [](int) { return false; }, [](int, int) { return false; });
    Completion cold_done(cold, 4);
    assert(!cold_done.can_combine());
    cold_done.complete_cpu();
    assert(cold_done.can_combine());

    const float invalid = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try {
        (void) plan_window(1, 1, experts, &invalid, 8, layout,
                           [](int) { return true; }, [](int, int) { return true; });
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try {
        (void) plan_window(2, 6, two_tokens, two_weights, 5, layout,
                           [](int) { return true; }, [](int, int) { return true; });
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::vector<std::pair<int, int>> chunks;
    for_each_chunk(24, 16, [&](int first, int count) { chunks.push_back({first, count}); });
    assert(chunks.size() == 2 && chunks[0].first == 0 && chunks[0].second == 16);
    assert(chunks[1].first == 16 && chunks[1].second == 8);
    rejected = false;
    try { for_each_chunk(24, 17, [](int, int) {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::puts("DS4 EP routing: owner coverage, three-tier picks, repeated experts and fail-closed completion OK");
    return 0;
}
