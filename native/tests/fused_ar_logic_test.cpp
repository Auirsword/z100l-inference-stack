#include "fused_ar.hpp"
#include "queue_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace {

namespace fused = strata::ds4::p2p::fused;
using fused::QueueInputLayout;
using fused::QueuePolicy;
using fused::make_queue_input_layout;
using fused::make_queue_policy;
using fused::queue_input_offset;
using fused::validate_queue_arguments;
int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool same_float(float a, float b) {
    std::uint32_t aa = 0, bb = 0;
    std::memcpy(&aa, &a, sizeof(aa));
    std::memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

void test_shapes() {
    fused::Config config;
    config.n = 16384;
    std::string error = "stale";
    expect(fused::validate_config(config, 2, error) && error.empty(), "TP2 shape is valid");
    expect(fused::validate_config(config, 4, error) && error.empty(), "TP4 shape is valid");
    for (int members : {-1, 0, 1, 3, 5}) {
        expect(!fused::validate_config(config, members, error) && !error.empty(), "unsupported member count is rejected");
    }
    config.n = 0;
    expect(!fused::validate_config(config, 2, error), "zero elements are rejected");
    config.n = 6;
    expect(!fused::validate_config(config, 2, error), "non-float4 element count is rejected");
    config.n = 4;
    config.slots = 0;
    expect(!fused::validate_config(config, 2, error), "zero slots are rejected");
    config.slots = 2;
    config.max_blocks = 0;
    expect(!fused::validate_config(config, 2, error), "zero blocks are rejected");
    config.max_blocks = 65;
    expect(!fused::validate_config(config, 2, error), "blocks above fixed limit are rejected");
    config.max_blocks = 64;
    config.timeout_cycles = 0;
    expect(!fused::validate_config(config, 2, error), "zero timeout budget is rejected");
    expect(fused::kErrorRingTimeout != fused::kErrorReadyTimeout &&
               fused::kErrorReadyTimeout != fused::kErrorPeer && fused::kErrorNone == 0,
           "device errors have distinct nonzero codes");
}

void test_initialization_stages() {
    fused::Config config;
    config.n = 16384;
    expect(config.init_stage == 4, "full initialization is the default stage");
    std::string error;
    for (int stage = 0; stage <= 4; ++stage) {
        config.init_stage = stage;
        expect(fused::validate_config(config, 2, error) && error.empty(), "every diagnostic initialization stage has a valid shape");
        expect(fused::stage_allows_launch(stage) == (stage == 4), "only complete initialization may launch");
    }
    for (int stage : {-1, 5}) {
        config.init_stage = stage;
        expect(!fused::validate_config(config, 2, error) && !error.empty(), "out-of-range initialization stages are rejected");
        expect(!fused::stage_allows_launch(stage), "invalid initialization stages cannot launch");
    }
}

void test_slots_and_read_complete_boundary() {
    for (int slots : {1, 2, 4}) {
        for (unsigned int seq = 1; seq <= static_cast<unsigned int>(3 * slots + 2); ++seq) {
            int slot = -1;
            expect(fused::slot_for_sequence(seq, slots, slot), "nonzero sequence maps to a slot");
            expect(slot == static_cast<int>(seq % static_cast<unsigned int>(slots)),
                   "slot mapping matches existing AR seq modulo slots");
            unsigned int previous = 0xdeadbeefu;
            expect(fused::reuse_sequence(seq, slots, previous), "valid reuse query succeeds");
            if (seq <= static_cast<unsigned int>(slots)) {
                expect(previous == 0xdeadbeefu, "first slot occupancy has no read-complete dependency");
            } else {
                expect(previous == seq - static_cast<unsigned int>(slots),
                       "reuse depends on the previous occupant of exactly this slot");
                expect(!fused::sequence_at_least(previous - 1, previous),
                       "one-generation-old done is insufficient for reuse");
                expect(fused::sequence_at_least(previous, previous) &&
                           fused::sequence_at_least(previous + 1, previous),
                       "equal or newer done releases slot reuse");
            }
        }
    }
    int slot = -1;
    unsigned int previous = 0;
    expect(!fused::slot_for_sequence(0, 2, slot), "sequence zero is rejected");
    expect(!fused::slot_for_sequence(1, 0, slot), "zero-slot mapping is rejected");
    expect(!fused::reuse_sequence(0, 2, previous), "sequence zero has no valid reuse rule");
    expect(fused::sequence_at_least(0u, 0xffffffffu), "signed sequence comparison handles uint32 wrap");
    expect(!fused::sequence_at_least(0xfffffff0u, 16u), "old pre-wrap flag is not newer than post-wrap sequence");
}

void test_layout_and_launch_shape() {
    fused::Config config;
    config.n = 1024;
    config.slots = 3;
    std::size_t stride = 0, bytes = 0;
    expect(fused::outbox_layout(config, stride, bytes), "guarded outbox layout is valid");
    expect(stride == 1024 + 2 * fused::kGuardFloats, "slot stride includes both guards");
    expect(bytes == 3 * stride * sizeof(float), "allocation covers every guarded slot");
    for (int slot = 0; slot < config.slots; ++slot) {
        expect(fused::outbox_payload_offset(slot, config.n) == static_cast<std::size_t>(slot) * stride +
                                                                  fused::kGuardFloats,
               "payload follows the selected slot leading guard");
    }
    config.n = 6;
    expect(!fused::outbox_layout(config, stride, bytes), "invalid float shape has no outbox layout");
    config.n = std::numeric_limits<int>::max() - 3;
    config.slots = std::numeric_limits<int>::max();
    expect(!fused::outbox_layout(config, stride, bytes), "overflowing outbox layout is rejected");

    config = fused::Config{};
    config.n = 4 * fused::kThreads * 100;
    config.max_blocks = 64;
    expect(fused::choose_blocks(config, 60, 1) == 60, "launch shape respects co-resident block capacity");
    expect(fused::choose_blocks(config, 60, 2) == 64, "launch shape respects configured block cap");
    expect(fused::choose_blocks(config, 0, 2) == 0 && fused::choose_blocks(config, 60, 0) == 0,
           "invalid occupancy is rejected rather than assuming a grid barrier");
    config.n = 4 * fused::kThreads;
    expect(fused::choose_blocks(config, 60, 2) == 1, "small shape uses one block");
}

void test_fixed_order_host_reference() {
    const float tp4[] = {1.0e20f, -1.0e20f, 3.0f, 4.0f};
    expect(same_float(fused::reference_sum(tp4, 4), 7.0f), "host reference preserves member 0-to-3 FP32 order");
    const float tp2[] = {1.25f, -2.5f};
    expect(same_float(fused::reference_sum(tp2, 2), -1.25f), "TP2 host reference has exact expected bits");
}

void test_queued_policy() {
    const QueuePolicy normal = make_queue_policy(false, 8);
    expect(!normal.queue_mode && normal.queue_depth == 0 && normal.intermediate_outputs_checked,
           "default mode retains per-sequence output checks and has no queue depth");
    const QueuePolicy queued = make_queue_policy(true, 80);
    expect(queued.queue_mode && queued.queue_depth == 80 && !queued.intermediate_outputs_checked,
           "queued report declares depth 80 and excludes intermediate output verification");
    std::string error;
    expect(validate_queue_arguments(false, 1, error) && error.empty(), "non-queued skew remains valid");
    expect(validate_queue_arguments(true, 0, error) && error.empty(), "queued without a skew kernel is valid");
    expect(!validate_queue_arguments(true, 1, error) && error == "--queued requires --skew-cycles 0",
           "queued rejects every nonzero skew setting");
}

void test_queued_input_layout() {
    constexpr std::size_t image_bytes = (16384 + 2 * fused::kGuardFloats) * sizeof(float);
    QueueInputLayout layout;
    expect(make_queue_input_layout(2, 80, image_bytes, layout) && layout.total_bytes == 160 * image_bytes,
           "TP2 queue of 80 allocates exactly 160 guarded images");
    std::size_t offset = 1;
    expect(queue_input_offset(layout, 0, 0, offset) && offset == 0, "first member starts at the pinned allocation base");
    expect(queue_input_offset(layout, 0, 1, offset) && offset == image_bytes, "member inputs are contiguous and stable");
    expect(queue_input_offset(layout, 79, 1, offset) && offset == 159 * image_bytes && offset + image_bytes == layout.total_bytes,
           "last member of the last sequence ends at the validated allocation size");
    expect(!queue_input_offset(layout, 80, 0, offset) && offset == 0, "an index past the queued sequence range is refused");
    expect(!queue_input_offset(layout, 0, 2, offset) && offset == 0, "an index past the TP2 member range is refused");
    expect(!make_queue_input_layout(0, 80, image_bytes, layout) && !make_queue_input_layout(2, 0, image_bytes, layout) &&
               !make_queue_input_layout(2, 80, 0, layout),
           "empty member, sequence and image dimensions are refused");
    const std::size_t overflowing_image = std::numeric_limits<std::size_t>::max() / 2 + 1;
    expect(!make_queue_input_layout(2, 1, overflowing_image, layout), "pinned allocation byte overflow is refused");
}

}  // namespace

int main() {
    test_shapes();
    test_initialization_stages();
    test_slots_and_read_complete_boundary();
    test_layout_and_launch_shape();
    test_fixed_order_host_reference();
    test_queued_policy();
    test_queued_input_layout();
    if (failures == 0)
        std::puts("fused AR host logic: shapes, init stages, ring, layout, reference and queued-policy/input-layout checks passed");
    return failures == 0 ? 0 : 1;
}
