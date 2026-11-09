#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::ds4::p2p::fused {

struct QueuePolicy {
    bool queue_mode = false;
    int queue_depth = 0;
    bool intermediate_outputs_checked = true;
};

inline QueuePolicy make_queue_policy(bool queued, int sequences) {
    return QueuePolicy{queued, queued ? sequences : 0, !queued};
}

inline bool validate_queue_arguments(bool queued, std::uint64_t skew_cycles, std::string& error) {
    if (queued && skew_cycles != 0) {
        error = "--queued requires --skew-cycles 0";
        return false;
    }
    return true;
}

struct QueueInputLayout {
    int members = 0;
    int sequences = 0;
    std::size_t image_bytes = 0;
    std::size_t total_bytes = 0;
};

inline bool make_queue_input_layout(int members, int sequences, std::size_t image_bytes, QueueInputLayout& layout) {
    layout = QueueInputLayout{};
    if (members <= 0 || sequences <= 0 || image_bytes == 0) return false;
    const std::size_t m = static_cast<std::size_t>(members), n = static_cast<std::size_t>(sequences);
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (m > maximum / n) return false;
    const std::size_t count = m * n;
    if (count > maximum / image_bytes) return false;
    layout = QueueInputLayout{members, sequences, image_bytes, count * image_bytes};
    return true;
}

inline bool queue_input_offset(const QueueInputLayout& layout, int sequence_index, int member, std::size_t& offset) {
    offset = 0;
    QueueInputLayout valid;
    if (!make_queue_input_layout(layout.members, layout.sequences, layout.image_bytes, valid) ||
        sequence_index < 0 || sequence_index >= layout.sequences || member < 0 || member >= layout.members)
        return false;
    const std::size_t index = static_cast<std::size_t>(sequence_index) * static_cast<std::size_t>(layout.members) +
                              static_cast<std::size_t>(member);
    offset = index * layout.image_bytes;
    return true;
}

}  // namespace strata::ds4::p2p::fused
