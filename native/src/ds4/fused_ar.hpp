#pragma once

#include "ds4_p2p.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::ds4::p2p::fused {

constexpr int kMaxMembers = 4;
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 64;
constexpr std::size_t kGuardBytes = 64;
constexpr std::size_t kGuardFloats = kGuardBytes / sizeof(float);

struct Config {
    int n = 0;
    int slots = 2;
    int max_blocks = kMaxBlocks;
    std::uint64_t timeout_cycles = static_cast<std::uint64_t>(1) << 34;
    bool fp16_wire = false;
    int init_stage = 4;
};

inline bool stage_allows_launch(int stage) { return stage == 4; }

struct DeviceError {
    unsigned int code = 0;
    unsigned int seq = 0;
    unsigned int stage = 0;
    unsigned int detail = 0;
};

enum ErrorCode : unsigned int {
    kErrorNone = 0,
    kErrorInvalid = 1,
    kErrorRingTimeout = 2,
    kErrorReadyTimeout = 3,
    kErrorPeer = 4,
};

enum ErrorStage : unsigned int {
    kStageNone = 0,
    kStageEntry = 1,
    kStageRing = 2,
    kStageReady = 3,
};

// Device-resident UVA pointers. The host writes this table once during initialization.
struct PeerTable {
    float* outboxes[kMaxMembers] = {};
    unsigned int* ready[kMaxMembers] = {};
    unsigned int* done[kMaxMembers] = {};
    DeviceError* errors[kMaxMembers] = {};
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_FUSED_HOST_DEVICE __host__ __device__
#else
#define DS4_FUSED_HOST_DEVICE
#endif

DS4_FUSED_HOST_DEVICE inline bool sequence_at_least(unsigned int observed, unsigned int required) {
    return static_cast<int>(observed - required) >= 0;
}

DS4_FUSED_HOST_DEVICE inline bool slot_for_sequence(unsigned int seq, int slots, int& slot) {
    if (seq == 0 || slots <= 0) return false;
    slot = static_cast<int>(seq % static_cast<unsigned int>(slots));
    return true;
}

DS4_FUSED_HOST_DEVICE inline bool reuse_sequence(unsigned int seq, int slots, unsigned int& previous) {
    if (seq == 0 || slots <= 0) return false;
    if (seq <= static_cast<unsigned int>(slots)) return true;
    previous = seq - static_cast<unsigned int>(slots);
    return true;
}

inline bool validate_config(const Config& config, int members, std::string& error) {
    error.clear();
    if (members != 2 && members != 4) {
        error = "fused AR supports exactly TP2 or TP4";
        return false;
    }
    if (config.n <= 0 || config.n % 4 != 0) {
        error = "fused AR requires a positive FP32 element count divisible by four";
        return false;
    }
    if (config.slots <= 0) {
        error = "fused AR requires at least one ring slot";
        return false;
    }
    if (config.max_blocks <= 0 || config.max_blocks > kMaxBlocks) {
        error = "fused AR max_blocks must be in [1,64]";
        return false;
    }
    if (config.timeout_cycles == 0) {
        error = "fused AR timeout cycle budget must be nonzero";
        return false;
    }
    if (config.init_stage < 0 || config.init_stage > 4) {
        error = "fused AR initialization stage must be in [0,4]";
        return false;
    }
    return true;
}

inline bool outbox_layout(const Config& config, std::size_t& slot_stride_floats,
                          std::size_t& allocation_bytes) {
    if (config.n <= 0 || config.n % 4 != 0 || config.slots <= 0) return false;
    const std::size_t n = static_cast<std::size_t>(config.n);
    const std::size_t slots = static_cast<std::size_t>(config.slots);
    const std::size_t maximum = std::vector<float>().max_size();
    if (n > maximum - 2 * kGuardFloats) return false;
    slot_stride_floats = n + 2 * kGuardFloats;
    if (slots > maximum / slot_stride_floats) return false;
    const std::size_t floats = slots * slot_stride_floats;
    if (floats > static_cast<std::size_t>(-1) / sizeof(float)) return false;
    allocation_bytes = floats * sizeof(float);
    return true;
}

DS4_FUSED_HOST_DEVICE inline std::size_t outbox_payload_offset(int slot, int n) {
    return static_cast<std::size_t>(slot) * (static_cast<std::size_t>(n) + 2 * kGuardFloats) + kGuardFloats;
}

#undef DS4_FUSED_HOST_DEVICE

inline int choose_blocks(const Config& config, int multiprocessors, int occupancy_per_multiprocessor) {
    if (config.n <= 0 || config.max_blocks <= 0 || multiprocessors <= 0 || occupancy_per_multiprocessor <= 0)
        return 0;
    const int n4 = config.n / 4;
    const int desired = (n4 + kThreads - 1) / kThreads;
    const long long capacity = static_cast<long long>(multiprocessors) * occupancy_per_multiprocessor;
    const int capped = capacity > kMaxBlocks ? kMaxBlocks : static_cast<int>(capacity);
    const int blocks = desired < config.max_blocks ? desired : config.max_blocks;
    return blocks < capped ? blocks : capped;
}

inline float reference_sum(const float* values, int members) {
    volatile float sum = values[0];
    for (int member = 1; member < members; ++member) sum = sum + values[member];
    return sum;
}

// One instance represents the same fixed member order as PeerAccess::devices(). Launch seq 1,2,... on every member,
// serialize launches per member, and do not run competing kernels that reduce the occupancy-capped resident grid.
// The caller must keep PeerAccess alive and enabled until after close(), and quiesce launches before close/destruction.
// x must be 16-byte aligned, must not alias any outbox, and stream must belong to the member's device.
// launch() performs one kernel launch and no data/flag transfer or synchronization.
// synchronize() is a diagnostic/validation boundary: it synchronizes and reads the sticky device error, not AR data.
// Check every member and reject the whole group's result if any fails; after a device error, reinitialize the group.
// HIP outboxes and synchronization/error state require fine-grained allocations, with release/acquire flag access;
// unsupported fine-grained allocation is an error, never a coarse-memory fallback.
class Group final {
public:
    Group(PeerAccess& peers, const Config& config);
    ~Group();

    Group(const Group&) = delete;
    Group& operator=(const Group&) = delete;
    Group(Group&&) = delete;
    Group& operator=(Group&&) = delete;

    bool initialize(std::string& error);
    bool launch(int member, float* x, unsigned int seq, Stream stream, std::string& error);
    bool synchronize(int member, Stream stream, DeviceError& device_error, std::string& error);
    bool close(std::string& error);

    bool initialized() const { return initialized_; }
    int members() const { return static_cast<int>(members_.size()); }
    int blocks(int member) const;
    float* outbox_raw(int member) const;
    std::size_t outbox_bytes() const { return outbox_bytes_; }

private:
    struct MemberResources {
        int device = -1;
        int blocks = 0;
        float* outbox = nullptr;
        unsigned int* ready = nullptr;
        unsigned int* done = nullptr;
        unsigned int* counters = nullptr;
        DeviceError* error = nullptr;
        PeerTable* table = nullptr;
    };

    bool release_resources(std::string& error);

    PeerAccess* peers_;
    Config config_;
    std::vector<MemberResources> members_;
    std::size_t outbox_bytes_ = 0;
    bool initialized_ = false;
};

}  // namespace strata::ds4::p2p::fused
