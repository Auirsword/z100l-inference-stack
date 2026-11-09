#include "ds4_p2p.hpp"
#include "p2p_verify.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::ds4::p2p::DirectedPair;
using strata::ds4::p2p::Error;
using strata::ds4::p2p::PeerAccess;
using strata::ds4::p2p::Runtime;
using strata::ds4::p2p::Stream;
using Pair = std::pair<int, int>;
using Bytes = std::vector<std::uint8_t>;

int failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::string key(const Pair& pair) {
    return std::to_string(pair.first) + ">" + std::to_string(pair.second);
}

class FakeRuntime final : public Runtime {
public:
    static constexpr Error kAlreadyEnabled = 704;
    static constexpr Error kInvalid = 701;
    static constexpr Error kInjected = 799;

    Error get_device(int* device) override {
        if (get_status != 0) return get_status;
        if (device == nullptr) return kInvalid;
        *device = current;
        return 0;
    }

    Error set_device(int device) override {
        if (device < 0 || device >= 8 || set_status != 0) return kInvalid;
        current = device;
        return 0;
    }

    Error can_access_peer(int* can_access, int device, int peer_device) override {
        operations.push_back("can:" + key({device, peer_device}));
        if (can_errors.count({device, peer_device}) != 0) return kInjected;
        if (can_access == nullptr || device < 0 || peer_device < 0 || device >= 8 || peer_device >= 8 ||
            device == peer_device)
            return kInvalid;
        *can_access = denied.count({device, peer_device}) == 0 ? 1 : 0;
        return 0;
    }

    Error enable_peer_access(int peer_device, unsigned int flags) override {
        const Pair pair{current, peer_device};
        operations.push_back("enable:" + key(pair));
        if (flags != 0 || peer_device < 0 || peer_device >= 8 || current == peer_device) return kInvalid;
        if (enable_errors.count(pair) != 0) return kInjected;
        if (active.count(pair) != 0) return kAlreadyEnabled;
        active.insert(pair);
        return 0;
    }

    Error disable_peer_access(int peer_device) override {
        const Pair pair{current, peer_device};
        operations.push_back("disable:" + key(pair));
        if (disable_errors.count(pair) != 0) return kInjected;
        if (active.erase(pair) == 0) return kInvalid;
        return 0;
    }

    Error memcpy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                            std::size_t bytes, Stream stream) override {
        (void) dst;
        (void) src;
        (void) stream;
        operations.push_back("copy:" + key({src_device, dst_device}) + ":" + std::to_string(bytes));
        ++copy_calls;
        last_bytes = bytes;
        last_copy_device = current;
        if (copy_status != 0) return copy_status;
        if (current != dst_device || active.count({dst_device, src_device}) == 0) return kInvalid;
        return 0;
    }

    Error peer_access_already_enabled() const override { return kAlreadyEnabled; }

    const char* error_string(Error error) const override {
        if (error == 0) return "success";
        if (error == kAlreadyEnabled) return "already enabled";
        if (error == kInjected) return "injected failure";
        return "invalid fake runtime call";
    }

    int current = 0;
    Error get_status = 0;
    Error set_status = 0;
    Error copy_status = 0;
    int copy_calls = 0;
    int last_copy_device = -1;
    std::size_t last_bytes = 0;
    std::set<Pair> active;
    std::set<Pair> denied;
    std::set<Pair> can_errors;
    std::set<Pair> enable_errors;
    std::set<Pair> disable_errors;
    std::vector<std::string> operations;
};

std::size_t operation_count(const FakeRuntime& fake, const std::string& prefix) {
    return static_cast<std::size_t>(std::count_if(fake.operations.begin(), fake.operations.end(),
                                                   [&](const std::string& operation) {
                                                       return operation.compare(0, prefix.size(), prefix) == 0;
                                                   }));
}

void test_directed_pairs_and_copy() {
    FakeRuntime fake;
    {
        PeerAccess peers(fake, {2, 0, 3, 1});
        std::string error;
        expect(peers.enable_all(error), "four explicit devices enable");
        expect(error.empty(), "four explicit devices enable without diagnostic");
        expect(operation_count(fake, "can:") == 12, "four devices produce twelve directed capability queries");
        expect(operation_count(fake, "enable:") == 12, "four devices produce twelve directed enables");
        expect(fake.active.size() == 12, "all directed links are active");
        expect(peers.ready(), "fabric becomes ready");
        expect(peers.is_enabled(0, 2) && peers.is_enabled(2, 0), "directions are independent");
        expect(fake.current == 0, "enable restores the caller device");

        int x = 0, y = 0;
        std::size_t bytes = 0;
        expect(peers.copy_peer_async(&y, 1, &x, 2, 17, nullptr, bytes, error), "explicit peer copy enqueues");
        expect(bytes == 17 && fake.last_bytes == 17, "peer copy reports exact byte count");
        expect(fake.copy_calls == 1 && fake.last_copy_device == 1, "peer copy is issued on the destination device");
        expect(fake.current == 0, "peer copy restores the caller device");

        expect(peers.enable_all(error) == false, "duplicate enable is refused");
        expect(peers.disable_all(error), "full teardown succeeds");
        expect(fake.active.empty(), "owned links are disabled");
        expect(operation_count(fake, "disable:") == 12, "twelve owned links are disabled");
        std::vector<std::string> disables;
        for (const std::string& operation : fake.operations)
            if (operation.compare(0, 8, "disable:") == 0) disables.push_back(operation);
        expect(!disables.empty() && disables.front() == "disable:1>3", "teardown is reverse enable order");
        expect(!disables.empty() && disables.back() == "disable:2>0", "teardown ends at first enabled link");
        const std::size_t count = fake.operations.size();
        expect(peers.disable_all(error), "repeated teardown is a successful no-op");
        expect(fake.operations.size() == count, "repeated teardown makes no runtime calls");
    }
    expect(fake.operations.size() == 37, "destructor does not repeat an explicit complete teardown");
}

void test_illegal_devices_and_copies() {
    for (const std::vector<int>& devices : {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{0, 0},
                                            std::vector<int>{-1, 0}}) {
        FakeRuntime fake;
        PeerAccess peers(fake, devices);
        std::string error;
        expect(!peers.enable_all(error), "invalid explicit device list is rejected");
        expect(!error.empty(), "invalid explicit device list has a diagnostic");
        expect(fake.operations.empty(), "invalid explicit device list causes no peer operation");
    }

    FakeRuntime fake;
    PeerAccess peers(fake, {0, 1});
    std::string error;
    int value = 0;
    std::size_t bytes = 99;
    expect(!peers.copy_peer_async(&value, 1, &value, 0, 1, nullptr, bytes, error),
           "copy before enable is refused");
    expect(bytes == 0 && fake.copy_calls == 0, "refused copy has zero bytes and no runtime fallback");
    expect(peers.enable_all(error), "valid pair enables");
    expect(!peers.copy_peer_async(&value, 1, &value, 1, 1, nullptr, bytes, error), "same-device peer copy is refused");
    expect(!peers.copy_peer_async(nullptr, 1, &value, 0, 1, nullptr, bytes, error), "null destination is refused");
    expect(!peers.copy_peer_async(&value, 7, &value, 0, 1, nullptr, bytes, error), "unlisted destination is refused");
    expect(!peers.copy_peer_async(&value, 1, &value, 7, 1, nullptr, bytes, error), "unlisted source is refused");
    expect(bytes == 0 && fake.copy_calls == 0, "all invalid copies avoid the runtime and host staging");
    expect(peers.disable_all(error), "invalid-copy test teardown");
}

void test_capability_and_enable_failures_cleanup() {
    {
        FakeRuntime fake;
        fake.denied.insert({1, 0});
        PeerAccess peers(fake, {0, 1});
        std::string error;
        expect(!peers.enable_all(error), "unsupported directed pair fails setup");
        expect(error.find("can_access=0") != std::string::npos, "unsupported pair reports direction and capability");
        expect(fake.active.empty(), "unsupported pair rolls back earlier owned link");
        expect(operation_count(fake, "disable:") == 1, "unsupported pair cleanup disables only the earlier link");
        expect(!peers.ready(), "failed setup is not ready");
    }
    {
        FakeRuntime fake;
        fake.can_errors.insert({0, 1});
        PeerAccess peers(fake, {0, 1});
        std::string error;
        expect(!peers.enable_all(error), "capability API error fails setup");
        expect(error.find("799") != std::string::npos, "capability API error preserves status");
        expect(operation_count(fake, "enable:") == 0, "capability failure does not attempt enable");
    }
    {
        FakeRuntime fake;
        fake.enable_errors.insert({1, 0});
        PeerAccess peers(fake, {0, 1});
        std::string error;
        expect(!peers.enable_all(error), "enable API error fails setup");
        expect(error.find("799") != std::string::npos, "enable API error preserves status");
        expect(fake.active.empty(), "enable failure rolls back earlier owned link");
        expect(operation_count(fake, "disable:") == 1, "enable failure performs one cleanup");
    }
}

void test_preexisting_link_is_not_owned() {
    FakeRuntime fake;
    fake.active.insert({0, 1});
    {
        PeerAccess peers(fake, {0, 1});
        std::string error;
        expect(peers.enable_all(error), "already-enabled link is accepted");
        expect(peers.is_enabled(0, 1), "already-enabled link is usable");
        int x = 0, y = 0;
        std::size_t bytes = 0;
        expect(peers.copy_peer_async(&y, 0, &x, 1, 9, nullptr, bytes, error),
               "already-enabled destination-to-source link supports explicit copy");
        expect(peers.disable_all(error), "mixed ownership teardown succeeds");
        expect(fake.active.count({0, 1}) == 1, "already-enabled external link remains enabled");
        expect(fake.active.count({1, 0}) == 0, "owned reverse link is disabled");
        expect(operation_count(fake, "disable:") == 1, "external link is never disabled");
    }
    expect(fake.active.count({0, 1}) == 1, "destructor still leaves external link alone");
}

void test_disable_failure_and_retry() {
    FakeRuntime fake;
    fake.disable_errors.insert({1, 0});
    PeerAccess peers(fake, {0, 1, 2});
    std::string error;
    expect(peers.enable_all(error), "disable-retry setup enables six links");
    expect(!peers.disable_all(error), "disable failure is reported");
    expect(error.find("disable_peer_access") != std::string::npos, "disable failure identifies operation");
    expect(operation_count(fake, "disable:") == 6, "teardown attempts every owned link despite one failure");
    expect(fake.active.size() == 1 && fake.active.count({1, 0}) == 1, "only failed link remains active");
    const std::size_t before_retry = fake.operations.size();
    expect(!peers.disable_all(error), "failed link is retained for a visible retry");
    expect(operation_count(fake, "disable:") == 7, "retry does not duplicate successful releases");
    expect(fake.operations.size() == before_retry + 1, "retry touches only the failed peer link");
    fake.disable_errors.clear();
    expect(peers.disable_all(error), "retry succeeds after runtime recovers");
    expect(fake.active.empty(), "successful retry releases final link");
    const std::size_t final_count = fake.operations.size();
    expect(peers.disable_all(error), "post-retry teardown is idempotent");
    expect(fake.operations.size() == final_count, "post-retry teardown has no duplicate free");
}

void test_copy_runtime_error_has_no_fallback() {
    FakeRuntime fake;
    fake.copy_status = FakeRuntime::kInjected;
    PeerAccess peers(fake, {0, 1});
    std::string error;
    expect(peers.enable_all(error), "copy-error setup enables");
    int x = 0, y = 0;
    std::size_t bytes = 123;
    expect(!peers.copy_peer_async(&y, 1, &x, 0, 4096, nullptr, bytes, error), "peer runtime error is returned");
    expect(bytes == 0 && fake.last_bytes == 4096, "failed peer enqueue reports zero completed enqueue bytes");
    expect(error.find("memcpy_peer_async") != std::string::npos, "copy error names the explicit peer API");
    expect(operation_count(fake, "copy:") == 1, "copy failure is not retried through another path");
    expect(peers.disable_all(error), "copy-error teardown succeeds");
}

void test_guards_alignment_and_bounds() {
    namespace verify = strata::ds4::p2p::verify;
    for (std::size_t alignment : {std::size_t{1}, std::size_t{4}, std::size_t{16}}) {
        const std::size_t offset = 64 + alignment;
        const std::uint32_t seed = static_cast<std::uint32_t>(alignment);
        Bytes actual = verify::make_region(offset, 257, 64, seed);
        const Bytes expected = actual;
        expect(actual.size() == offset + 257 + 64, "aligned image has exact bounds");
        expect(actual[offset] == verify::pattern_byte(0, seed) &&
                   actual[offset + 256] == verify::pattern_byte(256, seed),
               "unaligned payload starts and ends at requested positions");
        expect(actual[offset - 1] == verify::guard_byte(offset - 1, seed),
               "leading guard reaches the unaligned payload");
        expect(actual[offset + 257] == verify::guard_byte(0, seed ^ 0x7f4a7c15u),
               "trailing guard starts immediately after payload");

        verify::Mismatch mismatch;
        actual[offset - 1] ^= 1;
        expect(!verify::equal(actual, expected, mismatch) && mismatch.offset == offset - 1,
               "leading guard corruption is detected exactly");
        actual = expected;
        actual[offset + 257] ^= 1;
        expect(!verify::equal(actual, expected, mismatch) && mismatch.offset == offset + 257,
               "trailing guard corruption is detected exactly");
        actual = expected;
        actual[offset + 256] ^= 1;
        expect(!verify::equal(actual, expected, mismatch) && mismatch.offset == offset + 256,
               "last payload byte corruption is detected exactly");
    }

    Bytes too_small(32, 0);
    expect(!verify::initialize_region(too_small, 16, 20, 4, 1), "out-of-bounds payload is rejected");
    expect(!verify::initialize_region(too_small, 16, 12, 5, 1), "out-of-bounds trailing guard is rejected");
    expect(verify::make_region(std::numeric_limits<std::size_t>::max(), 2, 2, 1).empty(),
           "size arithmetic overflow is rejected");
    Bytes shorter(3, 0), longer(4, 0);
    verify::Mismatch mismatch;
    expect(!verify::equal(shorter, longer, mismatch) && mismatch.offset == 3, "size mismatch is detected");
}

void test_repeated_ring_expectations_detect_stale_data() {
    namespace verify = strata::ds4::p2p::verify;
    constexpr std::size_t payload = 64;
    constexpr std::size_t guard = 8;
    constexpr std::size_t stride = guard + payload + guard;
    Bytes source(4 * stride, 0), destination(4 * stride, 0);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        expect(verify::initialize_region(source, slot * stride + guard, payload, guard,
                                         static_cast<std::uint32_t>(slot + 1)),
               "source ring slot initializes");
        expect(verify::initialize_region(destination, slot * stride + guard, payload, guard,
                                         static_cast<std::uint32_t>(slot + 11)),
               "destination ring slot initializes");
    }
    for (std::size_t slot = 0; slot < 4; ++slot)
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(slot * stride + guard), payload,
                    destination.begin() + static_cast<std::ptrdiff_t>(slot * stride + guard));
    verify::Mismatch mismatch;
    expect(verify::equal(destination, destination, mismatch), "ring expectation self-check passes");
    Bytes stale = destination;
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(guard), payload,
                stale.begin() + static_cast<std::ptrdiff_t>(2 * stride + guard));
    expect(!verify::equal(stale, destination, mismatch) && mismatch.offset == 2 * stride + guard,
           "revisited ring slot exposes stale or wrong payload");
}

}  // namespace

int main() {
    test_directed_pairs_and_copy();
    test_illegal_devices_and_copies();
    test_capability_and_enable_failures_cleanup();
    test_preexisting_link_is_not_owned();
    test_disable_failure_and_retry();
    test_copy_runtime_error_has_no_fallback();
    test_guards_alignment_and_bounds();
    test_repeated_ring_expectations_detect_stale_data();
    if (failures == 0) std::puts("ds4_p2p host tests: all fake-runtime and verification checks passed");
    return failures == 0 ? 0 : 1;
}
