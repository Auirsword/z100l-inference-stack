#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace strata::ds4::p2p {

using Error = int;
using Stream = void*;

class Runtime {
public:
    virtual ~Runtime() = default;

    virtual Error get_device(int* device) = 0;
    virtual Error set_device(int device) = 0;
    virtual Error can_access_peer(int* can_access, int device, int peer_device) = 0;
    virtual Error enable_peer_access(int peer_device, unsigned int flags) = 0;
    virtual Error disable_peer_access(int peer_device) = 0;
    virtual Error memcpy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                                    std::size_t bytes, Stream stream) = 0;
    virtual Error peer_access_already_enabled() const = 0;
    virtual const char* error_string(Error error) const = 0;
};

// The process-wide CUDA/HIP runtime adapter. It has no mutable state of its own.
Runtime& cuda_runtime();

struct DirectedPair {
    int device = -1;
    int peer = -1;
};

// Owns only peer links that it actually enabled. Links that returned
// peer-access-already-enabled remain available for copies but are never disabled here.
//
// The caller must quiesce every stream and kernel using peer mappings before disable_all()
// or destruction, and must keep them quiescent until teardown completes. Neither function
// synchronizes the GPU. copy_peer_async() only enqueues work; its caller must establish
// source-stream ordering and synchronize the destination stream before reading the result.
class PeerAccess final {
public:
    explicit PeerAccess(const std::vector<int>& devices);
    PeerAccess(Runtime& runtime, const std::vector<int>& devices);
    ~PeerAccess();

    PeerAccess(const PeerAccess&) = delete;
    PeerAccess& operator=(const PeerAccess&) = delete;
    PeerAccess(PeerAccess&&) = delete;
    PeerAccess& operator=(PeerAccess&&) = delete;

    bool enable_all(std::string& error);
    bool disable_all(std::string& error);

    // Uses the destination device's stream and requires the destination->source peer link.
    // On success bytes_enqueued equals bytes; it does not mean the copy has completed.
    bool copy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                         std::size_t bytes, Stream stream, std::size_t& bytes_enqueued,
                         std::string& error);

    bool ready() const { return ready_; }
    bool is_enabled(int device, int peer_device) const;
    const std::vector<int>& devices() const { return devices_; }

private:
    struct Link {
        DirectedPair pair;
        bool enabled = false;
        bool owned = false;
    };

    int find_link(int device, int peer_device) const;
    bool validate_devices(std::string& error) const;
    bool disable_owned(std::string& error);
    bool restore_device(int device, std::string& error) const;

    Runtime* runtime_;
    std::vector<int> devices_;
    std::vector<Link> links_;
    std::vector<std::size_t> owned_order_;
    bool ready_ = false;
};

}  // namespace strata::ds4::p2p
