#include "ds4_p2p.hpp"

#include <cuda_runtime.h>

namespace strata::ds4::p2p {
namespace {

class CudaRuntime final : public Runtime {
public:
    Error get_device(int* device) override { return cudaGetDevice(device); }

    Error set_device(int device) override { return cudaSetDevice(device); }

    Error can_access_peer(int* can_access, int device, int peer_device) override {
        return cudaDeviceCanAccessPeer(can_access, device, peer_device);
    }

    Error enable_peer_access(int peer_device, unsigned int flags) override {
        return cudaDeviceEnablePeerAccess(peer_device, flags);
    }

    Error disable_peer_access(int peer_device) override {
#if defined(STRATA_USE_HIP)
        // The project's HIP compatibility header has no cudaDeviceDisablePeerAccess alias.
        return hipDeviceDisablePeerAccess(peer_device);
#else
        return cudaDeviceDisablePeerAccess(peer_device);
#endif
    }

    Error memcpy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                            std::size_t bytes, Stream stream) override {
        return cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, static_cast<cudaStream_t>(stream));
    }

    Error peer_access_already_enabled() const override { return cudaErrorPeerAccessAlreadyEnabled; }

    const char* error_string(Error error) const override {
        return cudaGetErrorString(static_cast<cudaError_t>(error));
    }
};

}  // namespace

PeerAccess::PeerAccess(const std::vector<int>& devices) : PeerAccess(cuda_runtime(), devices) {}

Runtime& cuda_runtime() {
    static CudaRuntime runtime;
    return runtime;
}

}  // namespace strata::ds4::p2p
