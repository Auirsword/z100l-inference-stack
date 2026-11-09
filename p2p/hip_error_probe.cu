#include "cuda_runtime.h"
#include <cstdint>
#include <cstdio>

__global__ void delay_kernel(std::uint64_t cycles) {
    const unsigned long long started = static_cast<unsigned long long>(clock64());
    while (static_cast<unsigned long long>(clock64()) - started < cycles) {}
}

int main() {
    cudaStream_t stream = nullptr;
    cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "stream creation: %d\n", static_cast<int>(status));
        return 1;
    }
    delay_kernel<<<1, 1, 0, stream>>>(static_cast<std::uint64_t>(1) << 31);
    const cudaError_t launch = cudaGetLastError();
    status = cudaStreamQuery(stream);
    std::printf("query=%d (%s), expected_not_ready=%d (%s), matches=%d, launch=%d\n",
                static_cast<int>(status), cudaGetErrorString(status), static_cast<int>(cudaErrorNotReady),
                cudaGetErrorString(cudaErrorNotReady), status == cudaErrorNotReady, static_cast<int>(launch));
    const cudaError_t pending_set = cudaSetDevice(0);
    const cudaError_t after_set = cudaStreamQuery(stream);
    std::printf("pending_set_device=%d (%s), query_after_set=%d (%s)\n", static_cast<int>(pending_set),
                cudaGetErrorString(pending_set), static_cast<int>(after_set), cudaGetErrorString(after_set));
    const cudaError_t sync = cudaStreamSynchronize(stream);
    const cudaError_t destroy = cudaStreamDestroy(stream);
    return launch == cudaSuccess && sync == cudaSuccess && destroy == cudaSuccess && status == cudaErrorNotReady ? 0 : 1;
}
