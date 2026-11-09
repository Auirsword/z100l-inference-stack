#include "fused_ar.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <sstream>

namespace strata::ds4::p2p::fused {
namespace {

__device__ __forceinline__ unsigned int fresh_u32(const unsigned int* pointer) {
#if defined(STRATA_USE_HIP)
    return __hip_atomic_load(pointer, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return __ldcv(pointer);
#endif
}

__device__ __forceinline__ void publish_u32(unsigned int* pointer, unsigned int value) {
#if defined(STRATA_USE_HIP)
    __hip_atomic_store(pointer, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    *reinterpret_cast<volatile unsigned int*>(pointer) = value;
#endif
}

#if defined(STRATA_USE_HIP)
using U32x2 = unsigned int __attribute__((ext_vector_type(2)));
using U32x4 = unsigned int __attribute__((ext_vector_type(4)));
#endif

__device__ __forceinline__ float4 fresh_float4(const float4* pointer) {
#if defined(STRATA_USE_HIP)
    const U32x4 value = *reinterpret_cast<const volatile U32x4*>(pointer);
    return make_float4(__uint_as_float(value.x), __uint_as_float(value.y),
                       __uint_as_float(value.z), __uint_as_float(value.w));
#else
    return __ldcv(pointer);
#endif
}

__device__ __forceinline__ uint2 fresh_uint2(const uint2* pointer) {
#if defined(STRATA_USE_HIP)
    const U32x2 value = *reinterpret_cast<const volatile U32x2*>(pointer);
    return make_uint2(value.x, value.y);
#else
    return __ldcv(pointer);
#endif
}

__device__ __forceinline__ uint2 to_h4(float4 value) {
    const __half2 a = __floats2half2_rn(value.x, value.y);
    const __half2 b = __floats2half2_rn(value.z, value.w);
    return make_uint2(*reinterpret_cast<const unsigned int*>(&a), *reinterpret_cast<const unsigned int*>(&b));
}

__device__ __forceinline__ float4 from_h4(uint2 value) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&value.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&value.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ float4 sat4(float4 value) {
    const float limit = 65504.0f;
    return make_float4(fminf(fmaxf(value.x, -limit), limit), fminf(fmaxf(value.y, -limit), limit),
                       fminf(fmaxf(value.z, -limit), limit), fminf(fmaxf(value.w, -limit), limit));
}

__device__ void set_error(DeviceError* error, unsigned int code, unsigned int seq,
                          unsigned int stage, unsigned int detail) {
    if (atomicCAS(&error->code, 0u, code) == 0u) {
        error->seq = seq;
        error->stage = stage;
        error->detail = detail;
        __threadfence_system();
    }
}

__device__ bool wait_sequence(const unsigned int* flag, const DeviceError* peer_error,
                              DeviceError* local_error, unsigned int seq, unsigned int required,
                              std::uint64_t budget, unsigned int timeout_code,
                              unsigned int stage, unsigned int detail) {
    const unsigned long long start = static_cast<unsigned long long>(clock64());
    while (!sequence_at_least(fresh_u32(flag), required)) {
        const unsigned int local_code = fresh_u32(&local_error->code);
        if (local_code != 0u) return false;
        if (peer_error != local_error) {
            const unsigned int peer_code = fresh_u32(&peer_error->code);
            if (peer_code != 0u) {
                set_error(local_error, kErrorPeer, seq, stage, peer_code);
                return false;
            }
        }
        const unsigned long long now = static_cast<unsigned long long>(clock64());
        if (now - start > static_cast<unsigned long long>(budget)) {
            set_error(local_error, timeout_code, seq, stage, detail);
            return false;
        }
    }
    if (peer_error != local_error) {
        const unsigned int peer_code = fresh_u32(&peer_error->code);
        if (peer_code != 0u) {
            set_error(local_error, kErrorPeer, seq, stage, peer_code);
            return false;
        }
    }
    __threadfence_system();
    return true;
}

__global__ void __launch_bounds__(kThreads) fused_ar_kernel(
    float* x, int n, int members, int member, int slots, unsigned int seq,
    std::uint64_t timeout_cycles, bool fp16_wire, unsigned int* counters, const PeerTable* table) {
    __shared__ int proceed;
    int slot = 0;
    if (!slot_for_sequence(seq, slots, slot)) {
        if (threadIdx.x == 0) set_error(table->errors[member], kErrorInvalid, seq, kStageEntry, 0u);
        return;
    }
    DeviceError* local_error = table->errors[member];

    if (threadIdx.x == 0) {
        proceed = fresh_u32(&local_error->code) == 0u;
        unsigned int previous = 0;
        if (proceed && reuse_sequence(seq, slots, previous) && seq > static_cast<unsigned int>(slots)) {
            for (int reader = 0; reader < members && proceed; ++reader) {
                if (reader == member) continue;
                const unsigned int* flag = table->done[member] + static_cast<std::size_t>(slot) * members + reader;
                proceed = wait_sequence(flag, table->errors[reader], local_error, seq, previous,
                                        timeout_cycles, kErrorRingTimeout, kStageRing,
                                        static_cast<unsigned int>(reader));
            }
        }
    }
    __syncthreads();
    if (!proceed) return;

    const std::size_t payload_offset = outbox_payload_offset(slot, n);
    float4* mine = reinterpret_cast<float4*>(table->outboxes[member] + payload_offset);
    const float4* input = reinterpret_cast<const float4*>(x);
    const int n4 = n / 4;
    if (fp16_wire) {
        uint2* wire_mine = reinterpret_cast<uint2*>(mine);
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x)
            wire_mine[q] = to_h4(sat4(input[q]));
    } else {
        for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x)
            mine[q] = input[q];
    }

    // Each block orders its outbox writes before the completion atomic. The last block publishes ready only after
    // all blocks arrived, so no peer can observe ready for a partially written local slot.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(counters, 1u) == gridDim.x - 1) {
        counters[0] = 0u;
        __threadfence_system();
        publish_u32(table->ready[member] + slot, seq);
    }

    if (threadIdx.x == 0) {
        proceed = 1;
        for (int other = 0; other < members && proceed; ++other) {
            proceed = wait_sequence(table->ready[other] + slot, table->errors[other], local_error,
                                    seq, seq, timeout_cycles, kErrorReadyTimeout, kStageReady,
                                    static_cast<unsigned int>(other));
        }
    }
    __syncthreads();
    if (!proceed) return;

    float4* output = reinterpret_cast<float4*>(x);
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n4; q += gridDim.x * blockDim.x) {
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int other = 0; other < members; ++other) {
            const float* base = table->outboxes[other] + payload_offset;
            float4 partial;
            if (fp16_wire) {
                const uint2* wire_base = reinterpret_cast<const uint2*>(base);
                partial = from_h4(fresh_uint2(wire_base + q));
            } else {
                partial = fresh_float4(reinterpret_cast<const float4*>(base) + q);
            }
            if (other == 0) {
                value = partial;
            } else {
                value.x = __fadd_rn(value.x, partial.x);
                value.y = __fadd_rn(value.y, partial.y);
                value.z = __fadd_rn(value.z, partial.z);
                value.w = __fadd_rn(value.w, partial.w);
            }
        }
        if (fp16_wire) value = from_h4(to_h4(sat4(value)));
        output[q] = value;
    }

    // A second grid completion protects reuse: the last reader block acknowledges this reader's seq in every
    // owner's device-resident done table. The owner's seq > slots wait covers all peer readers before overwriting.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(counters + 1, 1u) == gridDim.x - 1) {
        counters[1] = 0u;
        __threadfence_system();
        for (int owner = 0; owner < members; ++owner) {
            if (owner == member) continue;
            unsigned int* done = table->done[owner] + static_cast<std::size_t>(slot) * members + member;
            publish_u32(done, seq);
        }
    }
}

std::string cuda_error(const char* operation, cudaError_t status) {
    std::ostringstream out;
    out << operation << " failed with runtime error " << static_cast<int>(status) << " ("
        << cudaGetErrorString(status) << ")";
    return out.str();
}

void append_error(std::string& error, const std::string& next) {
    if (next.empty()) return;
    if (!error.empty()) error += "; ";
    error += next;
}

cudaError_t allocate_peer(void** pointer, std::size_t bytes) {
#if defined(STRATA_USE_HIP)
    return hipExtMallocWithFlags(pointer, bytes, hipDeviceMallocFinegrained);
#else
    return cudaMalloc(pointer, bytes);
#endif
}

bool restore_device(int device, std::string& error) {
    const cudaError_t status = cudaSetDevice(device);
    if (status == cudaSuccess) return true;
    append_error(error, cuda_error("cudaSetDevice(restore)", status));
    return false;
}

}  // namespace

Group::Group(PeerAccess& peers, const Config& config) : peers_(&peers), config_(config) {}

Group::~Group() {
    std::string ignored;
    close(ignored);
}

int Group::blocks(int member) const {
    return member >= 0 && member < static_cast<int>(members_.size()) ? members_[member].blocks : 0;
}

float* Group::outbox_raw(int member) const {
    return member >= 0 && member < static_cast<int>(members_.size()) ? members_[member].outbox : nullptr;
}

bool Group::initialize(std::string& error) {
    error.clear();
    if (initialized_ || !members_.empty()) {
        error = "fused AR group is already initialized or has unreleased resources";
        return false;
    }
    const std::vector<int>& devices = peers_->devices();
    const int count = static_cast<int>(devices.size());
    if (!peers_->ready() || !validate_config(config_, count, error)) {
        if (error.empty()) error = "fused AR requires an enabled PeerAccess group";
        return false;
    }
    for (int accessor = 0; accessor < count; ++accessor) {
        for (int peer = 0; peer < count; ++peer) {
            if (accessor != peer && !peers_->is_enabled(devices[accessor], devices[peer])) {
                error = "fused AR missing directed peer access from device " + std::to_string(devices[accessor]) +
                        " to " + std::to_string(devices[peer]);
                return false;
            }
        }
    }
    std::size_t slot_stride = 0;
    if (!outbox_layout(config_, slot_stride, outbox_bytes_)) {
        error = "fused AR outbox layout overflows host allocation arithmetic";
        return false;
    }

    int original_device = -1;
    const cudaError_t get_status = cudaGetDevice(&original_device);
    if (get_status != cudaSuccess) {
        error = cuda_error("cudaGetDevice before fused AR initialization", get_status);
        return false;
    }

    members_.resize(static_cast<std::size_t>(count));
    for (int member = 0; member < count; ++member) members_[member].device = devices[member];
    bool ok = true;
    for (int member = 0; member < count && ok; ++member) {
        MemberResources& resources = members_[member];
        cudaError_t status = cudaSetDevice(resources.device);
        if (status != cudaSuccess) {
            error = cuda_error("cudaSetDevice for fused AR initialization", status);
            ok = false;
            break;
        }
        int multiprocessors = 0, occupancy = 0;
        status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, resources.device);
        if (status == cudaSuccess)
            status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, fused_ar_kernel, kThreads, 0);
        if (status != cudaSuccess) {
            error = cuda_error("fused AR occupancy query", status);
            ok = false;
            break;
        }
        resources.blocks = choose_blocks(config_, multiprocessors, occupancy);
        if (resources.blocks <= 0) {
            error = "fused AR kernel has no valid co-resident launch shape";
            ok = false;
            break;
        }
        if (config_.init_stage >= 1) {
            status = allocate_peer(reinterpret_cast<void**>(&resources.outbox), outbox_bytes_);
            if (status == cudaSuccess)
                status = allocate_peer(reinterpret_cast<void**>(&resources.ready),
                                       static_cast<std::size_t>(config_.slots) * sizeof(unsigned int));
            if (status == cudaSuccess)
                status = allocate_peer(reinterpret_cast<void**>(&resources.done),
                                       static_cast<std::size_t>(config_.slots) * count * sizeof(unsigned int));
            if (status == cudaSuccess)
                status = cudaMalloc(reinterpret_cast<void**>(&resources.counters), 2 * sizeof(unsigned int));
            if (status == cudaSuccess)
                status = allocate_peer(reinterpret_cast<void**>(&resources.error), sizeof(DeviceError));
            if (status == cudaSuccess) status = cudaMalloc(reinterpret_cast<void**>(&resources.table), sizeof(PeerTable));
            if (status != cudaSuccess) {
                error = cuda_error("allocate fused AR peer state (HIP requires fine-grained memory)", status);
                ok = false;
                break;
            }
        }
        if (config_.init_stage >= 2) {
            status = cudaMemset(resources.outbox, 0, outbox_bytes_);
            if (status == cudaSuccess)
                status = cudaMemset(resources.ready, 0, static_cast<std::size_t>(config_.slots) * sizeof(unsigned int));
            if (status == cudaSuccess)
                status = cudaMemset(resources.done, 0,
                                    static_cast<std::size_t>(config_.slots) * count * sizeof(unsigned int));
            if (status == cudaSuccess) status = cudaMemset(resources.counters, 0, 2 * sizeof(unsigned int));
            if (status == cudaSuccess) status = cudaMemset(resources.error, 0, sizeof(DeviceError));
            if (status != cudaSuccess) {
                error = cuda_error("cudaMemset for fused AR device state", status);
                ok = false;
                break;
            }
        }
    }

    if (ok && config_.init_stage >= 3) {
        PeerTable host_table;
        for (int member = 0; member < count; ++member) {
            host_table.outboxes[member] = members_[member].outbox;
            host_table.ready[member] = members_[member].ready;
            host_table.done[member] = members_[member].done;
            host_table.errors[member] = members_[member].error;
        }
        for (int member = 0; member < count && ok; ++member) {
            cudaError_t status = cudaSetDevice(members_[member].device);
            if (status == cudaSuccess)
                status = cudaMemcpy(members_[member].table, &host_table, sizeof(host_table), cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                error = cuda_error("initialize fused AR device peer table", status);
                ok = false;
            }
        }
    }

    if (!ok) {
        std::string cleanup_error;
        release_resources(cleanup_error);
        append_error(error, cleanup_error);
        members_.clear();
    } else {
        initialized_ = stage_allows_launch(config_.init_stage);
    }
    if (!restore_device(original_device, error)) ok = false;
    return ok;
}

bool Group::launch(int member, float* x, unsigned int seq, Stream stream, std::string& error) {
    error.clear();
    if (!initialized_ || !stage_allows_launch(config_.init_stage) || member < 0 ||
        member >= static_cast<int>(members_.size())) {
        error = "fused AR launch received an invalid member or uninitialized group";
        return false;
    }
    int slot = 0;
    if (x == nullptr || reinterpret_cast<std::uintptr_t>(x) % 16 != 0 ||
        !slot_for_sequence(seq, config_.slots, slot)) {
        error = "fused AR launch requires 16-byte-aligned non-null x and a nonzero sequence";
        return false;
    }
    MemberResources& resources = members_[member];
    int original_device = -1;
    cudaError_t status = cudaGetDevice(&original_device);
    if (status != cudaSuccess) {
        error = cuda_error("cudaGetDevice before fused AR launch", status);
        return false;
    }
    status = cudaSetDevice(resources.device);
    if (status == cudaSuccess) {
        fused_ar_kernel<<<resources.blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            x, config_.n, static_cast<int>(members_.size()), member, config_.slots, seq,
            config_.timeout_cycles, config_.fp16_wire, resources.counters, resources.table);
        status = cudaGetLastError();
    }
    if (status != cudaSuccess) error = cuda_error("fused AR kernel launch", status);
    const bool ok = status == cudaSuccess;
    if (!restore_device(original_device, error)) return false;
    return ok;
}

bool Group::synchronize(int member, Stream stream, DeviceError& device_error, std::string& error) {
    error.clear();
    device_error = DeviceError{};
    if (!initialized_ || !stage_allows_launch(config_.init_stage) || member < 0 ||
        member >= static_cast<int>(members_.size())) {
        error = "fused AR synchronize received an invalid member or uninitialized group";
        return false;
    }
    MemberResources& resources = members_[member];
    int original_device = -1;
    cudaError_t status = cudaGetDevice(&original_device);
    if (status == cudaSuccess) status = cudaSetDevice(resources.device);
    if (status == cudaSuccess) status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (status == cudaSuccess)
        status = cudaMemcpy(&device_error, resources.error, sizeof(device_error), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) error = cuda_error("fused AR synchronize/status read", status);
    bool ok = status == cudaSuccess;
    if (original_device >= 0 && !restore_device(original_device, error)) ok = false;
    if (ok && device_error.code != kErrorNone) {
        std::ostringstream out;
        out << "fused AR device error code=" << device_error.code << ",seq=" << device_error.seq
            << ",stage=" << device_error.stage << ",detail=" << device_error.detail;
        error = out.str();
        ok = false;
    }
    return ok;
}

bool Group::release_resources(std::string& error) {
    bool ok = true;
    if (members_.empty()) return true;
    int original_device = -1;
    cudaError_t status = cudaGetDevice(&original_device);
    if (status != cudaSuccess) {
        append_error(error, cuda_error("cudaGetDevice before fused AR release", status));
        return false;
    }
    for (MemberResources& resources : members_) {
        status = cudaSetDevice(resources.device);
        if (status != cudaSuccess) {
            append_error(error, cuda_error("cudaSetDevice for fused AR release", status));
            ok = false;
            continue;
        }
        auto release = [&](auto*& pointer, const char* what) {
            if (pointer == nullptr) return;
            const cudaError_t free_status = cudaFree(static_cast<void*>(pointer));
            if (free_status == cudaSuccess) {
                pointer = nullptr;
            } else {
                append_error(error, cuda_error(what, free_status));
                ok = false;
            }
        };
        release(resources.table, "cudaFree fused AR peer table");
        release(resources.error, "cudaFree fused AR error");
        release(resources.counters, "cudaFree fused AR counters");
        release(resources.done, "cudaFree fused AR done");
        release(resources.ready, "cudaFree fused AR ready");
        release(resources.outbox, "cudaFree fused AR outbox");
    }
    if (!restore_device(original_device, error)) ok = false;
    return ok;
}

bool Group::close(std::string& error) {
    error.clear();
    if (members_.empty()) {
        initialized_ = false;
        return true;
    }
    const bool ok = release_resources(error);
    if (ok) {
        members_.clear();
        outbox_bytes_ = 0;
        initialized_ = false;
    }
    return ok;
}

}  // namespace strata::ds4::p2p::fused
