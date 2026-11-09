#include "ds4_p2p.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using strata::ds4::p2p::PeerAccess;
constexpr int kElements = 16384;
constexpr std::uint64_t kTimeoutCycles = static_cast<std::uint64_t>(1) << 30;

struct Result {
    unsigned int observed = 0;
    unsigned int error = 0;
    unsigned int mismatch = 0;
    unsigned int reserved = 0;
};

struct Resources {
    int device = -1;
    cudaStream_t stream = nullptr;
    unsigned int* flag = nullptr;
    unsigned int* fine_flag = nullptr;
    unsigned int* payload = nullptr;
    unsigned int* fine_payload = nullptr;
    Result* result = nullptr;
};

__device__ unsigned int pattern(unsigned int index, unsigned int seed) {
    return seed ^ (index * 2654435761u);
}

__device__ unsigned int read_flag(const unsigned int* flag, int variant) {
    if (variant >= 3 && variant != 6)
        return __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    if (variant == 2) return atomicAdd(const_cast<unsigned int*>(flag), 0u);
    return *reinterpret_cast<const volatile unsigned int*>(flag);
}

__global__ void publish(unsigned int* payload, unsigned int* flag, unsigned int seed, int variant) {
    for (int i = 0; i < kElements; ++i) payload[i] = pattern(i, seed);
    __threadfence_system();
    if (variant == 8) __hip_atomic_store(flag, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    else if (variant == 0 || variant == 5 || variant == 6) *reinterpret_cast<volatile unsigned int*>(flag) = 1u;
    else if (variant >= 3) atomicExch_system(flag, 1u);
    else atomicExch(flag, 1u);
    __threadfence_system();
}

__global__ void consume(const unsigned int* peer_flag, const unsigned int* peer_payload,
                        unsigned int* output, Result* result, unsigned int seed, int variant) {
    const unsigned long long start = static_cast<unsigned long long>(clock64());
    unsigned int observed = 0;
    while ((observed = read_flag(peer_flag, variant)) != 1u) {
        if (static_cast<unsigned long long>(clock64()) - start > kTimeoutCycles) {
            result->observed = observed;
            result->error = 1u;
            return;
        }
    }
    __threadfence_system();
    unsigned int mismatch = kElements;
    for (int i = 0; i < kElements; ++i) {
        const unsigned int value = *reinterpret_cast<const volatile unsigned int*>(peer_payload + i);
        output[i] = value;
        if (value != pattern(i, seed) && mismatch == static_cast<unsigned int>(kElements)) mismatch = i;
    }
    result->observed = observed;
    result->error = mismatch == static_cast<unsigned int>(kElements) ? 0u : 2u;
    result->mismatch = mismatch;
}

std::string cuda_error(const char* operation, cudaError_t status) {
    std::ostringstream out;
    out << operation << ": " << static_cast<int>(status) << " " << cudaGetErrorString(status);
    return out.str();
}

bool cuda_ok(cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) return true;
    if (!error.empty()) error += "; ";
    error += cuda_error(operation, status);
    return false;
}

bool initialize(Resources& resources, std::string& error) {
    return cuda_ok(cudaSetDevice(resources.device), "set_device", error) &&
           cuda_ok(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "create_stream", error) &&
           cuda_ok(cudaMalloc(reinterpret_cast<void**>(&resources.flag), sizeof(unsigned int)), "allocate_flag", error) &&
           cuda_ok(cudaMalloc(reinterpret_cast<void**>(&resources.payload), kElements * sizeof(unsigned int)),
                   "allocate_payload", error) &&
           cuda_ok(cudaMalloc(reinterpret_cast<void**>(&resources.result), sizeof(Result)), "allocate_result", error);
}

bool synchronize(Resources& resources, std::string& error) {
    return cuda_ok(cudaSetDevice(resources.device), "set_device_for_sync", error) &&
           cuda_ok(cudaStreamSynchronize(resources.stream), "synchronize", error);
}

bool release(Resources& resources, std::string& error) {
    bool ok = cuda_ok(cudaSetDevice(resources.device), "set_device_for_cleanup", error);
    ok = cuda_ok(cudaDeviceSynchronize(), "device_cleanup_sync", error) && ok;
    ok = cuda_ok(cudaFree(resources.result), "free_result", error) && ok;
    ok = cuda_ok(cudaFree(resources.payload), "free_payload", error) && ok;
    ok = cuda_ok(cudaFree(resources.flag), "free_flag", error) && ok;
    if (resources.fine_flag) ok = cuda_ok(cudaFree(resources.fine_flag), "free_fine_flag", error) && ok;
    if (resources.fine_payload) ok = cuda_ok(cudaFree(resources.fine_payload), "free_fine_payload", error) && ok;
    ok = cuda_ok(cudaStreamDestroy(resources.stream), "destroy_stream", error) && ok;
    resources.result = nullptr;
    resources.payload = nullptr;
    resources.fine_payload = nullptr;
    resources.flag = nullptr;
    resources.fine_flag = nullptr;
    resources.stream = nullptr;
    return ok;
}

const char* variant_name(int variant) {
    if (variant == 0) return "coarse_volatile_volatile";
    if (variant == 1) return "coarse_agent_exchange_volatile";
    if (variant == 2) return "coarse_agent_exchange_agent_atomic";
    if (variant == 3) return "coarse_system_exchange_system_acquire";
    if (variant == 4) return "fine_system_exchange_system_acquire";
    if (variant == 5) return "fine_volatile_volatile";
    if (variant == 6) return "fine_volatile_fine_payload";
    if (variant == 7) return "fine_system_exchange_fine_payload";
    return "fine_release_acquire_fine_payload";
}

bool run_case(std::vector<Resources>& resources, int sender, int variant, std::ostream& log, std::string& json) {
    std::string error;
    Resources& source = resources[sender];
    Resources& receiver = resources[1 - sender];
    unsigned int* source_flag = variant >= 4 ? source.fine_flag : source.flag;
    unsigned int* receiver_flag = variant >= 4 ? receiver.fine_flag : receiver.flag;
    unsigned int* source_payload = variant >= 6 ? source.fine_payload : source.payload;
    const unsigned int seed = 0x61300000u + static_cast<unsigned int>(sender * 16 + variant);
    unsigned int local_flag = 0;
    Result result;
    bool harness_ok = true;
    const bool skipped = source_flag == nullptr || receiver_flag == nullptr || source_payload == nullptr;
    if (!skipped) {
        for (Resources* item : {&source, &receiver}) {
            unsigned int* flag = variant >= 4 ? item->fine_flag : item->flag;
            harness_ok = cuda_ok(cudaSetDevice(item->device), "set_device_for_reset", error) && harness_ok;
            harness_ok = cuda_ok(cudaMemset(flag, 0, sizeof(unsigned int)), "reset_flag", error) && harness_ok;
            harness_ok = cuda_ok(cudaMemset(item->payload, 0, kElements * sizeof(unsigned int)),
                                 "reset_payload", error) && harness_ok;
            if (variant >= 6)
                harness_ok = cuda_ok(cudaMemset(item->fine_payload, 0, kElements * sizeof(unsigned int)),
                                     "reset_fine_payload", error) && harness_ok;
            harness_ok = cuda_ok(cudaMemset(item->result, 0, sizeof(Result)), "reset_result", error) && harness_ok;
        }
    }
    if (harness_ok && !skipped) {
        harness_ok = cuda_ok(cudaSetDevice(receiver.device), "set_receiver", error);
        if (harness_ok) {
            consume<<<1, 1, 0, receiver.stream>>>(source_flag, source_payload, receiver.payload,
                                                  receiver.result, seed, variant);
            harness_ok = cuda_ok(cudaGetLastError(), "launch_consume", error);
        }
        harness_ok = cuda_ok(cudaSetDevice(source.device), "set_sender", error) && harness_ok;
        if (harness_ok) {
            publish<<<1, 1, 0, source.stream>>>(source_payload, source_flag, seed, variant);
            harness_ok = cuda_ok(cudaGetLastError(), "launch_publish", error);
        }
    }
    if (!skipped) {
        log << "FLAG_PROBE synchronize receiver=" << receiver.device << " sender=" << sender
            << " variant=" << variant_name(variant) << std::endl;
        harness_ok = synchronize(receiver, error) && harness_ok;
        harness_ok = synchronize(source, error) && harness_ok;
    }
    if (harness_ok && !skipped) {
        harness_ok = cuda_ok(cudaSetDevice(receiver.device), "read_result_device", error) &&
                     cuda_ok(cudaMemcpy(&result, receiver.result, sizeof(Result), cudaMemcpyDeviceToHost),
                             "read_result", error);
        harness_ok = cuda_ok(cudaSetDevice(source.device), "read_flag_device", error) &&
                     cuda_ok(cudaMemcpy(&local_flag, source_flag, sizeof(unsigned int), cudaMemcpyDeviceToHost),
                             "read_local_flag", error) && harness_ok;
    }
    const bool passed = !skipped && harness_ok && local_flag == 1 && result.observed == 1 && result.error == 0;
    json += "    {\"sender\":" + std::to_string(sender) + ",\"receiver\":" + std::to_string(1 - sender) +
            ",\"variant\":\"" + variant_name(variant) + "\",\"passed\":" + (passed ? "true" : "false") +
            ",\"skipped\":" + (skipped ? "true" : "false") + ",\"local_flag\":" + std::to_string(local_flag) +
            ",\"observed\":" + std::to_string(result.observed) + ",\"error_code\":" + std::to_string(result.error) +
            ",\"mismatch\":" + std::to_string(result.mismatch) + ",\"runtime_error\":\"" + error + "\"}";
    log << "FLAG_PROBE result passed=" << passed << " skipped=" << skipped << " local_flag=" << local_flag
        << " observed=" << result.observed << " error_code=" << result.error << " runtime_error=" << error << std::endl;
    return harness_ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string json_path;
    bool allow_peer_rmw = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--allow-peer-rmw-atomics") allow_peer_rmw = true;
        else if (argument == "--json" && i + 1 < argc) json_path = argv[++i];
        else {
            std::cerr << "Usage: p2p_flag_probe [--json PATH] [--allow-peer-rmw-atomics]\n"
                         "Diagnostic only; not fused AR acceptance. Peer RMW is disabled by default.\n";
            return 2;
        }
    }
    std::string error;
    bool harness_ok = true;
    for (int device = 0; device < 2; ++device) {
        char bdf[64] = {};
        const char* expected = device == 0 ? "0000:03:00.0" : "0000:43:00.0";
        harness_ok = cuda_ok(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), device), "device_bdf", error) && harness_ok;
        if (harness_ok && expected != std::string(bdf)) {
            error += "unexpected device BDF";
            harness_ok = false;
        }
    }
    std::vector<Resources> resources(2);
    resources[0].device = 0;
    resources[1].device = 1;
    if (harness_ok) {
        PeerAccess peers({0, 1});
        harness_ok = peers.enable_all(error);
        if (harness_ok) {
            for (Resources& item : resources) harness_ok = initialize(item, error) && harness_ok;
        }
        int fine_allocations = 0;
        int fine_payload_allocations = 0;
        if (harness_ok) {
            for (Resources& item : resources) {
                const cudaError_t status = hipExtMallocWithFlags(reinterpret_cast<void**>(&item.fine_flag),
                                                                  sizeof(unsigned int), hipDeviceMallocFinegrained);
                if (status == cudaSuccess) ++fine_allocations;
                else {
                    std::cerr << "fine-grained flag allocation unavailable on device " << item.device << ": "
                              << cudaGetErrorString(status) << std::endl;
                    (void)cudaGetLastError();
                }
                const cudaError_t payload_status = hipExtMallocWithFlags(
                    reinterpret_cast<void**>(&item.fine_payload), kElements * sizeof(unsigned int),
                    hipDeviceMallocFinegrained);
                if (payload_status == cudaSuccess) ++fine_payload_allocations;
                else {
                    std::cerr << "fine-grained payload allocation unavailable on device " << item.device << ": "
                              << cudaGetErrorString(payload_status) << std::endl;
                    (void)cudaGetLastError();
                }
            }
        }
        std::string cases;
        bool first_case = true;
        if (harness_ok) {
            for (int variant = 0; variant < 9 && harness_ok; ++variant) {
                if (!allow_peer_rmw && ((variant >= 1 && variant <= 4) || variant == 7)) continue;
                for (int sender = 0; sender < 2; ++sender) {
                    if (!first_case) cases += ",\n";
                    first_case = false;
                    harness_ok = run_case(resources, sender, variant, std::cerr, cases) && harness_ok;
                }
            }
        }
        for (Resources& item : resources) harness_ok = release(item, error) && harness_ok;
        harness_ok = peers.disable_all(error) && harness_ok;
        std::ostringstream report;
        report << "{\"schema\":\"ds4-p2p-flag-diagnostic-v1\",\"completed\":" << (harness_ok ? "true" : "false")
               << ",\"fused_ar_accepted\":false,\"peer_rmw_atomics_enabled\":" << (allow_peer_rmw ? "true" : "false")
               << ",\"fine_grained_allocations\":" << fine_allocations
               << ",\"fine_payload_allocations\":" << fine_payload_allocations
               << ",\"elements\":" << kElements
               << ",\"timeout_cycles\":" << kTimeoutCycles << ",\"cases\":[\n" << cases << "\n],\"error\":\"" << error
               << "\"}\n";
        if (!json_path.empty()) {
            std::ofstream output(json_path);
            output << report.str();
            if (!output) {
                std::cerr << "could not write JSON report\n";
                harness_ok = false;
            }
        }
        std::cout << report.str();
    }
    if (!error.empty()) std::cerr << error << std::endl;
    return harness_ok ? 0 : 1;
}
