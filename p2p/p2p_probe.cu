#include "ds4_p2p.hpp"
#include "p2p_verify.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using strata::ds4::p2p::DirectedPair;
using strata::ds4::p2p::Error;
using strata::ds4::p2p::PeerAccess;
using strata::ds4::p2p::Runtime;
using strata::ds4::p2p::Stream;
using Bytes = std::vector<std::uint8_t>;

constexpr std::size_t kPayloadBytes = static_cast<std::size_t>(4) << 20;
constexpr std::size_t kGuardBytes = 64;
constexpr std::size_t kMaxAlignment = 16;
constexpr std::size_t kMainAllocationBytes = kGuardBytes + kMaxAlignment + kPayloadBytes + kGuardBytes;
constexpr std::size_t kRingPayloadBytes = 4096;
constexpr std::size_t kRingSlots = 4;
constexpr std::size_t kRingRounds = 2;
constexpr std::size_t kRingSlotStride = kGuardBytes + kRingPayloadBytes + kGuardBytes;
constexpr std::size_t kRingAllocationBytes = kRingSlots * kRingSlotStride;
constexpr unsigned int kThreads = 256;

struct DeviceSpec {
    int ordinal = -1;
    std::string bdf;
    std::string actual_bdf;
    bool mapping_ok = false;
};

struct Event {
    std::string operation;
    int device = -1;
    int peer = -1;
    int src = -1;
    int dst = -1;
    std::size_t bytes = 0;
    Error status = 0;
    int can_access = -1;
    bool ok = false;
};

struct Check {
    std::string name;
    int src = -1;
    int dst = -1;
    int alignment = -1;
    int slot = -1;
    int round = -1;
    bool ok = false;
    std::string error;
};

struct PairResult {
    int src = -1;
    int dst = -1;
    bool ran = false;
    bool ok = false;
};

struct Report {
    std::vector<DeviceSpec> devices;
    std::vector<Event> events;
    std::vector<Check> checks;
    std::vector<PairResult> pairs;
    std::string error;
};

void append_error(std::string& error, const std::string& next) {
    if (next.empty()) return;
    if (!error.empty()) error += "; ";
    error += next;
}

void mark(const std::string& message) {
    std::fprintf(stderr, "P2P_PROBE %s\n", message.c_str());
    std::fflush(stderr);
}

std::string pair_name(int src, int dst) {
    return "src=" + std::to_string(src) + ",dst=" + std::to_string(dst);
}

std::string cuda_error(const std::string& operation, Error status) {
    return operation + " failed with runtime error " + std::to_string(status) + " (" +
           cudaGetErrorString(static_cast<cudaError_t>(status)) + ")";
}

class ProbeLog {
public:
    explicit ProbeLog(Report& report) : report_(report) {}

    template <typename F>
    bool cuda(const std::string& operation, F&& call, std::string& error) {
        mark(operation + " start");
        const Error status = call();
        if (status != 0) {
            append_error(error, cuda_error(operation, status));
            mark(operation + " FAIL: " + cudaGetErrorString(static_cast<cudaError_t>(status)));
            return false;
        }
        mark(operation + " ok");
        return true;
    }

    template <typename F>
    bool check(const std::string& name, int src, int dst, int alignment, int slot, int round, F&& call) {
        std::ostringstream label;
        label << name;
        if (src >= 0) label << " " << pair_name(src, dst);
        if (alignment >= 0) label << " alignment=" << alignment;
        if (slot >= 0) label << " slot=" << slot;
        if (round >= 0) label << " round=" << round;
        mark("CHECK " + label.str() + " start");
        std::string error;
        const bool ok = call(error);
        if (!ok && error.empty()) error = "check failed without a diagnostic";
        report_.checks.push_back(Check{name, src, dst, alignment, slot, round, ok, error});
        mark(std::string("CHECK ") + label.str() + (ok ? " ok" : " FAIL: " + error));
        return ok;
    }

private:
    Report& report_;
};

class LoggingRuntime final : public Runtime {
public:
    LoggingRuntime(Runtime& runtime, Report& report) : runtime_(runtime), report_(report) {}

    Error get_device(int* device) override {
        const Error status = runtime_.get_device(device);
        if (status == 0 && device != nullptr) current_ = *device;
        return status;
    }

    Error set_device(int device) override {
        const Error status = runtime_.set_device(device);
        if (status == 0) current_ = device;
        return status;
    }

    Error can_access_peer(int* can_access, int device, int peer_device) override {
        const std::string label = "can_access_peer device=" + std::to_string(device) +
                                  ",peer=" + std::to_string(peer_device);
        mark(label + " start");
        const Error status = runtime_.can_access_peer(can_access, device, peer_device);
        const int can = (status == 0 && can_access != nullptr) ? *can_access : -1;
        const bool ok = status == 0 && can == 1;
        report_.events.push_back(Event{"can_access_peer", device, peer_device, -1, -1, 0, status, can, ok});
        mark(label + (ok ? " ok can_access=1" : " FAIL status=" + std::to_string(status) +
                                              " can_access=" + std::to_string(can)));
        return status;
    }

    Error enable_peer_access(int peer_device, unsigned int flags) override {
        const std::string label = "enable_peer_access device=" + std::to_string(current_) +
                                  ",peer=" + std::to_string(peer_device) + ",flags=" + std::to_string(flags);
        mark(label + " start");
        const Error status = runtime_.enable_peer_access(peer_device, flags);
        const bool already = status == runtime_.peer_access_already_enabled();
        const bool ok = status == 0 || already;
        report_.events.push_back(Event{"enable_peer_access", current_, peer_device, -1, -1, 0, status, -1, ok});
        mark(label + (ok ? (already ? " ok already-enabled (not owned)" : " ok enabled (owned)")
                         : " FAIL status=" + std::to_string(status)));
        return status;
    }

    Error disable_peer_access(int peer_device) override {
        const std::string label = "disable_peer_access device=" + std::to_string(current_) +
                                  ",peer=" + std::to_string(peer_device);
        mark(label + " start");
        const Error status = runtime_.disable_peer_access(peer_device);
        report_.events.push_back(Event{"disable_peer_access", current_, peer_device, -1, -1, 0, status, -1, status == 0});
        mark(label + (status == 0 ? " ok" : " FAIL status=" + std::to_string(status)));
        return status;
    }

    Error memcpy_peer_async(void* dst, int dst_device, const void* src, int src_device,
                            std::size_t bytes, Stream stream) override {
        const std::string label = "cudaMemcpyPeerAsync " + pair_name(src_device, dst_device) +
                                  ",bytes=" + std::to_string(bytes);
        mark(label + " start");
        const Error status = runtime_.memcpy_peer_async(dst, dst_device, src, src_device, bytes, stream);
        report_.events.push_back(
            Event{"cudaMemcpyPeerAsync", -1, -1, src_device, dst_device, bytes, status, -1, status == 0});
        mark(label + (status == 0 ? " enqueued (not yet complete)" : " FAIL status=" + std::to_string(status)));
        return status;
    }

    Error peer_access_already_enabled() const override { return runtime_.peer_access_already_enabled(); }
    const char* error_string(Error error) const override { return runtime_.error_string(error); }

private:
    Runtime& runtime_;
    Report& report_;
    int current_ = -1;
};

__global__ void uva_read_kernel(const std::uint8_t* peer, std::uint8_t* local, std::size_t bytes) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) local[i] = strata::ds4::p2p::verify::transformed_byte(peer[i]);
}

__global__ void uva_write_kernel(std::uint8_t* peer, std::size_t bytes, std::uint32_t seed) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < bytes) peer[i] = strata::ds4::p2p::verify::pattern_byte(i, seed);
}

struct DeviceResources {
    struct Slot {
        int device = -1;
        cudaStream_t stream = nullptr;
        std::uint8_t* main = nullptr;
        std::uint8_t* ring = nullptr;
    };

    DeviceResources(const std::vector<DeviceSpec>& devices, ProbeLog& log) : log(log) {
        slots.reserve(devices.size());
        for (const DeviceSpec& device : devices) slots.push_back(Slot{device.ordinal, nullptr, nullptr, nullptr});
    }

    Slot& get(int device) {
        for (Slot& slot : slots) {
            if (slot.device == device) return slot;
        }
        std::fprintf(stderr, "P2P_PROBE internal error: unknown device %d\n", device);
        std::abort();
    }

    bool initialize(std::string& error) {
        for (Slot& slot : slots) {
            if (!log.cuda("set_device(" + std::to_string(slot.device) + ") for allocation",
                          [&] { return cudaSetDevice(slot.device); }, error))
                return false;
            if (!log.cuda("create stream on device " + std::to_string(slot.device),
                          [&] { return cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking); }, error))
                return false;
            if (!log.cuda("allocate main buffer on device " + std::to_string(slot.device),
                          [&] { return cudaMalloc(reinterpret_cast<void**>(&slot.main), kMainAllocationBytes); }, error))
                return false;
            if (!log.cuda("allocate ring buffer on device " + std::to_string(slot.device),
                          [&] { return cudaMalloc(reinterpret_cast<void**>(&slot.ring), kRingAllocationBytes); }, error))
                return false;
        }
        return true;
    }

    bool close(std::string& error) {
        if (closed) return true;
        closed = true;
        bool ok = true;
        for (Slot& slot : slots) {
            std::string step_error;
            ok = log.cuda("set_device(" + std::to_string(slot.device) + ") for quiescence",
                          [&] { return cudaSetDevice(slot.device); }, step_error) && ok;
            ok = log.cuda("device synchronize before teardown on " + std::to_string(slot.device),
                          [&] { return cudaDeviceSynchronize(); }, step_error) && ok;
            append_error(error, step_error);
        }
        for (Slot& slot : slots) {
            std::string step_error;
            ok = log.cuda("set_device(" + std::to_string(slot.device) + ") for release",
                          [&] { return cudaSetDevice(slot.device); }, step_error) && ok;
            if (slot.main != nullptr) {
                ok = log.cuda("free main buffer on " + std::to_string(slot.device),
                              [&] { return cudaFree(slot.main); }, step_error) && ok;
                slot.main = nullptr;
            }
            if (slot.ring != nullptr) {
                ok = log.cuda("free ring buffer on " + std::to_string(slot.device),
                              [&] { return cudaFree(slot.ring); }, step_error) && ok;
                slot.ring = nullptr;
            }
            if (slot.stream != nullptr) {
                ok = log.cuda("destroy stream on " + std::to_string(slot.device),
                              [&] { return cudaStreamDestroy(slot.stream); }, step_error) && ok;
                slot.stream = nullptr;
            }
            append_error(error, step_error);
        }
        return ok;
    }

    ~DeviceResources() {
        std::string ignored;
        close(ignored);
    }

    ProbeLog& log;
    std::vector<Slot> slots;
    bool closed = false;
};

bool upload_main(DeviceResources::Slot& slot, const Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device(" + std::to_string(slot.device) + ") for main upload",
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("upload main image to device " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(slot.main, image.data(), image.size(), cudaMemcpyHostToDevice); }, error);
}

bool upload_ring(DeviceResources::Slot& slot, const Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device(" + std::to_string(slot.device) + ") for ring upload",
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("upload ring image to device " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(slot.ring, image.data(), image.size(), cudaMemcpyHostToDevice); }, error);
}

bool download_main(DeviceResources::Slot& slot, Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device(" + std::to_string(slot.device) + ") for main download",
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("download main image from device " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(image.data(), slot.main, image.size(), cudaMemcpyDeviceToHost); }, error);
}

bool download_ring(DeviceResources::Slot& slot, Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device(" + std::to_string(slot.device) + ") for ring download",
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("download ring image from device " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(image.data(), slot.ring, image.size(), cudaMemcpyDeviceToHost); }, error);
}

bool compare_images(const Bytes& actual_source, const Bytes& expected_source,
                    const Bytes& actual_destination, const Bytes& expected_destination,
                    std::string& error) {
    strata::ds4::p2p::verify::Mismatch mismatch;
    if (!strata::ds4::p2p::verify::equal(actual_source, expected_source, mismatch)) {
        std::ostringstream out;
        out << "source payload/guard mismatch at byte " << mismatch.offset << ": expected "
            << static_cast<unsigned int>(mismatch.expected) << ", got " << static_cast<unsigned int>(mismatch.actual);
        error = out.str();
        return false;
    }
    if (!strata::ds4::p2p::verify::equal(actual_destination, expected_destination, mismatch)) {
        std::ostringstream out;
        out << "destination payload/guard mismatch at byte " << mismatch.offset << ": expected "
            << static_cast<unsigned int>(mismatch.expected) << ", got " << static_cast<unsigned int>(mismatch.actual);
        error = out.str();
        return false;
    }
    return true;
}

bool verify_main(DeviceResources::Slot& source, DeviceResources::Slot& destination,
                 const Bytes& expected_source, const Bytes& expected_destination,
                 ProbeLog& log, std::string& error) {
    Bytes actual_source(expected_source.size()), actual_destination(expected_destination.size());
    return download_main(source, actual_source, log, error) &&
           download_main(destination, actual_destination, log, error) &&
           compare_images(actual_source, expected_source, actual_destination, expected_destination, error);
}

bool verify_ring(DeviceResources::Slot& source, DeviceResources::Slot& destination,
                 const Bytes& expected_source, const Bytes& expected_destination,
                 ProbeLog& log, std::string& error) {
    Bytes actual_source(expected_source.size()), actual_destination(expected_destination.size());
    return download_ring(source, actual_source, log, error) &&
           download_ring(destination, actual_destination, log, error) &&
           compare_images(actual_source, expected_source, actual_destination, expected_destination, error);
}

std::uint32_t seed_for(int src, int dst, int kind, int a, int b = 0, int c = 0) {
    return static_cast<std::uint32_t>(src + 16) * 0x10204081u ^ static_cast<std::uint32_t>(dst + 16) * 0x20408103u ^
           static_cast<std::uint32_t>(kind + 1) * 0x40810207u ^ static_cast<std::uint32_t>(a + 32) * 0x08102041u ^
           static_cast<std::uint32_t>(b + 32) * 0x01020419u ^ static_cast<std::uint32_t>(c + 32) * 0x10204083u;
}

bool runtime_copy_case(DeviceResources& resources, PeerAccess& peers, int src, int dst,
                       std::size_t alignment, ProbeLog& log, std::string& error) {
    DeviceResources::Slot& source = resources.get(src);
    DeviceResources::Slot& destination = resources.get(dst);
    const std::size_t offset = kGuardBytes + alignment;
    const std::uint32_t source_seed = seed_for(src, dst, 0, static_cast<int>(alignment));
    Bytes source_image = strata::ds4::p2p::verify::make_region(offset, kPayloadBytes, kGuardBytes, source_seed);
    Bytes destination_image = strata::ds4::p2p::verify::make_region(
        offset, kPayloadBytes, kGuardBytes, source_seed ^ 0x51ed270bu);
    if (source_image.empty() || destination_image.empty()) {
        error = "failed to construct bounded main test images";
        return false;
    }
    Bytes expected_destination = destination_image;
    std::copy_n(source_image.begin() + static_cast<std::ptrdiff_t>(offset), kPayloadBytes,
                expected_destination.begin() + static_cast<std::ptrdiff_t>(offset));
    if (!upload_main(source, source_image, log, error) || !upload_main(destination, destination_image, log, error))
        return false;

    std::size_t bytes_enqueued = 0;
    std::string copy_error;
    if (!peers.copy_peer_async(destination.main + offset, dst, source.main + offset, src, kPayloadBytes,
                               destination.stream, bytes_enqueued, copy_error)) {
        error = copy_error;
        return false;
    }
    if (bytes_enqueued != kPayloadBytes) {
        error = "peer copy byte count mismatch: expected " + std::to_string(kPayloadBytes) + ", got " +
                std::to_string(bytes_enqueued);
        return false;
    }
    if (!log.cuda("set_device(" + std::to_string(dst) + ") for runtime-copy synchronization",
                  [&] { return cudaSetDevice(dst); }, error) ||
        !log.cuda("synchronize runtime peer copy " + pair_name(src, dst),
                  [&] { return cudaStreamSynchronize(destination.stream); }, error))
        return false;
    return verify_main(source, destination, source_image, expected_destination, log, error);
}

bool uva_read_case(DeviceResources& resources, int src, int dst, std::size_t alignment,
                   ProbeLog& log, std::string& error) {
    DeviceResources::Slot& source = resources.get(src);
    DeviceResources::Slot& destination = resources.get(dst);
    const std::size_t offset = kGuardBytes + alignment;
    const std::uint32_t source_seed = seed_for(src, dst, 1, static_cast<int>(alignment));
    Bytes source_image = strata::ds4::p2p::verify::make_region(offset, kPayloadBytes, kGuardBytes, source_seed);
    Bytes destination_image = strata::ds4::p2p::verify::make_region(
        offset, kPayloadBytes, kGuardBytes, source_seed ^ 0x68bc21ebu);
    if (source_image.empty() || destination_image.empty()) {
        error = "failed to construct bounded UVA-read images";
        return false;
    }
    Bytes expected_destination = destination_image;
    for (std::size_t i = 0; i < kPayloadBytes; ++i)
        expected_destination[offset + i] = strata::ds4::p2p::verify::transformed_byte(source_image[offset + i]);
    if (!upload_main(source, source_image, log, error) || !upload_main(destination, destination_image, log, error))
        return false;
    if (!log.cuda("set_device(" + std::to_string(dst) + ") for UVA peer read",
                  [&] { return cudaSetDevice(dst); }, error))
        return false;
    const unsigned int blocks = static_cast<unsigned int>((kPayloadBytes + kThreads - 1) / kThreads);
    const std::uint8_t* peer = source.main + offset;
    std::uint8_t* local = destination.main + offset;
    cudaStream_t stream = destination.stream;
    if (!log.cuda("launch UVA kernel on destination reading source " + pair_name(src, dst),
                  [&] {
                      uva_read_kernel<<<blocks, kThreads, 0, stream>>>(peer, local, kPayloadBytes);
                      return cudaGetLastError();
                  }, error) ||
        !log.cuda("synchronize UVA peer read " + pair_name(src, dst),
                  [&] { return cudaStreamSynchronize(stream); }, error))
        return false;
    return verify_main(source, destination, source_image, expected_destination, log, error);
}

bool uva_write_case(DeviceResources& resources, int src, int dst, std::size_t alignment,
                    ProbeLog& log, std::string& error) {
    DeviceResources::Slot& source = resources.get(src);
    DeviceResources::Slot& destination = resources.get(dst);
    const std::size_t offset = kGuardBytes + alignment;
    const std::uint32_t source_seed = seed_for(src, dst, 2, static_cast<int>(alignment));
    const std::uint32_t write_seed = source_seed ^ 0x02f5e1b7u;
    Bytes source_image = strata::ds4::p2p::verify::make_region(offset, kPayloadBytes, kGuardBytes, source_seed);
    Bytes destination_image = strata::ds4::p2p::verify::make_region(
        offset, kPayloadBytes, kGuardBytes, source_seed ^ 0x6a09e667u);
    if (source_image.empty() || destination_image.empty()) {
        error = "failed to construct bounded UVA-write images";
        return false;
    }
    Bytes expected_destination = destination_image;
    strata::ds4::p2p::verify::fill_payload(expected_destination, offset, kPayloadBytes, write_seed);
    if (!upload_main(source, source_image, log, error) || !upload_main(destination, destination_image, log, error))
        return false;
    if (!log.cuda("set_device(" + std::to_string(src) + ") for UVA peer write",
                  [&] { return cudaSetDevice(src); }, error))
        return false;
    const unsigned int blocks = static_cast<unsigned int>((kPayloadBytes + kThreads - 1) / kThreads);
    std::uint8_t* peer = destination.main + offset;
    cudaStream_t stream = source.stream;
    if (!log.cuda("launch UVA kernel on source writing destination " + pair_name(src, dst),
                  [&] {
                      uva_write_kernel<<<blocks, kThreads, 0, stream>>>(peer, kPayloadBytes, write_seed);
                      return cudaGetLastError();
                  }, error) ||
        !log.cuda("synchronize UVA peer write " + pair_name(src, dst),
                  [&] { return cudaStreamSynchronize(stream); }, error))
        return false;
    return verify_main(source, destination, source_image, expected_destination, log, error);
}

Bytes ring_image(int src, int dst, int round, int salt) {
    Bytes image(kRingAllocationBytes, 0);
    for (std::size_t slot = 0; slot < kRingSlots; ++slot) {
        const std::size_t offset = slot * kRingSlotStride + kGuardBytes;
        const std::uint32_t seed = seed_for(src, dst, 3 + salt, round, static_cast<int>(slot));
        if (!strata::ds4::p2p::verify::initialize_region(image, offset, kRingPayloadBytes, kGuardBytes, seed))
            return Bytes{};
    }
    return image;
}

bool ring_case(DeviceResources& resources, PeerAccess& peers, int src, int dst, int round, int slot,
               ProbeLog& log, std::string& error) {
    DeviceResources::Slot& source = resources.get(src);
    DeviceResources::Slot& destination = resources.get(dst);
    const int destination_slot = (slot + 1) % static_cast<int>(kRingSlots);
    Bytes source_image = ring_image(src, dst, round, 0);
    Bytes destination_image = ring_image(src, dst, round, 1);
    if (source_image.empty() || destination_image.empty()) {
        error = "failed to construct bounded ring images";
        return false;
    }
    const std::size_t source_offset = static_cast<std::size_t>(slot) * kRingSlotStride + kGuardBytes;
    const std::size_t destination_offset = static_cast<std::size_t>(destination_slot) * kRingSlotStride + kGuardBytes;
    Bytes expected_destination = destination_image;
    std::copy_n(source_image.begin() + static_cast<std::ptrdiff_t>(source_offset), kRingPayloadBytes,
                expected_destination.begin() + static_cast<std::ptrdiff_t>(destination_offset));
    if (!upload_ring(source, source_image, log, error) || !upload_ring(destination, destination_image, log, error))
        return false;

    std::size_t bytes_enqueued = 0;
    std::string copy_error;
    if (!peers.copy_peer_async(destination.ring + destination_offset, dst, source.ring + source_offset, src,
                               kRingPayloadBytes, destination.stream, bytes_enqueued, copy_error)) {
        error = copy_error;
        return false;
    }
    if (bytes_enqueued != kRingPayloadBytes) {
        error = "ring peer copy byte count mismatch";
        return false;
    }
    if (!log.cuda("set_device(" + std::to_string(dst) + ") for ring synchronization",
                  [&] { return cudaSetDevice(dst); }, error) ||
        !log.cuda("synchronize ring peer copy " + pair_name(src, dst),
                  [&] { return cudaStreamSynchronize(destination.stream); }, error))
        return false;
    return verify_ring(source, destination, source_image, expected_destination, log, error);
}

bool run_pair(DeviceResources& resources, PeerAccess& peers, int src, int dst, ProbeLog& log) {
    mark("DIRECTED PAIR " + pair_name(src, dst) + " start");
    for (std::size_t alignment : {std::size_t{1}, std::size_t{4}, std::size_t{16}}) {
        if (!log.check("runtime_peer_copy", src, dst, static_cast<int>(alignment), -1, -1,
                       [&](std::string& error) {
                           return runtime_copy_case(resources, peers, src, dst, alignment, log, error);
                       }))
            return false;
        if (!log.check("uva_kernel_read", src, dst, static_cast<int>(alignment), -1, -1,
                       [&](std::string& error) { return uva_read_case(resources, src, dst, alignment, log, error); }))
            return false;
        if (!log.check("uva_kernel_write", src, dst, static_cast<int>(alignment), -1, -1,
                       [&](std::string& error) { return uva_write_case(resources, src, dst, alignment, log, error); }))
            return false;
    }
    for (int round = 0; round < static_cast<int>(kRingRounds); ++round) {
        for (int slot = 0; slot < static_cast<int>(kRingSlots); ++slot) {
            if (!log.check("runtime_peer_ring_copy", src, dst, 16, slot, round,
                           [&](std::string& error) { return ring_case(resources, peers, src, dst, round, slot, log, error); }))
                return false;
        }
    }
    mark("DIRECTED PAIR " + pair_name(src, dst) + " ok");
    return true;
}

std::string normalize_bdf(const std::string& input) {
    if (input.size() != 12 || input[4] != ':' || input[7] != ':' || input[10] != '.') return {};
    std::string output = input;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i == 4 || i == 7 || i == 10) continue;
        const unsigned char c = static_cast<unsigned char>(output[i]);
        if (!std::isxdigit(c)) return {};
        output[i] = static_cast<char>(std::tolower(c));
    }
    return output;
}

bool parse_ordinal(const std::string& text, int& ordinal) {
    if (text.empty()) return false;
    const char* first = text.data();
    const char* last = first + text.size();
    const auto result = std::from_chars(first, last, ordinal);
    return result.ec == std::errc() && result.ptr == last && ordinal >= 0;
}

void usage(std::ostream& out) {
    out << "Usage: p2p_probe --device ORDINAL=0000:BB:DD.F --device ORDINAL=0000:BB:DD.F "
           "--device ORDINAL=0000:BB:DD.F --device ORDINAL=0000:BB:DD.F [--json PATH]\n"
           "Exactly four manually assigned devices are required. Human logs go to stderr; the final JSON goes to stdout.\n"
           "This executable performs real GPU peer operations; compilation and host unit tests do not validate hardware.\n";
}

bool parse_arguments(int argc, char** argv, std::vector<DeviceSpec>& devices, std::string& json_path,
                     bool& help, std::string& error) {
    help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            help = true;
            return true;
        }
        if (argument == "--json") {
            if (++i >= argc) {
                error = "--json requires a path";
                return false;
            }
            json_path = argv[i];
            if (json_path.empty()) {
                error = "--json path is empty";
                return false;
            }
            continue;
        }
        if (argument != "--device") {
            error = "unknown argument: " + argument;
            return false;
        }
        if (++i >= argc) {
            error = "--device requires ORDINAL=0000:BB:DD.F";
            return false;
        }
        const std::string specification = argv[i];
        const std::size_t equals = specification.find('=');
        if (equals == std::string::npos || specification.find('=', equals + 1) != std::string::npos) {
            error = "invalid --device specification: " + specification;
            return false;
        }
        DeviceSpec device;
        if (!parse_ordinal(specification.substr(0, equals), device.ordinal)) {
            error = "invalid device ordinal in: " + specification;
            return false;
        }
        device.bdf = normalize_bdf(specification.substr(equals + 1));
        if (device.bdf.empty()) {
            error = "BDF must use canonical 0000:BB:DD.F hexadecimal form: " + specification;
            return false;
        }
        for (const DeviceSpec& existing : devices) {
            if (existing.ordinal == device.ordinal) {
                error = "device ordinal repeated: " + std::to_string(device.ordinal);
                return false;
            }
            if (existing.bdf == device.bdf) {
                error = "BDF repeated: " + device.bdf;
                return false;
            }
        }
        devices.push_back(device);
    }
    if (devices.size() != 4) {
        error = "exactly four --device ORDINAL=BDF mappings are required; got " + std::to_string(devices.size());
        return false;
    }
    return true;
}

bool check_device_mappings(Report& report, ProbeLog& log, std::string& error) {
    for (std::size_t i = 0; i < report.devices.size(); ++i) {
        DeviceSpec& device = report.devices[i];
        if (!log.check("manual_bdf_mapping", device.ordinal, -1, -1, -1, -1, [&](std::string& check_error) {
                char buffer[64] = {};
                if (!log.cuda("query BDF for explicit device " + std::to_string(device.ordinal),
                              [&] { return cudaDeviceGetPCIBusId(buffer, static_cast<int>(sizeof(buffer)), device.ordinal); },
                              check_error))
                    return false;
                device.actual_bdf = normalize_bdf(buffer);
                if (device.actual_bdf.empty()) {
                    check_error = "runtime returned non-canonical BDF: " + std::string(buffer);
                    return false;
                }
                device.mapping_ok = device.actual_bdf == device.bdf;
                if (!device.mapping_ok) {
                    check_error = "manual BDF mismatch for device " + std::to_string(device.ordinal) + ": supplied " +
                                  device.bdf + ", runtime reports " + device.actual_bdf;
                    return false;
                }
                return true;
            })) {
            error = "explicit device/BDF mapping validation failed";
            return false;
        }
    }
    return true;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (unsigned char c : text) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char* digits = "0123456789abcdef";
                    out << "\\u00" << digits[c >> 4] << digits[c & 15];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

const char* boolean(bool value) { return value ? "true" : "false"; }

std::string report_json(const Report& report, bool success) {
    std::ostringstream out;
    out << "{\n  \"schema\": \"ds4-p2p-probe-v1\",\n  \"success\": " << boolean(success) << ",\n";
    out << "  \"hardware_acceptance\": \"" << (success ? "all-explicit-directed-checks-passed" : "not-established") << "\",\n";
    out << "  \"devices\": [";
    for (std::size_t i = 0; i < report.devices.size(); ++i) {
        const DeviceSpec& device = report.devices[i];
        out << (i ? "," : "") << "\n    {\"ordinal\": " << device.ordinal << ", \"bdf\": \""
            << json_escape(device.bdf) << "\", \"actual_bdf\": \"" << json_escape(device.actual_bdf)
            << "\", \"mapping_ok\": " << boolean(device.mapping_ok) << "}";
    }
    out << "\n  ],\n  \"events\": [";
    for (std::size_t i = 0; i < report.events.size(); ++i) {
        const Event& event = report.events[i];
        out << (i ? "," : "") << "\n    {\"operation\": \"" << json_escape(event.operation) << "\", \"device\": "
            << event.device << ", \"peer\": " << event.peer << ", \"src\": " << event.src << ", \"dst\": " << event.dst
            << ", \"bytes\": " << event.bytes << ", \"status\": " << event.status << ", \"can_access\": "
            << event.can_access << ", \"ok\": " << boolean(event.ok) << "}";
    }
    out << "\n  ],\n  \"checks\": [";
    for (std::size_t i = 0; i < report.checks.size(); ++i) {
        const Check& check = report.checks[i];
        out << (i ? "," : "") << "\n    {\"name\": \"" << json_escape(check.name) << "\", \"src\": " << check.src
            << ", \"dst\": " << check.dst << ", \"alignment\": " << check.alignment << ", \"slot\": " << check.slot
            << ", \"round\": " << check.round << ", \"ok\": " << boolean(check.ok) << ", \"error\": \""
            << json_escape(check.error) << "\"}";
    }
    std::size_t passed = 0;
    out << "\n  ],\n  \"directed_pairs\": [";
    for (std::size_t i = 0; i < report.pairs.size(); ++i) {
        const PairResult& pair = report.pairs[i];
        if (pair.ok) ++passed;
        out << (i ? "," : "") << "\n    {\"src\": " << pair.src << ", \"dst\": " << pair.dst << ", \"ran\": "
            << boolean(pair.ran) << ", \"ok\": " << boolean(pair.ok) << "}";
    }
    std::size_t checks_failed = 0;
    for (const Check& check : report.checks) {
        if (!check.ok) ++checks_failed;
    }
    out << "\n  ],\n  \"summary\": {\"required_directed_pairs\": 12, \"passed_directed_pairs\": " << passed
        << ", \"checks_total\": " << report.checks.size() << ", \"checks_failed\": " << checks_failed << "},\n";
    out << "  \"error\": \"" << json_escape(report.error) << "\"\n}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    Report report;
    std::string json_path;
    std::string error;
    bool help = false;
    if (!parse_arguments(argc, argv, report.devices, json_path, help, error)) {
        std::fprintf(stderr, "P2P_PROBE argument error: %s\n", error.c_str());
        usage(std::cerr);
        return 2;
    }
    if (help) {
        usage(std::cout);
        return 0;
    }

    for (const DeviceSpec& source : report.devices) {
        for (const DeviceSpec& destination : report.devices) {
            if (source.ordinal != destination.ordinal)
                report.pairs.push_back(PairResult{source.ordinal, destination.ordinal, false, false});
        }
    }

    ProbeLog log(report);
    if (check_device_mappings(report, log, error)) {
        std::vector<int> ordinals;
        for (const DeviceSpec& device : report.devices) ordinals.push_back(device.ordinal);
        LoggingRuntime logging_runtime(strata::ds4::p2p::cuda_runtime(), report);
        {
            PeerAccess peers(logging_runtime, ordinals);
            std::string setup_error;
            if (!peers.enable_all(setup_error)) {
                append_error(error, setup_error);
            } else {
                DeviceResources resources(report.devices, log);
                std::string resource_error;
                if (!resources.initialize(resource_error)) {
                    append_error(error, resource_error);
                } else {
                    for (PairResult& pair : report.pairs) {
                        pair.ran = true;
                        if (!run_pair(resources, peers, pair.src, pair.dst, log)) {
                            append_error(error, "directed pair failed: " + pair_name(pair.src, pair.dst));
                            break;
                        }
                        pair.ok = true;
                    }
                }
                std::string close_error;
                if (!resources.close(close_error)) append_error(error, "resource teardown failed: " + close_error);
            }
            std::string disable_error;
            if (!peers.disable_all(disable_error)) append_error(error, "peer teardown failed: " + disable_error);
        }
    }

    const bool all_pairs = report.pairs.size() == 12 &&
                           std::all_of(report.pairs.begin(), report.pairs.end(), [](const PairResult& pair) {
                               return pair.ran && pair.ok;
                           });
    bool success = error.empty() && all_pairs;
    if (!all_pairs) append_error(error, "success requires all 12 directed pairs to pass");
    success = error.empty() && all_pairs;
    std::string json = report_json(report, success);
    if (!json_path.empty()) {
        std::ofstream output(json_path, std::ios::binary | std::ios::trunc);
        if (!output || !(output << json) || !output.flush()) {
            append_error(error, "could not write JSON report: " + json_path);
            success = false;
            json = report_json(report, false);
        }
    }
    std::cout << json;
    std::cout.flush();
    mark(success ? "SUCCESS: all 12 directed pairs passed payload, guard, alignment, ring, runtime and UVA checks"
                 : "FAILURE: hardware acceptance not established");
    return success ? 0 : 1;
}
