#include "queue_policy.hpp"

#if !defined(FUSED_AR_PROBE_LOGIC_ONLY)
#include "fused_ar.hpp"
#include "p2p_verify.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fused = strata::ds4::p2p::fused;
namespace verify = strata::ds4::p2p::verify;
using strata::ds4::p2p::PeerAccess;
using Bytes = std::vector<std::uint8_t>;
using fused::QueueInputLayout;
using fused::QueuePolicy;
using fused::make_queue_input_layout;
using fused::make_queue_policy;
using fused::queue_input_offset;
using fused::validate_queue_arguments;

#ifndef DS4_FUSED_PROBE_ELEMENTS
#define DS4_FUSED_PROBE_ELEMENTS 16384
#endif
constexpr int kElements = DS4_FUSED_PROBE_ELEMENTS;
static_assert(kElements > 0 && kElements % 4 == 0, "probe elements must be positive float4 values");
constexpr std::size_t kXBytes = static_cast<std::size_t>(kElements) * sizeof(float);
constexpr std::size_t kXAllocationBytes = fused::kGuardBytes + kXBytes + fused::kGuardBytes;
constexpr unsigned int kMaxSequences = 256;

struct DeviceSpec {
    int ordinal = -1;
    std::string bdf;
    std::string actual_bdf;
    bool mapping_ok = false;
};

struct Check {
    std::string name;
    unsigned int seq = 0;
    int member = -1;
    bool ok = false;
    std::string error;
};

struct SequenceResult {
    unsigned int seq = 0;
    bool ok = false;
};

struct Report {
    std::vector<DeviceSpec> devices;
    std::vector<Check> checks;
    std::vector<SequenceResult> sequences;
    int slots = 0;
    int requested_sequences = 0;
    std::uint64_t timeout_cycles = 0;
    std::uint64_t skew_cycles = 0;
    bool fp16_wire = false;
    bool queue_mode = false;
    int queue_depth = 0;
    bool intermediate_outputs_checked = true;
    std::size_t pinned_input_bytes = 0;
    bool pinned_inputs_retained = false;
    std::string pinned_input_status = "not_allocated";
    std::string error;
};

void append_error(std::string& error, const std::string& next) {
    if (next.empty()) return;
    if (!error.empty()) error += "; ";
    error += next;
}

void mark(const std::string& message) {
    std::fprintf(stderr, "FUSED_AR_PROBE %s\n", message.c_str());
    std::fflush(stderr);
}

std::string cuda_error(const std::string& operation, cudaError_t status) {
    return operation + " failed with runtime error " + std::to_string(static_cast<int>(status)) + " (" +
           cudaGetErrorString(status) + ")";
}

class ProbeLog {
public:
    explicit ProbeLog(Report& report) : report_(report) {}

    template <typename F>
    bool cuda(const std::string& operation, F&& call, std::string& error) {
        mark(operation + " start");
        const cudaError_t status = call();
        if (status != cudaSuccess) {
            append_error(error, cuda_error(operation, status));
            mark(operation + " FAIL: " + cudaGetErrorString(status));
            return false;
        }
        mark(operation + " ok");
        return true;
    }

    template <typename F>
    bool check(const std::string& name, unsigned int seq, int member, F&& call) {
        mark("CHECK " + name + " seq=" + std::to_string(seq) + ",member=" + std::to_string(member) + " start");
        std::string error;
        const bool ok = call(error);
        if (!ok && error.empty()) error = "check failed without a diagnostic";
        report_.checks.push_back(Check{name, seq, member, ok, error});
        mark("CHECK " + name + " seq=" + std::to_string(seq) + ",member=" + std::to_string(member) +
             (ok ? " ok" : " FAIL: " + error));
        return ok;
    }

private:
    Report& report_;
};

__global__ void skew_delay_kernel(std::uint64_t cycles) {
    const unsigned long long start = static_cast<unsigned long long>(clock64());
    while (static_cast<unsigned long long>(clock64()) - start <= cycles) {}
}

struct ProbeBuffers {
    struct Slot {
        int device = -1;
        cudaStream_t stream = nullptr;
        std::uint8_t* x = nullptr;
    };

    ProbeBuffers(const std::vector<DeviceSpec>& devices, ProbeLog& probe_log) : log(probe_log) {
        for (const DeviceSpec& device : devices) slots.push_back(Slot{device.ordinal, nullptr, nullptr});
    }

    bool initialize(std::string& error) {
        for (Slot& slot : slots) {
            if (!log.cuda("set_device for fused probe buffers " + std::to_string(slot.device),
                          [&] { return cudaSetDevice(slot.device); }, error) ||
                !log.cuda("create fused probe stream " + std::to_string(slot.device),
                          [&] { return cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking); }, error) ||
                !log.cuda("allocate guarded fused x " + std::to_string(slot.device),
                          [&] { return cudaMalloc(reinterpret_cast<void**>(&slot.x), kXAllocationBytes); }, error))
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
            ok = log.cuda("quiesce fused probe device " + std::to_string(slot.device),
                          [&] { return cudaSetDevice(slot.device); }, step_error) && ok;
            ok = log.cuda("synchronize fused probe device " + std::to_string(slot.device),
                          [&] { return cudaDeviceSynchronize(); }, step_error) && ok;
            append_error(error, step_error);
        }
        for (Slot& slot : slots) {
            std::string step_error;
            ok = log.cuda("set_device for fused probe release " + std::to_string(slot.device),
                          [&] { return cudaSetDevice(slot.device); }, step_error) && ok;
            if (slot.x != nullptr) {
                ok = log.cuda("free fused probe x " + std::to_string(slot.device),
                              [&] { return cudaFree(slot.x); }, step_error) && ok;
                slot.x = nullptr;
            }
            if (slot.stream != nullptr) {
                ok = log.cuda("destroy fused probe stream " + std::to_string(slot.device),
                              [&] { return cudaStreamDestroy(slot.stream); }, step_error) && ok;
                slot.stream = nullptr;
            }
            append_error(error, step_error);
        }
        return ok;
    }

    ~ProbeBuffers() {
        std::string ignored;
        close(ignored);
    }

    ProbeLog& log;
    std::vector<Slot> slots;
    bool closed = false;
};

class QueuedInputs {
public:
    QueuedInputs(Report& report, ProbeLog& log) : report_(report), log_(log) {}

    bool initialize(int members, int sequences, std::string& error) {
        QueueInputLayout next;
        if (!make_queue_input_layout(members, sequences, kXAllocationBytes, next)) {
            report_.pinned_input_status = "invalid_layout";
            error = "queued pinned input layout is invalid or overflows";
            return false;
        }
        layout_ = next;
        report_.pinned_input_bytes = layout_.total_bytes;
        void* memory = nullptr;
        if (!log_.cuda("allocate pinned queued input storage", [&] {
                return cudaHostAlloc(&memory, layout_.total_bytes, cudaHostAllocPortable);
            }, error)) {
            report_.pinned_input_status = "allocation_failed";
            return false;
        }
        data_ = static_cast<std::uint8_t*>(memory);
        report_.pinned_input_status = "allocated";
        return true;
    }

    bool store(int sequence_index, int member, const Bytes& image, std::string& error) {
        std::size_t offset = 0;
        if (data_ == nullptr || image.size() != layout_.image_bytes ||
            !queue_input_offset(layout_, sequence_index, member, offset)) {
            error = "queued pinned input store is outside its validated layout";
            return false;
        }
        std::memcpy(data_ + offset, image.data(), image.size());
        return true;
    }

    const std::uint8_t* source(int sequence_index, int member) const {
        std::size_t offset = 0;
        if (data_ == nullptr || !queue_input_offset(layout_, sequence_index, member, offset)) return nullptr;
        return data_ + offset;
    }

    bool close(bool safe_to_release, std::string& error) {
        if (finished_) return true;
        finished_ = true;
        if (data_ == nullptr) return true;
        if (!safe_to_release) {
            report_.pinned_inputs_retained = true;
            report_.pinned_input_status = "retained_after_failure_or_without_quiesce";
            append_error(error, "pinned queued inputs retained because safe device quiescence was not established");
            mark("pinned queued input storage retained; no free was attempted");
            data_ = nullptr;
            return true;
        }
        const bool ok = log_.cuda("release pinned queued input storage", [&] { return cudaFreeHost(data_); }, error);
        if (ok) {
            data_ = nullptr;
            report_.pinned_input_status = "released";
            return true;
        }
        report_.pinned_inputs_retained = true;
        report_.pinned_input_status = "retained_after_free_failure";
        data_ = nullptr;
        return false;
    }

    ~QueuedInputs() {
        std::string ignored;
        close(false, ignored);
    }

private:
    Report& report_;
    ProbeLog& log_;
    QueueInputLayout layout_;
    std::uint8_t* data_ = nullptr;
    bool finished_ = false;
};

float* x_payload(ProbeBuffers::Slot& slot) {
    return reinterpret_cast<float*>(slot.x + fused::kGuardBytes);
}

bool upload_x(ProbeBuffers::Slot& slot, const Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device for fused x upload " + std::to_string(slot.device),
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("upload guarded fused x " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(slot.x, image.data(), image.size(), cudaMemcpyHostToDevice); }, error);
}

bool upload_x_queued(ProbeBuffers::Slot& slot, const std::uint8_t* image, ProbeLog& log, std::string& error) {
    if (image == nullptr) {
        error = "queued upload requires a valid pinned input address";
        return false;
    }
    return log.cuda("set_device for queued fused x upload " + std::to_string(slot.device),
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("queue pinned fused x upload " + std::to_string(slot.device),
                    [&] {
                        return cudaMemcpyAsync(slot.x, image, kXAllocationBytes, cudaMemcpyHostToDevice, slot.stream);
                    }, error);
}

bool download_x(ProbeBuffers::Slot& slot, Bytes& image, ProbeLog& log, std::string& error) {
    return log.cuda("set_device for fused x download " + std::to_string(slot.device),
                    [&] { return cudaSetDevice(slot.device); }, error) &&
           log.cuda("download guarded fused x " + std::to_string(slot.device),
                    [&] { return cudaMemcpy(image.data(), slot.x, image.size(), cudaMemcpyDeviceToHost); }, error);
}

void store_float(Bytes& image, std::size_t byte_offset, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "FP32 probe requires 32-bit float");
    std::memcpy(image.data() + byte_offset, &value, sizeof(value));
}

float load_float(const Bytes& image, std::size_t byte_offset) {
    float value = 0.0f;
    std::memcpy(&value, image.data() + byte_offset, sizeof(value));
    return value;
}

std::uint16_t fp16_wire_bits(float value) {
    const float saturated = std::fmin(std::fmax(value, -65504.0f), 65504.0f);
    return strata::kernels::f16_from_f32(saturated);
}

float fp16_wire_round(float value) {
    return strata::kernels::f32_from_f16(fp16_wire_bits(value));
}

float reference_fp16_wire(const float* values, int members) {
    float partials[fused::kMaxMembers] = {};
    for (int member = 0; member < members; ++member) partials[member] = fp16_wire_round(values[member]);
    return fp16_wire_round(fused::reference_sum(partials, members));
}

void store_fp16(Bytes& image, std::size_t byte_offset, float value) {
    const std::uint16_t bits = fp16_wire_bits(value);
    std::memcpy(image.data() + byte_offset, &bits, sizeof(bits));
}

float input_value(int member, int members, unsigned int seq, int index) {
    if (index % 31 == 0) {
        if (member == 0) return 1.0e20f;
        if (member == 1) return -1.0e20f;
        if (member == 2) return 3.0f;
        if (member == 3) return 4.0f;
    }
    const std::uint32_t bits = static_cast<std::uint32_t>(index) * 2654435761u + seq * 2246822519u +
                               static_cast<std::uint32_t>(member + 1) * 3266489917u;
    const int integer = static_cast<int>(bits % 20001u) - 10000;
    const float scale = member & 1 ? -0.0009765625f : 0.0009765625f;
    return static_cast<float>(integer) * scale + static_cast<float>(seq & 15u) * 0.125f +
           static_cast<float>(members) * 0.0625f;
}

Bytes make_input(int member, int members, unsigned int seq) {
    Bytes image = verify::make_region(fused::kGuardBytes, kXBytes, fused::kGuardBytes,
                                      seq * 17u + static_cast<unsigned int>(member));
    if (image.empty()) return image;
    for (int i = 0; i < kElements; ++i)
        store_float(image, fused::kGuardBytes + static_cast<std::size_t>(i) * sizeof(float),
                    input_value(member, members, seq, i));
    return image;
}

std::string mismatch_error(const char* what, const verify::Mismatch& mismatch) {
    std::ostringstream out;
    out << what << " mismatch at byte " << mismatch.offset << ": expected "
        << static_cast<unsigned int>(mismatch.expected) << ", got " << static_cast<unsigned int>(mismatch.actual);
    return out.str();
}

bool upload_outboxes(fused::Group& group, const std::vector<DeviceSpec>& devices,
                     const std::vector<Bytes>& expected, ProbeLog& log, std::string& error) {
    for (int member = 0; member < group.members(); ++member) {
        if (!log.cuda("set_device for outbox initialization " + std::to_string(devices[member].ordinal),
                      [&] { return cudaSetDevice(devices[member].ordinal); }, error) ||
            !log.cuda("initialize guarded outbox " + std::to_string(devices[member].ordinal),
                      [&] {
                          return cudaMemcpy(group.outbox_raw(member), expected[member].data(),
                                            expected[member].size(), cudaMemcpyHostToDevice);
                      }, error))
            return false;
    }
    return true;
}

bool run_sequence(fused::Group& group, ProbeBuffers& buffers, const std::vector<DeviceSpec>& devices,
                  unsigned int seq, int slots, std::uint64_t skew_cycles, bool fp16_wire,
                  std::vector<Bytes>& expected_outboxes, ProbeLog& log, std::string& error,
                  QueuedInputs* queued_inputs, unsigned int final_seq) {
    const int members = group.members();
    const bool queued = queued_inputs != nullptr;
    const bool last = !queued || seq == final_seq;
    std::vector<Bytes> inputs(static_cast<std::size_t>(members));
    std::vector<Bytes> expected_x(static_cast<std::size_t>(members));
    for (int member = 0; member < members; ++member) {
        inputs[member] = make_input(member, members, seq);
        if (inputs[member].empty()) {
            append_error(error, "could not create/upload guarded input");
            return false;
        }
        const std::uint8_t* source = nullptr;
        if (queued) {
            const int sequence_index = static_cast<int>(seq) - 1;
            if (!queued_inputs->store(sequence_index, member, inputs[member], error)) return false;
            source = queued_inputs->source(sequence_index, member);
        }
        const bool uploaded = queued ? upload_x_queued(buffers.slots[member], source, log, error)
                                     : upload_x(buffers.slots[member], inputs[member], log, error);
        if (!uploaded) {
            append_error(error, "could not create/upload guarded input");
            return false;
        }
    }

    std::vector<float> expected(kElements, 0.0f);
    std::vector<float> partials(static_cast<std::size_t>(members), 0.0f);
    for (int i = 0; i < kElements; ++i) {
        for (int member = 0; member < members; ++member)
            partials[member] = load_float(inputs[member], fused::kGuardBytes + static_cast<std::size_t>(i) * sizeof(float));
        expected[i] = fp16_wire ? reference_fp16_wire(partials.data(), members)
                                : fused::reference_sum(partials.data(), members);
    }
    for (int member = 0; member < members; ++member) {
        expected_x[member] = inputs[member];
        for (int i = 0; i < kElements; ++i)
            store_float(expected_x[member], fused::kGuardBytes + static_cast<std::size_t>(i) * sizeof(float), expected[i]);
    }

    int slot = 0;
    if (!fused::slot_for_sequence(seq, slots, slot)) {
        error = "internal ring shape error";
        return false;
    }
    for (int member = 0; member < members; ++member) {
        Bytes& outbox = expected_outboxes[member];
        const std::size_t float_offset = fused::outbox_payload_offset(slot, kElements);
        if (fp16_wire) {
            for (int i = 0; i < kElements; ++i) {
                const float partial = load_float(inputs[member], fused::kGuardBytes +
                                                                 static_cast<std::size_t>(i) * sizeof(float));
                store_fp16(outbox, float_offset * sizeof(float) + static_cast<std::size_t>(i) * sizeof(std::uint16_t),
                           partial);
            }
        } else {
            std::copy_n(inputs[member].begin() + fused::kGuardBytes, kXBytes,
                        outbox.begin() + static_cast<std::ptrdiff_t>(float_offset * sizeof(float)));
        }
    }

    if (skew_cycles != 0) {
        const int delayed = static_cast<int>((seq - 1) % static_cast<unsigned int>(members));
        ProbeBuffers::Slot& slot_resources = buffers.slots[delayed];
        cudaStream_t stream = slot_resources.stream;
        const std::uint64_t cycles = skew_cycles;
        if (!log.cuda("set_device for deliberate skew " + std::to_string(devices[delayed].ordinal),
                      [&] { return cudaSetDevice(devices[delayed].ordinal); }, error) ||
            !log.cuda("launch short device-side skew on member " + std::to_string(delayed),
                      [&] {
                          skew_delay_kernel<<<1, 1, 0, stream>>>(cycles);
                          return cudaGetLastError();
                      }, error))
            return false;
    }

    std::vector<bool> launched(static_cast<std::size_t>(members), false);
    for (int order = 0; order < members; ++order) {
        const int member = static_cast<int>((seq + static_cast<unsigned int>(order)) % static_cast<unsigned int>(members));
        std::string launch_error;
        if (!log.check(queued ? "queued_fused_launch" : "fused_launch", seq, member, [&](std::string& check_error) {
                const bool ok = group.launch(member, x_payload(buffers.slots[member]), seq,
                                             buffers.slots[member].stream, check_error);
                if (!ok) launch_error = check_error;
                return ok;
            })) {
            append_error(error, launch_error);
            return false;
        }
        launched[member] = true;
    }
    if (!last) return true;

    bool synchronized = true;
    for (int member = 0; member < members; ++member) {
        if (!launched[member]) continue;
        std::string sync_error;
        if (!log.check(queued ? "queued_final_synchronize" : "fused_synchronize", seq, member,
                       [&](std::string& check_error) {
                           fused::DeviceError device_error;
                           const bool ok = group.synchronize(member, buffers.slots[member].stream, device_error, check_error);
                           if (!ok) sync_error = check_error;
                           return ok;
                       })) {
            if (queued) append_error(error, sync_error);
            synchronized = false;
        }
    }
    if (!synchronized) {
        if (queued) append_error(error, "fused AR queued device error: reject every member output");
        else error = "fused AR device error: reject every member output for this sequence";
        return false;
    }

    for (int member = 0; member < members; ++member) {
        if (!log.check(queued ? "queued_final_guarded_output_bitwise" : "guarded_output_bitwise", seq, member,
                       [&](std::string& check_error) {
                Bytes actual(kXAllocationBytes, 0);
                if (!download_x(buffers.slots[member], actual, log, check_error)) return false;
                verify::Mismatch mismatch;
                if (!verify::equal(actual, expected_x[member], mismatch)) {
                    check_error = mismatch_error(fp16_wire ? "FP16-wire output/guard bitwise" : "FP32 output/guard bitwise",
                                                 mismatch);
                    return false;
                }
                return true;
            }))
            return false;
        if (!log.check(queued ? "queued_final_outbox_slots_and_guards" : "all_outbox_slots_and_guards", seq, member,
                       [&](std::string& check_error) {
                Bytes actual(group.outbox_bytes(), 0);
                if (!log.cuda("set_device for outbox verification " + std::to_string(devices[member].ordinal),
                              [&] { return cudaSetDevice(devices[member].ordinal); }, check_error) ||
                    !log.cuda("download outbox for verification " + std::to_string(devices[member].ordinal),
                              [&] {
                                  return cudaMemcpy(actual.data(), group.outbox_raw(member), actual.size(),
                                                    cudaMemcpyDeviceToHost);
                              }, check_error))
                    return false;
                verify::Mismatch mismatch;
                if (!verify::equal(actual, expected_outboxes[member], mismatch)) {
                    check_error = mismatch_error("outbox payload/guard", mismatch);
                    return false;
                }
                return true;
            }))
            return false;
    }
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

template <typename T>
bool parse_unsigned(const std::string& text, T& value) {
    if (text.empty()) return false;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc() && result.ptr == text.data() + text.size();
}

void usage(std::ostream& out) {
    out << "Usage: fused_ar_probe (--device ORDINAL=0000:BB:DD.F){2|4} [--slots N] [--sequences N] "
           "[--skew-cycles N] [--timeout-cycles N] [--fp16-wire] [--queued] [--json PATH]\n"
           "Two or four explicit devices select TP2/TP4. Logs go to stderr and final JSON to stdout.\n"
           "--fp16-wire clamps and rounds every partial and the final sum to FP16 while keeping x as FP32.\n"
           "--queued requires --skew-cycles 0: upload/launch every sequence before the final synchronization; only the "
           "last output and final outbox state are checked, intermediate outputs are not read.\n"
           "Queued inputs use one pinned allocation made before the first launch; it is freed only after safe quiescence, "
           "otherwise it is retained and reported.\n"
           "This executable performs real GPU fused allreduce; host tests do not establish hardware correctness.\n";
}

bool parse_arguments(int argc, char** argv, Report& report, int& slots, int& sequences,
                     std::string& json_path, bool& help, std::string& error) {
    help = false;
    slots = 2;
    sequences = 0;
    report.skew_cycles = static_cast<std::uint64_t>(1) << 20;
    report.timeout_cycles = static_cast<std::uint64_t>(1) << 34;
    report.fp16_wire = false;
    report.queue_mode = false;
    report.queue_depth = 0;
    report.intermediate_outputs_checked = true;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            help = true;
            return true;
        }
        if (argument == "--fp16-wire") {
            report.fp16_wire = true;
            continue;
        }
        if (argument == "--queued") {
            report.queue_mode = true;
            continue;
        }
        if (argument == "--device") {
            if (++i >= argc) return error = "--device requires ORDINAL=BDF", false;
            const std::string specification = argv[i];
            const std::size_t equals = specification.find('=');
            DeviceSpec device;
            int ordinal = -1;
            if (equals == std::string::npos || !parse_unsigned(specification.substr(0, equals), ordinal) || ordinal < 0)
                return error = "invalid --device specification: " + specification, false;
            device.ordinal = ordinal;
            device.bdf = normalize_bdf(specification.substr(equals + 1));
            if (device.bdf.empty()) return error = "BDF must use 0000:BB:DD.F form", false;
            for (const DeviceSpec& existing : report.devices)
                if (existing.ordinal == device.ordinal || existing.bdf == device.bdf)
                    return error = "duplicate device ordinal or BDF", false;
            report.devices.push_back(device);
            continue;
        }
        if (argument == "--json" || argument == "--slots" || argument == "--sequences" ||
            argument == "--skew-cycles" || argument == "--timeout-cycles") {
            if (++i >= argc) return error = argument + " requires a value", false;
            const std::string value = argv[i];
            if (argument == "--json") json_path = value;
            else if (argument == "--slots" && !parse_unsigned(value, slots)) return error = "invalid slot count", false;
            else if (argument == "--sequences" && !parse_unsigned(value, sequences)) return error = "invalid sequence count", false;
            else if (argument == "--skew-cycles" && !parse_unsigned(value, report.skew_cycles))
                return error = "invalid skew cycle count", false;
            else if (argument == "--timeout-cycles" && !parse_unsigned(value, report.timeout_cycles))
                return error = "invalid timeout cycle count", false;
            continue;
        }
        return error = "unknown argument: " + argument, false;
    }
    if (report.devices.size() != 2 && report.devices.size() != 4)
        return error = "exactly two or four explicit --device mappings are required", false;
    if (slots <= 0 || slots > 16) return error = "slots must be in [1,16]", false;
    if (sequences == 0) sequences = 3 * slots + 2;
    if (sequences <= 0 || sequences > static_cast<int>(kMaxSequences))
        return error = "sequences must be in [1,256]", false;
    if (report.timeout_cycles == 0) return error = "timeout cycles must be nonzero", false;
    if (report.skew_cycles > (static_cast<std::uint64_t>(1) << 24))
        return error = "skew cycles above 2^24 are refused as a stress test", false;
    if (!validate_queue_arguments(report.queue_mode, report.skew_cycles, error)) return false;
    const QueuePolicy queue = make_queue_policy(report.queue_mode, sequences);
    report.queue_mode = queue.queue_mode;
    report.queue_depth = queue.queue_depth;
    report.intermediate_outputs_checked = queue.intermediate_outputs_checked;
    report.slots = slots;
    report.requested_sequences = sequences;
    return true;
}

bool check_mappings(Report& report, ProbeLog& log, std::string& error) {
    for (std::size_t i = 0; i < report.devices.size(); ++i) {
        DeviceSpec& device = report.devices[i];
        if (!log.check("manual_bdf_mapping", 0, static_cast<int>(i), [&](std::string& check_error) {
                char buffer[64] = {};
                if (!log.cuda("query fused probe BDF " + std::to_string(device.ordinal),
                              [&] { return cudaDeviceGetPCIBusId(buffer, sizeof(buffer), device.ordinal); }, check_error))
                    return false;
                device.actual_bdf = normalize_bdf(buffer);
                device.mapping_ok = !device.actual_bdf.empty() && device.actual_bdf == device.bdf;
                if (!device.mapping_ok) {
                    check_error = "manual BDF mismatch: supplied " + device.bdf + ", runtime reports " + buffer;
                    return false;
                }
                return true;
            })) {
            error = "explicit fused device/BDF mapping failed";
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
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char* digits = "0123456789abcdef";
                    out << "\\u00" << digits[c >> 4] << digits[c & 15];
                } else out << static_cast<char>(c);
        }
    }
    return out.str();
}

const char* boolean(bool value) { return value ? "true" : "false"; }

std::string report_json(const Report& report, bool success) {
    std::ostringstream out;
    out << "{\n  \"schema\": \"ds4-fused-ar-probe-v1\",\n  \"success\": " << boolean(success) << ",\n";
    out << "  \"members\": " << report.devices.size() << ", \"elements_per_member\": " << kElements
        << ", \"slots\": " << report.slots << ", \"requested_sequences\": " << report.requested_sequences
        << ", \"timeout_cycles\": " << report.timeout_cycles << ", \"skew_cycles\": " << report.skew_cycles
        << ", \"fp16_wire\": " << boolean(report.fp16_wire) << ",\n";
    if (report.queue_mode) {
        out << "  \"queue_mode\": true, \"queue_depth\": " << report.queue_depth
            << ", \"intermediate_outputs_checked\": " << boolean(report.intermediate_outputs_checked) << ",\n";
        out << "  \"pinned_input_bytes\": " << report.pinned_input_bytes << ", \"pinned_inputs_retained\": "
            << boolean(report.pinned_inputs_retained) << ", \"pinned_input_status\": \""
            << json_escape(report.pinned_input_status) << "\",\n";
    }
    out << "  \"devices\": [";
    for (std::size_t i = 0; i < report.devices.size(); ++i) {
        const DeviceSpec& device = report.devices[i];
        out << (i ? "," : "") << "\n    {\"member\": " << i << ", \"ordinal\": " << device.ordinal
            << ", \"bdf\": \"" << json_escape(device.bdf) << "\", \"actual_bdf\": \""
            << json_escape(device.actual_bdf) << "\", \"mapping_ok\": " << boolean(device.mapping_ok) << "}";
    }
    out << "\n  ],\n  \"sequences\": [";
    for (std::size_t i = 0; i < report.sequences.size(); ++i)
        out << (i ? "," : "") << "\n    {\"seq\": " << report.sequences[i].seq << ", \"ok\": "
            << boolean(report.sequences[i].ok) << "}";
    out << "\n  ],\n  \"checks\": [";
    for (std::size_t i = 0; i < report.checks.size(); ++i) {
        const Check& check = report.checks[i];
        out << (i ? "," : "") << "\n    {\"name\": \"" << json_escape(check.name) << "\", \"seq\": " << check.seq
            << ", \"member\": " << check.member << ", \"ok\": " << boolean(check.ok) << ", \"error\": \""
            << json_escape(check.error) << "\"}";
    }
    out << "\n  ],\n  \"error\": \"" << json_escape(report.error) << "\"\n}\n";
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    Report report;
    std::string json_path, error;
    int slots = 0, sequences = 0;
    bool help = false;
    if (!parse_arguments(argc, argv, report, slots, sequences, json_path, help, error)) {
        std::fprintf(stderr, "FUSED_AR_PROBE argument error: %s\n", error.c_str());
        usage(std::cerr);
        return 2;
    }
    if (help) {
        usage(std::cout);
        return 0;
    }
    ProbeLog log(report);
    if (check_mappings(report, log, error)) {
        std::vector<int> ordinals;
        for (const DeviceSpec& device : report.devices) ordinals.push_back(device.ordinal);
        PeerAccess peers(ordinals);
        std::string peer_error;
        if (!peers.enable_all(peer_error)) {
            append_error(error, peer_error);
        } else {
            fused::Config config;
            config.n = kElements;
            config.slots = slots;
            config.timeout_cycles = report.timeout_cycles;
            config.fp16_wire = report.fp16_wire;
            fused::Group group(peers, config);
            std::string group_error;
            if (!group.initialize(group_error)) {
                append_error(error, group_error);
            } else {
                ProbeBuffers buffers(report.devices, log);
                QueuedInputs queued_inputs(report, log);
                std::string buffer_error;
                if (!buffers.initialize(buffer_error)) {
                    append_error(error, buffer_error);
                } else {
                    std::vector<Bytes> expected_outboxes(static_cast<std::size_t>(group.members()));
                    for (int member = 0; member < group.members(); ++member) {
                        Bytes& image = expected_outboxes[member];
                        image.resize(group.outbox_bytes());
                        for (std::size_t i = 0; i < image.size(); ++i)
                            image[i] = verify::guard_byte(i, 0x51ed270bu ^ static_cast<unsigned int>(member));
                    }
                    if (!upload_outboxes(group, report.devices, expected_outboxes, log, error)) {
                        append_error(error, "outbox initialization failed");
                    } else {
                        const bool inputs_ready = !report.queue_mode || queued_inputs.initialize(group.members(), sequences, error);
                        if (inputs_ready) {
                            for (int sequence = 1; sequence <= sequences; ++sequence) {
                                const unsigned int seq = static_cast<unsigned int>(sequence);
                                std::string sequence_error;
                                QueuedInputs* input_storage = report.queue_mode ? &queued_inputs : nullptr;
                                const bool ok = run_sequence(group, buffers, report.devices, seq, slots, report.skew_cycles,
                                                             report.fp16_wire, expected_outboxes, log, sequence_error, input_storage,
                                                             static_cast<unsigned int>(sequences));
                                if (!report.queue_mode || sequence == sequences)
                                    report.sequences.push_back(SequenceResult{seq, ok});
                                append_error(error, sequence_error);
                                if (!ok) break;
                            }
                        }
                    }
                }
                std::string close_error;
                const bool buffers_closed = buffers.close(close_error);
                if (!buffers_closed) append_error(error, close_error);
                (void) queued_inputs.close(report.queue_mode && error.empty() && buffers_closed, error);
            }
            std::string group_close_error;
            if (!group.close(group_close_error)) append_error(error, group_close_error);
        }
        std::string disable_error;
        if (!peers.disable_all(disable_error)) append_error(error, disable_error);
    }

    const bool all_sequences = report.queue_mode
                                   ? report.sequences.size() == 1 &&
                                         report.sequences.front().seq == static_cast<unsigned int>(sequences) &&
                                         report.sequences.front().ok
                                   : report.sequences.size() == static_cast<std::size_t>(sequences) &&
                                         std::all_of(report.sequences.begin(), report.sequences.end(),
                                                     [](const SequenceResult& sequence) { return sequence.ok; });
    if (!all_sequences)
        append_error(error, report.queue_mode ? "queued success requires the final TP sequence and its output checks to pass"
                                             : "success requires every requested TP sequence to pass");
    bool success = error.empty() && all_sequences;
    report.error = error;
    std::string json = report_json(report, success);
    if (!json_path.empty()) {
        std::ofstream output(json_path, std::ios::binary | std::ios::trunc);
        if (!output || !(output << json) || !output.flush()) {
            report.error = "could not write JSON report: " + json_path;
            success = false;
            json = report_json(report, false);
        }
    }
    std::cout << json;
    std::cout.flush();
    if (success && report.queue_mode)
        mark(report.fp16_wire
                 ? "SUCCESS: queued FP16-wire final output and guards passed; intermediate outputs not checked"
                 : "SUCCESS: queued FP32 final output and guards passed; intermediate outputs not checked");
    else
        mark(success ? (report.fp16_wire
                            ? "SUCCESS: fused FP16-wire TP ring, skew, bitwise outputs and guards passed"
                            : "SUCCESS: fused FP32 TP ring, skew, bitwise outputs and guards passed")
                     : "FAILURE: fused AR hardware acceptance not established");
    return success ? 0 : 1;
}
#endif  // FUSED_AR_PROBE_LOGIC_ONLY
