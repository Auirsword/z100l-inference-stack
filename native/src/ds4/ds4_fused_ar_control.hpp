#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace strata::ds4::fused_ar_control {

enum class WindowSource { kPrompt, kStep, kVerify, kMulti };

inline int parse_mode(const char* value) {
    if (value == nullptr) return 0;
    if (value[0] >= '0' && value[0] <= '2' && value[1] == 0) return value[0] - '0';
    return -1;
}

inline int parse_init_mask(const char* value) {
    if (value == nullptr) return 7;
    if (value[0] >= '0' && value[0] <= '7' && value[1] == 0) return value[0] - '0';
    return -1;
}

inline int parse_group_stage(const char* value) {
    if (value == nullptr) return 4;
    if (value[0] >= '0' && value[0] <= '4' && value[1] == 0) return value[0] - '0';
    return -1;
}

inline int parse_test_mode(const char* value) {
    if (value == nullptr || value[0] == 0) return 0;
    if (std::string(value) == "drop-member-1") return 1;
    return -1;
}

inline bool window_enabled(int mode, WindowSource source, int T) {
    return (mode == 1 || mode == 2) && source == WindowSource::kStep && T == 1;
}

inline bool group_enabled(int mode, std::size_t members) {
    return (mode == 1 || mode == 2) && (members == 2 || members == 4);
}

inline bool cuda_pdl_allowed(int mode, const char* pdl) {
    return mode == 0 || (pdl != nullptr && pdl[0] == '0' && pdl[1] == 0);
}

inline bool physical_devices(const std::vector<int>& members, const std::vector<int>& device_ids,
                             std::vector<int>& mapped) {
    mapped.clear();
    for (int member : members) {
        if (member < 0 || static_cast<std::size_t>(member) >= device_ids.size()) {
            mapped.clear();
            return false;
        }
        mapped.push_back(device_ids[static_cast<std::size_t>(member)]);
    }
    return true;
}

inline unsigned next_sequence(unsigned& sequence) { return ++sequence; }

inline void append_error(std::string& error, const std::string& next) {
    if (next.empty()) return;
    if (!error.empty()) error += "; ";
    error += next;
}

class ErrorState {
public:
    void fail(const std::string& error) {
        std::lock_guard<std::mutex> lock(mu);
        append_error(error_, error);
    }

    std::string error() const {
        std::lock_guard<std::mutex> lock(mu);
        return error_;
    }

private:
    mutable std::mutex mu;
    std::string error_;
};

}  // namespace strata::ds4::fused_ar_control
