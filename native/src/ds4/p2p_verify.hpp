#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::ds4::p2p::verify {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DS4_P2P_HOST_DEVICE __host__ __device__
#else
#define DS4_P2P_HOST_DEVICE
#endif

DS4_P2P_HOST_DEVICE inline std::uint8_t pattern_byte(std::size_t index, std::uint32_t seed) {
    std::uint32_t x = static_cast<std::uint32_t>(index) * 2654435761u + seed * 2246822519u;
    x ^= x >> 15;
    x *= 3266489917u;
    x ^= x >> 16;
    return static_cast<std::uint8_t>(x >> 8);
}

DS4_P2P_HOST_DEVICE inline std::uint8_t transformed_byte(std::uint8_t value) {
    return static_cast<std::uint8_t>(value ^ 0x5au);
}

#undef DS4_P2P_HOST_DEVICE

inline std::uint8_t guard_byte(std::size_t index, std::uint32_t seed) {
    return static_cast<std::uint8_t>(pattern_byte(index, seed ^ 0x9e3779b9u) ^ 0xc3u);
}

inline void fill_payload(std::vector<std::uint8_t>& image, std::size_t offset, std::size_t bytes,
                         std::uint32_t seed) {
    for (std::size_t i = 0; i < bytes; ++i) image[offset + i] = pattern_byte(i, seed);
}

inline bool initialize_region(std::vector<std::uint8_t>& image, std::size_t payload_offset,
                              std::size_t payload_bytes, std::size_t trailing_guard,
                              std::uint32_t seed) {
    if (payload_offset > image.size() || payload_bytes > image.size() - payload_offset) return false;
    const std::size_t end = payload_offset + payload_bytes;
    if (trailing_guard > image.size() - end) return false;
    for (std::size_t i = 0; i < payload_offset; ++i) image[i] = guard_byte(i, seed);
    fill_payload(image, payload_offset, payload_bytes, seed);
    for (std::size_t i = 0; i < trailing_guard; ++i) image[end + i] = guard_byte(i, seed ^ 0x7f4a7c15u);
    return true;
}

inline std::vector<std::uint8_t> make_region(std::size_t leading_guard, std::size_t payload_bytes,
                                             std::size_t trailing_guard, std::uint32_t seed) {
    std::vector<std::uint8_t> image;
    const std::size_t maximum = image.max_size();
    if (payload_bytes > maximum || leading_guard > maximum - payload_bytes) return image;
    const std::size_t end = leading_guard + payload_bytes;
    if (trailing_guard > maximum - end) return image;
    image.resize(end + trailing_guard);
    if (!initialize_region(image, leading_guard, payload_bytes, trailing_guard, seed)) image.clear();
    return image;
}

struct Mismatch {
    std::size_t offset = 0;
    std::uint8_t expected = 0;
    std::uint8_t actual = 0;
};

inline bool equal(const std::vector<std::uint8_t>& actual, const std::vector<std::uint8_t>& expected,
                  Mismatch& mismatch) {
    if (actual.size() != expected.size()) {
        mismatch.offset = actual.size() < expected.size() ? actual.size() : expected.size();
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            mismatch = Mismatch{i, expected[i], actual[i]};
            return false;
        }
    }
    return true;
}

}  // namespace strata::ds4::p2p::verify
