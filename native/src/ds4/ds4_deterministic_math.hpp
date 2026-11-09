#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define DS4_DET_HD __host__ __device__
#else
#define DS4_DET_HD
#endif

namespace strata::ds4::gguf::native_canonical {

/// 0: unset or off, 1: on, -1: invalid (model loading must reject it).
inline int parse(const char* value) {
    if (value == nullptr || std::strcmp(value, "0") == 0) return 0;
    return std::strcmp(value, "1") == 0 ? 1 : -1;
}

}  // namespace strata::ds4::gguf::native_canonical

namespace strata::ds4::gguf::native_ordered {

/// 0: unset or off, 1: on, -1: invalid (model loading must reject it).
inline int parse(const char* value) {
    if (value == nullptr || std::strcmp(value, "0") == 0) return 0;
    return std::strcmp(value, "1") == 0 ? 1 : -1;
}

}  // namespace strata::ds4::gguf::native_ordered

namespace strata::ds4::gguf::deterministic {

inline bool enabled() {
    const char* value = std::getenv("DS4_DETERMINISTIC");
    return value != nullptr && std::atoi(value) != 0;
}

inline bool canonical_math(bool deterministic_enabled, const char* native_value) {
    return deterministic_enabled || native_canonical::parse(native_value) == 1;
}
inline bool canonical_math() { return canonical_math(enabled(), std::getenv("DS4_NATIVE_CANONICAL")); }

inline bool ordered_merge(bool deterministic_enabled, const char* native_value) {
    return deterministic_enabled || native_ordered::parse(native_value) == 1;
}
inline bool ordered_merge() { return ordered_merge(enabled(), std::getenv("DS4_NATIVE_ORDERED_COMBINE")); }

inline bool supported_type(int type) { return type == 16 || type == 18 || type == 21 || type == 22; }

DS4_DET_HD inline double add(double a, double b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    volatile double result = a + b;
    return result;
#endif
}

DS4_DET_HD inline double sub(double a, double b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    volatile double result = a - b;
    return result;
#endif
}

DS4_DET_HD inline double mul(double a, double b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    volatile double result = a * b;
    return result;
#endif
}

DS4_DET_HD inline double fma(double a, double b, double c) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fma_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

DS4_DET_HD inline float divide(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    volatile float result = a / b;
    return result;
#endif
}

DS4_DET_HD inline float exp_negative(float x) {
    const double y = -(double) x;
    if (y < -745.0) return 0.0f;
    if (y > 709.0) return __builtin_huge_valf();
    const double log2e = 1.44269504088896340736;
    const int n = (int) floor(fma(y, log2e, 0.5));
    double r = sub(y, mul((double) n, 0.693147180559945286227));
    r = sub(r, mul((double) n, 2.31904681384629956e-17));
    constexpr double coefficients[17] = {
        1.0,
        1.0,
        0.5,
        0.166666666666666666667,
        0.0416666666666666666667,
        0.00833333333333333333333,
        0.00138888888888888888889,
        0.000198412698412698412698,
        0.0000248015873015873015873,
        0.00000275573192239858906526,
        0.000000275573192239858906526,
        0.0000000250521083854417187751,
        0.00000000208767569878680989792,
        0.000000000160590438368216145994,
        0.0000000000114707455977297247139,
        0.00000000000076471637318198164759,
        0.0000000000000477947733238738529744,
    };
    double p = coefficients[16];
#pragma unroll
    for (int i = 15; i >= 0; --i) p = fma(p, r, coefficients[i]);
    const int64_t bits = (int64_t) (n + 1023) << 52;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return (float) mul(p, __longlong_as_double(bits));
#else
    double scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return (float) mul(p, scale);
#endif
}

DS4_DET_HD inline float swiglu(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = gate > limit ? limit : gate;
        up = up > limit ? limit : up < -limit ? -limit : up;
    }
    const float sigmoid = divide(gate, 1.0f + exp_negative(gate));
    volatile float product = sigmoid * up;
    return product;
}

}  // namespace strata::ds4::gguf::deterministic
