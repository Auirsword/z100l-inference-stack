// AR payload accounting: how many all-reduces the prompt path issues and how big each one is.
// Enabled by DS4_AR_SITES=1; dumped (and reset) by DS4_PREFILL_STATS at the end of a prefill.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int kBuckets = 40;
std::atomic<unsigned long long> g_calls[2][kBuckets];
std::atomic<unsigned long long> g_bytes[2][kBuckets];

bool on() {
    static const bool v = std::getenv("DS4_AR_SITES") != nullptr && std::atoi(std::getenv("DS4_AR_SITES")) != 0;
    return v;
}

int bucket(size_t bytes) {
    int b = 0;
    size_t v = bytes;
    while (v > 1 && b < kBuckets - 1) { v >>= 1; ++b; }
    return b;
}
}  // namespace

extern "C" void ds4_ar_note(size_t bytes, int kind) {
    if (!on()) return;
    const int k = (kind == 0 ? 0 : 1);
    const int b = bucket(bytes);
    g_calls[k][b].fetch_add(1, std::memory_order_relaxed);
    g_bytes[k][b].fetch_add(bytes, std::memory_order_relaxed);
}

extern "C" void ds4_ar_dump() {
    if (!on()) return;
    static const char* name[2] = {"kernel path", "DMA path"};
    for (int k = 0; k < 2; ++k) {
        unsigned long long tot_c = 0, tot_b = 0;
        for (int b = 0; b < kBuckets; ++b) {
            tot_c += g_calls[k][b].load();
            tot_b += g_bytes[k][b].load();
        }
        if (tot_c == 0) continue;
        std::fprintf(stderr, "DS4_AR_SITES %s: %llu calls, %.2f GB total\n", name[k], tot_c, (double) tot_b / 1e9);
        for (int b = 0; b < kBuckets; ++b) {
            const unsigned long long c = g_calls[k][b].exchange(0);
            const unsigned long long by = g_bytes[k][b].exchange(0);
            if (c) std::fprintf(stderr, "  %s ~%llu KB: %llu calls, %.2f GB\n", name[k],
                                (unsigned long long) ((size_t) 1 << (b > 10 ? b - 10 : 0)), c, (double) by / 1e9);
        }
    }
}

extern "C" int ds4_ar_pieces() {
    static const int v = [] {
        const char* e = std::getenv("DS4_AR_PIECES");
        int n = e != nullptr ? std::atoi(e) : 4;
        if (n < 1) n = 1;
        if (n > 8) n = 8;
        return n;
    }();
    return v;
}
