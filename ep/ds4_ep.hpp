#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::ds4::ep {

enum class Tier { main_vram, owner_vram, cpu_cold };

struct Config {
    int ranks = 4;
    bool gguf = true;
    int layers = 40;
    int experts = 384;
    int top_k = 6;
    int tp = 1;
    int seqs = 1;
    int prefill_chunk = 0;
    bool dspark = false;
    bool vision = false;
    bool threads = false;
    bool sync = true;
    std::vector<int> gpus = {0, 1, 2, 3};
};

inline bool validate_ep4(const Config& c, std::string& error) {
    const bool shape = c.ranks == 4 && c.gguf && c.layers == 40 && c.experts == 384 && c.top_k == 6;
    const bool run = c.tp == 1 && c.seqs == 1 && c.prefill_chunk == 0 && !c.dspark && !c.vision && !c.threads &&
                     c.sync && c.gpus == std::vector<int>{0, 1, 2, 3};
    if (shape && run) {
        error.clear();
        return true;
    }
    error = "EP4 requires a GGUF 40-layer/384-expert/top6 model, --gpus 0,1,2,3, --tp 1, --parallel 1, "
            "--dspark none, --vision off, --prefill-chunk 0, DS4_SYNC=1 and no device threads";
    return false;
}

struct Layout {
    int experts = 0;
    int ranks = 0;

    Layout(int n, int r) : experts(n), ranks(r) {
        if (n <= 0 || r <= 0 || n % r) throw std::invalid_argument("EP experts must be divisible by ranks");
    }
    int owner(int expert) const {
        if (expert < 0 || expert >= experts) throw std::out_of_range("EP expert index");
        return expert / (experts / ranks);
    }
};

struct Pick {
    int token = 0;
    int pick = 0;
    int expert = 0;
    float weight = 0;
    Tier tier = Tier::cpu_cold;
};

struct Group {
    int owner = 0;
    int expert = 0;
    std::vector<size_t> picks;
};

struct Plan {
    std::vector<Pick> picks;
    std::vector<Group> groups;
    std::vector<size_t> cpu_picks;
};

inline int output_row(const Pick& pick, int pick_stride) {
    if (pick.token < 0 || pick.pick < 0 || pick.pick >= pick_stride)
        throw std::out_of_range("EP output row: token " + std::to_string(pick.token) + ", pick " +
                                std::to_string(pick.pick) + ", stride " + std::to_string(pick_stride));
    return pick.token * pick_stride + pick.pick;
}

// One expert can be selected by more than 16 token/pick rows. Each matrix call must stay within x3::kMaxMvCols.
template <class F>
void for_each_chunk(int rows, int width, F&& fn) {
    if (rows < 0 || width <= 0 || width > 16) throw std::invalid_argument("EP activation chunk shape");
    for (int first = 0; first < rows; first += width) fn(first, std::min(width, rows - first));
}

// Compact, unweighted owner rows -> a zeroed full global-row buffer for the main card's combine.  No route weights are
// accepted here: the owner cannot turn this into a weighted partial sum or pass compact rows off as the global layout.
inline void scatter_unweighted_rows(const float* compact, const int* global_rows, int nrows, int dim, float* full,
                                    int full_rows, int valid_rows) {
    if (compact == nullptr || global_rows == nullptr || full == nullptr || compact == full || nrows < 0 ||
        nrows > valid_rows || dim <= 0 || full_rows <= 0 || valid_rows <= 0 || valid_rows > full_rows)
        throw std::invalid_argument("EP host scatter shape");
    std::fill(full, full + (size_t) full_rows * dim, 0.0f);
    std::vector<unsigned char> seen((size_t) full_rows, 0);
    for (int i = 0; i < nrows; ++i) {
        const int row = global_rows[i];
        if (row < 0 || row >= valid_rows) throw std::out_of_range("EP host scatter row");
        if (seen[(size_t) row] != 0) throw std::out_of_range("EP duplicate host scatter row");
        seen[(size_t) row] = 1;
        std::copy(compact + (size_t) i * dim, compact + (size_t) (i + 1) * dim, full + (size_t) row * dim);
    }
}

// Input routes have pick_stride slots per token; Plan::picks, Group::picks and cpu_picks are dense indices.
// output_row deliberately maps the latter to the padded global-row layout used by owner results and combine.
template <typename MainHot, typename OwnerHot>
Plan plan_window(int tokens, int k, const int* experts, const float* weights, int pick_stride, const Layout& layout,
                 MainHot main_hot, OwnerHot owner_hot) {
    if (tokens <= 0 || k <= 0 || tokens > 16 || k > pick_stride || pick_stride > 8)
        throw std::invalid_argument("EP window shape");
    Plan result;
    const size_t count = (size_t) tokens * k;
    result.picks.reserve(count);
    std::vector<std::vector<std::vector<size_t>>> grouped(
        (size_t) layout.ranks, std::vector<std::vector<size_t>>((size_t) layout.experts));
    for (int token = 0; token < tokens; ++token) {
        for (int pick = 0; pick < k; ++pick) {
            const size_t index = (size_t) token * pick_stride + pick;
            const int expert = experts[index];
            const int owner = layout.owner(expert);
            const float weight = weights[index];
            if (!std::isfinite(weight)) throw std::invalid_argument("EP nonfinite route weight");
            Tier tier = Tier::cpu_cold;
            if (owner == 0 && main_hot(expert)) tier = Tier::main_vram;
            else if (owner_hot(owner, expert)) tier = Tier::owner_vram;
            const size_t dense = result.picks.size();
            result.picks.push_back({token, pick, expert, weight, tier});
            if (tier == Tier::cpu_cold) result.cpu_picks.push_back(dense);
            else grouped[(size_t) owner][(size_t) expert].push_back(dense);
        }
    }
    for (int owner = 0; owner < layout.ranks; ++owner) {
        for (int expert = 0; expert < layout.experts; ++expert) {
            auto& indices = grouped[(size_t) owner][(size_t) expert];
            if (!indices.empty()) result.groups.push_back({owner, expert, std::move(indices)});
        }
    }
    return result;
}

class Completion {
public:
    Completion(const Plan& plan, int ranks) : owner_done_((size_t) ranks, true), has_work_((size_t) ranks, false),
                                             cpu_done_(plan.cpu_picks.empty()) {
        if (ranks <= 0) throw std::invalid_argument("EP completion ranks");
        for (const Group& group : plan.groups) {
            owner_done_[(size_t) group.owner] = false;
            has_work_[(size_t) group.owner] = true;
        }
    }

    void complete_owner(int owner) {
        if (owner < 0 || (size_t) owner >= owner_done_.size() || !has_work_[(size_t) owner])
            throw std::out_of_range("EP completion owner");
        if (!failed_) owner_done_[(size_t) owner] = true;
    }
    void complete_cpu() { if (!failed_) cpu_done_ = true; }
    void fail(std::string error) {
        failed_ = true;
        error_ = std::move(error);
    }
    bool can_combine() const {
        return !failed_ && cpu_done_ && std::all_of(owner_done_.begin(), owner_done_.end(), [](bool done) { return done; });
    }
    const std::string& error() const { return error_; }

private:
    std::vector<bool> owner_done_;
    std::vector<bool> has_work_;
    bool cpu_done_ = false;
    bool failed_ = false;
    std::string error_;
};

}  // namespace strata::ds4::ep
