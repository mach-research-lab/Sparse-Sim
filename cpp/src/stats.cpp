#include "stats.hpp"

namespace sparse_sim {

KernelStats make_stats_with_models(const std::vector<std::string>& names) {
    KernelStats stats;
    for (const auto& name : names) {
        stats.model_stats[name] = ModelStats{};
    }
    for (const auto& instruction : {
             "idm_ld",
             "idm_st",
             "idm_init",
             "idm_evict_simd",
             "idm_evict",
             "idm_rst",
             "idm_val",
         }) {
        stats.instruction_count[instruction] = 0;
    }
    return stats;
}

}  // namespace sparse_sim
