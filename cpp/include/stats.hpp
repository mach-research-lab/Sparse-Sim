#ifndef SPARSE_SIM_STATS_HPP
#define SPARSE_SIM_STATS_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sparse_sim {

struct ModelStats {
    std::uint64_t cycles = 0;
    std::uint64_t checks = 0;
};

struct KernelStats {
    std::map<std::string, ModelStats> model_stats;
    std::map<std::string, std::uint64_t> instruction_count;
    std::uint64_t reads = 0;
    std::uint64_t zeroes_found = 0;
    std::uint64_t writes = 0;
    std::uint64_t pairs_initialized = 0;
    std::uint64_t pairs_evicted = 0;
    float avg_utilization = 0.0F;
    float max_utilization = 0.0F;
    float ovf_utilization = 0.0F;
    std::uint8_t overflow = 0;
};

using SpGemmStats = KernelStats;

KernelStats make_stats_with_models(const std::vector<std::string>& names);

}  // namespace sparse_sim

#endif  // SPARSE_SIM_STATS_HPP
