#ifndef SPARSE_SIM_IDM_MODEL_GROUP_HPP
#define SPARSE_SIM_IDM_MODEL_GROUP_HPP

#include <memory>
#include <vector>

#include "idm_model.hpp"
#include "stats.hpp"

namespace sparse_sim {

class IdmModelGroup {
public:
    explicit IdmModelGroup(std::vector<std::unique_ptr<IdmModel>> models);

    std::vector<data_t> idm_ld(const std::vector<idx_t>& indices, KernelStats& stats);
    bool idm_st(const std::vector<idx_t>& indices,
                const std::vector<data_t>& values,
                KernelStats& stats);
    bool idm_init(const std::vector<idx_t>& indices,
                  const std::vector<data_t>& values,
                  KernelStats& stats);
    IdmEvictResult idm_evict_simd(KernelStats& stats);
    IdmEvictResult idm_evict(KernelStats& stats);
    void idm_rst(KernelStats& stats);
    std::uint64_t idm_val(KernelStats& stats);
    void sample_row_utilization(KernelStats& stats) const;

private:
    float primary_utilization() const;

    std::vector<std::unique_ptr<IdmModel>> models_;
};

}  // namespace sparse_sim

#endif  // SPARSE_SIM_IDM_MODEL_GROUP_HPP
