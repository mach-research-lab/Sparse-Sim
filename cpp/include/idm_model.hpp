#ifndef SPARSE_SIM_IDM_MODEL_HPP
#define SPARSE_SIM_IDM_MODEL_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace sparse_sim {

struct IdmResult {
    std::uint64_t cycles = 0;
    std::uint64_t checks = 0;
    bool overflow = false;
};

struct IdmLoadResult : IdmResult {
    std::vector<data_t> values;
};

struct IdmStoreResult : IdmResult {};
struct IdmInitResult : IdmResult {};
struct IdmResetResult : IdmResult {};

struct IdmEvictResult : IdmResult {
    std::vector<idx_t> indices;
    std::vector<data_t> values;
};

struct IdmValResult : IdmResult {
    std::uint64_t valid_pairs = 0;
};

class IdmModel {
public:
    virtual ~IdmModel() = default;

    virtual const std::string& name() const = 0;
    virtual IdmLoadResult idm_ld(const std::vector<idx_t>& indices) = 0;
    virtual IdmStoreResult idm_st(const std::vector<idx_t>& indices,
                                  const std::vector<data_t>& values) = 0;
    virtual IdmInitResult idm_init(const std::vector<idx_t>& indices,
                                   const std::vector<data_t>& values) = 0;
    virtual IdmEvictResult idm_evict_simd() = 0;
    virtual IdmEvictResult idm_evict() = 0;
    virtual IdmResetResult idm_rst() = 0;
    virtual IdmValResult idm_val() const = 0;
    virtual float utilization() const = 0;
    virtual bool overflow() const = 0;
    virtual bool terminal_overflow() const { return false; }
};

}  // namespace sparse_sim

#endif  // SPARSE_SIM_IDM_MODEL_HPP
