#ifndef SPARSE_SIM_MODELS_HPP
#define SPARSE_SIM_MODELS_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "config.hpp"
#include "h3.hpp"
#include "idm_model.hpp"

namespace sparse_sim {

class ArrayBackedModel : public IdmModel {
public:
    explicit ArrayBackedModel(std::string name);
    IdmResetResult idm_rst() override;
    IdmValResult idm_val() const override;
    float utilization() const override;
    bool overflow() const override;

protected:
    std::uint64_t valid_count() const;
    IdmEvictResult evict_all_with_cycles(std::uint64_t cycles_per_round);
    IdmEvictResult evict_one_per_logical_lane(std::uint32_t lanes, std::uint64_t cycles);

    std::string name_;
    std::vector<float> mem_array_;
    std::vector<std::uint32_t> idx_array_;
    std::vector<std::uint8_t> valid_;
    bool overflow_ = false;
};

class SimaModel final : public ArrayBackedModel {
public:
    SimaModel(std::string name, std::uint32_t issue_width, H3Matrix h3_matrix);
    const std::string& name() const override;
    IdmLoadResult idm_ld(const std::vector<idx_t>& indices) override;
    IdmStoreResult idm_st(const std::vector<idx_t>& indices,
                          const std::vector<data_t>& values) override;
    IdmInitResult idm_init(const std::vector<idx_t>& indices,
                           const std::vector<data_t>& values) override;
    IdmEvictResult idm_evict_simd() override;
    IdmEvictResult idm_evict() override;
    bool terminal_overflow() const override;

private:
    std::size_t hash(idx_t key) const;

    std::uint32_t issue_width_;
    H3Matrix h3_matrix_;
};

class ViaModel final : public ArrayBackedModel {
public:
    ViaModel(std::string name, std::uint32_t lanes);
    const std::string& name() const override;
    IdmLoadResult idm_ld(const std::vector<idx_t>& indices) override;
    IdmStoreResult idm_st(const std::vector<idx_t>& indices,
                          const std::vector<data_t>& values) override;
    IdmInitResult idm_init(const std::vector<idx_t>& indices,
                           const std::vector<data_t>& values) override;
    IdmEvictResult idm_evict_simd() override;
    IdmEvictResult idm_evict() override;

private:
    std::uint32_t lanes_;
};

class InnerSpModel final : public ArrayBackedModel {
public:
    InnerSpModel(std::string name, std::uint32_t banks, H3Matrix h3_matrix);
    const std::string& name() const override;
    IdmLoadResult idm_ld(const std::vector<idx_t>& indices) override;
    IdmStoreResult idm_st(const std::vector<idx_t>& indices,
                          const std::vector<data_t>& values) override;
    IdmInitResult idm_init(const std::vector<idx_t>& indices,
                           const std::vector<data_t>& values) override;
    IdmEvictResult idm_evict_simd() override;
    IdmEvictResult idm_evict() override;

private:
    std::size_t hash(idx_t key) const;

    std::uint32_t banks_;
    H3Matrix h3_matrix_;
};

class AsaModel final : public IdmModel {
public:
    explicit AsaModel(std::string name);
    const std::string& name() const override;
    IdmLoadResult idm_ld(const std::vector<idx_t>& indices) override;
    IdmStoreResult idm_st(const std::vector<idx_t>& indices,
                          const std::vector<data_t>& values) override;
    IdmInitResult idm_init(const std::vector<idx_t>& indices,
                           const std::vector<data_t>& values) override;
    IdmEvictResult idm_evict_simd() override;
    IdmEvictResult idm_evict() override;
    IdmResetResult idm_rst() override;
    IdmValResult idm_val() const override;
    float utilization() const override;
    bool overflow() const override;

private:
    std::string name_;
    std::vector<float> values_;
    std::vector<std::uint32_t> keys_;
    std::vector<std::uint8_t> valid_;
};

}  // namespace sparse_sim

#endif  // SPARSE_SIM_MODELS_HPP
