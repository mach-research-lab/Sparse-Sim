#include "models.hpp"

#include <algorithm>

namespace sparse_sim {

AsaModel::AsaModel(std::string name)
    : name_(std::move(name)),
      values_(default_config().elements(), 0.0F),
      keys_(default_config().elements(), 0),
      valid_(default_config().elements(), 0) {}

const std::string& AsaModel::name() const {
    return name_;
}

IdmLoadResult AsaModel::idm_ld(const std::vector<idx_t>& indices) {
    IdmLoadResult result;
    result.values.resize(indices.size(), 0.0F);
    result.cycles = indices.size();

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto it = std::find(keys_.begin(), keys_.end(), static_cast<std::uint32_t>(indices[i]));
        const auto addr = static_cast<std::size_t>(std::distance(keys_.begin(), it));
        if (it != keys_.end() && valid_[addr]) {
            result.values[i] = values_[addr];
        }
    }

    return result;
}

IdmStoreResult AsaModel::idm_st(const std::vector<idx_t>& indices,
                                const std::vector<data_t>& values) {
    IdmStoreResult result;
    result.cycles = indices.size();

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto it = std::find(keys_.begin(), keys_.end(), static_cast<std::uint32_t>(indices[i]));
        auto addr = static_cast<std::size_t>(std::distance(keys_.begin(), it));

        if (it != keys_.end() && valid_[addr]) {
            values_[addr] = values[i];
        } else {
            auto free_it = std::find(valid_.begin(), valid_.end(), static_cast<std::uint8_t>(0));
            if (free_it == valid_.end()) {
                return {};
            }
            addr = static_cast<std::size_t>(std::distance(valid_.begin(), free_it));
            keys_[addr] = static_cast<std::uint32_t>(indices[i]);
            values_[addr] = values[i];
            valid_[addr] = 1;
        }
    }

    return result;
}

IdmInitResult AsaModel::idm_init(const std::vector<idx_t>& indices,
                                 const std::vector<data_t>& values) {
    auto result = idm_st(indices, values);
    return {IdmResult{result.cycles, result.checks, result.overflow}};
}

IdmEvictResult AsaModel::idm_evict_simd() {
    const auto target = std::min<std::size_t>(default_config().simd, idm_val().valid_pairs);
    IdmEvictResult result;
    result.cycles = target;

    for (std::size_t addr = 0; addr < valid_.size() && result.indices.size() < target; ++addr) {
        if (!valid_[addr]) {
            continue;
        }
        result.indices.push_back(static_cast<idx_t>(keys_[addr]));
        result.values.push_back(values_[addr]);
        valid_[addr] = 0;
    }

    return result;
}

IdmEvictResult AsaModel::idm_evict() {
    IdmEvictResult result;
    while (idm_val().valid_pairs > 0) {
        auto chunk = idm_evict_simd();
        result.cycles += chunk.cycles;
        result.indices.insert(result.indices.end(), chunk.indices.begin(), chunk.indices.end());
        result.values.insert(result.values.end(), chunk.values.begin(), chunk.values.end());
    }
    return result;
}

IdmResetResult AsaModel::idm_rst() {
    std::fill(valid_.begin(), valid_.end(), 0);
    return {};
}

IdmValResult AsaModel::idm_val() const {
    return {IdmResult{}, static_cast<std::uint64_t>(
        std::count(valid_.begin(), valid_.end(), static_cast<std::uint8_t>(1)))};
}

float AsaModel::utilization() const {
    return 0.0F;
}

bool AsaModel::overflow() const {
    return false;
}

}  // namespace sparse_sim
