#include "models.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sparse_sim {

ViaModel::ViaModel(std::string name, std::uint32_t lanes)
    : ArrayBackedModel(std::move(name)), lanes_(lanes) {
    if (lanes_ == 0 || lanes_ > default_config().simd) {
        throw std::invalid_argument("VIA lanes must be in range 1..SIMD");
    }
}

const std::string& ViaModel::name() const {
    return name_;
}

IdmLoadResult ViaModel::idm_ld(const std::vector<idx_t>& indices) {
    const auto& config = default_config();
    IdmLoadResult result;
    result.values.resize(indices.size(), 0.0F);

    if (lanes_ == config.simd) {
        result.cycles = 1 + config.pipelined;
    } else {
        result.cycles = static_cast<std::uint64_t>(
            std::ceil(static_cast<float>(indices.size()) / static_cast<float>(lanes_))) +
            config.pipelined;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto it = std::find(idx_array_.begin(), idx_array_.end(),
                                  static_cast<std::uint32_t>(indices[i]));
        const auto addr = static_cast<std::size_t>(std::distance(idx_array_.begin(), it));
        if (it != idx_array_.end() && valid_[addr]) {
            result.values[i] = mem_array_[addr];
        }
    }

    return result;
}

IdmStoreResult ViaModel::idm_st(const std::vector<idx_t>& indices,
                                const std::vector<data_t>& values) {
    const auto& config = default_config();
    if (indices.size() != values.size()) {
        throw std::invalid_argument("VIA write index/value size mismatch");
    }

    IdmStoreResult result;
    if (lanes_ == config.simd) {
        result.cycles = 1 + config.pipelined;
    } else {
        result.cycles = static_cast<std::uint64_t>(
            std::ceil(static_cast<float>(indices.size()) / static_cast<float>(lanes_))) +
            config.pipelined;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto it = std::find(idx_array_.begin(), idx_array_.end(),
                            static_cast<std::uint32_t>(indices[i]));
        auto addr = static_cast<std::size_t>(std::distance(idx_array_.begin(), it));

        if (it != idx_array_.end() && valid_[addr]) {
            mem_array_[addr] = values[i];
        } else {
            auto free_it = std::find(valid_.begin(), valid_.end(), static_cast<std::uint8_t>(0));
            if (free_it == valid_.end()) {
                overflow_ = true;
                return {IdmResult{0, 0, true}};
            }
            addr = static_cast<std::size_t>(std::distance(valid_.begin(), free_it));
            idx_array_[addr] = static_cast<std::uint32_t>(indices[i]);
            mem_array_[addr] = values[i];
            valid_[addr] = 1;
        }
    }

    return result;
}

IdmInitResult ViaModel::idm_init(const std::vector<idx_t>& indices,
                                 const std::vector<data_t>& values) {
    if (indices.size() != values.size()) {
        throw std::invalid_argument("VIA init index/value size mismatch");
    }

    std::uint64_t cycles = 0;
    for (std::size_t offset = 0; offset < indices.size(); offset += lanes_) {
        const auto chunk = std::min<std::size_t>(lanes_, indices.size() - offset);
        std::vector<idx_t> idx_chunk(indices.begin() + offset, indices.begin() + offset + chunk);
        std::vector<data_t> val_chunk(values.begin() + offset, values.begin() + offset + chunk);
        auto result = idm_st(idx_chunk, val_chunk);
        cycles += result.cycles;
        if (result.overflow) {
            return {IdmResult{cycles, result.checks, true}};
        }
    }

    return {IdmResult{cycles, 0, false}};
}

IdmEvictResult ViaModel::idm_evict_simd() {
    IdmEvictResult result;
    const auto target = std::min<std::uint64_t>(lanes_, valid_count());
    result.cycles = target == 0 ? 0 : static_cast<std::uint64_t>(
        std::ceil(static_cast<float>(target) / static_cast<float>(lanes_)));

    for (std::size_t addr = 0; addr < valid_.size() && result.indices.size() < target; ++addr) {
        if (!valid_[addr]) {
            continue;
        }
        result.indices.push_back(static_cast<idx_t>(idx_array_[addr]));
        result.values.push_back(mem_array_[addr]);
        valid_[addr] = 0;
    }

    return result;
}

IdmEvictResult ViaModel::idm_evict() {
    IdmEvictResult result;
    while (valid_count() > 0) {
        auto chunk = idm_evict_simd();
        result.cycles += chunk.cycles;
        result.indices.insert(result.indices.end(), chunk.indices.begin(), chunk.indices.end());
        result.values.insert(result.values.end(), chunk.values.begin(), chunk.values.end());
    }
    return result;
}

}  // namespace sparse_sim
