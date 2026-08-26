#include "models.hpp"

#include <algorithm>

namespace sparse_sim {

ArrayBackedModel::ArrayBackedModel(std::string name)
    : name_(std::move(name)),
      mem_array_(default_config().elements(), 0.0F),
      idx_array_(default_config().elements(), 0),
      valid_(default_config().elements(), 0) {}

IdmResetResult ArrayBackedModel::idm_rst() {
    std::fill(valid_.begin(), valid_.end(), 0);
    overflow_ = false;
    return {};
}

IdmValResult ArrayBackedModel::idm_val() const {
    return {IdmResult{}, valid_count()};
}

float ArrayBackedModel::utilization() const {
    const auto used = valid_count();
    return 100.0F * static_cast<float>(used) / static_cast<float>(valid_.size());
}

bool ArrayBackedModel::overflow() const {
    return overflow_;
}

std::uint64_t ArrayBackedModel::valid_count() const {
    return static_cast<std::uint64_t>(
        std::count(valid_.begin(), valid_.end(), static_cast<std::uint8_t>(1)));
}

IdmEvictResult ArrayBackedModel::evict_one_per_logical_lane(std::uint32_t lanes,
                                                            std::uint64_t cycles) {
    IdmEvictResult result;
    result.cycles = valid_count() == 0 ? 0 : cycles;

    std::vector<std::uint8_t> emitted(lanes, 0);
    for (std::size_t addr = 0; addr < valid_.size(); ++addr) {
        if (!valid_[addr]) {
            continue;
        }

        const auto lane = idx_array_[addr] & (lanes - 1);
        if (emitted[lane]) {
            continue;
        }

        emitted[lane] = 1;
        result.indices.push_back(static_cast<idx_t>(idx_array_[addr]));
        result.values.push_back(mem_array_[addr]);
        valid_[addr] = 0;
    }

    return result;
}

IdmEvictResult ArrayBackedModel::evict_all_with_cycles(std::uint64_t cycles_per_round) {
    IdmEvictResult result;
    while (valid_count() > 0) {
        auto chunk = evict_one_per_logical_lane(default_config().simd, cycles_per_round);
        result.cycles += chunk.cycles;
        result.indices.insert(result.indices.end(), chunk.indices.begin(), chunk.indices.end());
        result.values.insert(result.values.end(), chunk.values.begin(), chunk.values.end());
    }
    return result;
}

}  // namespace sparse_sim
