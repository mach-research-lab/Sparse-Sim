#include "models.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sparse_sim {

InnerSpModel::InnerSpModel(std::string name, std::uint32_t banks, H3Matrix h3_matrix)
    : ArrayBackedModel(std::move(name)), banks_(banks), h3_matrix_(std::move(h3_matrix)) {
    if (banks_ == 0 || default_config().elements() % banks_ != 0) {
        throw std::invalid_argument("InnerSP banks must evenly divide memory elements");
    }
}

std::size_t InnerSpModel::hash(idx_t key) const {
    return h3_hash(static_cast<std::uint64_t>(key), h3_matrix_);
}

const std::string& InnerSpModel::name() const {
    return name_;
}

IdmLoadResult InnerSpModel::idm_ld(const std::vector<idx_t>& indices) {
    const auto& config = default_config();
    const auto bank_elems = config.elements() / banks_;
    const auto bank_mask = banks_ - 1;
    const auto bank_shift = log2_floor(banks_);

    std::vector<std::uint32_t> port_queue(config.simd, 0);
    std::vector<std::uint32_t> bank_queue(banks_, 0);
    std::vector<std::size_t> hashes(indices.size());

    std::transform(indices.begin(), indices.end(), hashes.begin(), [this](idx_t key) { return hash(key); });
    for (const auto hash : hashes) {
        port_queue[config.port_mask() & hash]++;
    }

    IdmLoadResult result;
    result.values.resize(indices.size(), 0.0F);

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & bank_mask;
        bool found = false;

        for (std::size_t j = 0; j < bank_elems; ++j) {
            bank_queue[base_mem_addr]++;
            const auto addr = base_mem_addr + (j << bank_shift);

            if (!valid_[addr]) {
                break;
            }
            if (valid_[addr] && idx_array_[addr] == static_cast<std::uint32_t>(indices[i])) {
                result.values[i] = mem_array_[addr];
                found = true;
                break;
            }
        }

        if (!found) {
            result.values[i] = 0.0F;
        }
    }

    const auto collisions = *std::max_element(port_queue.begin(), port_queue.end());
    const auto serial = *std::max_element(bank_queue.begin(), bank_queue.end());
    result.cycles = std::max(collisions, serial);
    result.checks = std::accumulate(bank_queue.begin(), bank_queue.end(), std::uint64_t{0});
    return result;
}

IdmStoreResult InnerSpModel::idm_st(const std::vector<idx_t>& indices,
                                    const std::vector<data_t>& values) {
    const auto& config = default_config();
    if (indices.size() != values.size()) {
        throw std::invalid_argument("InnerSP write index/value size mismatch");
    }

    const auto bank_elems = config.elements() / banks_;
    const auto bank_mask = banks_ - 1;
    const auto bank_shift = log2_floor(banks_);

    std::vector<std::uint32_t> port_queue(config.simd, 0);
    std::vector<std::uint32_t> bank_queue(banks_, 0);
    std::vector<std::size_t> hashes(indices.size());

    std::transform(indices.begin(), indices.end(), hashes.begin(), [this](idx_t key) { return hash(key); });
    for (const auto hash : hashes) {
        port_queue[config.port_mask() & hash]++;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & bank_mask;
        bool stored = false;

        for (std::size_t j = 0; j < bank_elems; ++j) {
            bank_queue[base_mem_addr]++;
            const auto addr = base_mem_addr + (j << bank_shift);

            if ((valid_[addr] && idx_array_[addr] == static_cast<std::uint32_t>(indices[i])) ||
                !valid_[addr]) {
                idx_array_[addr] = static_cast<std::uint32_t>(indices[i]);
                mem_array_[addr] = values[i];
                valid_[addr] = 1;
                stored = true;
                break;
            }
        }

        if (!stored) {
            overflow_ = true;
            return {IdmResult{0, 0, true}};
        }
    }

    const auto collisions = *std::max_element(port_queue.begin(), port_queue.end());
    const auto serial = *std::max_element(bank_queue.begin(), bank_queue.end());
    return {std::max(collisions, serial),
            std::accumulate(bank_queue.begin(), bank_queue.end(), std::uint64_t{0})};
}

IdmInitResult InnerSpModel::idm_init(const std::vector<idx_t>& indices,
                                     const std::vector<data_t>& values) {
    const auto& config = default_config();
    if (indices.size() != values.size()) {
        throw std::invalid_argument("InnerSP init index/value size mismatch");
    }

    const auto bank_elems = config.elements() / banks_;
    const auto bank_mask = banks_ - 1;
    const auto bank_shift = log2_floor(banks_);
    std::uint64_t cycles = 0;
    std::uint64_t checks = 0;

    for (std::size_t offset = 0; offset < indices.size(); offset += config.simd) {
        const auto chunk = std::min<std::size_t>(config.simd, indices.size() - offset);
        std::vector<std::uint32_t> first_level_queue(16, 0);
        std::vector<std::uint32_t> second_level_queue(16, 0);
        std::vector<std::size_t> hashes(chunk);

        std::transform(indices.begin() + offset, indices.begin() + offset + chunk,
                       hashes.begin(), [this](idx_t key) { return hash(key); });

        for (std::size_t i = 0; i < chunk; ++i) {
            const auto bank = static_cast<std::uint32_t>(hashes[i]) & bank_mask;
            first_level_queue[(bank >> 4U) & 0xFU]++;
            second_level_queue[bank & 0xFU]++;
        }

        const auto first_pressure = *std::max_element(first_level_queue.begin(), first_level_queue.end());
        const auto second_pressure = *std::max_element(second_level_queue.begin(), second_level_queue.end());
        cycles += std::max(first_pressure, second_pressure);

        for (std::size_t i = 0; i < chunk; ++i) {
            const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & bank_mask;
            bool stored = false;

            for (std::size_t j = 0; j < bank_elems; ++j) {
                checks++;
                const auto addr = base_mem_addr + (j << bank_shift);
                if (!valid_[addr]) {
                    idx_array_[addr] = static_cast<std::uint32_t>(indices[offset + i]);
                    mem_array_[addr] = values[offset + i];
                    valid_[addr] = 1;
                    stored = true;
                    break;
                }
            }

            if (!stored) {
                overflow_ = true;
                return {IdmResult{cycles, checks, true}};
            }
        }
    }

    return {IdmResult{cycles, checks, false}};
}

IdmEvictResult InnerSpModel::idm_evict_simd() {
    return evict_one_per_logical_lane(static_cast<std::uint32_t>(default_config().simd), 1);
}

IdmEvictResult InnerSpModel::idm_evict() {
    return evict_all_with_cycles(1);
}

}  // namespace sparse_sim
