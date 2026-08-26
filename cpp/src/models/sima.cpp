#include "models.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace sparse_sim {

SimaModel::SimaModel(std::string name, std::uint32_t issue_width, H3Matrix h3_matrix)
    : ArrayBackedModel(std::move(name)), issue_width_(issue_width), h3_matrix_(std::move(h3_matrix)) {
    if (issue_width_ == 0) {
        throw std::invalid_argument("SIMA issue width must be non-zero");
    }
}

std::size_t SimaModel::hash(idx_t key) const {
    return h3_hash(static_cast<std::uint64_t>(key), h3_matrix_);
}

const std::string& SimaModel::name() const {
    return name_;
}

IdmLoadResult SimaModel::idm_ld(const std::vector<idx_t>& indices) {
    const auto& config = default_config();
    std::vector<std::uint32_t> port_queue(config.simd, 0);
    std::vector<std::size_t> hashes(indices.size());

    std::transform(indices.begin(), indices.end(), hashes.begin(), [this](idx_t key) { return hash(key); });
    for (const auto hash : hashes) {
        port_queue[config.port_mask() & hash]++;
    }

    const auto collisions = *std::max_element(port_queue.begin(), port_queue.end());
    std::uint64_t cycles = static_cast<std::uint64_t>(
        std::ceil(static_cast<float>(collisions) / static_cast<float>(issue_width_))) +
        config.pipelined;

    std::vector<std::uint8_t> port_elems_cnt(config.simd, 0);
    std::vector<std::uint8_t> port_hit(config.simd, 0);
    std::vector<std::uint8_t> port_extra_delay(config.simd, 0);

    IdmLoadResult result;
    result.values.resize(indices.size(), 0.0F);

    const auto bucket_shift = log2_floor(config.buckets());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & (config.buckets() - 1);
        bool found = false;

        for (std::size_t j = 0; j < config.hash_depth; ++j) {
            const auto addr = base_mem_addr + (j << bucket_shift);
            if (valid_[addr] && idx_array_[addr] == static_cast<std::uint32_t>(indices[i])) {
                result.values[i] = mem_array_[addr];
                found = true;
                break;
            }
        }

        if (issue_width_ > 1) {
            const auto port = config.port_mask() & static_cast<std::uint32_t>(indices[i]);
            if (found) {
                if (port_hit[port] > 0) {
                    port_extra_delay[port]++;
                }
                port_hit[port]++;
            }
            port_elems_cnt[port]++;
            if (port_elems_cnt[port] >= issue_width_) {
                port_elems_cnt[port] = 0;
                port_hit[port] = 0;
            }
        }
    }

    if (issue_width_ > 1) {
        cycles += *std::max_element(port_extra_delay.begin(), port_extra_delay.end());
    }

    result.cycles = cycles;
    return result;
}

IdmStoreResult SimaModel::idm_st(const std::vector<idx_t>& indices,
                                 const std::vector<data_t>& values) {
    const auto& config = default_config();
    if (indices.size() != values.size()) {
        throw std::invalid_argument("SIMA write index/value size mismatch");
    }

    std::vector<std::uint32_t> port_queue(config.simd, 0);
    std::vector<std::size_t> hashes(indices.size());
    std::transform(indices.begin(), indices.end(), hashes.begin(), [this](idx_t key) { return hash(key); });
    for (const auto hash : hashes) {
        port_queue[config.port_mask() & hash]++;
    }

    const auto collisions = *std::max_element(port_queue.begin(), port_queue.end());
    const auto bucket_shift = log2_floor(config.buckets());

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & (config.buckets() - 1);
        bool stored = false;

        for (std::size_t j = 0; j < config.hash_depth; ++j) {
            const auto addr = base_mem_addr + (j << bucket_shift);
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

    return {static_cast<std::uint64_t>(collisions) + config.pipelined, 0};
}

IdmInitResult SimaModel::idm_init(const std::vector<idx_t>& indices,
                                  const std::vector<data_t>& values) {
    const auto& config = default_config();
    if (indices.size() != values.size()) {
        throw std::invalid_argument("SIMA init index/value size mismatch");
    }

    std::vector<std::uint32_t> port_queue(config.simd, 0);
    std::vector<std::size_t> hashes(indices.size());
    std::transform(indices.begin(), indices.end(), hashes.begin(), [this](idx_t key) { return hash(key); });
    for (const auto hash : hashes) {
        port_queue[config.port_mask() & hash]++;
    }

    const auto collisions = port_queue.empty() ? 0 : *std::max_element(port_queue.begin(), port_queue.end());
    const auto bucket_shift = log2_floor(config.buckets());

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const auto base_mem_addr = static_cast<std::uint32_t>(hashes[i]) & (config.buckets() - 1);
        bool stored = false;

        for (std::size_t j = 0; j < config.hash_depth; ++j) {
            const auto addr = base_mem_addr + (j << bucket_shift);
            if (!valid_[addr]) {
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

    const auto cycles = static_cast<std::uint64_t>(
        std::ceil(static_cast<float>(collisions) / static_cast<float>(issue_width_))) +
        config.pipelined;
    return {IdmResult{cycles, 0, false}};
}

IdmEvictResult SimaModel::idm_evict_simd() {
    return evict_one_per_logical_lane(static_cast<std::uint32_t>(default_config().simd), 1);
}

IdmEvictResult SimaModel::idm_evict() {
    return evict_all_with_cycles(1);
}

bool SimaModel::terminal_overflow() const {
    return true;
}

}  // namespace sparse_sim
