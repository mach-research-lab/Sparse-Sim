#ifndef SPARSE_SIM_CONFIG_HPP
#define SPARSE_SIM_CONFIG_HPP

#include <cstddef>
#include <cstdint>

#include "h3.hpp"
#include "types.hpp"

namespace sparse_sim {

struct SimConfig {
    std::size_t simd = 16;
    std::size_t memory_size_bytes = 128 * 1024;
    std::size_t element_size_bytes = sizeof(float);
    std::size_t hash_depth = 16;
    std::size_t pipelined = 0;
    H3MatrixId h3_matrix = H3MatrixId::Modulo;
    std::size_t h3_hash_bits = 12;

    std::size_t elements() const;
    std::size_t ports() const;
    std::size_t port_mask() const;
    std::size_t buckets() const;
    std::size_t bucket_bits() const;
};

const SimConfig& default_config();

std::uint32_t log2_floor(std::uint64_t value);

}  // namespace sparse_sim

#endif  // SPARSE_SIM_CONFIG_HPP
