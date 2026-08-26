#include "config.hpp"

#include <stdexcept>

namespace sparse_sim {

std::size_t SimConfig::elements() const {
    return memory_size_bytes / element_size_bytes;
}

std::size_t SimConfig::ports() const {
    return simd;
}

std::size_t SimConfig::port_mask() const {
    return ports() - 1;
}

std::size_t SimConfig::buckets() const {
    return elements() / hash_depth;
}

std::size_t SimConfig::bucket_bits() const {
    return log2_floor(buckets());
}

const SimConfig& default_config() {
    static const SimConfig config;
    return config;
}

std::uint32_t log2_floor(std::uint64_t value) {
    if (value == 0) {
        throw std::invalid_argument("log2_floor requires a non-zero value");
    }

    std::uint32_t result = 0;
    while (value >>= 1U) {
        ++result;
    }
    return result;
}

}  // namespace sparse_sim
