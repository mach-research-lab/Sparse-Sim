#ifndef SPARSE_SIM_H3_HPP
#define SPARSE_SIM_H3_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sparse_sim {

enum class H3MatrixId {
    Modulo,
    Ovf,
    Accel,
};

struct H3Matrix {
    std::string name;
    std::size_t input_bits = 0;
    std::size_t hash_bits = 0;
    std::vector<std::uint64_t> columns;
};

std::string_view h3_matrix_name(H3MatrixId matrix_id);
H3MatrixId h3_matrix_id_from_name(std::string_view name);
std::size_t h3_hash(std::uint64_t key, const H3Matrix& matrix);
H3Matrix select_h3_matrix(H3MatrixId matrix_id,
                          std::size_t input_bits,
                          std::size_t hash_bits,
                          std::size_t bucket_bits);

}  // namespace sparse_sim

#endif  // SPARSE_SIM_H3_HPP
