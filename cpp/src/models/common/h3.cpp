#include "h3.hpp"

#include <stdexcept>
#include <string>

namespace sparse_sim {

std::string_view h3_matrix_name(H3MatrixId matrix_id) {
    switch (matrix_id) {
        case H3MatrixId::Modulo:
            return "modulo";
        case H3MatrixId::Ovf:
            return "ovf";
        case H3MatrixId::Accel:
            return "accel";
    }
    throw std::invalid_argument("Unknown H3 matrix id");
}

H3MatrixId h3_matrix_id_from_name(std::string_view name) {
    if (name == "modulo") {
        return H3MatrixId::Modulo;
    }
    if (name == "ovf") {
        return H3MatrixId::Ovf;
    }
    if (name == "accel") {
        return H3MatrixId::Accel;
    }
    throw std::invalid_argument("Unknown H3 matrix: " + std::string(name));
}

std::size_t h3_hash(std::uint64_t key, const H3Matrix& matrix) {
    if (matrix.columns.size() != matrix.hash_bits) {
        throw std::invalid_argument("H3 matrix column count must match hash bit count");
    }

    std::size_t hash = 0;
    for (unsigned bit = 0; bit < matrix.hash_bits; ++bit) {
        if (__builtin_parityll(key & matrix.columns[bit])) {
            hash |= 1ULL << bit;
        }
    }
    return hash;
}

}  // namespace sparse_sim
