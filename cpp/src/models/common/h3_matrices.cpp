#include "h3.hpp"

#include <sstream>
#include <stdexcept>

namespace sparse_sim {

namespace {

void validate_matrix_dimensions(std::size_t input_bits, std::size_t hash_bits) {
    if (input_bits > 64 || hash_bits > 64) {
        throw std::invalid_argument("H3 matrices currently support up to 64 input and hash bits");
    }
    if (hash_bits > input_bits) {
        throw std::invalid_argument("H3 hash bits cannot exceed input bits");
    }
}

H3Matrix make_modulo_matrix(std::size_t input_bits, std::size_t hash_bits) {
    validate_matrix_dimensions(input_bits, hash_bits);

    H3Matrix matrix;
    matrix.name = std::string(h3_matrix_name(H3MatrixId::Modulo));
    matrix.input_bits = input_bits;
    matrix.hash_bits = hash_bits;
    matrix.columns.reserve(hash_bits);

    // This pseudo-identity H3 matrix projects the low output bits of the key.
    // For a power-of-two bucket count, this behaves like key % buckets.
    for (std::size_t bit = 0; bit < hash_bits; ++bit) {
        matrix.columns.push_back(1ULL << bit);
    }
    return matrix;
}

H3Matrix make_ovf_matrix(std::size_t input_bits, std::size_t hash_bits, std::size_t bucket_bits) {
    validate_matrix_dimensions(input_bits, hash_bits);
    if (bucket_bits == 0 || bucket_bits + hash_bits > input_bits) {
        throw std::invalid_argument("OVF H3 matrix requires bucket_bits + hash_bits <= input_bits");
    }

    H3Matrix matrix;
    matrix.name = std::string(h3_matrix_name(H3MatrixId::Ovf));
    matrix.input_bits = input_bits;
    matrix.hash_bits = hash_bits;
    matrix.columns.reserve(hash_bits);

    // Split the key into a low bucket-sized slice and the next bucket-sized
    // slice, then XOR matching bit positions. This reproduces:
    //   (key & (buckets - 1)) ^ ((key >> log2(buckets)) & (buckets - 1))
    for (std::size_t bit = 0; bit < hash_bits; ++bit) {
        matrix.columns.push_back((1ULL << bit) | (1ULL << (bit + bucket_bits)));
    }
    return matrix;
}

H3Matrix make_accel_matrix(std::size_t input_bits, std::size_t hash_bits) {
    validate_matrix_dimensions(input_bits, hash_bits);
    constexpr std::size_t kWindowBits = 8;
    if (hash_bits + kWindowBits > input_bits) {
        throw std::invalid_argument("Accel H3 matrix requires hash_bits + 8 <= input_bits");
    }

    H3Matrix matrix;
    matrix.name = std::string(h3_matrix_name(H3MatrixId::Accel));
    matrix.input_bits = input_bits;
    matrix.hash_bits = hash_bits;
    matrix.columns.reserve(hash_bits);

    for (std::size_t bit = 0; bit < hash_bits; ++bit) {
        std::uint64_t column = 0;
        for (std::size_t input_bit = bit + 1U; input_bit <= bit + kWindowBits; ++input_bit) {
            column |= 1ULL << input_bit;
        }
        matrix.columns.push_back(column);
    }
    return matrix;
}

}  // namespace

H3Matrix select_h3_matrix(H3MatrixId matrix_id,
                          std::size_t input_bits,
                          std::size_t hash_bits,
                          std::size_t bucket_bits) {
    switch (matrix_id) {
        case H3MatrixId::Modulo:
            return make_modulo_matrix(input_bits, hash_bits);
        case H3MatrixId::Ovf:
            return make_ovf_matrix(input_bits, hash_bits, bucket_bits);
        case H3MatrixId::Accel:
            return make_accel_matrix(input_bits, hash_bits);
    }

    std::ostringstream message;
    message << "Unknown H3 matrix id for " << input_bits << " input bits and "
            << hash_bits << " output bits";
    throw std::invalid_argument(message.str());
}

}  // namespace sparse_sim
