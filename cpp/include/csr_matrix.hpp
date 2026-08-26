#ifndef SPARSE_SIM_CSR_MATRIX_HPP
#define SPARSE_SIM_CSR_MATRIX_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sparse_sim {

class CsrMatrix {
public:
    CsrMatrix() = default;
    CsrMatrix(std::vector<int> indptr,
              std::vector<int> indices,
              std::vector<float> data,
              int rows,
              int cols);

    int rows() const;
    int cols() const;
    std::size_t nnz() const;

    const std::vector<int>& indptr() const;
    const std::vector<int>& indices() const;
    const std::vector<float>& data() const;

private:
    std::vector<int> indptr_;
    std::vector<int> indices_;
    std::vector<float> data_;
    int rows_ = 0;
    int cols_ = 0;
};

}  // namespace sparse_sim

#endif  // SPARSE_SIM_CSR_MATRIX_HPP
