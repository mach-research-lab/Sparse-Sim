#include "csr_matrix.hpp"

#include <stdexcept>

namespace sparse_sim {

CsrMatrix::CsrMatrix(std::vector<int> indptr,
                     std::vector<int> indices,
                     std::vector<float> data,
                     int rows,
                     int cols)
    : indptr_(std::move(indptr)),
      indices_(std::move(indices)),
      data_(std::move(data)),
      rows_(rows),
      cols_(cols) {
    if (rows_ <= 0 || cols_ <= 0) {
        throw std::invalid_argument("CSR matrix dimensions must be positive");
    }
    if (indptr_.size() != static_cast<std::size_t>(rows_) + 1) {
        throw std::invalid_argument("CSR indptr size does not match row count");
    }
    if (indices_.size() != data_.size()) {
        throw std::invalid_argument("CSR indices and data sizes differ");
    }
    if (indptr_.empty() || indptr_.front() != 0) {
        throw std::invalid_argument("CSR indptr must start at zero");
    }
    if (indptr_.back() != static_cast<int>(indices_.size())) {
        throw std::invalid_argument("CSR indptr end does not match nnz");
    }
}

int CsrMatrix::rows() const { return rows_; }
int CsrMatrix::cols() const { return cols_; }
std::size_t CsrMatrix::nnz() const { return data_.size(); }

const std::vector<int>& CsrMatrix::indptr() const { return indptr_; }
const std::vector<int>& CsrMatrix::indices() const { return indices_; }
const std::vector<float>& CsrMatrix::data() const { return data_; }

}  // namespace sparse_sim
