#ifndef SPARSE_SIM_SPGEMM_HPP
#define SPARSE_SIM_SPGEMM_HPP

#include "csr_matrix.hpp"
#include "h3.hpp"
#include "stats.hpp"

namespace sparse_sim {

SpGemmStats simulate_spgemm(const CsrMatrix& a, const CsrMatrix& b, H3MatrixId h3_matrix);

}  // namespace sparse_sim

#endif  // SPARSE_SIM_SPGEMM_HPP
