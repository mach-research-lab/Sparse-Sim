#include "spgemm.hpp"

#include <algorithm>
#include <stdexcept>

#include "config.hpp"
#include "idm_model_group.hpp"
#include "model_registry.hpp"

namespace sparse_sim {

SpGemmStats simulate_spgemm(const CsrMatrix& a, const CsrMatrix& b, H3MatrixId h3_matrix) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("SpGEMM matrix sizes are incompatible");
    }

    auto stats = make_stats_with_models(model_names());
    IdmModelGroup models(create_default_models(h3_matrix));

    for (int row = 0; row < a.rows(); ++row) {
        const int a_row_start = a.indptr()[row];
        const int a_row_end = a.indptr()[row + 1];

        for (int a_pos = a_row_start; a_pos < a_row_end; ++a_pos) {
            const int b_row = a.indices()[a_pos];
            const float a_value = a.data()[a_pos];

            const int b_row_start = b.indptr()[b_row];
            const int b_row_end = b.indptr()[b_row + 1];
            const int b_row_len = b_row_end - b_row_start;

            for (int offset = 0; offset < b_row_len; offset += static_cast<int>(default_config().simd)) {
                const int chunk = std::min(static_cast<int>(default_config().simd), b_row_len - offset);
                const auto idx_begin = b.indices().begin() + b_row_start + offset;
                const auto data_begin = b.data().begin() + b_row_start + offset;

                std::vector<int> indices(idx_begin, idx_begin + chunk);
                std::vector<float> b_values(data_begin, data_begin + chunk);
                std::vector<float> accum = models.idm_ld(indices, stats);

                for (int i = 0; i < chunk; ++i) {
                    accum[i] += a_value * b_values[i];
                }

                if (!models.idm_st(indices, accum, stats)) {
                    stats.avg_utilization /= static_cast<float>(a.rows());
                    return stats;
                }
            }
        }

        models.sample_row_utilization(stats);
        models.idm_rst(stats);
    }

    stats.avg_utilization /= static_cast<float>(a.rows());
    return stats;
}

}  // namespace sparse_sim
