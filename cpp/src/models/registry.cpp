#include "model_registry.hpp"

#include "config.hpp"
#include "h3.hpp"
#include "models.hpp"

namespace sparse_sim {

std::vector<std::string> model_names() {
    std::vector<std::string> names;
#define SPARSE_ADD_MODEL_NAME(identifier, label, kind, parameter) names.emplace_back(label);
    SPARSE_SIM_MODEL_VARIANTS(SPARSE_ADD_MODEL_NAME)
#undef SPARSE_ADD_MODEL_NAME
    return names;
}

std::vector<std::unique_ptr<IdmModel>> create_default_models() {
    return create_default_models(default_config().h3_matrix);
}

std::vector<std::unique_ptr<IdmModel>> create_default_models(H3MatrixId h3_matrix_id) {
    std::vector<std::unique_ptr<IdmModel>> models;
    const auto& config = default_config();
    const auto h3_matrix = select_h3_matrix(h3_matrix_id,
                                            8U * sizeof(idx_t),
                                            config.h3_hash_bits,
                                            config.bucket_bits());

#define SPARSE_CREATE_MODEL(identifier, label, kind, parameter) \
    if constexpr (kind == ModelKind::Sima) { \
        models.emplace_back(std::make_unique<SimaModel>(label, parameter, h3_matrix)); \
    } else if constexpr (kind == ModelKind::Via) { \
        models.emplace_back(std::make_unique<ViaModel>(label, parameter)); \
    } else if constexpr (kind == ModelKind::InnerSp) { \
        models.emplace_back(std::make_unique<InnerSpModel>(label, parameter, h3_matrix)); \
    } else if constexpr (kind == ModelKind::Asa) { \
        models.emplace_back(std::make_unique<AsaModel>(label)); \
    }

    SPARSE_SIM_MODEL_VARIANTS(SPARSE_CREATE_MODEL)
#undef SPARSE_CREATE_MODEL

    return models;
}

}  // namespace sparse_sim
