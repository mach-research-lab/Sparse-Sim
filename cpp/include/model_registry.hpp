#ifndef SPARSE_SIM_MODEL_REGISTRY_HPP
#define SPARSE_SIM_MODEL_REGISTRY_HPP

#include <memory>
#include <string>
#include <vector>

#include "h3.hpp"
#include "idm_model.hpp"

namespace sparse_sim {

enum class ModelKind {
    Sima,
    Via,
    InnerSp,
    Asa,
};

#define SPARSE_SIM_MODEL_VARIANTS(X) \
    X(sima_1, "sima_1", ModelKind::Sima, 1) \
    X(sima_2, "sima_2", ModelKind::Sima, 2) \
    X(sima_4, "sima_4", ModelKind::Sima, 4) \
    X(via_4, "via_4", ModelKind::Via, 4) \
    X(via_16, "via_16", ModelKind::Via, 16) \
    X(innersp_256, "innersp_256", ModelKind::InnerSp, 256) \
    X(asa, "asa", ModelKind::Asa, 1)

std::vector<std::string> model_names();
std::vector<std::unique_ptr<IdmModel>> create_default_models();
std::vector<std::unique_ptr<IdmModel>> create_default_models(H3MatrixId h3_matrix);

}  // namespace sparse_sim

#endif  // SPARSE_SIM_MODEL_REGISTRY_HPP
