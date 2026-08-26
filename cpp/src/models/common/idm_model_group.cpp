#include "idm_model_group.hpp"

#include <algorithm>
#include <stdexcept>

namespace sparse_sim {

namespace {

void add_model_result(const IdmModel& model, const IdmResult& result, KernelStats& stats) {
    auto& model_stats = stats.model_stats[model.name()];
    model_stats.cycles += result.cycles;
    model_stats.checks += result.checks;
}

bool has_terminal_overflow(const std::vector<std::unique_ptr<IdmModel>>& models) {
    return std::any_of(models.begin(), models.end(), [](const auto& model) {
        return model->terminal_overflow() && model->overflow();
    });
}

}  // namespace

IdmModelGroup::IdmModelGroup(std::vector<std::unique_ptr<IdmModel>> models)
    : models_(std::move(models)) {
    if (models_.empty()) {
        throw std::invalid_argument("IdmModelGroup requires at least one model");
    }
}

std::vector<data_t> IdmModelGroup::idm_ld(const std::vector<idx_t>& indices,
                                          KernelStats& stats) {
    std::vector<data_t> reference;
    bool have_reference = false;

    for (auto& model : models_) {
        auto result = model->idm_ld(indices);
        add_model_result(*model, result, stats);

        if (!have_reference) {
            reference = std::move(result.values);
            have_reference = true;
        } else if (result.values != reference) {
            throw std::runtime_error("IDM load values diverged at " + model->name());
        }
    }

    stats.instruction_count["idm_ld"]++;
    stats.reads += indices.size();
    stats.zeroes_found += static_cast<std::uint64_t>(
        std::count(reference.begin(), reference.end(), 0.0F));
    return reference;
}

bool IdmModelGroup::idm_st(const std::vector<idx_t>& indices,
                           const std::vector<data_t>& values,
                           KernelStats& stats) {
    for (auto& model : models_) {
        auto result = model->idm_st(indices, values);
        add_model_result(*model, result, stats);
    }

    stats.instruction_count["idm_st"]++;
    stats.writes += indices.size();
    if (has_terminal_overflow(models_)) {
        stats.ovf_utilization = primary_utilization();
        stats.overflow = 1;
        return false;
    }
    return true;
}

bool IdmModelGroup::idm_init(const std::vector<idx_t>& indices,
                             const std::vector<data_t>& values,
                             KernelStats& stats) {
    for (auto& model : models_) {
        auto result = model->idm_init(indices, values);
        add_model_result(*model, result, stats);
    }

    stats.instruction_count["idm_init"]++;
    stats.pairs_initialized += indices.size();
    stats.writes += indices.size();
    if (has_terminal_overflow(models_)) {
        stats.ovf_utilization = primary_utilization();
        stats.overflow = 1;
        return false;
    }
    return true;
}

IdmEvictResult IdmModelGroup::idm_evict_simd(KernelStats& stats) {
    IdmEvictResult reference;
    bool have_reference = false;

    for (auto& model : models_) {
        auto result = model->idm_evict_simd();
        add_model_result(*model, result, stats);

        if (!have_reference) {
            reference = result;
            have_reference = true;
        }
    }

    stats.instruction_count["idm_evict_simd"]++;
    stats.pairs_evicted += reference.indices.size();
    return reference;
}

IdmEvictResult IdmModelGroup::idm_evict(KernelStats& stats) {
    IdmEvictResult reference;
    bool have_reference = false;

    for (auto& model : models_) {
        auto result = model->idm_evict();
        add_model_result(*model, result, stats);

        if (!have_reference) {
            reference = result;
            have_reference = true;
        }
    }

    stats.instruction_count["idm_evict"]++;
    stats.pairs_evicted += reference.indices.size();
    return reference;
}

void IdmModelGroup::idm_rst(KernelStats& stats) {
    for (auto& model : models_) {
        auto result = model->idm_rst();
        add_model_result(*model, result, stats);
    }
    stats.instruction_count["idm_rst"]++;
}

std::uint64_t IdmModelGroup::idm_val(KernelStats& stats) {
    std::uint64_t reference = 0;
    bool have_reference = false;

    for (auto& model : models_) {
        auto result = model->idm_val();
        add_model_result(*model, result, stats);
        if (!have_reference) {
            reference = result.valid_pairs;
            have_reference = true;
        } else if (result.valid_pairs != reference) {
            throw std::runtime_error("IDM valid-pair counts diverged at " + model->name());
        }
    }

    stats.instruction_count["idm_val"]++;
    return reference;
}

void IdmModelGroup::sample_row_utilization(KernelStats& stats) const {
    const auto utilization = primary_utilization();
    stats.avg_utilization += utilization;
    stats.max_utilization = std::max(stats.max_utilization, utilization);
}

float IdmModelGroup::primary_utilization() const {
    for (const auto& model : models_) {
        if (model->terminal_overflow()) {
            return model->utilization();
        }
    }
    return models_.front()->utilization();
}

}  // namespace sparse_sim
