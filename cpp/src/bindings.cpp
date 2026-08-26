#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "csr_matrix.hpp"
#include "h3.hpp"
#include "model_registry.hpp"
#include "spgemm.hpp"

namespace py = pybind11;

namespace {

template <typename T>
std::vector<T> to_vector(py::array_t<T, py::array::c_style | py::array::forcecast> array) {
    const auto info = array.request();
    const auto* begin = static_cast<T*>(info.ptr);
    return std::vector<T>(begin, begin + info.size);
}

py::dict stats_to_dict(const sparse_sim::SpGemmStats& stats) {
    py::dict cycles;
    py::dict model_stats;
    for (const auto& name : sparse_sim::model_names()) {
        const auto& stats_for_model = stats.model_stats.at(name);
        cycles[py::str(name)] = stats_for_model.cycles;

        py::dict model_entry;
        model_entry["cycles"] = stats_for_model.cycles;
        model_entry["checks"] = stats_for_model.checks;
        model_stats[py::str(name)] = model_entry;
    }

    py::dict instruction_count;
    for (const auto& [instruction, count] : stats.instruction_count) {
        instruction_count[py::str(instruction)] = count;
    }

    py::dict out;
    out["cycles"] = cycles;
    out["model_stats"] = model_stats;
    out["instruction_count"] = instruction_count;
    out["reads"] = stats.reads;
    out["zeroes_found"] = stats.zeroes_found;
    out["writes"] = stats.writes;
    out["pairs_initialized"] = stats.pairs_initialized;
    out["pairs_evicted"] = stats.pairs_evicted;
    out["innersp_checks"] = stats.model_stats.at("innersp_256").checks;
    out["avg_util"] = stats.avg_utilization;
    out["max_util"] = stats.max_utilization;
    out["ovf_util"] = stats.ovf_utilization;
    out["overflow"] = stats.overflow;
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Clean SIMA SpGEMM simulator core";
    m.def("model_names", &sparse_sim::model_names);
    m.def(
        "spgemm",
        [](py::array_t<int, py::array::c_style | py::array::forcecast> a_indptr,
           py::array_t<int, py::array::c_style | py::array::forcecast> a_indices,
           py::array_t<float, py::array::c_style | py::array::forcecast> a_data,
           std::pair<int, int> a_shape,
           py::array_t<int, py::array::c_style | py::array::forcecast> b_indptr,
           py::array_t<int, py::array::c_style | py::array::forcecast> b_indices,
           py::array_t<float, py::array::c_style | py::array::forcecast> b_data,
           std::pair<int, int> b_shape,
           const std::string& h3_matrix) {
            sparse_sim::CsrMatrix a(to_vector<int>(a_indptr),
                                    to_vector<int>(a_indices),
                                    to_vector<float>(a_data),
                                    a_shape.first,
                                    a_shape.second);
            sparse_sim::CsrMatrix b(to_vector<int>(b_indptr),
                                    to_vector<int>(b_indices),
                                    to_vector<float>(b_data),
                                    b_shape.first,
                                    b_shape.second);
            return stats_to_dict(sparse_sim::simulate_spgemm(
                a, b, sparse_sim::h3_matrix_id_from_name(h3_matrix)));
        },
        py::arg("a_indptr"),
        py::arg("a_indices"),
        py::arg("a_data"),
        py::arg("a_shape"),
        py::arg("b_indptr"),
        py::arg("b_indices"),
        py::arg("b_data"),
        py::arg("b_shape"),
        py::arg("h3_matrix") = "modulo");
}
