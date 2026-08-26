cmake_minimum_required(VERSION 3.18)
project(sparse_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPARSE_SIM_BUILD_PYTHON "Build the pybind11 extension" ON)

add_library(sparse_sim_core
    cpp/src/csr_matrix.cpp
    cpp/src/models/asa.cpp
    cpp/src/models/inner_sp.cpp
    cpp/src/models/registry.cpp
    cpp/src/models/sima.cpp
    cpp/src/models/via.cpp
    cpp/src/models/common/array_backed_model.cpp
    cpp/src/models/common/config.cpp
    cpp/src/models/common/h3.cpp
    cpp/src/models/common/h3_matrices.cpp
    cpp/src/models/common/idm_model_group.cpp
    cpp/src/spgemm.cpp
    cpp/src/stats.cpp
)

set_target_properties(sparse_sim_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sparse_sim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_compile_options(sparse_sim_core PRIVATE -Wall -Wextra -Wpedantic)

if(SPARSE_SIM_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    set(SPARSE_SIM_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11")
    if(EXISTS "${SPARSE_SIM_PYBIND11_DIR}/CMakeLists.txt")
        add_subdirectory("${SPARSE_SIM_PYBIND11_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11")
    else()
        find_package(pybind11 QUIET)
        if(NOT pybind11_FOUND)
            message(FATAL_ERROR "pybind11 not found. Install pybind11 or initialize lib/pybind11.")
        endif()
    endif()

    pybind11_add_module(_core cpp/src/bindings.cpp)
    target_link_libraries(_core PRIVATE sparse_sim_core)
    target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
