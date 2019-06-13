# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

cmake_minimum_required (VERSION 3.1)

include(ExternalProject)

ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.2.4"
        SOURCE_DIR "${CMAKE_BINARY_DIR}/pybind11"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

message("python CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("python CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
message("python CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")

set(BUILD_SH_IN "${CMAKE_SOURCE_DIR}/python/build_wheel.sh.in")
set(BUILD_SH    "${CMAKE_BINARY_DIR}/python/build_wheel.sh")
configure_file(${BUILD_SH_IN} ${BUILD_SH} @ONLY)

# he_seal_client
include_directories(
    SYSTEM
    #${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    #${NGRAPH_TF_INCLUDE_DIR}
)
link_directories(
    ${EXTERNAL_INSTALL_LIB_DIR}
    #${NGRAPH_TF_LIB_DIR}
    #${NGRAPH_TF_VENV_LIB_DIR}
)
message("Client infos")
message("NGRAPH_TF_LIB_DIR ${NGRAPH_TF_LIB_DIR}")
message("HE_TRANSFORMER_SOURCE_DIR ${HE_TRANSFORMER_SOURCE_DIR}")
message("EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_INCLUDE_DIR}")
message("EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_LIB_DIR}")
message("HE_INSTALL_LIB_DIR ${HE_INSTALL_LIB_DIR}")

set(PROJECT_ROOT_DIR "test_root_dir")

set(PYTHON_ARTIFACTS_DIR ${CMAKE_BINARY_DIR}/python/_install${CMAKE_INSTALL_PREFIX})
message("PYTHON_ARTIFACTS_DIR ${PYTHON_ARTIFACTS_DIR}")
set(PYTHON_LIB_DIR ${PYTHON_ARTIFACTS_DIR}/lib)
set(PYTHON_INCLUDE_DIR ${PYTHON_ARTIFACTS_DIR}/include)

add_library(he_seal_client SHARED ${CMAKE_SOURCE_DIR}/src/seal/he_seal_client.cpp
                                  ${CMAKE_SOURCE_DIR}/src/tcp/tcp_message.hpp)
# TODO: remove he_seal_backend dependency?
add_dependencies(he_seal_client ext_seal he_seal_backend)
target_link_libraries(he_seal_client seal libjson libboost ngraph)
install(TARGETS he_seal_client DESTINATION ${HE_INSTALL_LIB_DIR})

message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(
        OUTPUT python_dist #${CMAKE_BINARY_DIR}/python/dist/
        POST_BUILD
        DEPENDS he_seal_client
        #WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
        #COMMAND make -C ../ DESTDIR=python/_install install && bash build_wheel.sh
        COMMAND bash build_wheel.sh
)

add_custom_target(python_wheel DEPENDS he_seal_client he_seal_backend python_dist) # ${CMAKE_BINARY_DIR}/python/dist/)