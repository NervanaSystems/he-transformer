# *****************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# *****************************************************************************

# HE transformer sources
set(HE_SRC
    # main
    he_plain_tensor.cpp he_tensor.cpp node_wrapper.cpp
    # pass
    pass/he_fusion.cpp pass/he_liveness.cpp
    # op
    op/bounded_relu.cpp
    # seal kernels
    seal/kernel/constant_seal.cpp
    seal/kernel/pad_seal.cpp
    seal/kernel/subtract_seal.cpp
    seal/kernel/result_seal.cpp
    seal/kernel/add_seal.cpp
    seal/kernel/multiply_seal.cpp
    seal/kernel/negate_seal.cpp
    # seal backend
    seal/seal_util.cpp
    seal/he_seal_cipher_tensor.cpp
    seal/he_seal_executable.cpp
    seal/he_seal_backend.cpp
    seal/he_seal_client.cpp)

# Target library
add_library(he_seal_backend SHARED ${HE_SRC})
target_include_directories(he_seal_backend
                           PUBLIC ${HE_TRANSFORMER_SOURCE_DIR}
                                  ${EXTERNAL_INSTALL_INCLUDE_DIR}
                                  ${NGRAPH_TF_INCLUDE_DIR})
target_link_libraries(he_seal_backend
                      PUBLIC libjson
                             libboost
                             libseal
                             libngraph_tf
                             ngraph)

message("HE_TRANSFORMER_SOURCE_DIR ${HE_TRANSFORMER_SOURCE_DIR}")
message("EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_INCLUDE_DIR}")
# Get library names
set(HE_SEAL_BACKEND_LIB_NAME "libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

message("NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_LIB_DIR}")
message("NGRAPH_TF_LIB_DIR ${NGRAPH_TF_LIB_DIR}")

# Install he-transformer (the HE libraries are installed separately) at `make
# install`
install(TARGETS he_seal_backend DESTINATION ${EXTERNAL_INSTALL_LIB_DIR})

message(
  "Copying CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} to ${EXTERNAL_INSTALL_INCLUDE_DIR}"
  )
message("PROJECT_ROOT_DIR ${PROJECT_ROOT_DIR}")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

# Create symbolic links for he_seal_backend, to allow ngraph and ngraph-tf to
# recognize find it.
add_custom_target(
  he_seal_backend_soft_link ALL
  DEPENDS he_seal_backend
  COMMAND
    ${CMAKE_COMMAND}
    -E
    create_symlink
    ${EXTERNAL_INSTALL_LIB_DIR}/libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${NGRAPH_TF_VENV_LIB_DIR}/libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}
  COMMAND
    ${CMAKE_COMMAND}
    -E
    create_symlink
    ${EXTERNAL_INSTALL_LIB_DIR}/libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${NGRAPH_TF_LIB_DIR}/libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX})
