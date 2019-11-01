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

add_subdirectory(protos)

# HE transformer sources
set(HE_SRC
    # main
    he_op_annotations.cpp
    he_tensor.cpp
    he_type.cpp
    node_wrapper.cpp
    util.cpp
    he_plaintext.cpp
    # pass
    pass/he_fusion.cpp
    pass/he_liveness.cpp
    pass/propagate_he_annotations.cpp
    pass/supported_ops.cpp
    # op
    op/bounded_relu.cpp
    # seal kernels
    seal/kernel/add_seal.cpp
    seal/kernel/bounded_relu_seal.cpp
    seal/kernel/dot_seal.cpp
    seal/kernel/convolution_seal.cpp
    seal/kernel/constant_seal.cpp
    seal/kernel/divide_seal.cpp
    seal/kernel/exp_seal.cpp
    seal/kernel/minimum_seal.cpp
    seal/kernel/multiply_seal.cpp
    seal/kernel/negate_seal.cpp
    seal/kernel/pad_seal.cpp
    seal/kernel/power_seal.cpp
    seal/kernel/relu_seal.cpp
    seal/kernel/rescale_seal.cpp
    seal/kernel/softmax_seal.cpp
    seal/kernel/subtract_seal.cpp
    # seal backend
    seal/he_seal_backend.cpp
    seal/he_seal_client.cpp
    seal/he_seal_encryption_parameters.cpp
    seal/he_seal_executable.cpp
    seal/seal_util.cpp
    # protobuf files
    ${message_proto_srcs})

set_source_files_properties(${message_proto_srcs}
                            PROPERTIES
                            GENERATED
                            TRUE)

message("message_proto_srcs ${message_proto_srcs}")

# Target library
add_library(he_seal_backend SHARED ${HE_SRC})

target_include_directories(he_seal_backend
                           PUBLIC ${HE_TRANSFORMER_SOURCE_DIR}
                                  ${NGRAPH_TF_INCLUDE_DIR})

target_link_libraries(he_seal_backend
                      PUBLIC libjson
                             libboost
                             libseal
                             libngraph_tf
                             protobuf::libprotobuf
                             generated_proto_files
                             ngraph)

# SEAL uses zlib
find_package(ZLIB 1.2.11 EXACT)
if(ZLIB_FOUND)
  message(STATUS "ZLIB_LIBRARIES ${ZLIB_LIBRARIES}")
  target_link_libraries(he_seal_backend PUBLIC ZLIB::ZLIB)
endif()

message("HE_TRANSFORMER_SOURCE_DIR ${HE_TRANSFORMER_SOURCE_DIR}")
message("EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_INCLUDE_DIR}")
# Get library names
set(HE_SEAL_BACKEND_LIB_NAME "libhe_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

message("NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_LIB_DIR}")
message("NGRAPH_TF_LIB_DIR ${NGRAPH_TF_LIB_DIR}")

# Install he-transformer to installation location and into ngraph-tf
install(TARGETS he_seal_backend DESTINATION ${EXTERNAL_INSTALL_LIB_DIR})
install(TARGETS he_seal_backend DESTINATION ${NGRAPH_TF_VENV_LIB_DIR})
install(TARGETS he_seal_backend DESTINATION ${NGRAPH_TF_LIB_DIR})
install(TARGETS he_seal_backend DESTINATION ${NGRAPH_HE_INSTALL_LIB})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${NGRAPH_HE_INSTALL_LIB}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

# Install protobuf files
install(DIRECTORY ${CMAKE_BINARY_DIR}/protos
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${NGRAPH_HE_INSTALL_LIB}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

message(STATUS "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR}")
