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
    he_tensor.cpp
    he_type.cpp
    node_wrapper.cpp
    he_util.cpp
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
    # tcp
    tcp/tcp_message.cpp
    tcp/tcp_client.cpp
    tcp/tcp_session.cpp
    # protobuf files
    ${message_proto_srcs})

set_source_files_properties(${message_proto_srcs}
                            PROPERTIES
                            GENERATED
                            TRUE)

# Create library and link to dependencies
add_library(he_seal_backend SHARED ${HE_SRC})
target_include_directories(he_seal_backend PUBLIC ${HE_TRANSFORMER_SOURCE_DIR})

find_package(Threads)
target_link_libraries(he_seal_backend
                      PUBLIC libjson
                             libboost
                             libseal
                             libngraph_tf
                             protobuf::libprotobuf
                             generated_proto_files
                             ngraph)

# TODO: better condition if we don't link with OpenMP
if (NGRAPH_HE_CODE_COVERAGE)
        find_package(Threads)
        target_link_libraries(he_seal_backend PUBLIC  ${CMAKE_THREAD_LIBS_INIT})
endif()

foreach(INSTALL_LIB_DIR ${INSTALL_LIB_DIRS})
  install(TARGETS he_seal_backend DESTINATION ${INSTALL_LIB_DIR})
  add_custom_command(TARGET he_seal_backend POST_BUILD
                     COMMAND ${CMAKE_COMMAND}
                             -E
                             copy
                             ${Protobuf_LIBRARIES}
                             ${INSTALL_LIB_DIR}
                     DEPENDS protobuf::libprotobuf)
endforeach()

foreach(INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIRS})
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
          DESTINATION ${INSTALL_INCLUDE_DIR}
          FILES_MATCHING
          PATTERN "*.hpp"
          PATTERN "*.h")
  install(DIRECTORY ${Protobuf_INCLUDE_DIR}/
          DESTINATION ${INSTALL_INCLUDE_DIR}
          FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hpp"
          PATTERN "*.inc") # For port_def.inc
  install(DIRECTORY ${json_INCLUDE_DIR}/
          DESTINATION ${INSTALL_INCLUDE_DIR}
          FILES_MATCHING
          PATTERN "*.h"
          PATTERN "*.hpp")

endforeach()
