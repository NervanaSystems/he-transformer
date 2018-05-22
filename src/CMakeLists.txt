# *****************************************************************************
# Copyright 2017-2018 Intel Corporation
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
# *****************************************************************************

# HE transformer sources
set (HE_SRC
    # kernels
    kernel/add.cpp
    kernel/broadcast.cpp
    kernel/concat.cpp
    kernel/constant.cpp
    kernel/dot.cpp
    kernel/multiply.cpp
    kernel/one_hot.cpp
    kernel/relinearize.cpp
    kernel/reshape.cpp
    kernel/result.cpp
    kernel/slice.cpp
    kernel/subtract.cpp
    kernel/sum.cpp
    # main
    create_backend.cpp
    he_backend.cpp
    he_call_frame.cpp
    he_cipher_tensor_view.cpp
    he_ciphertext.cpp
    he_plaintext.cpp
    he_tensor_view.cpp
    he_parameter.cpp
    he_plain_tensor_view.cpp
    int_call_frame.cpp
    # pass
    pass/insert_relinearize.cpp
    # heaan
    he_heaan_backend.cpp
    he_heaan_parameter.cpp
    heaan_ciphertext_wrapper.cpp
    heaan_plaintext_wrapper.cpp
    # seal
    he_seal_backend.cpp
    he_seal_parameter.cpp
    seal_ciphertext_wrapper.cpp
    seal_plaintext_wrapper.cpp
)

# Paths for the upstream ngraph
message(STATUS "NGRAPH_INCLUDE_DIR in parent scope: ${NGRAPH_INCLUDE_DIR}")
message(STATUS "NGRAPH_LIB_DIR in parent scope: ${NGRAPH_LIB_DIR}")

include_directories(
    SYSTEM
    ${NGRAPH_INCLUDE_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
)
link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
)

# Target library
add_library(he_backend SHARED ${HE_SRC})
add_dependencies(he_backend ext_ngraph ext_seal)
target_link_libraries(he_backend ngraph seal)

# Install he-transformer (the HE libraries are installed separately) at `make install`
# By default, this installs to `~/ngraph_dist`
install(TARGETS he_backend DESTINATION ${NGRAPH_INSTALL_LIB_DIR})
install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE_DIR}
    FILES_MATCHING PATTERN "*.hpp"
)
install(DIRECTORY
    ${EXTERNAL_INSTALL_LIB_DIR}/
    DESTINATION ${NGRAPH_INSTALL_LIB_DIR}
)
install(DIRECTORY
    ${EXTERNAL_INSTALL_INCLUDE_DIR}/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE_DIR}
)
