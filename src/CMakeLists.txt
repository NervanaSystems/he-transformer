# *****************************************************************************
# Copyright 2018-2019 Intel Corporation
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
    kernel/constant.cpp
    kernel/multiply.cpp
    kernel/negate.cpp
    kernel/pad.cpp
    kernel/subtract.cpp
    kernel/result.cpp

    # seal
    seal/kernel/add_seal.cpp
    seal/kernel/multiply_seal.cpp
    seal/kernel/negate_seal.cpp
    seal/kernel/subtract_seal.cpp
    seal/he_seal_backend.cpp
    seal/seal_plaintext_wrapper.cpp
    seal/seal_ciphertext_wrapper.cpp
    seal/bfv/he_seal_bfv_backend.cpp
    seal/bfv/kernel/add_seal_bfv.cpp
    seal/bfv/kernel/multiply_seal_bfv.cpp
    seal/ckks/he_seal_ckks_backend.cpp
    seal/ckks/kernel/add_seal_ckks.cpp
    seal/ckks/kernel/multiply_seal_ckks.cpp

    # main
    he_backend.cpp
    he_executable.cpp
    he_cipher_tensor.cpp
    he_plain_tensor.cpp
    he_tensor.cpp
    node_wrapper.cpp
)
include_directories(
    SYSTEM
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    ${NGRAPH_TF_INCLUDE_DIR}
)
link_directories(
    ${EXTERNAL_INSTALL_LIB_DIR}
    ${NGRAPH_TF_LIB_DIR}
)

message("Linking directories: ${NGRAPH_TF_LIB_DIR}, ${EXTERNAL_INSTALL_LIB_DIR}")

# Target library
add_library(he_backend SHARED ${HE_SRC})
add_dependencies(he_backend ext_ngraph_tf ext_seal)
target_link_libraries(he_backend seal libjson ngraph)

if(NGRAPH_HE_SANITIZE_ADDRESS)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

# Get library names
set(HE_BACKEND_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(HE_SEAL_CKKS_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_ckks_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(HE_SEAL_BFV_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_bfv_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

message("NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_LIB_DIR}")

# Install he-transformer (the HE libraries are installed separately) at `make install`
install(TARGETS he_backend DESTINATION ${EXTERNAL_INSTALL_LIB_DIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${HE_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")

# Create symbolic links for CKKS and BFV backends, to allow ngraph and ngraph-tf to recognize these backends.
add_custom_target(he_seal_ckks_backend ALL DEPENDS he_backend
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${EXTERNAL_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}he_backend${CMAKE_SHARED_LIBRARY_SUFFIX}
            ${NGRAPH_TF_VENV_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_ckks_backend${CMAKE_SHARED_LIBRARY_SUFFIX})

add_custom_target(he_seal_bfv_backend ALL DEPENDS he_backend
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${EXTERNAL_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}he_backend${CMAKE_SHARED_LIBRARY_SUFFIX}
        ${NGRAPH_TF_VENV_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_bfv_backend${CMAKE_SHARED_LIBRARY_SUFFIX})
