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
    he_cipher_tensor.cpp
    he_plain_tensor.cpp
    he_tensor.cpp
    node_wrapper.cpp
)

# Paths for the upstream ngraph
message(STATUS "NGRAPH_INCLUDE_DIR in parent scope: ${NGRAPH_INCLUDE_DIR}")
message(STATUS "NGRAPH_LIB_DIR in parent scope: ${NGRAPH_LIB_DIR}")

include_directories(
    SYSTEM
    ${NGRAPH_INCLUDE_DIR}
    ${HE_TRANSFORMER_SOURCE_DIR}
    ${EXTERNAL_INSTALL_INCLUDE_DIR}
    ${NGRAPH_TF_INCLUDE_DIR}
)
link_directories(
    ${NGRAPH_LIB_DIR}
    ${EXTERNAL_INSTALL_LIB_DIR}
    ${NGRAPH_TF_LIB_DIR}
)

message("Linking directories: ${NGRAPH_LIB_DIR}, ${EXTERNAL_INSTALL_LIB_DIR}")

# Target library
add_library(he_backend SHARED ${HE_SRC})
add_dependencies(he_backend ext_ngraph_tf ext_seal)
target_link_libraries(he_backend seal libjson)

set_target_properties(he_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY
    ${EXTERNAL_NGRAPH_INSTALL_DIR}/lib
)

# Note, this will flag many false positives near openmp parallel for loops.
if(NGRAPH_HE_SANITIZE_ADDRESS)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

# Get library names
set(HE_BACKEND_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(HE_SEAL_CKKS_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_ckks_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(HE_SEAL_BFV_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_bfv_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Install he_backend to the python library where ng-tf loads from
if (ENABLE_TF)
    message("PY_NGRAPH_LIB_DIR ${PY_NGRAPH_LIB_DIR}")
    install(TARGETS he_backend DESTINATION ${PY_NGRAPH_LIB_DIR})

    # Workaround to ensure BFV and CKKS backends are recognized by ng-tf
    # TODO: investigate use of backend configuration strings
    install(FILES ${PY_NGRAPH_LIB_DIR}/${HE_BACKEND_LIB_NAME}
            DESTINATION ${PY_NGRAPH_LIB_DIR}
            RENAME ${HE_SEAL_CKKS_LIB_NAME})

    install(FILES ${PY_NGRAPH_LIB_DIR}/${HE_BACKEND_LIB_NAME}
            DESTINATION ${PY_NGRAPH_LIB_DIR}
            RENAME ${HE_SEAL_BFV_LIB_NAME})

endif()

# Install he-transformer (the HE libraries are installed separately) at `make install`
message("NGRAPH_INSTALL_LIB_DIR" ${NGRAPH_INSTALL_LIB_DIR})

install(TARGETS he_backend DESTINATION ${NGRAPH_INSTALL_LIB_DIR} RENAME ${HE_SEAL_CKKS_LIB_NAME})
install(TARGETS he_backend DESTINATION ${NGRAPH_INSTALL_LIB_DIR} RENAME ${HE_SEAL_BFV_LIB_NAME})
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
