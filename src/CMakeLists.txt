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
    seal/he_seal_backend.cpp
    # main
    he_plain_tensor.cpp
    he_tensor.cpp
    node_wrapper.cpp

    seal/seal_util.cpp
    seal/he_seal_cipher_tensor.cpp


    seal/he_seal_executable.cpp

    # seal kernels
    seal/kernel/constant_seal.cpp
    #seal/kernel/pad_seal.cpp
    #seal/kernel/subtract_seal.cpp
    #seal/kernel/result_seal.cpp
    seal/kernel/add_seal.cpp
    #seal/kernel/multiply_seal.cpp
    #seal/kernel/negate_seal.cpp
    #seal/kernel/subtract_seal.cpp

    # seal backend





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
add_library(he_seal_backend SHARED ${HE_SRC})
add_dependencies(he_seal_backend ext_ngraph_tf ext_seal)
target_link_libraries(he_seal_backend seal libjson ngraph)

if(NGRAPH_HE_SANITIZE_ADDRESS)
    message(STATUS "Enable Address Sanitizer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

# Get library names
set(HE_SEAL_BACKEND_LIB_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}he_seal_backend${CMAKE_SHARED_LIBRARY_SUFFIX}")

message("NGRAPH_TF_VENV_LIB_DIR ${NGRAPH_TF_VENV_LIB_DIR}")

# Install he-transformer (the HE libraries are installed separately) at `make install`
install(TARGETS he_seal_backend DESTINATION ${EXTERNAL_INSTALL_LIB_DIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        DESTINATION ${HE_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")